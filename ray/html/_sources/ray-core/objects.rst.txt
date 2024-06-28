.. _objects-in-ray:

对象
=======

在 Ray 中，任务和 actor 创建和计算对象。
我们将这些对象称为 **远程对象** 因为他们可能存储在 Ray 集群的任何位置，我们使用 **对象引用** 来引入他们。 
远程对象缓存在 Ray 的分布式 `共享内存 <https://en.wikipedia.org/wiki/Shared_memory>`__ **对象存储** 中，集群中的每个节点都有一个对象存储。
在集群设置中，一个远程对象可以存在于一个或多个节点上，与持有对象引用的节点无关。

**对象引用** 是一个指针或唯一 ID，
可以用来引用远程对象而不需要看到它的值。
如果你熟悉 futures，Ray 对象引用在概念上是类似的。

对象应用可以通过两种方式创建：

  1. 他们通过远程方法调用返回。
  2. 他们通过 :func:`ray.put() <ray.put>` 返回。

.. tab-set::

    .. tab-item:: Python

      .. testcode::

        import ray

        # Put an object in Ray's object store.
        y = 1
        object_ref = ray.put(y)

    .. tab-item:: Java

      .. code-block:: java

        // Put an object in Ray's object store.
        int y = 1;
        ObjectRef<Integer> objectRef = Ray.put(y);

    .. tab-item:: C++

      .. code-block:: c++

        // Put an object in Ray's object store.
        int y = 1;
        ray::ObjectRef<int> object_ref = ray::Put(y);

.. note::

    远程对象是不可变的。
    也就是说，他们的值在创建后不能被改变。
    这允许远程对象在多个对象存储中复制而无需同步副本。


获取对象数据
--------------------

你可以传递一个对象引用或对象引用列表给 ``ray.get``。
如果当前节点的对象存储中不包含对象，对象将被下载。

.. tab-set::

    .. tab-item:: Python

        如果对象是一个 `numpy 数组 <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`__ 或者是
        一个 numpy 数组的集合，``get`` 调用是零拷贝的，返回的数组是由共享对象存储内存支持的。
        否则，我们将对象数据反序列化为 Python 对象。

        .. testcode::

          import ray
          import time

          # Get the value of one object ref.
          obj_ref = ray.put(1)
          assert ray.get(obj_ref) == 1

          # Get the values of multiple object refs in parallel.
          assert ray.get([ray.put(i) for i in range(3)]) == [0, 1, 2]

          # You can also set a timeout to return early from a ``get``
          # that's blocking for too long.
          from ray.exceptions import GetTimeoutError
          # ``GetTimeoutError`` is a subclass of ``TimeoutError``.

          @ray.remote
          def long_running_function():
              time.sleep(8)

          obj_ref = long_running_function.remote()
          try:
              ray.get(obj_ref, timeout=4)
          except GetTimeoutError:  # You can capture the standard "TimeoutError" instead
              print("`get` timed out.")

        .. testoutput::

          `get` timed out.

    .. tab-item:: Java

        .. code-block:: java

          // Get the value of one object ref.
          ObjectRef<Integer> objRef = Ray.put(1);
          Assert.assertTrue(objRef.get() == 1);
          // You can also set a timeout(ms) to return early from a ``get`` that's blocking for too long.
          Assert.assertTrue(objRef.get(1000) == 1);

          // Get the values of multiple object refs in parallel.
          List<ObjectRef<Integer>> objectRefs = new ArrayList<>();
          for (int i = 0; i < 3; i++) {
            objectRefs.add(Ray.put(i));
          }
          List<Integer> results = Ray.get(objectRefs);
          Assert.assertEquals(results, ImmutableList.of(0, 1, 2));

          // Ray.get timeout example: Ray.get will throw an RayTimeoutException if time out.
          public class MyRayApp {
            public static int slowFunction() throws InterruptedException {
              TimeUnit.SECONDS.sleep(10);
              return 1;
            }
          }
          Assert.assertThrows(RayTimeoutException.class,
            () -> Ray.get(Ray.task(MyRayApp::slowFunction).remote(), 3000));

    .. tab-item:: C++

        .. code-block:: c++

          // Get the value of one object ref.
          ray::ObjectRef<int> obj_ref = ray::Put(1);
          assert(*obj_ref.Get() == 1);

          // Get the values of multiple object refs in parallel.
          std::vector<ray::ObjectRef<int>> obj_refs;
          for (int i = 0; i < 3; i++) {
            obj_refs.emplace_back(ray::Put(i));
          }
          auto results = ray::Get(obj_refs);
          assert(results.size() == 3);
          assert(*results[0] == 0);
          assert(*results[1] == 1);
          assert(*results[2] == 2);

传递对象参数
------------------------

Ray 对象应用可以自由地在 Ray 应用程序中传递。这意味着他们可以作为任务、actor 方法的参数传递，甚至存储在其他对象中。对象通过 *分布式引用计数* 跟踪，一旦对象的所有引用被删除，对象数据将自动释放。

这里有两种不同的方式可以将对象传递给 Ray 任务或方法。根据对象传递的方式，Ray 将决定是否在任务执行之前对对象进行 *解引用*。

**将对象作为顶级参数传递**: 当一个对象直接作为一个顶级参数传递给任务时，Ray 会解引用对象。这意味着 Ray 将获取所有顶级对象引用参数的底层数据，直到对象数据完全可用。

.. literalinclude:: doc_code/obj_val.py

**将对象作为嵌套参数传递**: When an object is passed within a nested object, 当对象在欠怼对象内被传递，比如，在一个 Python 列表内，Ray *不会* 解引用它。这意味着任务需要调用 ``ray.get()`` 来获取具体值。然而，如果任务从不调用 ``ray.get()``，那么对象值永远不需要传输到任务所在的机器上。我们推荐尽可能将对象作为顶级参数传递，但是嵌套参数可以用于将对象传递给其他任务而不需要查看数据。

.. literalinclude:: doc_code/obj_ref.py

顶级与非顶级传递约定也适用于 actor 构造函数和 actor 方法调用:

.. testcode::

    @ray.remote
    class Actor:
      def __init__(self, arg):
        pass

      def method(self, arg):
        pass

    obj = ray.put(2)

    # 传递给 actor 构造函数的示例
    actor_handle = Actor.remote(obj)  # by-value
    actor_handle = Actor.remote([obj])  # by-reference

    # 传递给 actor 方法的示例
    actor_handle.method.remote(obj)  # by-value
    actor_handle.method.remote([obj])  # by-reference

对象的闭包捕获
--------------------------

你可以通过 *闭包捕获* 将对象传递给任务。这在你有一个大对象想要在多个任务或 actor 之间共享时非常方便，而不想重复传递它。但是要注意，定义一个闭包捕获对象引用的任务会通过引用计数固定对象，因此对象将不会被释放直到任务完成。

.. literalinclude:: doc_code/obj_capture.py

嵌套对象引用
--------------

Ray 也支持嵌套对象引用。这允许你构建复合对象，这些对象本身持有对更多子对象的引用。

.. testcode::

    # 对象可相互在其内部嵌套。Ray 通过引用技术会保持其内部对象可用
    # 直到所有外部的对象引用被删除
    object_ref_2 = ray.put([object_ref])

容错
---------------

Ray 可以通过 :ref:`lineage reconstruction <fault-tolerance-objects-reconstruction>` 自动恢复对象数据丢失，但不能从 :ref:`所有者 <fault-tolerance-ownership>` 故障中恢复。
有关更多详细信息，请参阅 :ref:`Ray 容错 <fault-tolerance>` 。

关于 Ray 对象的更多信息
----------------------

.. toctree::
    :maxdepth: 1

    objects/serialization.rst
    objects/object-spilling.rst
