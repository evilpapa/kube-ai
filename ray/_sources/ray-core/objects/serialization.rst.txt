.. _serialization-guide:

序列化
=============

由于 Ray 进程不共享内存空间，因此在工作程序和节点之间传输的数据需要 **序列化** 和 **反序列化**。Ray 使用 `Plasma 对象存储  <https://arrow.apache.org/blog/2017/08/08/plasma-in-memory-object-store/>`_ 来高效地跨不同进程和不同节点传输对象。对象存储中的 Numpy 数组在同一节点上的工作程序之间共享（零拷贝反序列化）。

概述
--------

Ray 已决定使用自定义的 `Pickle 协议版本 5 <https://www.python.org/dev/peps/pep-0574/>`_ 向移植来替换原始的 PyArrow 序列化器。这消除了之前的几个限制（例如无法序列化递归对象）。

Ray 目前兼容 Pickle 协议版本 5，同时 Ray 借助 cloudpickle 支持更广泛的对象（例如 lambda & 嵌套函数、动态类）的序列化。

.. _plasma-store:

Plasma 对象存储
~~~~~~~~~~~~~~~~~~~

Plasma 是一种内存对象存储。它最初是作为 Apache Arrow 的一部分开发的。在 Ray 的 1.0.0 版本发布之前，Ray 将 Arrow 的 Plasma 代码分叉到 Ray 的代码库中，以便根据 Ray 的架构和性能需求分解并继续开发。

Plasma 用于在不同进程和不同节点之间高效传输对象。Plasma 对象存储中的所有对象都是 **不可变的**，并保存在共享内存中。这样，同一节点上的许多工作程序就可以高效地访问它们。

每个节点都有自己的对象存储。将数据放入对象存储后，不会自动广播到其他节点。数据将保留在写入器本地，直到另一个节点上的另一个task 或 actor请求。

Numpy 数组
~~~~~~~~~~~~

Ray 通过使用带外数据的 Pickle 协议 5 来优化 numpy 数组。
numpy 数组存储为只读对象，同一节点上的所有 Ray 工作进程都可以读取对象存储中的 numpy 数组而无需复制（零复制读取）。工作进程中的每个 numpy 数组对象都持有指向共享内存中保存的相关数组的指针。对只读对象的任何写入都需要用户先将其复制到本地进程内存中。

.. tip:: 您通常可以通过仅使用本机类型（例如，numpy 数组或 numpy 数组的列表/字典和其他原始类型）或通过使用 Actors 保存无法序列化的对象来避免序列化问题。

修复“分配目标是只读的” 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由于 Ray 将 numpy 数组放在对象存储中，因此当在远程函数中反序列化为参数时，它们将变为只读。例如，以下代码片段将崩溃：

.. literalinclude:: /ray-core/doc_code/deser.py

为避免此问题，如果您需要改变数组，则可以在目标位置手动复制该数组（``arr = arr.copy()``）。请注意，这实际上就像禁用 Ray 提供的零拷贝反序列化特性。

序列化说明
-------------------

- Ray 目前正在使用 Pickle 协议版本 5。大多数 Python 发行版使用的默认 Pickle 协议是协议 3。对于较大的对象，协议 4 和 5 比协议 3 更有效。

- 对于非本机对象，即使在对象中被多次引用，Ray 也会始终保留一份副本：

  .. testcode::

    import ray
    import numpy as np

    obj = [np.zeros(42)] * 99
    l = ray.get(ray.put(obj))
    assert l[0] is l[1]  # no problem!

- 只要有可能，就使用 numpy 数组或 numpy 数组的 Python 集合来获得最佳性能。

- 锁对象大多不可序列化，因为复制锁毫无意义，并且可能导致严重的并发问题。如果您的对象包含锁，您可能必须想出一种解决方法。

自定义序列化
------------------------

有时您可能想要自定义序列化过程，
因为 Ray 使用的默认序列化器（pickle5 + cloudpickle）
不适合您（无法序列化某些对象、对于某些对象来说太慢等）。

至少有 3 种方法可以定义自定义序列化过程：

1. 如果你想要自定义某类对象的序列化，并且有代码可读，那么可以在对应的类中定义 ``__reduce__``函数。
   大多数 Python 库都这样做。
   示例代码：

   .. testcode::

     import ray
     import sqlite3

     class DBConnection:
         def __init__(self, path):
             self.path = path
             self.conn = sqlite3.connect(path)

         # without '__reduce__', the instance is unserializable.
         def __reduce__(self):
             deserializer = DBConnection
             serialized_data = (self.path,)
             return deserializer, serialized_data

     original = DBConnection("/tmp/db")
     print(original.conn)

     copied = ray.get(ray.put(original))
     print(copied.conn)

  .. testoutput::

    <sqlite3.Connection object at ...>
    <sqlite3.Connection object at ...>


2. 如果你想要自定义一类对象的序列化，
   但是又不能访问或者修改对应的类，
   那么你可以向你使用的序列化器注册该类：

   .. testcode::

      import ray
      import threading

      class A:
          def __init__(self, x):
              self.x = x
              self.lock = threading.Lock()  # could not be serialized!

      try:
        ray.get(ray.put(A(1)))  # fail!
      except TypeError:
        pass

      def custom_serializer(a):
          return a.x

      def custom_deserializer(b):
          return A(b)

      # Register serializer and deserializer for class A:
      ray.util.register_serializer(
        A, serializer=custom_serializer, deserializer=custom_deserializer)
      ray.get(ray.put(A(1)))  # success!

      # You can deregister the serializer at any time.
      ray.util.deregister_serializer(A)
      try:
        ray.get(ray.put(A(1)))  # fail!
      except TypeError:
        pass

      # Nothing happens when deregister an unavailable serializer.
      ray.util.deregister_serializer(A)

   注意：序列化器是针对每个 Ray 工作器进行本地管理的。
   因此，对于每个 Ray 工作器，如果您想使用序列化器，则需要注册序列化器。
   取消注册序列化器也仅适用于本地。

   如果您为某个类注册了新的序列化器，
   则新的序列化器会立即在工作器中替换旧的序列化器。
   此 API 也是幂等的，重新注册相同的序列化器不会产生任何副作用。

3. 如果您想自定义特定对象的序列化，我们还为您提供了一个示例：

   .. testcode::

     import threading

     class A:
         def __init__(self, x):
             self.x = x
             self.lock = threading.Lock()  # could not serialize!

     try:
        ray.get(ray.put(A(1)))  # fail!
     except TypeError:
        pass

     class SerializationHelperForA:
         """A helper class for serialization."""
         def __init__(self, a):
             self.a = a

         def __reduce__(self):
             return A, (self.a.x,)

     ray.get(ray.put(SerializationHelperForA(A(1))))  # success!
     # the serializer only works for a specific object, not all A
     # instances, so we still expect failure here.
     try:
        ray.get(ray.put(A(1)))  # still fail!
     except TypeError:
        pass


故障排除
---------------

使用 ``ray.util.inspect_serializability`` 识别棘手的 pickling 问题。此函数可用于跟踪任何 Python 对象中潜在的不可序列化对象 - 无论是函数、类还是对象实例。

下面，我们在具有不可序列化对象（线程锁）的函数上演示这种行为：

.. testcode::

    from ray.util import inspect_serializability
    import threading

    lock = threading.Lock()

    def test():
        print(lock)

    inspect_serializability(test, name="test")

输出结果为：

.. testoutput::
  :options: +MOCK

    =============================================================
    Checking Serializability of <function test at 0x7ff130697e50>
    =============================================================
    !!! FAIL serialization: cannot pickle '_thread.lock' object
    Detected 1 global variables. Checking serializability...
        Serializing 'lock' <unlocked _thread.lock object at 0x7ff1306a9f30>...
        !!! FAIL serialization: cannot pickle '_thread.lock' object
        WARNING: Did not find non-serializable object in <unlocked _thread.lock object at 0x7ff1306a9f30>. This may be an oversight.
    =============================================================
    Variable:

    	FailTuple(lock [obj=<unlocked _thread.lock object at 0x7ff1306a9f30>, parent=<function test at 0x7ff130697e50>])

    was found to be non-serializable. There may be multiple other undetected variables that were non-serializable.
    Consider either removing the instantiation/imports of these variables or moving the instantiation into the scope of the function/class.
    =============================================================
    Check https://docs.ray.io/en/master/ray-core/objects/serialization.html#troubleshooting for more information.
    If you have any suggestions on how to improve this error message, please reach out to the Ray developers on github.com/ray-project/ray/issues/
    =============================================================

如需更详细的信息，请在导入 Ray 之前设置环境变量 ``RAY_PICKLE_VERBOSE_DEBUG='2'``。
这样就可以使用基于 Python 的后端（而不是 C-Pickle）进行序列化，因此您可以在序列化过程中调试 Python 代码。但是，这会使序列化速度慢得多。

已知的问题
------------

用户在使用某些 python3.8 和 3.9 版本时可能会遇到内存泄漏。这是由于 `python 的 pickle 模块中存在错误 <https://bugs.python.org/issue39492>`_ 造成的。

该问题已在 Python 3.8.2rc1、Python 3.9.0 alpha 4 或更高版本中得到解决。
