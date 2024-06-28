.. _ray-remote-classes:
.. _actor-guide:

Actors
======

Actor 扩展了 Ray API 从函数（tasks）到类。
一个 actor 是一个有状态的 worker（或者服务）。
当一个新的 actor 被实例化时，一个新的 worker 被创建，actor 的方法被调度到这个特定的 worker 上，并且可以访问和修改这个 worker 的状态。

.. tab-set::

    .. tab-item:: Python

        ``ray.remote`` 装饰器作用在 ``Counter`` 类上，表示这个类的实例将会是 actor。每个 actor 运行在自己的 Python 进程中。

        .. testcode::

          import ray

          @ray.remote
          class Counter:
              def __init__(self):
                  self.value = 0

              def increment(self):
                  self.value += 1
                  return self.value

              def get_counter(self):
                  return self.value

          # Create an actor from this class.
          counter = Counter.remote()

    .. tab-item:: Java

        ``Ray.actor`` 用于从普通的 Java 类创建 actor。

        .. code-block:: java

          // A regular Java class.
          public class Counter {

            private int value = 0;

            public int increment() {
              this.value += 1;
              return this.value;
            }
          }

          // Create an actor from this class.
          // `Ray.actor` takes a factory method that can produce
          // a `Counter` object. Here, we pass `Counter`'s constructor
          // as the argument.
          ActorHandle<Counter> counter = Ray.actor(Counter::new).remote();

    .. tab-item:: C++

        ``ray::Actor`` 用于从普通的 C++ 类创建 actor。

        .. code-block:: c++

          // A regular C++ class.
          class Counter {

          private:
              int value = 0;

          public:
            int Increment() {
              value += 1;
              return value;
            }
          };

          // Factory function of Counter class.
          static Counter *CreateCounter() {
              return new Counter();
          };

          RAY_REMOTE(&Counter::Increment, CreateCounter);

          // Create an actor from this class.
          // `ray::Actor` takes a factory method that can produce
          // a `Counter` object. Here, we pass `Counter`'s factory function
          // as the argument.
          auto counter = ray::Actor(CreateCounter).Remote();



使用 :ref:`状态 API <state-api-overview-ref>` 的 `ray list actors` 命令可以查看 actor 的状态：

.. code-block:: bash

  # This API is only available when you install Ray with `pip install "ray[default]"`.
  ray list actors

.. code-block:: bash

  ======== List: 2023-05-25 10:10:50.095099 ========
  Stats:
  ------------------------------
  Total: 1
  
  Table:
  ------------------------------
      ACTOR_ID                          CLASS_NAME    STATE      JOB_ID  NAME    NODE_ID                                                     PID  RAY_NAMESPACE
   0  9e783840250840f87328c9f201000000  Counter       ALIVE    01000000          13a475571662b784b4522847692893a823c78f1d3fd8fd32a2624923  38906  ef9de910-64fb-4575-8eb5-50573faa3ddf


指定所需资源
-----------------------------

.. _actor-resource-guide:

你可以在 actor 上指定所需的资源（查看 :ref:`resource-requirements` 获取更多信息）。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Specify required resources for an actor.
            @ray.remote(num_cpus=2, num_gpus=0.5)
            class Actor:
                pass

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources for an actor.
            Ray.actor(Counter::new).setResource("CPU", 2.0).setResource("GPU", 0.5).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources for an actor.
            ray::Actor(CreateCounter).SetResource("CPU", 2.0).SetResource("GPU", 0.5).Remote();


调用 actor
-----------------

我们可以通过调用 actor 的方法并使用 ``remote`` 操作符来与 actor 交互。
然后我们可以调用 ``get`` 方法来获取实际的值。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Call the actor.
            obj_ref = counter.increment.remote()
            print(ray.get(obj_ref))

        .. testoutput::

            1

    .. tab-item:: Java

        .. code-block:: java

            // Call the actor.
            ObjectRef<Integer> objectRef = counter.task(&Counter::increment).remote();
            Assert.assertTrue(objectRef.get() == 1);

    .. tab-item:: C++

        .. code-block:: c++

            // Call the actor.
            auto object_ref = counter.Task(&Counter::increment).Remote();
            assert(*object_ref.Get() == 1);

在不同的 actor 上调用的方法可以并行执行，而在同一个 actor 上调用的方法将按照调用的顺序串行执行。在同一个 actor 上的方法将共享状态，如下所示。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Create ten Counter actors.
            counters = [Counter.remote() for _ in range(10)]

            # Increment each Counter once and get the results. These tasks all happen in
            # parallel.
            results = ray.get([c.increment.remote() for c in counters])
            print(results)

            # Increment the first Counter five times. These tasks are executed serially
            # and share state.
            results = ray.get([counters[0].increment.remote() for _ in range(5)])
            print(results)

        .. testoutput::

            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            [2, 3, 4, 5, 6]

    .. tab-item:: Java

        .. code-block:: java

            // Create ten Counter actors.
            List<ActorHandle<Counter>> counters = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                counters.add(Ray.actor(Counter::new).remote());
            }

            // Increment each Counter once and get the results. These tasks all happen in
            // parallel.
            List<ObjectRef<Integer>> objectRefs = new ArrayList<>();
            for (ActorHandle<Counter> counterActor : counters) {
                objectRefs.add(counterActor.task(Counter::increment).remote());
            }
            // prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            System.out.println(Ray.get(objectRefs));

            // Increment the first Counter five times. These tasks are executed serially
            // and share state.
            objectRefs = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                objectRefs.add(counters.get(0).task(Counter::increment).remote());
            }
            // prints [2, 3, 4, 5, 6]
            System.out.println(Ray.get(objectRefs));

    .. tab-item:: C++

        .. code-block:: c++

            // Create ten Counter actors.
            std::vector<ray::ActorHandle<Counter>> counters;
            for (int i = 0; i < 10; i++) {
                counters.emplace_back(ray::Actor(CreateCounter).Remote());
            }

            // Increment each Counter once and get the results. These tasks all happen in
            // parallel.
            std::vector<ray::ObjectRef<int>> object_refs;
            for (ray::ActorHandle<Counter> counter_actor : counters) {
                object_refs.emplace_back(counter_actor.Task(&Counter::Increment).Remote());
            }
            // prints 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            auto results = ray::Get(object_refs);
            for (const auto &result : results) {
                std::cout << *result;
            }

            // Increment the first Counter five times. These tasks are executed serially
            // and share state.
            object_refs.clear();
            for (int i = 0; i < 5; i++) {
                object_refs.emplace_back(counters[0].Task(&Counter::Increment).Remote());
            }
            // prints 2, 3, 4, 5, 6
            results = ray::Get(object_refs);
            for (const auto &result : results) {
                std::cout << *result;
            }

传递 Actor 句柄
----------------------------

Actor 句柄可以传递给其他任务。我们可以定义使用 actor 句柄的远程函数（或 actor 方法）。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time

            @ray.remote
            def f(counter):
                for _ in range(10):
                    time.sleep(0.1)
                    counter.increment.remote()

    .. tab-item:: Java

        .. code-block:: java

            public static class MyRayApp {

              public static void foo(ActorHandle<Counter> counter) throws InterruptedException {
                for (int i = 0; i < 1000; i++) {
                  TimeUnit.MILLISECONDS.sleep(100);
                  counter.task(Counter::increment).remote();
                }
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            void Foo(ray::ActorHandle<Counter> counter) {
                for (int i = 0; i < 1000; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    counter.Task(&Counter::Increment).Remote();
                }
            }

如果我们实例化一个 actor，我们可以将句柄传递给不同的任务。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            counter = Counter.remote()

            # Start some tasks that use the actor.
            [f.remote(counter) for _ in range(3)]

            # Print the counter value.
            for _ in range(10):
                time.sleep(0.1)
                print(ray.get(counter.get_counter.remote()))

        .. testoutput::
            :options: +MOCK

            0
            3
            8
            10
            15
            18
            20
            25
            30
            30

    .. tab-item:: Java

        .. code-block:: java

            ActorHandle<Counter> counter = Ray.actor(Counter::new).remote();

            // Start some tasks that use the actor.
            for (int i = 0; i < 3; i++) {
              Ray.task(MyRayApp::foo, counter).remote();
            }

            // Print the counter value.
            for (int i = 0; i < 10; i++) {
              TimeUnit.SECONDS.sleep(1);
              System.out.println(counter.task(Counter::getCounter).remote().get());
            }

    .. tab-item:: C++

        .. code-block:: c++

            auto counter = ray::Actor(CreateCounter).Remote();

            // Start some tasks that use the actor.
            for (int i = 0; i < 3; i++) {
              ray::Task(Foo).Remote(counter);
            }

            // Print the counter value.
            for (int i = 0; i < 10; i++) {
              std::this_thread::sleep_for(std::chrono::seconds(1));
              std::cout << *counter.Task(&Counter::GetCounter).Remote().Get() << std::endl;
            }


取消 Actor 任务
----------------------

在返回的 `ObjectRef` 上调用 :func:`ray.cancel() <ray.cancel>` 可以取消 Actor 任务。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/actors.py
            :language: python
            :start-after: __cancel_start__
            :end-before: __cancel_end__


在 Ray 中，任务取消行为取决于任务的当前状态：

**未调度的任务**: 
如果 Actor 任务还没有被调度，Ray 会尝试取消调度。
当在这个阶段成功取消时，调用 ``ray.get(actor_task_ref)`` 会
产生一个 :class:`TaskCancelledError <ray.exceptions.TaskCancelledError>`。

**运行中的 Actor 任务（普通 Actor，线程 Actor）**:
针对单进程和多线程类别的 Actor，Ray 提供了中断机制。

**运行中的异步 Actor 任务**:
对于 `异步 Actor <_async-actors>` 类别的任务，Ray 会尝试取消关联的 `asyncio.Task`。
这些取消方法与 `asyncio 任务取消 <https://docs.python.org/3/library/asyncio-task.html#task-cancellation>`__ 中的标准一致。
注意，如果在异步函数中没有 `await`，`asyncio.Task` 不会在执行中间被中断。

**取消保证**:
Ray 尝试取消任务是 *尽力而为* 的，这意味着取消并不总是保证的。
例如，如果取消请求没有传递给执行器，
任务可能不会被取消。
你可以检查任务是否成功取消，使用 ``ray.get(actor_task_ref)``。

**递归取消**:
Ray 跟踪所有子任务和 Actor 任务。当给出 ``recursive=True`` 参数时，
它取消所有子任务和 Actor 任务。

调度
----------

对于每个 actor，Ray 会选择一个节点来运行它，调度决策基于一些因素，如 
:ref:`actor 的资源需求 <ray-scheduling-resources>` 
和 :ref:`指定的调度策略 <ray-scheduling-strategies>`。
有关更多详细信息，请参见 :ref:`Ray 调度 <ray-scheduling>`。

容错能力
---------------

默认情况下，Ray actor 不会被 :ref:`重启 <fault-tolerance-actors>`，
actor 任务不会在 actor 意外崩溃时重试。
你可以通过在 :func:`ray.remote() <ray.remote>` 和 :meth:`.options() <ray.actor.ActorClass.options>` 
中设置 ``max_restarts`` 和 ``max_task_retries`` 选项来更改此行为。
参考 :ref:`Ray fault tolerance <fault-tolerance>` 获取更多信息。

问答：Actor，Worker 以及资源
----------------------------------

worker 和 actor 之间有何不同？

每一个 "Ray worker" 都是一个 Python 进程。

Worker 却别于任务和 actor。任何 "Ray worker" 都是 1. 要么用于执行多个 Ray 任务，2. 作为一个专用的 Ray actor 启动。

* **Tasks**: 当 Ray 在一台机器上启动时，会自动启动一些 Ray workers（默认情况下每个 CPU 一个）。它们将被用于执行任务（类似于进程池）。如果你使用 `num_cpus=2` 执行 8 个任务，而总的 CPU 数量是 16 (`ray.cluster_resources()["CPU"] == 16`)，16 个 worker 将会有 8 个闲置。

* **Actor**: 
一个 Ray Actor 也是一个 "Ray worker"，但是在运行时实例化（在 `actor_cls.remote()` 时）。所有的方法都将在同一个进程中运行，使用相同的资源（在定义 Actor 时指定）。请注意，与任务不同，运行 Ray Actor 的 Python 进程不会被重用，并且在 Actor 被删除时将被终止。

要最大化利用资源，你需要最大化 worker 的工作时间。
你还需要分配足够的集群资源，以便你所有需要的 actor 都能运行，
以及你定义的任何其他任务都能运行。这也意味着任务的调度更加灵活，
如果你不需要 actor 的有状态部分，你大多数情况下最好使用任务。

更多关于 Ray Actor 的信息
---------------------

.. toctree::
    :maxdepth: 1

    actors/named-actors.rst
    actors/terminating-actors.rst
    actors/async_api.rst
    actors/concurrency_group_api.rst
    actors/actor-utils.rst
    actors/out-of-band-communication.rst
    actors/task-orders.rst
