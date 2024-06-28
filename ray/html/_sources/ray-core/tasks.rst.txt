.. _ray-remote-functions:

任务
=====

Ray 允许在单独的 Python worker 上异步执行任意函数。此类函数称为 **Ray 远程函数** ，其异步调用称为 **Ray 任务**。以下是示例。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __tasks_start__
            :end-before: __tasks_end__

        参阅 `ray.remote 包参考 <package-ref.html>`__ 页的具体文档来查看如何使用 ``ray.remote``。

    .. tab-item:: Java

        .. code-block:: java

          public class MyRayApp {
            // A regular Java static method.
            public static int myFunction() {
              return 1;
            }
          }

          // Invoke the above method as a Ray task.
          // This will immediately return an object ref (a future) and then create
          // a task that will be executed on a worker process.
          ObjectRef<Integer> res = Ray.task(MyRayApp::myFunction).remote();

          // The result can be retrieved with ``ObjectRef::get``.
          Assert.assertTrue(res.get() == 1);

          public class MyRayApp {
            public static int slowFunction() throws InterruptedException {
              TimeUnit.SECONDS.sleep(10);
              return 1;
            }
          }

          // Ray tasks are executed in parallel.
          // All computation is performed in the background, driven by Ray's internal event loop.
          for(int i = 0; i < 4; i++) {
            // This doesn't block.
            Ray.task(MyRayApp::slowFunction).remote();
          }

    .. tab-item:: C++

        .. code-block:: c++

          // A regular C++ function.
          int MyFunction() {
            return 1;
          }
          // Register as a remote function by `RAY_REMOTE`.
          RAY_REMOTE(MyFunction);

          // Invoke the above method as a Ray task.
          // This will immediately return an object ref (a future) and then create
          // a task that will be executed on a worker process.
          auto res = ray::Task(MyFunction).Remote();

          // The result can be retrieved with ``ray::ObjectRef::Get``.
          assert(*res.Get() == 1);

          int SlowFunction() {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            return 1;
          }
          RAY_REMOTE(SlowFunction);

          // Ray tasks are executed in parallel.
          // All computation is performed in the background, driven by Ray's internal event loop.
          for(int i = 0; i < 4; i++) {
            // This doesn't block.
            ray::Task(SlowFunction).Remote();
          a

从 :ref:`State API <state-api-overview-ref>` 使用 `ray summary tasks` 来查看运行中以及已完成的任务及总数：

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray summary tasks


.. code-block:: bash

  ======== Tasks Summary: 2023-05-26 11:09:32.092546 ========
  Stats:
  ------------------------------------
  total_actor_scheduled: 0
  total_actor_tasks: 0
  total_tasks: 5
  
  
  Table (group by func_name):
  ------------------------------------
      FUNC_OR_CLASS_NAME    STATE_COUNTS    TYPE
  0   slow_function         RUNNING: 4      NORMAL_TASK
  1   my_function           FINISHED: 1     NORMAL_TASK

指定所需资源
-----------------------------

您可以在任务中指定资源需求（有关更多详细信息，请参阅 :ref:`resource-requirements`。）

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __resource_start__
            :end-before: __resource_end__

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources.
            Ray.task(MyRayApp::myFunction).setResource("CPU", 4.0).setResource("GPU", 2.0).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources.
            ray::Task(MyFunction).SetResource("CPU", 4.0).SetResource("GPU", 2.0).Remote();

.. _ray-object-refs:

将对象引用传递给 Ray 任务
---------------------------------------

除了值之外，`对象引用 <objects.html>`__ 还可以传递到远程函数中。执行任务时，函数体内的 **参数将是底层值**。例如，以下函数：

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __pass_by_ref_start__
            :end-before: __pass_by_ref_end__

    .. tab-item:: Java

        .. code-block:: java

            public class MyRayApp {
                public static int functionWithAnArgument(int value) {
                    return value + 1;
                }
            }

            ObjectRef<Integer> objRef1 = Ray.task(MyRayApp::myFunction).remote();
            Assert.assertTrue(objRef1.get() == 1);

            // You can pass an object ref as an argument to another Ray task.
            ObjectRef<Integer> objRef2 = Ray.task(MyRayApp::functionWithAnArgument, objRef1).remote();
            Assert.assertTrue(objRef2.get() == 2);

    .. tab-item:: C++

        .. code-block:: c++

            static int FunctionWithAnArgument(int value) {
                return value + 1;
            }
            RAY_REMOTE(FunctionWithAnArgument);

            auto obj_ref1 = ray::Task(MyFunction).Remote();
            assert(*obj_ref1.Get() == 1);

            // You can pass an object ref as an argument to another Ray task.
            auto obj_ref2 = ray::Task(FunctionWithAnArgument).Remote(obj_ref1);
            assert(*obj_ref2.Get() == 2);

请注意以下行为：

  -  由于第二个任务依赖于第一个任务的输出，因此 Ray 只有在第一个任务完成后才会执行第二个任务。
  -  如果两个任务在不同的机器上调度，那么第一个任务的输出（对应的值 ``obj_ref1/objRef1``）
     将通过网络发送到调度第二个任务的机器。

等待部分结果
---------------------------

在 Ray 任务结果上调用 **ray.get** 将阻塞，直到任务完成执行。启动多个任务后，
您可能想知道哪些任务已完成执行，而无需阻塞所有任务。这可以通过 :func:`ray.wait() <ray.wait>` 来实现。该函数的工作原理如下：

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __wait_start__
            :end-before: __wait_end__

    .. tab-item:: Java

      .. code-block:: java

        WaitResult<Integer> waitResult = Ray.wait(objectRefs, /*num_returns=*/0, /*timeoutMs=*/1000);
        System.out.println(waitResult.getReady());  // List of ready objects.
        System.out.println(waitResult.getUnready());  // list of unready objects.

    .. tab-item:: C++

      .. code-block:: c++

        ray::WaitResult<int> wait_result = ray::Wait(object_refs, /*num_objects=*/0, /*timeout_ms=*/1000);

.. _ray-task-returns:

多返回值
----------------

默认情况下，Ray 任务仅返回单个 Object Ref。但是，您可以通过 ``num_returns`` 设置选项将 Ray 任务配置为返回多个 Object Ref。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __multiple_returns_start__
            :end-before: __multiple_returns_end__

对于返回多个对象的任务，Ray 还支持远程生成器，允许任务一次返回一个对象，以减少工作器的内存使用量。Ray 还支持动态设置返回值数量的选项，当任务调用者不知道预期返回值的数量时，此选项非常有用。有关用例的更多详细信息，请参阅 :ref:`用户指南 <generators>`。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __generator_start__
            :end-before: __generator_end__


任务取消
----------------

可以通过对返回的 Object ref 调用 :func:`ray.cancel() <ray.cancel>` 来取消 Ray 任务。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __cancel_start__
            :end-before: __cancel_end__


调度
----------

对于每个任务，Ray 将选择一个节点来运行它，并且调度决策基于一些因素，如 

:ref:`指定的调度策略 <ray-scheduling-strategies>`
和 :ref:`任务参数的位置 <ray-scheduling-locality>`。

有关更多详细信息，请参阅 :ref:`Ray 调度 <ray-scheduling>`。

容错
---------------

默认情况下，Ray 将 :ref:`重试 <task-retries>` 由于系统故障和指定的应用程序级故障而失败的任务。
你可以通过在 :func:`ray.remote() <ray.remote>` 和 :meth:`.options() <ray.remote_function.RemoteFunction.options>` 中设置 ``max_retries`` 和 ``retry_exceptions`` 选项来更改此行为。
参考 :ref:`Ray 容错 <fault-tolerance>` 以获取更多详细信息。


关于 Ray Tasks 的更多信息
--------------------

.. toctree::
    :maxdepth: 1

    tasks/nested-tasks.rst
    tasks/generators.rst
