Actor 的 AsyncIO / 并发
================================

在单个 actor 进程中，可以执行并发线程。

Ray 提供了两种 actor 内并发的方式：

 * :ref:`异步执行 <async-actors>`
 * :ref:`线程 <threaded-actors>`


请记住，Python 的 `全局解释器锁 (GIL) <https://wiki.python.org/moin/GlobalInterpreterLock>`_ 一次只允许一个线程的 Python 代码运行。

这意味着如果你只是并行化 Python 代码，你不会得到真正的并行性。如果你调用 Numpy、Cython、Tensorflow 或 PyTorch 代码，这些库在调用 C/C++ 函数时会释放 GIL。

:ref:`threaded-actors` 和 :ref:`async-actors` **模型都不允许你绕过 GIL 。**

.. _async-actors:

AsyncIO Actor
------------------

从 Python 3.5 开始，可以使用 ``async/await`` `语法 <https://docs.python.org/3/library/asyncio.html>`__ 编写并发代码。
Ray 与 asyncio 原生无缝集成。
你可以将 ray 与流行的异步框架一起使用，如 aiohttp、aioredis 等。

.. testcode::

    import ray
    import asyncio

    @ray.remote
    class AsyncActor:
        # multiple invocation of this method can be running in
        # the event loop at the same time
        async def run_concurrent(self):
            print("started")
            await asyncio.sleep(2) # concurrent workload here
            print("finished")

    actor = AsyncActor.remote()

    # regular ray.get
    ray.get([actor.run_concurrent.remote() for _ in range(4)])

    # async ray.get
    async def async_get():
        await actor.run_concurrent.remote()
    asyncio.run(async_get())

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished

.. testcode::
    :hide:

    # 注意：前面代码块的输出可能会出现在后续测试中。
    # 要防止不稳定性，我们等待异步调用完成。
    import time
    print("Sleeping...")
    time.sleep(3)

.. testoutput::

    ...

ObjectRefs 作为 asyncio.Futures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ObjectRefs 可以转换成 asyncio.Futures。
这使得在现有的并发应用中可以 ``await`` 的 ray 特性成为可能。

代替：

.. testcode::

    import ray

    @ray.remote
    def some_task():
        return 1

    ray.get(some_task.remote())
    ray.wait([some_task.remote()])

你可以这样：

.. testcode::

    import ray
    import asyncio

    @ray.remote
    def some_task():
        return 1

    async def await_obj_ref():
        await some_task.remote()
        await asyncio.wait([some_task.remote()])

    asyncio.run(await_obj_ref())

请参阅 `asyncio 文档 <https://docs.python.org/3/library/asyncio-task.html>`__
了解更多 `asyncio` 模式，包括超时控制和 ``asyncio.gather``。

如果需要直接访问未来对象，可以调用：

.. testcode::

    import asyncio

    async def convert_to_asyncio_future():
        ref = some_task.remote()
        fut: asyncio.Future = asyncio.wrap_future(ref.future())
        print(await fut)
    asyncio.run(convert_to_asyncio_future())

.. testoutput::

    1

.. _async-ref-to-futures:

ObjectRefs 作为 concurrent.futures.Futures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ObjectRefs 也可以被包装到 ``concurrent.futures.Future`` 对象。
这对于与现有API ``concurrent.futures`` 交互非常有用：

.. testcode::

    import concurrent

    refs = [some_task.remote() for _ in range(4)]
    futs = [ref.future() for ref in refs]
    for fut in concurrent.futures.as_completed(futs):
        assert fut.done()
        print(fut.result())

.. testoutput::

    1
    1
    1
    1

定义异步 Actor
~~~~~~~~~~~~~~~~~~~~~~~

使用 `async` 方法定义。Ray 会自动检测 actor 是否支持 `async` 调用。

.. testcode::

    import asyncio

    @ray.remote
    class AsyncActor:
        async def run_task(self):
            print("started")
            await asyncio.sleep(2) # Network, I/O task here
            print("ended")

    actor = AsyncActor.remote()
    # All 5 tasks should start at once. After 2 second they should all finish.
    # they should finish at the same time
    ray.get([actor.run_task.remote() for _ in range(5)])

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended

在底层，Ray 在单个 Python 事件循环中运行所有方法。
请注意，不允许在异步 actor 方法中运行阻塞的 ``ray.get`` 或 ``ray.wait``，因为 ``ray.get`` 会阻塞事件循环的执行。

在异步 actor 中，一次只能运行一个任务（尽管任务可以多路复用）。在 AsyncActor 中只有一个线程！如果你想要一个线程池，请参阅 :ref:`threaded-actors`。

在异步 actor 设置并发
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你可以通过 ``max_concurrency`` 选项来设置并发任务的数量。
默认情况下，可以同时运行 1000 个任务。

.. testcode::

    import asyncio

    @ray.remote
    class AsyncActor:
        async def run_task(self):
            print("started")
            await asyncio.sleep(1) # Network, I/O task here
            print("ended")

    actor = AsyncActor.options(max_concurrency=2).remote()

    # 只有两个任务会并发运行，一旦这两个任务结束，下两个才能运行
    ray.get([actor.run_task.remote() for _ in range(8)])

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended

.. _threaded-actors:

线程 Actors
---------------

有时，你可能需要一个线程池来执行多个任务。
比如，你可能有一个方法执行一些计算密集型任务，而不释放事件循环的控制权。这会影响异步 Actor 的性能，因为异步 Actor 一次只能执行一个任务，并依赖 ``await`` 来切换上下文。


相反，你可以使用 ``max_concurrency`` Actor 选项，而不使用任何异步方法，从而实现线程并发（如线程池）。


.. warning::
    当 actor 定义中至少有一个 ``async def`` 方法时，Ray 会将 actor 识别为 AsyncActor 而不是 ThreadedActor。


.. testcode::

    @ray.remote
    class ThreadedActor:
        def task_1(self): print("I'm running in a thread!")
        def task_2(self): print("I'm running in another thread!")

    a = ThreadedActor.options(max_concurrency=2).remote()
    ray.get([a.task_1.remote(), a.task_2.remote()])

.. testoutput::
    :options: +MOCK

    (ThreadedActor pid=4822) I'm running in a thread!
    (ThreadedActor pid=4822) I'm running in another thread!

线程 Actor 的每次调用都将在线程池中运行。线程池的大小受该 ``max_concurrency`` 值限制。

用于远程任务的 AsyncIO
------------------------

我们不支持使用 asyncio 执行远程任务。以下代码片段将失败：

.. testcode::
    :skipif: True

    @ray.remote
    async def f():
        pass

相反，您可以使用 ``async`` 包装该函数来同步运行该任务：

.. testcode::

    async def f():
        pass

    @ray.remote
    def wrapper():
        import asyncio
        asyncio.run(f())
