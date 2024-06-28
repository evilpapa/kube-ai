实用类
===============

Actor Pool
~~~~~~~~~~

.. tab-set::

    .. tab-item:: Python

        ``ray.util`` 模块包含了一个实用类 ``ActorPool``。
        这个类类似于 ``multiprocessing.Pool``，可以在固定的 actor 池中调度 Ray 任务。

        .. literalinclude:: ../doc_code/actor-pool.py

        参阅 :class:`包参考 <ray.util.ActorPool>` 获取更多信息。

    .. tab-item:: Java

        Actor pool 还未在 Java 实现。

    .. tab-item:: C++

        Actor pool 还未在 C++ 实现

使用 Ray Queue 传递消息
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

有时仅使用一个信号进行同步是不够的。如果需要在多个任务或 actor 之间发送数据，
可以使用 :class:`ray.util.queue.Queue <ray.util.queue.Queue>`。

.. literalinclude:: ../doc_code/actor-queue.py

Ray 的 Queue API 同 Python 的 ``asyncio.Queue`` 和 ``queue.Queue`` API 类似。
