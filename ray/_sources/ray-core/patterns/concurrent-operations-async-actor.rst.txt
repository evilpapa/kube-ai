模式：使用 asyncio 并发运行 actor 方法
========================================================

默认情况下，Ray :ref:`actor <ray-remote-classes>` 在单个线程中运行，Actor 方法调用按顺序执行。
这意味着长时间运行的方法调用会阻止所有后续方法调用。
在此模式中，我们使用 ``await`` 从长时间运行的方法调用中让出控制权，以便其他方法调用可以同时运行。
通常，当方法执行 IO 操作时，会让出控制权，但您也可以使用 ``await asyncio.sleep(0)`` 显式让出控制权。

.. note::
   您还可以使用 :ref:`threaded actors <threaded-actors>` 来实现并发。

用例
----------------

您有一个采用长轮询方法的 Actor，该方法会不断从远程存储中获取任务并执行它们。
您还想查询长轮询方法运行时执行的任务数。

使用默认 actor 时，代码将如下所示：

.. literalinclude:: ../doc_code/pattern_async_actor.py
    :language: python
    :start-after: __sync_actor_start__
    :end-before: __sync_actor_end__

这是有问题的，因为 ``TaskExecutor.run`` 方法会永远运行，并且永远不会交出控制权来运行其他方法。
以通过使用 :ref:`async actors <async-actors>` 并用 ``await`` 交出控制权来解决这个问题：

.. literalinclude:: ../doc_code/pattern_async_actor.py
    :language: python
    :start-after: __async_actor_start__
    :end-before: __async_actor_end__

这里，我们不是使用阻塞的 :func:`ray.get() <ray.get>`来获取 ObjectRef 的值，我们使用 ``await`` 来在等待对象被获取时产生控制权。

