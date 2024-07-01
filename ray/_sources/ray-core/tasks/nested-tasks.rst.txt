嵌套远程函数
=======================

远程函数可以调用其他远程函数，从而产生嵌套任务。例如，考虑以下示例。

.. literalinclude:: ../doc_code/nested-tasks.py
    :language: python
    :start-after: __nested_start__
    :end-before: __nested_end__

然后调用 ``g`` 和 ``h`` 会产生以下行为。

.. code:: python

    >>> ray.get(g.remote())
    [ObjectRef(b1457ba0911ae84989aae86f89409e953dd9a80e),
     ObjectRef(7c14a1d13a56d8dc01e800761a66f09201104275),
     ObjectRef(99763728ffc1a2c0766a2000ebabded52514e9a6),
     ObjectRef(9c2f372e1933b04b2936bb6f58161285829b9914)]

    >>> ray.get(h.remote())
    [1, 1, 1, 1]

**一个限制** 是 ``f`` 的定义必须在 ``g`` 和 ``h`` 的定义之前，
因为一旦定义了 ``g``，它将被 pickled 并发送到 worker，
所以如果 ``f`` 还没有被定义，定义将是不完整的。

阻塞时释放资源
--------------------------------

Ray 会在被阻塞时释放 CPU 资源。
这可以防止嵌套任务等待父任务持有的 CPU 资源的死锁情况。
考虑以下远程函数。

.. literalinclude:: ../doc_code/nested-tasks.py
    :language: python
    :start-after: __yield_start__
    :end-before: __yield_end__

当 ``g`` 任务正在执行时，在调用 ``ray.get`` 时被阻塞，它将释放其 CPU 资源。
当 ``ray.get`` 返回时，它将重新释放 CPU 资源。
它会在任务的整个生命周期内保留其 GPU 资源，因为该任务很可能会继续使用 GPU 内存。
