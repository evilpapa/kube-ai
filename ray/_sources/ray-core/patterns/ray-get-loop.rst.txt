.. _ray-get-loop:

反模式：循环中调用 ray.get 会损害并行性
=========================================================

**TLDR:** 避免在循环中调用 :func:`ray.get() <ray.get>`，因为它是一个阻塞调用；只在最终结果时使用 ``ray.get()``。

``ray.get()`` 调用可获取远程执行函数的结果。但是，它是一个阻塞调用，这意味着它总是等待直到请求的结果可用。
如果在循环中调用 ``ray.get()``，则循环将不会继续运行，直到 ``ray.get()`` 调用解决为止。

如果您还在同一循环中生成远程函数调用，那么您最终将没有任何并行性，因为您需要等待前一个函数调用完成（因为 ``ray.get()``），并且只在下一个循环迭代中生成下一个调用。
解决方案是将对 ``ray.get()`` 的调用与对远程函数的调用分开。这样，我们在等待结果之前就可以生成所有远程函数，并且可以在后台并行运行。此外，您可以将对象引用列表传递给 ``ray.get()``，而不是逐个调用它，以等待所有任务完成。

代码示例
------------

.. literalinclude:: ../doc_code/anti_pattern_ray_get_loop.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

.. figure:: ../images/ray-get-loop.svg

    Calling ``ray.get()`` in a loop

在调度远程工作后立即调用 ``ray.get()`` 时，循环会阻塞，直到结果被接收。因此，我们最终会得到顺序处理。
相反，我们应该首先调度所有远程调用，然后并行处理它们。调度工作后，我们可以一次请求所有结果。

其他与 ``ray.get()`` 相关的反模式包括：

- :doc:`unnecessary-ray-get`
- :doc:`ray-get-submission-order`
