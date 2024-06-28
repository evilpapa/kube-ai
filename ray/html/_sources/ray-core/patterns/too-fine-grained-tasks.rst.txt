反模式：过度并行化和太细力度的任务会损害性能
==========================================================================

**TLDR:** 避免过度并行化。并行化任务的开销比使用普通函数更高。

并行化或分配任务通常比普通函数调用具有更高的开销。因此，如果您并行化执行速度非常快的函数，则开销可能比实际函数调用花费的时间更长！

为了解决这个问题，我们应该小心不要进行过多的并行化。如果你的函数或任务太小，你可以使用一种称为 **批处理的** 技术，让你的任务在一次调用中完成更有意义的工作。


代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_too_fine_grained_tasks.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法：** 使用批处理

.. literalinclude:: ../doc_code/anti_pattern_too_fine_grained_tasks.py
    :language: python
    :start-after: __batching_start__
    :end-before: __batching_end__

从上面的例子中我们可以看出，过度并行的开销较大，程序运行速度比串行版本慢。
通过以适当的批处理大小进行批处理，我们能够摊销开销并实现预期的加速。
