.. _task-pattern-nested-tasks:

模式：使用嵌套任务实现嵌套并行
=========================================================

在此模式下，远程任务可以动态调用其他远程任务（包括其自身）以实现嵌套并行。
当子任务可以并行化时，这很有用。

但请记住，嵌套任务有其自身的成本：额外的工作进程、调度开销、bookkeeping 开销等。
要通过嵌套并行实现加速，请确保每个嵌套任务都执行大量工作。有关更多详细信息，请参阅 :doc:`too-fine-grained-tasks` 会损害加速。

用例
----------------

您想要对一大串数字进行快速排序。
通过使用嵌套任务，我们可以以分布式和并行的方式对列表进行排序。

.. figure:: ../images/tree-of-tasks.svg

    Tree of tasks


代码
------------

.. literalinclude:: ../doc_code/pattern_nested_tasks.py
    :language: python
    :start-after: __pattern_start__
    :end-before: __pattern_end__

我们在两个 ``quick_sort_distributed`` 函数调用都发生后调用  :func:`ray.get() <ray.get>`。
这允许您最大化工作负载中的并行性。 有关更多详细信息，请参阅 :doc:`ray-get-loop` 。

请注意，上面的执行时间表明，任务越小，非分布式版本的速度越快。
但是，随着任务执行时间的增加（即由于要排序的列表越大），分布式版本的速度就越快。
