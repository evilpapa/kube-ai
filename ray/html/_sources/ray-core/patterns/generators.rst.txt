.. _generator-pattern:

模式：使用生成器减少堆内存使用量
=====================================================

在此模式中，我们使用 Python 中的 **生成器** 来减少任务期间的总堆内存使用量。关键思想是，对于返回多个对象的任务，我们可以一次返回一个，而不是一次全部返回。这允许工作程序在返回下一个返回值之前释放上一个返回值使用的堆内存。

用例
----------------

您有一个返回多个大值的任务。另一种可能性是，该任务返回单个大值，但您希望通过将其分解为较小的块来通过 Ray 的对象存储流式传输该值。

使用普通的 Python 函数，我们可以编写这样的任务。下面是一个返回每个大小为 100MB 的 numpy 数组的示例：

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_start__
    :end-before: __large_values_end__

但是，这将要求任务在结束时同时将所有 ``num_returns`` 数组保存在任务的堆内存中。如果有很多返回值，这可能会导致堆内存使用量过高，并且可能会导致内存不足错误。

我们可以通过重写 ``large_values`` 为 **generator** 。而不是一次返回所有值作为元组或列表，我们可以一次返回一个值。

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_generator_start__
    :end-before: __large_values_generator_end__

代码示例
------------

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __program_start__

.. code-block:: text

    $ RAY_IGNORE_UNHANDLED_ERRORS=1 python test.py 100

    Using normal functions...
    ... -- A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker...
    Worker failed
    Using generators...
    (large_values_generator pid=373609) yielded return value 0
    (large_values_generator pid=373609) yielded return value 1
    (large_values_generator pid=373609) yielded return value 2
    ...
    Success!
