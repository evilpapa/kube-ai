.. _ray-pass-large-arg-by-value:

反模式：反复传递相同的大参数会损害性能
===================================================================================

**TLDR:** 避免将相同的大参数通过值传递给多个任务，使用 :func:`ray.put() <ray.put>` 并通过引用传递。

当将大型参数 (>100KB) 按值传递给任务时，
Ray 会隐式地将参数存储在对象存储中，工作进程会在运行任务之前从调用方的对象存储中将参数提取到本地对象存储中。
如果我们将同一个大型参数传递给多个任务，Ray 最终会在对象存储中存储该参数的多个副本，因为 Ray 不进行重复数据删除。

我们不应该将大参数按值传递给多个任务，
而应该使用 ``ray.put()`` 将参数存储到对象存储中一次并获取 ``ObjectRef``，
然后将参数引用传递给任务。这样，我们确保所有任务都使用参数的同一份副本，这样速度更快，占用的对象存储内存更少。

代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_pass_large_arg_by_value.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法：**

.. literalinclude:: ../doc_code/anti_pattern_pass_large_arg_by_value.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__
