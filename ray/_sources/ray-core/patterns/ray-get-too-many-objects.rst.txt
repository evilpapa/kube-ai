反模式：使用 ray.get 一次获取太多对象导致失败
===========================================================================

**TLDR:** 避免使用 :func:`ray.get() <ray.get>` 一次获取太多对象，因为这会导致堆内存不足或对象存储空间不足。相反，一次获取并处理一个批次。

如果您有大量的任务要并行运行，尝试一次性对所有任务执行 ``ray.get()`` 可能会导致堆内存不足或对象存储空间不足，因为 Ray 需要同时将所有对象提取到调用方。
相反，您应该一次获取并处理一个批次的结果。一旦处理了一个批次，Ray 将清除该批次中的对象，以为将来的批次腾出空间。

.. figure:: ../images/ray-get-too-many-objects.svg

    ``ray.get()`` 一次获取太多对象

代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_ray_get_too_many_objects.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法：**

.. literalinclude:: ../doc_code/anti_pattern_ray_get_too_many_objects.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

这里除了一次获取一个批次以避免失败之外，我们还使用 ``ray.wait()`` 以完成顺序而不是提交顺序处理结果，以减少运行时间。有关更多详细信息，请参见 :doc:`ray-get-submission-order`。
