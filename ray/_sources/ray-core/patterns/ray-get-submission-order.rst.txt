反模式：使用 ray.get 按提交顺序处理结果会增加运行时间
====================================================================================

**TLDR:** 避免按照提交顺序使用 :func:`ray.get() <ray.get>` 处理独立的结果，因为结果可能以与提交顺序不同。

一批任务被提交，我们需要在它们完成后逐个处理它们的结果。
如果每个任务完成所需的时间不同，并且我们按提交顺序处理结果，那么我们可能会浪费时间等待所有较慢（拖延者）的任务完成，而较快的任务已经完成。

相反，我们希望使用 :func:`ray.wait() <ray.wait>` 按完成顺序处理任务，以加快完成时间。

.. figure:: ../images/ray-get-submission-order.svg

    按提交顺序和按完成顺序处理结果


代码示例
------------

.. literalinclude:: ../doc_code/anti_pattern_ray_get_submission_order.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

其他与 ``ray.get()`` 相关的反模式包括：

- :doc:`unnecessary-ray-get`
- :doc:`ray-get-loop`
