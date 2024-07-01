.. _unnecessary-ray-get:

反模式：不必要的调用 ray.get 会影响性能
=============================================================

**TLDR:** 避免在中间步骤中不必要地调用 :func:`ray.get() <ray.get>`。直接使用对象引用进行操作，只在最后一步调用 ``ray.get()`` 获取最终结果。

当调用 ``ray.get()`` 时，对象必须传输到调用 ``ray.get()`` 的 worker/node。如果你不需要操作对象，你可能不需要调用 ``ray.get()``！

通常，最好在调用 ``ray.get()`` 之前等待尽可能长的时间，甚至设计程序以避免完全调用 ``ray.get()``。

代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_unnecessary_ray_get.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

.. figure:: ../images/unnecessary-ray-get-anti.svg

**更好的方法：**

.. literalinclude:: ../doc_code/anti_pattern_unnecessary_ray_get.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

.. figure:: ../images/unnecessary-ray-get-better.svg

请注意，在反模式示例中，我们调用 ``ray.get()``，这迫使我们将大型 rollout 传输到 driver，然后再传输到 *reduce* worker。

在修复版本中，我们只传递对象的引用给 *reduce* 任务。
``reduce`` worker 会隐式调用 ``ray.get()`` 从 ``generate_rollout`` worker 直接获取实际的 rollout 数据，避免了额外的拷贝到 driver。

其他与 ``ray.get()`` 相关的反模式包括：

- :doc:`ray-get-loop`
- :doc:`ray-get-submission-order`
