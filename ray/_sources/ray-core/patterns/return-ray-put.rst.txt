反模式：从任务返回 ray.put() ObjectRefs 会影响性能和容错性
==============================================================================================

**TLDR** 避免在任务返回值上调用 :func:`ray.put() <ray.put>` 并返回生成的 ObjectRefs。
相反，如果可能的话直接返回这些值。

返回 ray.put() ObjectRefs 被认为是反模式，原因如下：

- 它不允许内联小的返回值：Ray 有一个性能优化，可以直接将小的（<= 100KB）值内联返回给调用者，避免通过分布式对象存储。
  另一方面，``ray.put()`` 会无条件地将值存储到对象存储中，这使得对小返回值的优化变得不可能。
- 返回 ObjectRefs 会增加网络开销：返回值需要通过网络传输到调用者，而不是直接返回。
- 它的 :ref:`容错 <fault-tolerance>` 更差：调用 ``ray.put()`` 的 worker 进程是返回的 ``ObjectRef`` 的“所有者”，返回值的命运与所有者共享。如果 worker 进程死亡，返回值将丢失。
  相反，如果直接返回值，调用者进程（通常是 driver）是返回值的所有者。

代码示例：
------------

如果您想返回一个值，无论它是小还是大，您都应该直接返回它。

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_single_value_start__
    :end-before: __return_single_value_end__

如果您想要返回多个值并且在调用任务之前知道返回的数量，则应该使用 :ref:`num_returns <ray-task-returns>` 选项。

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_static_multi_values_start__
    :end-before: __return_static_multi_values_end__

如果在调用任务之前不知道返回的次数，则应尽可能使用 :ref:`dynamic generator <dynamic-generators>` 模式。

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_dynamic_multi_values_start__
    :end-before: __return_dynamic_multi_values_end__
