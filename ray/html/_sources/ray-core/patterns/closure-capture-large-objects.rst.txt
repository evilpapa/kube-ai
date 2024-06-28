反模式：闭包捕获大对象会损害性能
===============================================================

**TLDR:** 避免在远程函数或类中使用闭包捕获大对象，而是使用对象存储。

当定义一个 :func:`ray.remote <ray.remote>` 函数或类时，
很容易意外地在定义中隐式捕获大型（超过几 MB）对象。
这可能会导致性能下降甚至 OOM，因为 Ray 并非设计用于处理非常大的序列化函数或类。

对于如此大的对象，有两种方法可以解决这个问题：

- 使用 :func:`ray.put <ray.put>` 来将大对象放入 Ray 对象存储中，然后将对象引用作为参数传递给远程函数或类（下面的 *"更好的方法 #1"*）
- 在远程函数或类中创建大对象，通过 lambda 方法传递（下面的 *"更好的方法 #2"*）。这也是使用不可序列化对象的唯一选项。


代码示例
------------

**反模式**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法 #1:**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __better_approach_1_start__
    :end-before: __better_approach_1_end__

**更好的方法 #2:**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __better_approach_2_start__
    :end-before: __better_approach_2_end__
