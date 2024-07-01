反模式：使用全局变量在任务和 actor 之间共享状态
============================================================================

**TLDR:** 不要使用全局变量来共享状态，而是将全局变量封装在一个 actor 中，并将 actor 句柄传递给其他任务和 actor。

Ray 驱动程序、任务和 Actor 运行在不同的进程中，
因此它们不共享相同的地址空间。
这意味着如果您在一个进程中修改全局变量，
则更改不会反映在其他进程中。

解决方案是使用 actor 的实例变量来保存全局状态，并将 actor 句柄传递到需要修改或访问状态的地方。
请注意，不支持使用类变量来管理同一类的实例之间的状态。
每个 actor 实例都在其自己的进程中实例化，因此每个 actor 都有自己的类变量副本。

代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_global_variables.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法:**

.. literalinclude:: ../doc_code/anti_pattern_global_variables.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__
