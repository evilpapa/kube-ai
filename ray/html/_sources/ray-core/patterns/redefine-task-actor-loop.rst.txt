反模式：重新定义相同的远程函数或类会损害性能
============================================================================

**TLDR:** 避免重新定义相同的远程函数或类。

使用 :func:`ray.remote <ray.remote>` 装饰器多次装饰相同的函数或类会导致 Ray 性能下降。
对于每个 Ray 远程函数或类，Ray 都会对其进行 pickle 并上传到 GCS。
稍后，运行task 或 actor的工作程序将下载并解开它。
从 Ray 的角度来看，同一个函数或类的每次装饰都会生成一个新的远程函数或类。
因此，每次我们重新定义和运行远程函数或类时，都会发生 pickle、上传、下载和解开的工作。

代码示例
------------

**反模式：**

.. literalinclude:: ../doc_code/anti_pattern_redefine_task_actor_loop.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**更好的方法：**

.. literalinclude:: ../doc_code/anti_pattern_redefine_task_actor_loop.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

我们应该在循环外定义相同的远程函数或类，而不是在循环内定义多次，以便只对其进行一次 pickle 和上传。
