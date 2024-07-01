.. _core-patterns-limit-pending-tasks:

模式：使用 ray.wait 限制待处理任务数量
============================================================

在这种模式中，我们用 :func:`ray.wait() <ray.wait>` 来限制待处理任务的数量。

如果我们持续提交比其处理时间更快的任务，我们将在待处理任务队列中积累任务，最终可能导致 OOM。
使用 ``ray.wait()`` ，我们可以限制待处理任务的数量，以便待处理任务队列不会无限增长并导致 OOM。

.. note::

   如果我们提交的任务数量有限，那么我们不太可能遇到上述问题，因为每个任务仅使用少量内存来记录队列中的任务。
   当我们要运行的任务流无限时，这种情况更有可能发生。

.. note::

   此方法主要用于限制同时执行的任务数。
   它还可用于限制可以并发运行的任务数，但不建议这样做，因为它可能会损害调度性能。
   Ray 会根据资源可用性自动决定任务并行性，因此调整可以并发运行的任务数的推荐方法是修改 :ref:`每个任务的资源需求 <core-patterns-limit-running-tasks>`。

用例
----------------

您有一个工作 actor ，其以每秒 X 个任务的速率处理任务，并且您希望以低于 X 的速率向其提交任务以避免 OOM。

例如，Ray Serve 使用此模式来限制每个工作者待处理的查询数量。

.. figure:: ../images/limit-pending-tasks.svg

    限制待处理任务


代码
------------

**无 backpressure:**

.. literalinclude:: ../doc_code/limit_pending_tasks.py
    :language: python
    :start-after: __without_backpressure_start__
    :end-before: __without_backpressure_end__

**有 backpressure:**

.. literalinclude:: ../doc_code/limit_pending_tasks.py
    :language: python
    :start-after: __with_backpressure_start__
    :end-before: __with_backpressure_end__
