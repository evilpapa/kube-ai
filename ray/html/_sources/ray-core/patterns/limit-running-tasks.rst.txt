.. _core-patterns-limit-running-tasks:

模式：使用资源限制并发运行任务的数量
==========================================================================

在这种模式中，我们使用 :ref:`资源 <resource-requirements>` 来限制并发运行的任务数量。

默认情况下，每个 Ray 任务需要 1 个 CPU，每个 Ray Actor 需要 0 个 CPU，因此调度程序将任务并发限制为可用 CPU，将 Actor 并发限制为无限。
使用 1 个以上 CPU（例如通过多线程）的任务可能会因并发任务的干扰而变慢，但除此之外可以安全运行。

但是，如果任务或 actor 使用的内存超过其应有的份额，则可能会使节点过载并导致 OOM 等问题。
如果是这种情况，我们可以通过增加它们请求的资源量来减少每个节点上同时运行的任务或 actor 的数量。
这是有效的，因为 Ray 确保给定节点上所有同时运行的任务和 actor 的资源需求总和不超过该节点的总资源。

.. note::

   对于 actor 任务，正在运行的 actor 的数量限制了我们可以同时运行的 actor 任务的数量。

用例
----------------

您有一个数据处理工作负载，它使用 Ray :ref:`remote functions <ray-remote-functions>` 独立处理每个输入文件。
由于每个任务都需要将输入数据加载到堆内存中并进行处理，因此运行过多的任务可能会导致 OOM。
在这种情况下，您可以使用 ``memory`` 资源来限制并发运行的任务数量（使用其他资源如 ``num_cpus`` 也可以实现相同的目标）。
请注意，与 ``num_cpus`` 类似， ``memory`` 资源请求是 *物理逻辑* 的，这意味着如果每个任务的物理内存使用量超过此数量，Ray 将不会强制执行。

代码
------------

**无限制：**

.. literalinclude:: ../doc_code/limit_running_tasks.py
    :language: python
    :start-after: __without_limit_start__
    :end-before: __without_limit_end__

**有限制：**

.. literalinclude:: ../doc_code/limit_running_tasks.py
    :language: python
    :start-after: __with_limit_start__
    :end-before: __with_limit_end__
