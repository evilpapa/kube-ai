.. _fault-tolerance:

容错
===============

Ray 是一个分布式系统，这意味着故障可能会发生。通常，故障可以分为两类：1）应用程序级故障，2）系统级故障。
前者可能是由于用户级代码中的错误或外部系统失败引起的。
后者可能是由节点故障、网络故障或 Ray 中的错误引起的。
在这里，我们描述了 Ray 提供的机制，允许应用程序从故障中恢复。

要处理应用程序级故障，Ray 提供了机制来捕获错误、重试失败的代码和处理行为不当的代码。
有关这些机制的更多信息，请参见 :ref:`task <fault-tolerance-tasks>` 和 :ref:`actor <fault-tolerance-actors>` 容错性。

Ray 同样提供了一些机制来自动从内部系统级故障中恢复，例如 :ref:`节点故障 <fault-tolerance-nodes>`。
特别是，Ray 可以自动从 :ref:`分布式对象存储 <fault-tolerance-objects>` 中的一些故障中恢复。

如何编写容错 Ray 应用程序
--------------------------------------------

有几条建议可以使 Ray 应用程序具有容错性：

首先，如果 Ray 提供的容错机制不适用于您，
您可以捕获由故障引起的 :ref:`异常 <ray-core-exceptions>` 并手动恢复。

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __manual_retry_start__
    :end-before: __manual_retry_end__

其次，避免让 ``ObjectRef`` 超出其 :ref:`所有者 <fault-tolerance-objects>` 任务或 actor 的生命周期
（任务或 actor 通过调用 :meth:`ray.put() <ray.put>` 或 ``foo.remote()`` 创建初始 ``ObjectRef`` 的）。
只要仍有对对象的引用，对象的所有者 worker 就会在相应task 或 actor完成后继续运行。
如果对象的所有者 worker 失败，Ray :ref:`无法自动为尝试访问对象的用户 <fault-tolerance-ownership>`恢复对象。
从任务返回由 ``ray.put()`` 创建的 ``ObjectRef`` 是创建这种超出生命周期的对象的一个例子：

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __return_ray_put_start__
    :end-before: __return_ray_put_end__

上例中，对象 ``x`` 超出了其所有者任务 ``a`` 的生命周期。
如果 worker 进程运行任务 ``a`` 失败，之后调用 ``ray.get`` 获取 ``x_ref`` 将导致 ``OwnerDiedError`` 异常。

容错版本是直接返回 ``x``，这样它就由 driver 拥有，并且只在 driver 的生命周期内访问。
``x`` 如果丢失，Ray 可以通过 :ref:`lineage reconstruction <fault-tolerance-objects-reconstruction>` 自动恢复。
参考 :doc:`/ray-core/patterns/return-ray-put` 了解更多细节。

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __return_directly_start__
    :end-before: __return_directly_end__

第三，避免使用只能由特定节点满足的 :ref:`自定义资源需求 <custom-resources>`。
如果特定节点失败，正在运行的task 或 actor将无法重试。

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __node_ip_resource_start__
    :end-before: __node_ip_resource_end__

如果你倾向于在特定节点上运行任务，你可以使用 :class:`NodeAffinitySchedulingStrategy <ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy>`。
它允许你将亲和性作为软约束来指定，因此即使目标节点失败，任务仍然可以在其他节点上重试。

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: _node_affinity_scheduling_strategy_start__
    :end-before: __node_affinity_scheduling_strategy_end__


更多关于 Ray 容错性的信息
------------------------------

.. toctree::
    :maxdepth: 1

    fault_tolerance/tasks.rst
    fault_tolerance/actors.rst
    fault_tolerance/objects.rst
    fault_tolerance/nodes.rst
    fault_tolerance/gcs.rst
