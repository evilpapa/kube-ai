.. _ray-scheduling:

调度
==========

对于每个任务和 actor，Ray 会选择一个节点来运行它，调度决策基于以下因素。

.. _ray-scheduling-resources:

资源
---------

每个任务和节点有 :ref:`指定的资源需求 <resource-requirements>`。
鉴于此，一个节点可以处于以下状态之一：

- 可行：节点有足够的资源来运行任务或 actor。
  取决于这些资源的当前可用性，有两种子状态：

  - 可用：节点有所需的资源，现在是空闲的。
  - 不可用：节点有所需的资源，但它们当前被其他任务或 actor 使用。

- 不可行：此节点没有所需的资源。例如，对于 GPU 任务，只有 CPU 的节点是不可行的。

资源需求是 **硬性** 需求，这意味着只有可行的节点才有资格运行任务或 actor。
如果这个节点是可行的，Ray 将选择一个可用的节点，或者等待一个不可用的节点变为可用，具体取决于下面讨论的其他因素。
如果所有节点都是不可行的，那么任务或 actor 将无法调度，直到可行的节点被添加到集群中。

.. _ray-scheduling-strategies:

Scheduling Strategies 调度策略
---------------------

任务和 actor 支持一个 :func:`scheduling_strategy <ray.remote>` 选项，用于指定用于在可行节点中选择最佳节点的策略。
当前支持的策略如下。

"DEFAULT"
~~~~~~~~~

``"DEFAULT"`` Ray 的默认策略。
Ray 调度任务和 actor 到前 k 个节点的组中。
特殊的，节点首先按照已经调度的任务或 actor 来排序（为了本地性），然后按照资源利用率低的节点来排序（为了负载均衡）。
在前 k 组中，节点是随机选择的，以进一步改善负载均衡，并减轻大集群中冷启动的延迟。

在实现方面，Ray 根据集群中每个节点的逻辑资源利用率计算其分数。
如果利用率低于阈值（由 OS 环境变量 ``RAY_scheduler_spread_threshold`` 控制，默认值为 0.5），则分数为 0，否则为资源利用率本身（分数 1 表示节点完全被利用）。
Ray 通过从具有最低分数的前 k 个节点中随机选择来选择最佳节点进行调度。
``k`` 的值是（集群中节点的数量 * ``RAY_scheduler_top_k_fraction`` 环境变量）和 ``RAY_scheduler_top_k_absolute`` 环境变量的最大值。
默认情况下，它是总节点数的 20%。

当前，Ray 通过在集群中随机选择一个节点来处理不需要任何资源的 actor（即 ``num_cpus=0`` 且没有其他资源）。
Since nodes are randomly chosen, actors that don't require any resources are effectively SPREAD across the cluster. 由于节点是随机选择的，不需要任何资源的 actor 实际上是在集群中 SPREAD 的。

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __default_scheduling_strategy_start__
    :end-before: __default_scheduling_strategy_end__

"SPREAD"
~~~~~~~~

``"SPREAD"`` strategy will try to spread the tasks or actors among available nodes. ``"SPREAD"`` 策略将尝试在可用节点之间分配任务或 actor。

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __spread_scheduling_strategy_start__
    :end-before: __spread_scheduling_strategy_end__

PlacementGroupSchedulingStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy` 会将任务或 actor 调度到放置组所在的位置。
 这对于 actor gang scheduling 很有用。有关更多详细信息，请参见 :ref:`Placement Group <ray-placement-group-doc-ref>`。

NodeAffinitySchedulingStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy` 是一种低级策略，允许通过节点 ID 指定的特定节点调度任务或 actor。
``soft`` 标志指定如果指定的节点不存在（例如，如果节点死机）或不可行，因为它没有运行任务或 actor 所需的资源，则任务或 actor 是否允许在其他地方运行。
这种情况下，如果 ``soft`` 为 True，则任务或 actor 将被调度到另一个可行的节点。
 否则，任务或 actor 将失败，并显示 :py:class:`~ray.exceptions.TaskUnschedulableError` 或 :py:class:`~ray.exceptions.ActorUnschedulableError`。
只要指定节点是活动的和可行的，任务或 actor 将只在那里运行，
而不管 ``soft`` 标志如何。
这意味着如果节点当前没有可用资源，任务或 actor 将等待资源可用。
此策略应 *仅仅* 在其他高级调度策略（例如：:ref:`placement group <ray-placement-group-doc-ref>`）无法提供
所需的任务或 actor 放置时使用。它具有以下已知限制：

- 这是一种低级策略，阻止了智能调度器的优化。
- 它不能充分利用自动缩放集群，因为在创建任务或 actor 时必须知道节点 ID。
- 它可能很难做出最佳的静态放置决策，特别是在多租户集群中：例如，应用程序不知道其他什么正在调度到相同的节点上。

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __node_affinity_scheduling_strategy_start__
    :end-before: __node_affinity_scheduling_strategy_end__

.. _ray-scheduling-locality:

位置感知调度
-------------------------

默认的，Ray 会优先选择有大型任务参数的可用节点，以避免在网络上传输数据。
如果有多个大型任务参数，那么优先选择有最多对象字节的节点。
这优先于 ``"DEFAULT"`` 调度策略，这意味着 Ray 将尝试在位置首选节点上运行任务，而不管节点资源利用率如何。
 然而，如果位置首选节点不可用，Ray 可能会在其他地方运行任务。
当其他调度策略被指定时，它们具有更高的优先级，数据位置不再被考虑。

.. note::

  位置感知调度仅适用于任务，而不适用于 actor。

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __locality_aware_scheduling_start__
    :end-before: __locality_aware_scheduling_end__

更多关于 Ray 调度的内容
-------------------------

.. toctree::
    :maxdepth: 1

    resources
    ../tasks/using-ray-with-gpus
    placement-group
    memory-management
    ray-oom-prevention
