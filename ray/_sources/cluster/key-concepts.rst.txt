关键概念
============

.. _cluster-key-concepts:

本页介绍了 Ray 集群的关键概念：

.. contents::
    :local:

Ray 集群
-----------
Ray 集群由一个 :ref:`head node <cluster-head-node>` 和
任意数量的连接的 :ref:`worker nodes <cluster-worker-nodes>` 组成：

.. figure:: images/ray-cluster.svg
    :align: center
    :width: 600px

    *具有两个 worker 节点的 Ray 集群。
    每个节点都运行 Ray 辅助进程，以促进分布式调度和内存管理。
    头节点运行额外的控制进程（以蓝色突出显示）。*

 worker 节点的数量可以根据 Ray 集群配置指定的应用程序需求 *自动缩放*。 :ref:`自动缩放程序 <cluster-autoscaler>` 运行在头节点。

.. note::
    :ref:`在 Kubernetes 运行 <kuberay-index>` 时，Ray 节点以 pod 形式实现。

用户可以提交作业在 Ray 集群上执行，或者可以通过连接到头节点并运行 `ray.init` 来以交互方式使用集群。
有关更多信息，请参阅 :ref:`Ray Jobs <jobs-quickstart>`。

.. _cluster-head-node:

头节点
---------
每个 Ray 集群都有一个节点，该节点被指定为集群的 *头节点* 。
头节点与其他 worker 节点相同，不同之处在于它还运行负责集群管理的单例进程，例如
:ref:`自动缩放器 <cluster-autoscaler>`, :term:`GCS <GCS / Global Control Service>` 和运行 Ray 作业的
:ref:`Ray 驱动程序进程 <cluster-clients-and-jobs>`。 
Ray 可以像任何其他 worker 节点一样在头节点上调度task 和 actor，这在大型集群中是不可取的。
请参阅 :ref:`vms-large-cluster-configure-head-node` 以了解大型集群中的最佳实践。

.. _cluster-worker-nodes:

 worker 节点
------------
* worker 节点* 不运行任何头节点管理进程，仅用于运行 Ray 任务和 Actor 中的用户代码。它们参与分布式调度，以及 Ray 对象在 :ref:`集群内存` 中的存储和分发。

.. _cluster-autoscaler:

自动缩放
-----------

*Ray 自动扩缩器* 是运行在 :ref:`头节点 <cluster-head-node>` 的一个进程（如果使用 :ref:`Kubernetes <kuberay-index>`，则作为 Head Pod 中的 sidecar 容器运行）。
当 Ray 工作负载的资源需求超过集群的当前容量时，自动扩缩器将尝试增加 worker 节点的数量。当 worker 节点处于空闲状态时，自动扩缩器将从集群中删除 worker 节点。

重要的是要了解，自动缩放器仅对任务和 actor 资源请求做出反应，而不是应用程序指标或物理资源利用率。
要了解有关自动缩放的更多信息，请参阅 :ref:`VMs <cloud-vm-index>` 和 :ref:`Kubernetes <kuberay-index>` 上的 Ray 集群的用户指南。


.. _cluster-clients-and-jobs:

Ray 作业
--------

Ray 作业是一个单一应用程序：它是来自同一脚本的 Ray 任务、对象和 actor 的集合。
运行 Python 脚本的 worker 称为作业的 *驱动程序*。

有两种方法可以在 Ray 集群上运行 Ray 作业：

1. （推荐） 使用 :ref:`Ray Jobs API <jobs-overview>` 提交作业
2. 直接在 Ray 集群的任意节点上运行驱动程序脚本，进行交互式开发。

有关这些工作流程的详细信息，请参阅 :ref:`Ray Jobs API 指南 <jobs-overview>`。

.. figure:: images/ray-job-diagram.svg
    :align: center
    :width: 650px

    *在 Ray 集群上运行作业的两种方法。*
