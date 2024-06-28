.. _core-key-concepts:

核心概念
============

本节概述了 Ray 的关键概念。这些原语共同工作，使 Ray 能够灵活地支持广泛的分布式应用程序。

.. _task-key-concept:

任务
-----

Ray 允许在单独的 Python worker 上异步执行任意函数。这些异步 Ray 函数称为「任务」。Ray 允许任务根据 CPU、GPU 和自定义资源指定其资源需求。集群调度程序使用这些资源请求在集群中分配任务以实现并行执行。

请参阅 :ref:`任务的用户指南 <ray-remote-functions>`。

.. _actor-key-concept:

Actors
------

Actor 将 Ray API 从函数（任务）扩展到类。Actor 本质上是一个有状态的工作器（或服务）。当实例化新的 Actor 时，会创建一个新的工作器，并且 Actor 的方法会在该特定工作器上安排，并且可以访问和改变该工作器的状态。与任务一样，Actor 支持 CPU、GPU 和自定义资源要求。

请参阅 :ref:`Actor 的用户指南 <actor-guide>`。

对象
-------

在 Ray 中，任务和 actor 通过对象创建和计算。我们将这些对象称为 *远程对象*，因为它们可以存储在 Ray 集群中的任何位置，并且我们使用 *对象引用* 来引用它们。远程对象缓存在 Ray 的分布式 `共享内存 <https://en.wikipedia.org/wiki/Shared_memory>` *对象存储* 中，并且集群中每个节点都有一个对象存储。在集群设置中，远程对象可以存在于一个或多个节点上，与谁持有对象引用无关。

请参阅 :ref:`对象的用户指南 <objects-in-ray>`。

占位组 Placement Groups
----------------

占位组允许用户原子地在多个节点上预留资源组（即成组调度）。然后可以使用它们来调度 Ray 任务和 actor ，尽可能紧密地打包（PACK）或分散（SPREAD）。占位组通常用于成组调度 actor ，但也支持任务。

请参阅 :ref:`占位组用户指南 <ray-placement-group-doc-ref>`。

环境依赖
------------------------

当 Ray 在远程机器上执行task 和 actor时，它们的环境依赖项（例如 Python 包、本地文件、环境变量）必须可用才能运行代码。为了解决这个问题，您可以 (1) 使用 Ray :ref:`集群启动器 <vm-cluster-quick-start>`，或者 (2) 使用 Ray 的 :ref:`运行时环境 <runtime-environments>` 动态安装它们。

请参阅 :ref:`环境依赖关系的用户指南 <handling_dependencies>`。
