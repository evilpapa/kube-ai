(overview-overview)=

# 概述

Ray 是一个开源统一框架，用于扩展机器学习等 AI 和 Python 应用程序。 它提供了用于并行处理的计算层，因此您不需要成为分布式系统专家。 Ray 使用这些组件最大限度地降低了运行分布式个体和端到端机器学习工作流程的复杂性：
* 用于常见机器学习任务的可扩展库，例如数据预处理、分布式训练、超参数调整、强化学习和模型服务。
* 用于并行化和扩展 Python 应用程序的 Pythonic 分布式计算原语。
* 用于将 Ray 集群与现有工具和基础设施（例如 Kubernetes、AWS、GCP 和 Azure）集成和部署的集成和实用程序。

对于数据科学家和机器学习从业者来说，Ray 可以让您在不需要基础设施专业知识的情况下扩展工作：
* 跨多个节点和 GPU 轻松并行和分配 ML 工作负载。
* 通过本机和可扩展集成利用 ML 生态系统。

对于 ML 平台构建者和 ML 工程师，Ray：
* 提供计算抽象以创建可扩展且强大的机器学习平台。
* 提供统一的 ML API，简化与更广泛的 ML 生态系统的入门和集成。
* 通过使相同的 Python 代码能够从笔记本电脑无缝扩展到大型集群，减少开发和生产之间的摩擦。

对于分布式系统工程师来说，Ray 自动处理关键流程：
* 编排——管理分布式系统的各个组件。
* 调度——协调任务执行的时间和地点。
* 容错——无论不可避免的故障点如何，都确保任务完成。
* 自动缩放——调整分配给动态需求的资源数量。

## 你可以用 Ray 做什么

以下是个人、组织和公司利用 Ray 构建 AI 应用程序的一些常见 ML 工作负载：
* [在 CPU 和 GPU 批量预估](use-cases.html#batch-inference)
* [并行训练](use-cases.html#many-model-training)
* [模型服务](use-cases.html#model-serving)
* [大模型分布式训练](use-cases.html#distributed-training)
* [并行超参调优实验](use-cases.html#hyperparameter-tuning)
* [强化学习](use-cases.html#reinforcement-learning)
* [机器学习平台](use-cases.html#ml-platform)

## Ray 框架

|<img src="../images/map-of-ray.png" width="70%" loading="lazy">|
|:--:|
|Stack of Ray libraries - unified toolkit for ML workloads.|

Ray 的统一计算框架由三层组成：

1. **Ray AI 类库**--一组开源、Python、特定领域的库，为 ML 工程师、数据科学家和研究人员提供用于 ML 应用程序的可扩展且统一的工具包。
2. **Ray Core**--一个开源 Python 通用分布式计算库，使 ML 工程师和 Python 开发人员能够扩展 Python 应用程序并加速机器学习工作负载。
3. **Ray Clusters**--连接到公共 Ray 头节点的一组 worker 节点。 Ray 集群可以是固定大小的，也可以根据集群上运行的应用程序请求的资源自动缩放。

```{eval-rst}
.. grid:: 1 2 3 3
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::

        **扩展机器学习工作负载**
        ^^^
        使用分布式库工具包构建 ML 应用程序
        :doc:`数据处理 <../data/data>`, 
        :doc:`模型训练 <../train/train>`, 
        :doc:`调参 <../tune/index>`, 
        :doc:`强化学习 <../rllib/index>`, 
        :doc:`模型服务 <../serve/index>`, 
        以及 :doc:`更多 <../ray-more-libs/index>`。
        +++
        .. button-ref:: libraries-quickstart
            :color: primary
            :outline:
            :expand:

            Ray AI 类库

    .. grid-item-card::
        
        **构建分布式应用**
        ^^^
        使用如下方案构建并运行分布式应用
        :doc:`简单灵活的 API <../ray-core/walkthrough>`.
        :doc:很小甚至零改动 `并行化 <../ray-core/walkthrough>` 单机代码。
        +++
        .. button-ref:: ../ray-core/walkthrough
            :color: primary
            :outline:
            :expand:

            Ray 核心

    .. grid-item-card::
        
        **部署大规模工作负载**
        ^^^
        在 :doc:`AWS，GCP，Azure <../cluster/getting-started>` 或
        :doc:`现场 <../cluster/vms/user-guides/launching-clusters/on-premises>` 部署工作负载。
        使用 Ray 集群管理来在已有
        :doc:`Kubernetes <../cluster/kubernetes/index>` 或
        :doc:`YARN <../cluster/vms/user-guides/community/yarn>` 或
        或 :doc:`Slurm <../cluster/vms/user-guides/community/slurm>` 集群运行 Ray。
        +++
        .. button-ref:: ../cluster/getting-started
            :color: primary
            :outline:
            :expand:

            Ray 集群 
```

[Ray](../ray-air/getting-started) 的五个原生类库都分配一个特定的 ML 任务：
- [Data](../data/data): 跨训练、调整和预测的可扩展、与框架无关的数据加载和转换。
- [Train](../train/train): 具有容错能力的分布式多节点、多核模型训练，与流行的训练库集成。
- [Tune](../tune/index): 可扩展的超参数调整以优化模型性能。
- [Serve](../serve/index): 可扩展和可编程的模型服务，用于部署在线推理模型，并具有可选的微批处理来提高性能。
- [RLlib](../rllib/index): 可扩展的分布式强化学习工作负载。

Ray 的库既适合数据科学家，也适合机器学习工程师。 对于数据科学家来说，这些库可用于扩展个人工作负载以及端到端的机器学习应用程序。 对于机器学习工程师来说，这些库提供了可扩展的平台抽象，可用于轻松加载和集成更广泛的机器学习生态系统中的工具。

对于自定义应用，[Ray 核心](../ray-core/walkthrough) 类库使 Python 开发人员能够轻松构建可在笔记本电脑、集群、云或 Kubernetes 上运行的可扩展的分布式系统。它是 Ray AI 库和第三方集成（Ray 生态系统）的基础。

Ray 可以在任何机器、集群、云提供商和 Kubernetes 上运行，并且具有发展增长的
[社区整合生态系统](ray-libraries)。
