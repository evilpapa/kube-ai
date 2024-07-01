.. _cluster-index:

Ray 集群概述
=====================

Ray 可以将工作负载从笔记本电脑无缝扩展到大型集群。
虽然 Ray 可以在单台机器上直接运行，只需调用 ``ray.init``，但要在多个节点上运行 Ray 应用程序，您必须首先 *部署 Ray 集群*。

Ray 集群是一组连接到共同 :ref:`Ray head 节点 <cluster-head-node>` 的 worker 节点。
Ray 集群可以是固定大小的，也可以根据集群上运行的应用程序请求的资源 :ref:`自动扩展和缩减 <cluster-autoscaler>`。

我可以在哪里部署 Ray 集群？
--------------------------------

Ray 在以下技术栈上提供原生集群部署支持：

* 在 :ref:`AWS 和 GCP <cloud-vm-index>`。还有社区支持的 Azure 、 Aliyun 和 vSphere 继承。
* 在 :ref:`Kubernetes <kuberay-index>`，通过官方支持的 KubeRay 项目。

高级用户可能希望 :ref:`手动部署 Ray <on-prem>` 或部署到 :ref:`这里未列出的平台 <ref-cluster-setup>`。

.. note::

    多节点 Ray 集群仅支持 Linux。您可以在部署过程中设置
    环境变量 ``RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`` 来自行部署 Windows 和 OSX 集群。

接下来做什么？
------------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::

        **我想学习 Ray 集群的关键概念**
        ^^^
        了解与 Ray 集群交互的关键概念和主要方式。

        +++
        .. button-ref:: cluster-key-concepts
            :color: primary
            :outline:
            :expand:

            学习关键概念

    .. grid-item-card::

        **我想在 Kubernetes 上运行 Ray**
        ^^^
        将 Ray 应用程序部署到 Kubernetes 集群。您可以通过 Kind 在 Kubernetes 集群或笔记本电脑上运行本教程。

        +++
        .. button-ref:: kuberay-quickstart
            :color: primary
            :outline:
            :expand:

            在 Kubernetes 上开始使用 Ray

    .. grid-item-card::

        **我想在云提供商上运行 Ray**
        ^^^
        采用设计为在笔记本电脑上运行的示例应用程序，并在云中扩展它。
        需要访问 AWS 或 GCP 帐户。

        +++
        .. button-ref:: vm-cluster-quick-start
            :color: primary
            :outline:
            :expand:

            在虚拟机上开始使用 Ray

    .. grid-item-card::

        **我想在现有的 Ray 集群上运行我的应用程序**
        ^^^
        将应用程序作为作业提交到现有 Ray 集群的指南。

        +++
        .. button-ref:: jobs-quickstart
            :color: primary
            :outline:
            :expand:

            作业提交
