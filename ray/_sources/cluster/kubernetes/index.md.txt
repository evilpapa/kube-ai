# Ray 在 Kubernetes
(kuberay-index)=
## 简介

本章，我们会覆盖如何在 Kubernetes 集群上执行你的分布式 Ray 程序。

推荐的方式是使用 [KubeRay Operator](https://github.com/ray-project/kuberay) 。
该控制器提供了一个 Kubernetes 原生方式来管理 Ray clusters。
每个 Ray 集群由一个头节点 pod 以及一批 work 节点 pod 组成。
可选的自动缩放支持允许 KubeRay Operator 根据 Ray 工作负载的要求调整 Ray 集群的大小，并根据需要添加和删除 Ray pod。
KubeRay 支持异构计算节点（包括 GPU）以及在同一 Kubernetes 集群中运行具有不同 Ray 版本的多个 Ray 集群。

```{eval-rst}
.. image:: images/ray_on_kubernetes.png
    :align: center
..
  源文档可在这里找到：https://docs.google.com/drawings/d/1E3FQgWWLuj8y2zPdKXjoWKrfwgYXw6RV_FWRwK8dVlg/edit
```

KubeRay 引入了三种不同的 Kubernetes 自定义资源 (CRDs)： **RayCluster**, **RayJob**, and **RayService**.
这些 CRD 可帮助用户有效管理针对各种用例定制 Ray 集群。

查看 [入门](kuberay-quickstart) 来学习 KubeRay 基础，跟随入门指引在 Kubernetes 上通过 KubeRay 来运行你的第一个 Ray 应用。

* [RayCluster 入门](kuberay-raycluster-quickstart)
* [RayJob 入门](kuberay-rayjob-quickstart)
* [RayService 入门](kuberay-rayservice-quickstart)

## 了解更多

Ray 文档提供了开始在 Kubernetes 上运行 Ray 工作负载所需的所有信息。

```{eval-rst}
.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-3
    
    .. grid-item-card::

        **Getting Started**
        ^^^
    
        了解如何启动 Ray 集群并在 Kubernetes 上部署 Ray 应用程序。
    
        +++
        .. button-ref:: kuberay-quickstart
            :color: primary
            :outline:
            :expand:

            在 Kubernetes 上使用 Ray

    .. grid-item-card::

        **User Guides**
        ^^^
    
        了解在 Kubernetes 上配置 Ray 集群的最佳实践。
    
        +++
        .. button-ref:: kuberay-guides
            :color: primary
            :outline:
            :expand:

            阅读用户手册

    .. grid-item-card::

        **Examples**
        ^^^
    
        尝试 Kubernetes 上的 Ray 工作负载示例。
    
        +++
        .. button-ref:: kuberay-examples
            :color: primary
            :outline:
            :expand:

            尝试示例工作负载

    .. grid-item-card::

        **Ecosystem**
        ^^^
    
        将 KubeRay 与第三方 Kubernetes 生态系统工具集成。
    
        +++
        .. button-ref:: kuberay-ecosystem-integration
            :color: primary
            :outline:
            :expand:

            生态指引

    .. grid-item-card::

        **Benchmarks**
        ^^^
    
        检查 KubeRay 基准测试结果。
    
        +++
        .. button-ref:: kuberay-benchmarks
            :color: primary
            :outline:
            :expand:

            基准结果
    
    .. grid-item-card::

        **Troubleshooting**
        ^^^
    
        参阅 KubeRay 故障排除指南。
    
        +++
        .. button-ref:: kuberay-troubleshooting
            :color: primary
            :outline:
            :expand:

            问题引导
```
## 关于 KubeRay

Ray 的 Kubernetes 支持发布在 [KubeRay GitHub 仓库](https://github.com/ray-project/kuberay)，在 [Ray project](https://github.com/ray-project/) 板块。KubeRay 被多家公司用来运行 Ray 生产部署。

- 查看 [KubeRay GitHub repo](https://github.com/ray-project/kuberay) 跟踪进度、报告错误、提出新功能或为项目做出贡献。
