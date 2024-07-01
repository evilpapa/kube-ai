(kuberay-quickstart)=

# KubeRay 入门

## 自定义资源 (CRDs)

[KubeRay](https://github.com/ray-project/kuberay) 是一个强大的，开源的 Kubernetes 控制器，它简化了再 Kubernetes 上部署和管理 Ray 应用。
它提供了 3 种自定义资源 (CRDs)：

* **RayCluster**: KubeRay 完全管理 RayCluster 的生命周期，包括集群创建 / 删除，扩缩容，以及容错保证。

* **RayJob**: 通过 RayJob，KubeRay 自动创建一个 RayCluster 并在集群就绪提交任务。你同样也可以配置 RayJob 在任务完成之后自动删除 RayCluster。

* **RayService**: RayService 由两部分组成：一个 RayCluster 以及 Ray Serve 部署图。RayService 为 RayCluster 和高可用性提供零停机升级。

## 我该选择那种 CRD ？

使用 [RayService](kuberay-rayservice-quickstart) 来部署模型，使用 [RayCluster](kuberay-raycluster-quickstart) 来部署 Ray 应用是我们的明智推荐。
然而，使用场景不是模型服务或模型原型设计，我该如何在 [RayCluster](kuberay-raycluster-quickstart) 以及 [RayJob](kuberay-rayjob-quickstart) 之间进行选择？

### 问：是否能容忍集群升级的下线状态（例如：升级 Ray 版本）？

如果不，选择 RayJob。RayJob 可以配置为一旦任务完成就自动删除 RayCluster 集群。你可以为每个使用 RayJob 提交的任务指定 Ray 版本及配置。

如果可以，选择 RayCluster。Ray 并未原生支持滚动升级；因此，你需要手动停止并创建一个新的 RayCluster。

### 问：你是否部署在公有云上（例如：AWS，GCP，Azure）？

如果是，选择 RayJob。它允许在任务完成之上自动删除 RayCluster，可帮你节省支出。

### 问：您是否允许 RayCluster 转换带来的延迟？

如果是，选择 RayCluster。
和 RayJob 不同，它会在每个任务提交创建一个新的 RayCluster。RayCluster 只创建一次集群，可多次使用。

## 在 Kubernetes 创建你的第一个 Ray 应用！

* [RayCluster 入门](kuberay-raycluster-quickstart)
* [RayJob 入门](kuberay-rayjob-quickstart)
* [RayService 入门](kuberay-rayservice-quickstart)
