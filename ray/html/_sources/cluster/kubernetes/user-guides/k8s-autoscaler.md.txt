(ray-k8s-autoscaler-comparison)=
# （高级）了解 Kubernetes 背景下的 Ray Autoscaler
我们描述了 Ray 自动缩放器和 Kubernetes 生态系统中其他自动缩放器之间的关系。

## Ray Autoscaler 与 Horizo​​ntal Pod Autoscaler
Ray 自动缩放器调整 Ray 集群中的 Ray 节点数量。
在 Kubernetes 上，每个 Ray 节点都作为 Kubernetes Pod 运行。
因此，在 Kubernetes 的背景下，Ray 自动缩放器可缩放 Ray **Pod 数量**。
从这个意义上说，Ray Autoscaler 扮演的角色类似于 Kubernetes (https://kubernetes.io/docs/tasks/run-application/horizontal-Pod-autoscale/) (HPA)。
然而，Ray Autoscaler 与 HPA 的区别如下：

### 负载指标基于应用程序语义
Horizo​​ntal Pod Autoscaler 根据 CPU 和内存等物理使用指标确定规模。
相比之下，Ray 自动缩放器使用任务和 actor 注释中表达的逻辑资源。
例如，如果 RayCluster CR 中的每个 Ray 容器规范指示 10 个 CPU 的限制，并且您提交 20 个带有 `@ray.remote(num_cpus=5)` 注解的任务，
则将创建 10 个 Ray Pod 来满足 100 个 CPU 的资源需求。
在这方面，Ray 自动缩放器与
[Kubernetes Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler)类似，
后者根据容器资源请求中表达的逻辑资源做出缩放决策。

### 细粒度的缩容控制
为了适应 Ray 应用程序的状态性，Ray 自动缩放器比 Horizo​​ntal Pod Autoscaler 对缩小规模具有更细粒度的控制。
除了确定所需的比例之外，Ray Autoscaler 还能够精确选择要缩小的 Pod。然后 KubeRay operator 删除该 Pod。
相比之下，Horizo​​ntal Pod Autoscaler 只能减少副本数量，而无法控制删除哪些 Pod。
对于 Ray 应用程序来说，缩小随机 Pod 的规模可能是危险的。

### 架构：每个 Ray 集群一个 Ray Autoscaler
Horizontal Pod Autoscaling 由 Kubernetes 控制平面中的管理器集中控制；
管理器控制许多 Kubernetes 对象的规模。
相比之下，每个 Ray 集群都由自己的 Ray 自动缩放器进程管理，
作为 Ray head Pod 中的 sidecar 容器运行。这种设计选择是出于以下考虑：

- **可扩展性.** 自动缩放每个 Ray 集群需要处理来自该 Ray 集群的大量资源数据。
- **简化的版本控制和兼容性.** 自动缩放器和 Ray 都是作为 Ray 存储库的一部分开发的。
自动缩放器和 Ray 核心之间的接口很复杂。
为了支持在不同 Ray 版本上运行的多个 Ray 集群，最好匹配 Ray 和 Autoscaler 代码版本。
每个 Ray 集群运行一个自动缩放器并匹配代码版本可确保兼容性。

(kuberay-autoscaler-with-ray-autoscaler)=
## Ray Autoscaler 与 Kubernetes Cluster Autoscaler
The Ray Autoscaler 与 [Kubernetes Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler) 相辅相成。
Ray Autoscaler 决定创建 Ray Pod 后，Kubernetes Cluster Autoscaler
可以配置 Kubernetes 节点，以便放置 Pod。
同样，当 Ray 自动缩放器决定删除空闲 Pod 后，
Kubernetes Cluster Autoscaler 可以清理剩余的空闲 Kubernetes 节点。
建议配置 RayCluster，以便每个 Kubernetes 节点仅适合一个 Ray Pod。
如果遵循此模式，Ray Autoscaler Pod 缩放事件与集群​​自动缩放器节点缩放事件
大致一一对应。（我们说“大致”是因为在底层 Kubernetes 节点缩小之前，Ray Pod 可能会被删除并替换为新的 Ray Pod。）


## 垂直 Pod 自动缩放器
Ray Autoscaler 和 Kubernetes [Vertical Pod Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler) (VPA) 之间没有关系，
VPA 的目的是根据当前和过去的使用情况将各个 Pod 调整为适当的大小。
如果您发现单个 Ray Pod 上的负载过高，可以使用多种手动技术来减少负载，其中一种方法是通过增加 `ray.remote` 注释中指定的资源需求来减少每个节点的任务/actor。
例如，更改 `@ray.remote(num_cpus=2)` 为 `@ray.remote(num_cpus=4)`
将使给定 Ray Pod 中可以容纳的task 或 actor的数量减半。
