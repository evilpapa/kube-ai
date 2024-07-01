(kuberay-config)=

# RayCluster 配置

本指南涵盖了 Kubernetes 上 Ray 集群配置的关键方面。

## 介绍

Ray 在 Kubernetes 上的部署遵循 [operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)。 关键点是
- [自定义资源](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
    描述 Ray Cluster 状态的 `RayCluster` 。
- [自定义 controller](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/#custom-controllers)，
    KubeRay operator，它管理 Ray pods 以符合 `RayCluster` 规范。

要部署 Ray 集群，需要创建一个 `RayCluster` 自定义资源 (CR)：
```shell
kubectl apply -f raycluster.yaml
```

本指南涵盖  `RayCluster`  CR 配置的显著特征。

作为参考，这里是 yaml 格式的 `RayCluster` CR 精简示例。
```yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: raycluster-complete
spec:
  rayVersion: "2.3.0"
  enableInTreeAutoscaling: true
  autoscalerOptions:
     ...
  headGroupSpec:
    serviceType: ClusterIP # Options are ClusterIP, NodePort, and LoadBalancer
    rayStartParams:
      dashboard-host: "0.0.0.0"
      ...
    template: # Pod template
        metadata: # Pod metadata
        spec: # Pod spec
            containers:
            - name: ray-head
              image: rayproject/ray-ml:2.3.0
              resources:
                limits:
                  cpu: 14
                  memory: 54Gi
                requests:
                  cpu: 14
                  memory: 54Gi
              # Keep this preStop hook in each Ray container config.
              lifecycle:
                preStop:
                  exec:
                    command: ["/bin/sh","-c","ray stop"]
              ports: # Optional service port overrides
              - containerPort: 6379
                name: gcs
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client
              - containerPort: 8000
                name: serve
                ...
  workerGroupSpecs:
  - groupName: small-group
    replicas: 1
    minReplicas: 1
    maxReplicas: 5
    rayStartParams:
        ...
    template: # Pod template
      spec:
        ...
  # Another workerGroup
  - groupName: medium-group
    ...
  # Yet another workerGroup, with access to special hardware perhaps.
  - groupName: gpu-group
    ...
```

本指南的其余部分将讨论 `RayCluster` CR' 的配置字段。
另请参阅有关使用 KubeRay 配置 Ray 自动缩放的 [指南](kuberay-autoscaling-config) 。

(kuberay-config-ray-version)=
## Ray 版本
`rayVersion` 字段指定Ray集群中使用的Ray版本。
`rayVersion` 用于填充某些配置字段的默认值。
RayCluster CR 中指定的容器镜像版本与 CR 的 `rayVersion` 版本是相同的。
如果使用夜版或开发版镜像。则可以设置 `rayVersion` 为 Ray 的最新发行版本。

## Pod 配置: headGroupSpec 及 workerGroupSpecs

从较高层面来看，RayCluster 是 Kubernetes Pod 的集合，类似于 Kubernetes Deployment 或 StatefulSet。就像 Kubernetes 内置程序一样，配置的关键部分是
* Pod 规格
* 规模信息（需要多少个 Pod）

Deployment 和 `RayCluster` 之间的主要区别在于 `RayCluster` 是专门用于运行 Ray 应用的。
一个Ray 集群包括

* 负责 Ray 集群全局控制进程的 **head pod** 。
  Head Pod 同样可以运行 Ray 任务和 Actor。
* 任意数量的 **worker pods**，运行 Ray 任务和 actor。
  Worker 来自具有相同配置的 **worker groups** 。
  针对每个 worker 组，我们必须指定 **replicas**，即我们想要的组的 pod 数量。

head pod 的配置在 `headGroupSpec` 指定，而worker pod 的配置在 `workerGroupSpecs` 指定。
可能有多个工作组，每个组都有自己的配置。 
`workerGroupSpec` 中的 `replicas` 字段指定了该组在集群中需要保持的 worker pod 数量。
每个 `workerGroupSpec` 同样有 `minReplicas` 和
`maxReplicas` 字段，这些字段对于要启用 {ref}`自动缩放 <kuberay-autoscaling-config>` 非常重要。

### Pod 模板
`headGroupSpec` 和 `workerGroupSpec` 大部分配置都在 `template` 字段。
`template` 是 Kubernetes Pod 模板，他决定了组中的 pod 配置。
以下是 pod `template` 需注意的一些字段。

#### 容器
Ray Pod 模板至少指定一个容器，即运行 Ray 进程的容器。
Ray pod 模板还可以指定额外的 sidecar 容器，用于处理诸如 {ref}`log <kuberay-logging>` 等目的。
但是，KubeRay operator 假定容器列表中的第一个容器是主 Ray 容器。
因此，请确保在主 Ray 容器
**之后** 指定任何 sidecar 容器。换句话说，Ray 容器应该是 `containers` 列表中的 **第一个**。

#### 资源
为每个组规范指定容器 CPU 和内存请求和限制非常重要。
对于 GPU 工作负载，您可能还希望指定 GPU 限制。
例如，如果使用 Nvidia GPU 设备插件并且您希望指定一个可以访问 2 个 GPU 的 Pod，则设置 `nvidia.com/gpu:2`。
有关 GPU 支持的更多详细信息，请参阅 {ref}`本指南 <kuberay-gpu>` 。

理想的做法是调整每个 Ray pod 的大小，使其占据其调度所在的整个 Kubernetes 节点。
换句话说，最好每个 Kubernetes 节点运行一个大型 Ray pod。
一般来说，使用几个大型 Ray pod 比使用许多小型 Ray pod 更有效。较少大型 Ray pod 的模式具有以下优点：
- 更有效地使用每个 Ray pod 的共享内存对象存储
- 减少 Ray pod 之间的通信开销
- 减少每个 Pod Ray 控制结构（例如 Raylet）的冗余

Ray 容器配置中指定的CPU、GPU 和内存 **限制** 将自动通告给 Ray。
这些值将用作头或工作组中 Ray pod 的逻辑资源容量。
请注意，CPU 数量在转发给 Ray 之前将四舍五入到最接近的整数。
向 Ray 通告的资源容量可能会在 Ray 启动参数 {ref}`rayStartParams` 被覆盖。

另一方面，CPU、GPU 和内存 **请求** 将被 Ray 忽略。
因此，最好尽可能将 **资源请求设置为等于资源限制**。

#### 节点选择器和容忍度
可通过设置 Ray 工作组 pod 的 `nodeSelector` 和 `tolerations` 字段来控制调度。
具体来说，这些字段确定 Pod 可以调度到哪些 Kubernetes 节点上。
有关 Pod 到节点分配的更多信息，请参阅 [Kubernetes 文档](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)。

#### 镜像
`RayCluster` CR 的 Ray 容器镜像应与 CR 的 `spec.rayVersion` 版本一致。
如果使用夜版或开发版 Ray 镜像，最好指定 `spec.rayVersion` 为最新发布版本。

给定 Ray task 或 actor的代码依赖项必须安装在可能运行该task 或 actor的每个 Ray 节点上。
要实现这一点，最简单的方法是对 Ray 头和所有工作组使用相同的 Ray 镜像。
无论如何，请确保 CR 中的所有 Ray 图像都具有相同的 Ray 版本和 Python 版本。
要在集群中分发自定义代码依赖项，您可以使用 [官方 Ray 镜像](https://hub.docker.com/r/rayproject/ray>) 作为基础镜像来构建自定义容器镜像。
请参阅 {ref}`本指南 <docker-images>` 以了解有关官方 Ray 镜像的更多信息。
对于面向迭代和开发的动态依赖管理，您还可以使用 {ref}`运行时环境 <runtime-environments>`。

#### metadata.name 和 metadata.generateName
KubeRay operator 会忽略用户设置的 `metadata.name` and `metadata.generateName`。
KubeRay operator 会自动生成一个 `generateName` 以避免名称冲突。
参考 [KubeRay issue #587](https://github.com/ray-project/kuberay/pull/587) 获取更多信息。

(rayStartParams)=
## Ray 启动参数
每组规范中的 ``rayStartParams`` 字段是是 Ray 容器 `ray start` 入口点参数的字符串映射。有关参数的完整列表，请参阅 {ref}`ray start <ray-start-doc>` 文档。
我们特别注意以下参数：

### dashboard-host
对于大多数用例，对于 Ray head pod，该字段应设置为“0.0.0.0”。
这是在 Ray 集群外部公开 Ray 仪表板所必需的。 （未来版本可能会默认设置此参数。）

(kuberay-num-cpus)=
### num-cpus
此可选字段告诉 Ray 调度程序和自动缩放程序有多少个 CPU 可用于 Ray pod。
CPU 计数可以从组规范的 pod 中 `template` 指定的 Kubernetes 资源限制中自动检测。
但是，有时覆盖此自动检测到的值很有用。例如，`num-cpus:"0"` 的设置将阻止在头上调度具有非零 CPU 要求的 Ray 工作负载。
请注意，所有 Ray 启动参数的值（包括 `num-cpus`）必须以 **字符串** 形式提供。

### num-gpus
该字段指定 Ray 容器可用的 GPU 数量。
在未来的 KubeRay 版本中，GPU 的数量将从 Ray 容器资源限制中自动检测。
请注意，所有 Ray 启动参数的值（包括 `num-gpus`）必须以 **字符串** 形式提供。

### memory
Ray 可用的内存是从 Kubernetes 资源限制中自动检测的。如果您愿意，您可以通过在 `rayStartParams.memory` 下设置所需的内存值（以字节为单位）来覆盖此自动检测到的值。
请注意，所有 Ray 启动参数的值（包括 `memory`）必须以 **字符串** 形式提供。

### resources
该字段可用于指定 Ray pod 的自定义资源容量。这些资源容量将通告给 Ray 调度器和 Ray 自动缩放器。
例如，以下注释将 Ray pod 标记为具有 1 个 ``Custom1`` 容量单位和 5 个 ``Custom2`` 容量单位。
```yaml
rayStartParams:
    resources: '"{\"Custom1\": 1, \"Custom2\": 5}"'
```
然后，您可以使用诸如 `@ray.remote(resources={"Custom2": 1})` 之类的注释来注释任务和 actor。
 Ray 调度程序和自动缩放程序将采取适当的操作来调度此类任务。

请注意用于表达资源字符串的格式。特别要注意的是，反斜杠作为字符串中的实际字符出现。
如果您以编程方式指定 `RayCluster` 则可能必须
[转义反斜杠](https://github.com/ray-project/ray/blob/cd9cabcadf1607bcda1512d647d382728055e688/python/ray/tests/kuberay/test_autoscaling_e2e.py#L92) 以确保它们作为字符串的一部分进行处理。

`rayStartParams.resources` 字段只能用于自定义资源。
`CPU`、 `GPU` 和 `memory` 这些键是禁止的。
如果您需要为这些资源字段指定覆盖，
请使用 Ray 启动参 `num-cpus`、 `num-gpus` 和 `memory`。

(kuberay-networking)=
## 服务发现及网络
### Ray 头服务
KubeRay operator 自动配置 Kubernetes 服务，公开 Ray head pod 的多个服务的默认端口，包括
- Ray 客户端 (默认端口 10001)
- Ray 仪表板 (默认端口 8265)
- Ray GCS 服务器 (默认端口 6379)
- Ray Serve (默认端口 8000)
- Ray Prometheus 指标 (默认端口 8080)

RayCluster 配置的 Kubernetes 服务的名称是 `metadata.name` 跟着 <nobr>`head-svc`</nobr> 后缀。
对于本页给出的示例 CR，头服务的名称将为
<nobr>`raycluster-example-head-svc`</nobr>。 Kubernetes 网络 (`kube-dns`) 之后允许我们使用名称
<nobr>`raycluster-example-head-svc`</nobr> 访问 Ray Head 服务。
例如，可以使用以下命令从同一 Kubernetes 命名空间中的 pod 访问 Ray Client 服务器
```python
ray.init("ray://raycluster-example-head-svc:10001")
```
可以使用以下命令从另一个命名空间中的 Pod 访问 Ray Client 服务器
```python
ray.init("ray://raycluster-example-head-svc.default.svc.cluster.local:10001")
```
这假设 Ray 集群部署到默认的 Kubernetes 命名空间中。如果 Ray 集群部署在非默认命名空间中，请使用该命名空间代替 `default`.）

### ServiceType, Ingresses
Ray Client 和其他服务可以使用端口转发或入口暴露在 Kubernetes 集群外部。
访问 Ray head 服务的最简单方法是使用端口转发。

在集群外部公开头部服务的其他方法可能需要使用 LoadBalancer 或 NodePort 类型的服务。
设置 `headGroupSpec.serviceType`为适合您的应用程序的类型。

您可能希望设置一个入口以将 Ray head 的服务暴露在集群之外。有关详细信息，
请参阅 [KubeRay 文档][IngressDoc] 。

### 指定非默认端口
如果您希望覆盖 Ray head 服务公开的端口，您可以通过在  `headGroupSpec` 下指定 Ray head 容器的 `ports` 列表来实现。
头服务的非默认端口列表的示例。
```yaml
ports:
- containerPort: 6380
  name: gcs
- containerPort: 8266
  name: dashboard
- containerPort: 10002
  name: client
```
如果指定了头容器的 `ports` 列表，Ray头服务将精确暴露列表中的端口。
在上面的示例中，头服务将仅公开三个端口；
特别是，不会为 Ray Serve 暴露任何端口。

为了使 Ray 头实际使用端口列表中指定的非默认端口，您还必须指定相关的 `rayStartParams` 参数。 对于上面的例子，
```yaml
rayStartParams:
  port: "6380"
  dashboard-port: "8266"
  ray-client-server-port: "10002"
  ...
```
(kuberay-config-miscellaneous)=
## Pod 和容器生命周期：preStopHook

建议每个 Ray 容器的配置包含以下阻塞块：
```yaml
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh","-c","ray stop"]
```
为了确保正常终止，在 Ray pod 终止之前执行 `ray stop` 。

[IngressDoc]: kuberay-ingress
