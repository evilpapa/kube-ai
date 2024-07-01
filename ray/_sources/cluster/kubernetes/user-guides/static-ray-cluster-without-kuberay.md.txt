(deploy-a-static-ray-cluster-without-kuberay)=

# （高级）在没有 KubeRay 的情况下部署静态 Ray 集群

Ray 的这种部署方法不再需要使用 CustomResourceDefinitions (CRD)。
相比之下，CRD 是使用 KubeRay 的先决条件。 
KubeRay operator 是其关键组件之一，通过监视 Kubernetes 事件（创建/删除/更新）来管理 Ray 集群资源。
尽管 KubeRay 运算符可以在单个名称空间内运行，但 CRD 的使用具有集群范围内的范围。
如果没有必要的 Kubernetes 管理员权限来部署 KubeRay，本文档介绍了一种在不使用 KubeRay 的情况下将静态 Ray 集群部署到 Kubernetes 的方法。
但需要注意的是，这种部署方式缺少 KubeRay 提供的内置自动伸缩功能。

## 准备

### 安装最新的 Ray 版本

对于使用 {ref}`Ray Job Submission <jobs-overview>` 与远程集群进行交互，此步骤是必需的。

```
! pip install -U "ray[default]"
```

有关更多详细信息，请参阅 {ref}`安装`。

### 安装 kubectl

为了与 Kubernetes 交互，我们将使用 kubectl。安装说明可以在 [Kubernetes 文档](https://kubernetes.io/docs/tasks/tools/#kubectl) 找到。

### 访问 Kubernetes 集群

我们需要访问 Kubernetes 集群。有两种选择：

1. 配置对远程 Kubernetes 集群的访问
**或**

2.通过 [安装 kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)在本地运行示例。通过运行以下命令启动您的 [kind](https://kind.sigs.k8s.io/) 集群：

```
! kind create cluster
```

要执行本指南中的示例，请确保您的 Kubernetes 集群（或本地 Kind 集群）可以处理 3 个 CPU 和 3Gi 内存的额外资源请求。
另外，请确保您的 Kubernetes 集群和 Kubectl 版本至少为 1.19。

### 部署 Redis 以实现容错

请注意， [Kubernetes 部署配置文件](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster.with-fault-tolerance.yaml) 有一个部分用于将 Redis 部署到 Kubernetes，以便 Ray 头可以写入 GCS 元数据。
如果Kubernetes上已经部署了Redis，则本节可以省略。

## 部署静态 Ray 集群

在本节中，我们将在不使用 KubeRay 的情况下将静态 Ray 集群部署到 `default` 空间。
使用其他命名空间，请在 kubectl 命令中指定命名空间：

`kubectl -n <your-namespace> ...`

```
# Deploy a sample Ray Cluster from the Ray repo:

! kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster.with-fault-tolerance.yaml

# Note that the Ray cluster has fault tolerance enabled by default using the external Redis. 
# Please set the Redis IP address in the config.

# The password is currently set as '' for the external Redis. 
# Please download the config file and substitute the real password for the empty string if the external Redis has a password.
```

Ray 集群部署完成后，您可以通过运行以下命令查看头节点和 worker 节点的 Pod

```
! kubectl get pods

# NAME                                             READY   STATUS    RESTARTS   AGE
# deployment-ray-head-xxxxx                        1/1     Running   0          XXs
# deployment-ray-worker-xxxxx                      1/1     Running   0          XXs
# deployment-ray-worker-xxxxx                      1/1     Running   0          XXs
```

等待 Pod 达到该 `Running` 状态。这可能需要几分钟 - 大部分时间都花在下载 Ray 镜像上。
在单独的 shell 中，您可能希望使用以下命令实时观察 pod 的状态：

```
# If you're on MacOS, first `brew install watch`.
# Run in a separate shell:

! watch -n 1 kubectl get pod
```

如果您的 Pod 陷入 `Pending` 状态，您可以通过 `kubectl describe pod deployment-ray-head-xxxx-xxxxx`
检查错误并确保您的 Docker 资源限制设置得足够高。

请注意，在生产场景中，您将需要使用更大的 Ray pod。事实上，将每个 Ray pod 的大小调整为占用整个 Kubernetes 节点是有利的。请参阅 [配置指南](kuberay-config) 获取更多信息。

## 为静态 Ray 集群部署网络策略

如果您的 Kubernetes 对 Pod 有默认的拒绝网络策略，则需要手动创建网络策略以允许 Ray 集群中的头节点和 worker 节点之间进行双向通信，如 [端口配置文档](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations) 所述。

```
# Create a sample network policy for the static Ray cluster from the Ray repo:
! kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster-networkpolicy.yaml
```

部署网络策略后，您可以通过运行以下命令查看静态 Ray 集群的网络策略

```
! kubectl get networkpolicies

# NAME                               POD-SELECTOR                           AGE
# ray-head-egress                    app=ray-cluster-head                   XXs
# ray-head-ingress                   app=ray-cluster-head                   XXs
# ray-worker-egress                  app=ray-cluster-worker                 XXs
# ray-worker-ingress                 app=ray-cluster-worker                 XXs
```

### 外部 Redis 集成以实现容错

Ray 默认情况下使用内部键值存储，称为全局控制存储 (GCS)。 
GCS 运行在头节点上并存储集群元数据。这种方法的一个缺点是头节点如果崩溃就会丢失元数据。 
Ray 还可以将此元数据写入外部 Redis，以实现可靠性和高可用性。
通过此设置，静态 Ray 集群可以从头节点崩溃中恢复并容忍 GCS 故障，而不会丢失与 worker 节点的连接。

要使用此功能，我们需在 [Kubernetes 部署配置文件](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster.with-fault-tolerance.yaml) 的头节点部分传入 `RAY_REDIS_ADDRESS` 环境变量和 `--redis-password`。

## 在静态 Ray 集群上运行应用程序

在本节中，我们将与刚刚部署的静态 Ray 集群进行交互。

### 使用 kubectl exec 访问集群

与使用 KubeRay 部署的 Ray 集群相同，我们可以直接在 head pod 中执行并运行 Ray 程序。

首先，运行以下命令来获取 head pod：

```
! kubectl get pods --selector=app=ray-cluster-head

# NAME                                             READY   STATUS    RESTARTS   AGE
# deployment-ray-head-xxxxx                        1/1     Running   0          XXs
```

我们现在可以在之前识别的 Head Pod 上执行 Ray 程序。以下命令连接到 Ray Cluster，然后终止 Ray 程序。

```
# Substitute your output from the last cell in place of "deployment-ray-head-xxxxx"

! kubectl exec deployment-ray-head-xxxxx -it -c ray-head -- python -c "import ray; ray.init('auto')"
# 2022-08-10 11:23:17,093 INFO worker.py:1312 -- Connecting to existing Ray cluster at address: <IP address>:6380...
# 2022-08-10 11:23:17,097 INFO worker.py:1490 -- Connected to Ray cluster. View the dashboard at ...
```

尽管上述单元对于在 Ray 集群上偶尔执行非常有用，但在 Ray 集群上运行应用程序的推荐方法是使用 [Ray Jobs](jobs-quickstart)。

### Ray Job 提交

要为 Ray 作业提交设置 Ray 集群，必须确保客户端可以访问 Ray 作业端口。 
Ray 通过头节点上的 Dashboard 服务器接收作业请求。

首先，我们需要识别Ray头节点。静态 Ray 集群配置文件设置了一个针对 Ray head pod 的
[Kubernetes service](https://kubernetes.io/docs/concepts/services-networking/service/) 。
该服务让我们可以与 Ray 集群进行交互，而无需直接在 Ray 容器中执行命令。
要识别我们示例集群的 Ray head 服务，请运行：

```
! kubectl get service service-ray-cluster

# NAME                             TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                            AGE
# service-ray-cluster              ClusterIP   10.92.118.20   <none>        6380/TCP,8265/TCP,10001/TCP...     XXs
```

现在我们有了服务的名称，我们可以使用端口转发来访问 Ray Dashboard 端口（默认为 8265）。

```
# Execute this in a separate shell.
# Substitute the service name in place of service-ray-cluster

! kubectl port-forward service/service-ray-cluster 8265:8265
```

现在我们可以访问 Dashboard 端口，我们可以将作业提交到 Ray Cluster 来执行：

```
! ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## 清理

### 删除 Ray Cluster

删除静态 Ray 集群服务和部署

```
! kubectl delete -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster.with-fault-tolerance.yaml
```

删除静态 Ray 集群网络策略

```
! kubectl delete -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster-networkpolicy.yaml
```
