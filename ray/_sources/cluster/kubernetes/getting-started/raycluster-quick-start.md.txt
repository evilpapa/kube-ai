(kuberay-raycluster-quickstart)=

# RayCluster 快速开始

本指南向您展示如何管理 Kubernetes 上的 Ray 集群并与之交互。

## 准备

* 安装 [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) (>= 1.19)、 [Helm](https://helm.sh/docs/intro/install/) (>= v3.4), 以及 [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)。
* 确保您的 Kubernetes 集群至少有 4 个 CPU 和 4 GB RAM。

## Step 1: 创建Kubernetes集群

此步骤使用 [Kind](https://kind.sigs.k8s.io/) 创建一个本地 kubernetes 集群。如果您已经有 Kubernetes 集群，可以跳过此步骤。

```sh
kind create cluster --image=kindest/node:v1.23.0
```

(kuberay-operator-deploy)=
## Step 2: 部署 KubeRay Operator 

通过 [Helm chart 仓库](https://github.com/ray-project/kuberay-helm) 部署 KubeRay Operator 。

```sh
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install both CRDs and KubeRay operator v1.0.0-rc.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# Confirm that the operator is running in the namespace `default`.
kubectl get pods
# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          27s
```

KubeRay 为 Operator 安装提供了多种选项， 例如 Helm、Kustomize、以及一个独立命名空间的控制器。 如需了解更多信息，请参阅 [KubeRay 文档中的安装说明](https://ray-project.github.io/kuberay/deploy/installation/)。

## Step 3: 部署 RayCluster 自定义资源

一旦 KubeRay 操作器运行，我们就准备部署 RayCluster。为此，我们在 `default` 命名空间中创建 RayCluster 自定义资源 (CR)。

```sh
# Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
helm install raycluster kuberay/ray-cluster --version 1.0.0-rc.0

# Once the RayCluster CR has been created, you can view it by running:
kubectl get rayclusters

# NAME                 DESIRED WORKERS   AVAILABLE WORKERS   STATUS   AGE
# raycluster-kuberay   1                 1                   ready    72s
```

KubeRay operator 将检测 RayCluster 对象。然后，Operator 将通过创建 Head Pod 和 worker Pod 来启动 Ray 集群。要查看 Ray 集群的 pod，请运行以下命令：

```sh
# View the pods in the RayCluster named "raycluster-kuberay"
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay

# NAME                                          READY   STATUS    RESTARTS   AGE
# raycluster-kuberay-head-vkj4n                 1/1     Running   0          XXs
# raycluster-kuberay-worker-workergroup-xvfkr   1/1     Running   0          XXs
```

等待 Pod 达到“运行”状态。这可能需要几分钟 - 大部分时间都花在下载 Ray 镜像上。
如果您的 Pod 陷入 Pending 状态，您可以通过 `kubectl describe pod raycluster-kuberay-xxxx-xxxxx` 检查错误并确保您的 Docker 资源限制设置得足够高。
请注意，在生产场景中，您将需要使用更大的 Ray pod。 事实上，将每个 Ray pod 的大小调整为占用整个 Kubernetes 节点是有利的。 参阅 [配置指南](kuberay-config) 了解更多详细信息。

## Step 4: 在 RayCluster 上运行应用程序

现在，让我们与我们部署的 RayCluster 进行交互。

### 方法1：在 head Pod 中执行 Ray 作业

试验 RayCluster 最直接的方法是直接在 head pod 中执行。
首先，识别 RayCluster 的 Head Pod：

```sh
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
echo $HEAD_POD
# raycluster-kuberay-head-vkj4n

# Print the cluster resources.
kubectl exec -it $HEAD_POD -- python -c "import ray; ray.init(); print(ray.cluster_resources())"

# 2023-04-07 10:57:46,472 INFO worker.py:1243 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
# 2023-04-07 10:57:46,472 INFO worker.py:1364 -- Connecting to existing Ray cluster at address: 10.244.0.6:6379...
# 2023-04-07 10:57:46,482 INFO worker.py:1550 -- Connected to Ray cluster. View the dashboard at http://10.244.0.6:8265 
# {'object_store_memory': 802572287.0, 'memory': 3000000000.0, 'node:10.244.0.6': 1.0, 'CPU': 2.0, 'node:10.244.0.7': 1.0}
```

### Method 2: 通过 [ray job 提交 SDK](jobs-quickstart) 提交 Ray job 到 RayCluster

与方法 1 不同，此方法不需要您在 Ray head pod 中执行命令。
相反，您可以使用 [Ray 作业提交 SDK](jobs-quickstart) 通过 Ray 监听作业请求的 Ray Dashboard 端口（默认为 8265）将 Ray 作业提交到 RayCluster。
KubeRay operator 针对 Ray head Pod 配置了 [Kubernetes service](https://kubernetes.io/docs/concepts/services-networking/service/) 。

```sh
kubectl get service raycluster-kuberay-head-svc

# NAME                          TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)                                         AGE
# raycluster-kuberay-head-svc   ClusterIP   10.96.93.74   <none>        8265/TCP,8080/TCP,8000/TCP,10001/TCP,6379/TCP   15m
```

现在我们有了服务的名称，我们可以使用端口转发来访问 Ray Dashboard 端口（默认为 8265）。

```sh
# Execute this in a separate shell.
kubectl port-forward --address 0.0.0.0 service/raycluster-kuberay-head-svc 8265:8265

# Visit ${YOUR_IP}:8265 in your browser for the Dashboard (e.g. 127.0.0.1:8265)
```

注意：我们在本指南中使用端口转发作为试验 RayCluster 服务的简单方法。对于生产用例，您通常会
- 从 Kubernetes 集群内访问服务或
- 使用入口控制器将服务公开到集群外部。

有关详细信息，请参阅 {ref}`networking notes <kuberay-networking>` 。

现在我们可以访问 Dashboard 端口，我们可以将作业提交到 RayCluster：

```sh
# The following job's logs will show the Ray cluster's total resource capacity, including 2 CPUs.
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## Step 5: 清理

```sh
# [Step 5.1]: Delete the RayCluster CR
# Uninstall the RayCluster Helm chart
helm uninstall raycluster
# release "raycluster" uninstalled

# Note that it may take several seconds for the Ray pods to be fully terminated.
# Confirm that the RayCluster's pods are gone by running
kubectl get pods

# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          XXm

# [Step 5.2]: Delete the KubeRay operator
# Uninstall the KubeRay operator Helm chart
helm uninstall kuberay-operator
# release "kuberay-operator" uninstalled

# Confirm that the KubeRay operator pod is gone by running
kubectl get pods
# No resources found in default namespace.

# [Step 5.3]: Delete the Kubernetes cluster
kind delete cluster
```
