(kuberay-ml-example)=

# Kubernetes 上的 Ray Train XGBoostTrainer

:::{note}
要了解 Kubernetes 上 Ray 的基础知识，我们建议先查看
{ref}`入门指南 <kuberay-quickstart>`。
:::


在本指南中，我们向您展示如何在 Kubernetes 基础架构上运行 Ray 机器学习工作负载示例。

我们将使用 100 GB 的训练集运行 Ray 的 {ref}`XGBoost 训练基准 <xgboost-benchmark>` 。
要了解有关使用 Ray 的 XGBoostTrainer 的更多信息，请查看 {ref}`the XGBoostTrainer 文档 <train-gbdt-guide>`。

## GCP 上的 Kubernetes 基础设施设置

本文档提供了 GCP 创建 Kubernetes 集群的说明，但任何主要云提供商都可以使用类似的设置。
如果您想在其他云提供商上设置 Kubernetes 集群，请查看 {ref}`入门指南 <kuberay-k8s-setup>` 。
如果您有现有的 Kubernetes 集群，则可以忽略此步骤。

```shell
# Set up a cluster on Google Kubernetes Engine (GKE)
gcloud container clusters create autoscaler-ray-cluster \
    --num-nodes=10 --zone=us-central1-c --machine-type e2-standard-16 --disk-size 1000GB

# (Optional) Set up a cluster with autopilot on Google Kubernetes Engine (GKE).
# The following command creates an autoscaling node pool with a 1 node minimum and a 10 node maximum.
# The 1 static node will be used to run the Ray head pod. This node may also host the KubeRay
# operator and Kubernetes system components. After the workload is submitted, 9 additional nodes will
# scale up to accommodate Ray worker pods. These nodes will scale back down after the workload is complete.
gcloud container clusters create autoscaler-ray-cluster \
    --num-nodes=1 --min-nodes 1 --max-nodes 10 --enable-autoscaling \
    --zone=us-central1-c --machine-type e2-standard-16 --disk-size 1000GB
```

确保已连接到 Kubernetes 集群。对于 GCP，您可以通过以下方式执行此操作：
* 导航到您的 GKE 集群页面，然后单击“连接”按钮。然后复制“命令行访问”。
* `gcloud container clusters get-credentials <your-cluster-name> --region <your-region> --project <your-project>` ([链接](https://cloud.google.com/sdk/gcloud/reference/container/clusters/get-credentials))
* `kubectl config use-context` ([链接](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/))

对于本指南中的工作负载，建议使用具有以下属性的
Kubernetes 节点池或节点组：
- 共 10 个节点
- 每个节点容量为 16 个 CPU 和 64 Gi 内存。对于主要的云提供商，合适的实例类型包括
    * m5.4xlarge（亚马逊网络服务）
    * Standard_D5_v2 (Azure)
    * e2-standard-16 (Google Cloud)
- 每个节点应配置1000 GB的磁盘空间（用于存储训练集）。

## 部署 KubeRay operator

设置 Kubernetes 集群后，部署 KubeRay 控制器。
有关此步骤的说明，请参阅 {ref}`入门指南 <kuberay-operator-deploy>` 。

## 部署 Ray cluster

现在我们准备部署将执行工作负载的 Ray 集群。

:::{tip}
我们将部署的 Ray 集群配置为每个 16-CPU Kubernetes 节点安排一个 Ray pod。
鼓励但不强制要求每个 Kubernetes 节点安排一个 Ray pod。
从广义上讲，使用几个大型 Ray pod 比使用多个小型 pod 更有效率。
:::

我们建议查看以下命令中应用的 [配置文件][ConfigLink] 。
```shell
kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/kubernetes/configs/xgboost-benchmark.yaml
```

将创建一个 Ray head pod 和 9 个 Ray worker pod。


```{admonition} 可选：部署自动扩展 Ray 集群
如果您已设置自动缩放节点组或池，
您可能希望通过应用配置 [xgboost-benchmark-autoscaler.yaml][ConfigLinkAutoscaling] 来部署自动缩放集群。
将创建一个 Ray head pod。一旦工作负载启动，Ray 自动缩放器将触发 Ray worker pod 的创建。
然后，Kubernetes 自动缩放将创建节点来放置 Ray pod。
```

## 运行工作负载

要观察 Ray 头 pod 的启动进度，请运行以下命令。

```shell
# If you're on MacOS, first `brew install watch`.
watch -n 1 kubectl get pod
```

一旦 Ray head pod 进入 `Running` 状态，我们就可以执行 XGBoost 工作负载了。
我们将使用 {ref}`Ray Job Submission <jobs-overview>` 来启动工作负载。

### 连接到集群。

在本指南中，我们使用端口转发作为试验 Ray 集群服务的简单方法。
请参阅 {ref}`网络说明 <kuberay-networking>` 以了解生产用例。

```shell
# Run the following blocking command in a separate shell.
kubectl port-forward --address 0.0.0.0 service/raycluster-xgboost-benchmark-head-svc 8265:8265
```

### 提交工作负载。

我们将使用 {ref}`Ray Job Python SDK <ray-job-sdk>` 提交 XGBoost 工作负载。

```{literalinclude} /cluster/doc_code/xgboost_submit.py
:language: python
```

要提交工作负载，请运行上述 Python 脚本。
该脚本在 [Ray 存储卡][XGBSubmit]。

```shell
# Download the above script.
curl https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/doc_code/xgboost_submit.py -o xgboost_submit.py
# Run the script.
python xgboost_submit.py
```

### 观察进度。

基准测试可能需要长达 60 分钟才能运行。
使用以下工具观察其进度。

#### Job 日志

要跟踪作业的日志，请使用上述提交脚本打印的命令。
```shell
# Substitute the Ray Job's submission id.
ray job logs 'raysubmit_xxxxxxxxxxxxxxxx' --follow --address http://127.0.0.1:8265
```

#### Kubectl

使用以下命令观察集群中的 Pod
```shell
# If you're on MacOS, first `brew install watch`.
watch -n 1 kubectl get pod
```

#### Ray 仪表盘

在浏览器中查看 `localhost:8265` 以访问 Ray Dashboard。

#### Ray 状态

使用以下工具观察自动缩放状态和 Ray 资源使用情况
```shell
# Substitute the name of your Ray cluster's head pod.
watch -n 1 kubectl exec -it raycluster-xgboost-benchmark-head-xxxxx -- ray status
```

:::{note}
在某些情况下，对于某些云提供商，
K8s API 服务器可能会在 Kubernetes 集群
调整大小事件期间短暂不可用。

如果发生这种情况，请不要担心 - Ray 工作负载应该不会中断。
对于本指南中的示例，请等到 API 服务器恢复，重新启动端口转发过程，
然后重新运行作业日志命令。
:::

### Job 完成

#### 基准测试结果

基准测试完成后，作业日志将显示结果：

```
Results: {'training_time': 1338.488839321999, 'prediction_time': 403.36653568099973}
```

基准测试的性能对底层云基础设施很敏感——您可能无法匹配
{ref}`基准测试文档中引用的数字 <xgboost-benchmark>`。

#### 模型参数
Ray head pod 中的 `model.json` 文件包含训练模型的参数。
其他结果数据将在head pod 中的 `ray_results` 目录中提供。
有关详细信息，请参阅 {ref}`the XGBoostTrainer 文档 <train-gbdt-guide>` 。

```{admonition} Scale-down
如果启用了自动缩放功能，Ray 工作线程 Pod 将在 60 秒后缩小。
Ray 工作线程 Pod 消失后，您的 Kubernetes 基础架构应缩小托管这些 Pod 的节点。
```

#### 清理
使用以下命令删除您的 Ray 集群：
```shell
kubectl delete raycluster raycluster-xgboost-benchmark
```
如果您使用公共云，请不要忘记清理底层节点组和/或 Kubernetes 集群。

[ConfigLink]:https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/kubernetes/configs/xgboost-benchmark.yaml
[ConfigLinkAutoscaling]: https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/kubernetes/configs/xgboost-benchmark-autoscaler.yaml
[XGBSubmit]: https://github.com/ray-project/ray/blob/releases/2.0.0/doc/source/cluster/doc_code/xgboost_submit.py
