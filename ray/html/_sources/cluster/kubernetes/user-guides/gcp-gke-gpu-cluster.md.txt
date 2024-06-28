(kuberay-gke-gpu-cluster-setup)=

# 为 KubeRay 启动带有 GPU 的 Google Cloud GKE 集群

请参阅 <https://cloud.google.com/kubernetes-engine/docs/how-to/gpus> 了解完整详情，或继续阅读以快速入门。

## 步骤 1: 在 GKE 上创建 Kubernetes 集群

在本地计算机或 [Google Cloud Shell](https://cloud.google.com/shell) 运行和所有以下代码。如果运行在本机，需要安装 [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)。以下命令创建了一个名为 `kuberay-gpu-cluster` 带有 1 个 CPU 节点在 `us-west1-b` 区域的 Kubernetes 集群。本示例使用具有 4 个 vCPU 和 16 GB RAM 的  `e2-standard-4` 机器型号。

```sh
gcloud container clusters create kuberay-gpu-cluster \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-west1-b --machine-type e2-standard-4
```

> 注意：您还可以从 [Google Cloud Console](https://console.cloud.google.com/kubernetes/list) 创建集群。

## 步骤 2: 创建 GPU 节点池

运行以下命令为 Ray GPU 工作器创建 GPU 节点池。您也可以从 Google Cloud Console 创建它： <https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#console>

```sh
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-l4-vws,count=1,gpu-driver-version=default \
  --zone us-west1-b \
  --cluster kuberay-gpu-cluster \
  --num-nodes 1 \
  --min-nodes 0 \
  --max-nodes 1 \
  --enable-autoscaling \
  --machine-type g2-standard-4 \
```

该 `--accelerator` 标志指定节点池中每个节点的 GPU 类型和数量。此示例使用 [NVIDIA L4](https://cloud.google.com/compute/docs/gpus#l4-gpus) GPU。 `g2-standard-4` 机器类型有 1 个 GPU、24 GB GPU 内存、4 个 vCPU 和 16 GB RAM。

.. note::

    GKE automatically installs the GPU drivers for you.  For more details, see [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#create-gpu-pool-auto-drivers).

.. note::

    GKE automatically configures taints and tolerations so that only GPU pods are scheduled on GPU nodes.  For more details, see [GKE documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#create)

## 步骤 3: 配置 `kubectl` 连接集群

运行以下命令下载 Google Cloud 凭据并配置 Kubernetes CLI 以使用它们。

```sh
gcloud container clusters get-credentials kuberay-gpu-cluster --zone us-west1-b
```

有关更多详细信息，请参阅 [GKE 文档](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)。
