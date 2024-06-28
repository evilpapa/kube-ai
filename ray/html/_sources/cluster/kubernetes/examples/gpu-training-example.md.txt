(kuberay-gpu-training-example)=

# 在 Kubernetes 上使用 GPU 训练 PyTorch ResNet 模型
本指南在 Kubernetes 基础架构上使用 GPU 运行示例 Ray 机器学习训练工作负载。它使用 1 GB 的训练集运行 Ray 的 {ref}`PyTorch i图像训练基准 <pytorch_gpu_training_benchmark>`。

:::{note}
要了解 Kubernetes 上 Ray 的基础知识，我们建议先查看
{ref}`入门指南 <kuberay-quickstart>`。
:::

请注意，Kubernetes 和 Kubectl 至少需要 1.19 版本。

## 端到端工作流
以下脚本总结了 GPU 训练的端到端工作流程。这些说明适用于 GCP，但类似的设置适用于任何主要云提供商。以下脚本包括：
- 步骤 1：在 GCP 上设置 Kubernetes 集群。
- 第 2 步：使用 KubeRay Operator 在 Kubernetes 上部署 Ray 集群。
- 步骤 3：运行 PyTorch 图像训练基准。
```shell
# Step 1: Set up a Kubernetes cluster on GCP
# Create a node-pool for a CPU-only head node
# e2-standard-8 => 8 vCPU; 32 GB RAM
gcloud container clusters create gpu-cluster-1 \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-central1-c --machine-type e2-standard-8

# Create a node-pool for GPU. The node is for a GPU Ray worker node.
# n1-standard-8 => 8 vCPU; 30 GB RAM
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --zone us-central1-c --cluster gpu-cluster-1 \
  --num-nodes 1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
  --machine-type n1-standard-8

# Install NVIDIA GPU device driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Step 2: Deploy a Ray cluster on Kubernetes with the KubeRay operator.
# Please make sure you are connected to your Kubernetes cluster. For GCP, you can do so by:
#   (Method 1) Copy the connection command from the GKE console
#   (Method 2) "gcloud container clusters get-credentials <your-cluster-name> --region <your-region> --project <your-project>"
#   (Method 3) "kubectl config use-context ..."

# Install both CRDs and KubeRay operator v1.0.0-rc.0.
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# Create a Ray cluster
kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/ray-cluster.gpu.yaml

# Set up port-forwarding
kubectl port-forward --address 0.0.0.0 services/raycluster-head-svc 8265:8265

# Step 3: Run the PyTorch image training benchmark.
# Install Ray if needed
pip3 install -U "ray[default]"

# Download the Python script
curl https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/doc_code/pytorch_training_e2e_submit.py -o pytorch_training_e2e_submit.py

# Submit the training job to your ray cluster
python3 pytorch_training_e2e_submit.py

# Use the following command to follow this Job's logs:
# Substitute the Ray Job's submission id.
ray job logs 'raysubmit_xxxxxxxxxxxxxxxx' --address http://127.0.0.1:8265 --follow
```
在本文档的其余部分，我们将更详细地分解上述工作流程。

## 步骤 1：在 GCP 上设置 Kubernetes 集群。
在本节中，我们将设置一个带有 CPU 和 GPU 节点池的 Kubernetes 集群。这些说明适用于 GCP，但类似的设置适用于任何主要云提供商。如果您已有带有 GPU 的 Kubernetes 集群，则可以忽略此步骤。

如果您是 Kubernetes 新手，并且计划在托管的 Kubernetes 服务上部署 Ray 工作负载，
我们建议您先查看本 {ref}`入门指南 <kuberay-k8s-setup>` 。

没有必要使用具有那么多 RAM 的集群来运行此示例（以下命令中每个节点的 RAM 大于 30GB）。 请随意
更新选项 `machine-type` 和资源需求 `ray-cluster.gpu.yaml` 。

在第一个命令中，我们创建了了一个 `gpu-cluster-1` Kubernetes 集群，它有一个 CPU 节点 （`e2-standard-8`: 8 vCPU; 32 GB 内存）。在第二个命令，
我们添加了一个新节点 （`n1-standard-8`: 8 vCPU; 30 GB 内存）带有一个 GPU (`nvidia-tesla-t4`) 到集群。

```shell
# Step 1: Set up a Kubernetes cluster on GCP.
# e2-standard-8 => 8 vCPU; 32 GB RAM
gcloud container clusters create gpu-cluster-1 \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-central1-c --machine-type e2-standard-8

# Create a node-pool for GPU
# n1-standard-8 => 8 vCPU; 30 GB RAM
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --zone us-central1-c --cluster gpu-cluster-1 \
  --num-nodes 1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
  --machine-type n1-standard-8

# Install NVIDIA GPU device driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

## 步骤2：使用KubeRay Operator在Kubernetes上部署Ray集群

要执行以下步骤，请确保您已连接到 Kubernetes 集群。对于 GCP，您可以通过以下方式执行此操作：
* 从 GKE 控制台复制连接命令
* `gcloud container clusters get-credentials <your-cluster-name> --region <your-region> --project <your-project>` ([链接](https://cloud.google.com/sdk/gcloud/reference/container/clusters/get-credentials))
* `kubectl config use-context` ([链接](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/))

第一个命令将把 KubeRay（ray-operator）部署到你的 Kubernetes 集群。第二个命令将在 KubeRay 的帮助下创建一个 ray 集群。

第三个命令用于将 `ray-head` pod  的 8265 端口映射到 **127.0.0.1:8265**。您可以检查
**127.0.0.1:8265** 以查看仪表板。最后一个命令用于通过提交一个简单的作业来测试您的 Ray 集群。
它是可选的。

```shell
# Step 2: Deploy a Ray cluster on Kubernetes with the KubeRay operator.
# Create the KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# Create a Ray cluster
kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/ray-cluster.gpu.yaml

# port forwarding
kubectl port-forward --address 0.0.0.0 services/raycluster-head-svc 8265:8265

# Test cluster (optional)
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## 步骤 3：运行 PyTorch 图像训练基准。
我们将使用 [Ray Job Python SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#ray-job-sdk) 提交 PyTorch 工作负载。

```{literalinclude} /cluster/doc_code/pytorch_training_e2e_submit.py
:language: python
```

要提交工作负载，请运行上述 Python 脚本。该脚本可在 [Ray 仓库](https://github.com/ray-project/ray/tree/master/doc/source/cluster/doc_code/pytorch_training_e2e_submit.py) 找到。
```shell
# Step 3: Run the PyTorch image training benchmark.
# Install Ray if needed
pip3 install -U "ray[default]"

# Download the Python script
curl https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/doc_code/pytorch_training_e2e_submit.py -o pytorch_training_e2e_submit.py

# Submit the training job to your ray cluster
python3 pytorch_training_e2e_submit.py
# Example STDOUT:
# Use the following command to follow this Job's logs:
# ray job logs 'raysubmit_jNQxy92MJ4zinaDX' --follow

# Track job status
# Substitute the Ray Job's submission id.
ray job logs 'raysubmit_xxxxxxxxxxxxxxxx' --address http://127.0.0.1:8265 --follow
```

## 清理
使用以下命令删除您的 Ray 集群和 KubeRay：
```shell
kubectl delete raycluster raycluster

# Please make sure the ray cluster has already been removed before delete the operator.
helm uninstall kuberay-operator
```
如果您使用公共云，请不要忘记清理底层节点组
和/或 Kubernetes 集群。
