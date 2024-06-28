(kuberay-eks-gpu-cluster-setup)=

# 为 KubeRay 启动带有 GPU 的 Amazon EKS 集群

本指南将引导您完成专门为 KubeRay 创建具有 GPU 节点的 Amazon EKS 集群的步骤。
此处概述的配置可应用于文档中的大多数 KubeRay 示例。

## 步骤 1: 在 Amazon EKS 上创建 Kubernetes 集群

按照此 [AWS 文档](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#) 中的前两个步骤：
(1) 创建您的 Amazon EKS 集群 和 (2) 配置您的计算机以与您的集群进行通信。

## 步骤 2: 为 Amazon EKS 集群创建节点组

根据 [AWS 文档](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#) 按照 "步骤 3: 创建节点" 创建节点组。
以下部分提供了更详细的信息。

### 创建 CPU 节点组

通常，避免在 Ray head 上运行 GPU 工作负载。
为除 Ray GPU 工作器之外的所有 Pod（例如 KubeRay 运算符、Ray head 和 CoreDNS Pod）创建 CPU 节点组。

以下是适用于文档中大多数 KubeRay 示例的常用配置：
  * 实例类型： [**m5.xlarge**](https://aws.amazon.com/ec2/instance-types/m5/) (4 vCPU; 16 GB RAM)
  * 磁盘大小： 256 GB
  * 所需尺寸：1，最小尺寸：0，最大尺寸：1

### 创建 GPU 节点组

为 Ray GPU 工作者创建一个 GPU 节点组。

1. 以下是适用于文档中大多数 KubeRay 示例的常用配置：
   * AMI 类型：Bottlerocket NVIDIA (BOTTLEROCKET_x86_64_NVIDIA)
   * 实例类型： [**g5.xlarge**](https://aws.amazon.com/ec2/instance-types/g5/) （1 GPU；24 GB GPU 内存；4 vCPU；16 GB RAM）
   * 磁盘大小：1024 GB
   * 所需尺寸：1，最小尺寸：0，最大尺寸：1

> **注意:** 如果您遇到 `kubectl` 权限问题，请参考 [AWS 文档](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#) 按照 "步骤 2: 配置您的计算机以与您的集群通信" 进行操作

2. 请安装 NVIDIA 设备插件。 （注意：如果您在上面的步骤中使用了 `BOTTLEROCKET_x86_64_NVIDIA` AMI。）
   * 安装 DaemonSet for NVIDIA 设备插件以在 Amazon EKS 集群中运行支持 GPU 的容器。你可参考 [Amazon EKS 优化的加速 Amazon Linux AMIs](https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami)
   或 [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) 了解更多详细信息。
   * 如果 GPU 节点有污点，则添加 `tolerations` 到 `nvidia-device-plugin.yml` 启用 DaemonSet 在 GPU 节点上调度 Pod。

```sh
# Install the DaemonSet
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml

# Verify that your nodes have allocatable GPUs. If the GPU node fails to detect GPUs,
# please verify whether the DaemonSet schedules the Pod on the GPU node.
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# Example output:
# NAME                                GPU
# ip-....us-west-2.compute.internal   4
# ip-....us-west-2.compute.internal   <none>
```

3. 添加 Kubernetes 污点以防止在此 GPU 节点组上调度 CPU Pod。对于 KubeRay 示例，将以下污点添加到 GPU 节点： `Key: ray.io/node-type, Value: worker, Effect: NoSchedule`，并包含与 GPU Ray 工作 Pod 对应的污点 `tolerations`。

> 警告：GPU 节点非常昂贵。如果您不再需要该集群，请记得将其删除。

## 步骤 3: 验证节点组

> **注意:** 如果您遇到  `eksctl` 权限问题，请导航至您的 AWS 账户的网页
并从“命令行或编程访问” 页面复制
凭证环境变量，包括 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` 和 `AWS_SESSION_TOKEN`,

```sh
eksctl get nodegroup --cluster ${YOUR_EKS_NAME}

# CLUSTER         NODEGROUP       STATUS  CREATED                 MIN SIZE        MAX SIZE        DESIRED CAPACITY        INSTANCE TYPE   IMAGE ID                        ASG NAME                           TYPE
# ${YOUR_EKS_NAME}     cpu-node-group  ACTIVE  2023-06-05T21:31:49Z    0               1               1                       m5.xlarge       AL2_x86_64                      eks-cpu-node-group-...     managed
# ${YOUR_EKS_NAME}     gpu-node-group  ACTIVE  2023-06-05T22:01:44Z    0               1               1                       g5.12xlarge     BOTTLEROCKET_x86_64_NVIDIA      eks-gpu-node-group-...     managed
```
