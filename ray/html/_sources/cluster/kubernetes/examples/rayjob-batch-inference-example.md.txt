(kuberay-batch-inference-example)=

# RayJob 批量推理示例

此示例演示如何使用 RayJob 自定义资源在 Ray 集群上运行批量推理作业。

此示例使用图像分类工作负载，该工作负载基于 <https://docs.ray.io/en/latest/data/examples/huggingface_vit_batch_prediction.html>。请参阅该页面以获取代码的完整说明。

## 先决条件

您必须拥有一个正在运行的 Kubernetes 集群，并配置 `kubectl` 以使用它，并且有可用的 GPU。此示例提供了在 Google Kubernetes Engine (GKE) 上设置必要 GPU 的简短教程，但您可以使用任何带有 GPU 的 Kubernetes 集群。

## 步骤 0: 在 GKE 上创建 Kubernetes 集群（可选）

如果您已经拥有带有 GPU 的 Kubernetes 集群，则可以跳过此步骤。


否则，请按照 [本教程](kuberay-gke-gpu-cluster-setup)，但替换以下 GPU 节点池创建命令，以在 GKE 上创建具有四个 Nvidia T4 GPU 的 Kubernetes 集群：

```sh
gcloud container node-pools create gpu-node-pool \
  --accelerator type=nvidia-tesla-t4,count=4,gpu-driver-version=default \
  --zone us-west1-b \
  --cluster kuberay-gpu-cluster \
  --num-nodes 1 \
  --min-nodes 0 \
  --max-nodes 1 \
  --enable-autoscaling \
  --machine-type n1-standard-64
```

本示例使用四个 [Nvidia T4](https://cloud.google.com/compute/docs/gpus#nvidia_t4_gpus) GPU。机器类型为 `n1-standard-64`，具有 [64 vCPUs 和 240 GB 内存](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machine_types)。

## 步骤 1: 安装 KubeRay Operator

按照 [本文档](kuberay-operator-deploy) 从 Helm 存储库安装最新的稳定 KubeRay 控制器。

它应该在 CPU pod 上进行调度。

## 步骤 2: 提交 RayJob

创建 RayJob 自定义资源。 RayJob 规范定义在 [ray-job.batch-inference.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-job.batch-inference.yaml)。

使用 `curl` 下载文件：

```bash
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-job.batch-inference.yaml
```

请注意，该 `RayJob` 包含了创建 job 的 `RayCluster` 定义。在本教程中，我们使用具有 4 个 GPU 的单节点集群。对于生产用例，我们建议使用头节点没有 GPU 的多节点集群，以便 Ray 可以自动在工作节点上安排 GPU 工作负载，并且它们不会干扰头节点上的关键 Ray 进程。

请注意 `RayJob` 规范中的以下字段，它们指定了 Ray 镜像和 Ray 节点的 GPU 资源：

```yaml
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray-ml:2.6.3-gpu
              resources:
                limits:
                  nvidia.com/gpu: "4"
                  cpu: "54"
                  memory: "54Gi"
                requests:
                  nvidia.com/gpu: "4"
                  cpu: "54"
                  memory: "54Gi"
              volumeMounts:
                - mountPath: /home/ray/samples
                  name: code-sample
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-t4 # This is the GPU type we used in the GPU node pool.
```

要提交作业，请运行以下命令：

```bash
kubectl apply -f ray-job.batch-inference.yaml
```

使用 `kubectl describe rayjob rayjob-sample` 检查状态。

示例输出：

```
[...]
Status:
  Dashboard URL:          rayjob-sample-raycluster-j6t8n-head-svc.default.svc.cluster.local:8265
  End Time:               2023-08-22T22:48:35Z
  Job Deployment Status:  Running
  Job Id:                 rayjob-sample-ft8lh
  Job Status:             SUCCEEDED
  Message:                Job finished successfully.
  Observed Generation:    2
  Ray Cluster Name:       rayjob-sample-raycluster-j6t8n
  Ray Cluster Status:
    Endpoints:
      Client:        10001
      Dashboard:     8265
      Gcs - Server:  6379
      Metrics:       8080
    Head:
      Pod IP:             10.112.1.3
      Service IP:         10.116.1.93
    Last Update Time:     2023-08-22T22:47:44Z
    Observed Generation:  1
    State:                ready
  Start Time:             2023-08-22T22:48:02Z
Events:
  Type    Reason   Age   From               Message
  ----    ------   ----  ----               -------
  Normal  Created  36m   rayjob-controller  Created cluster rayjob-sample-raycluster-j6t8n
  Normal  Created  32m   rayjob-controller  Created k8s job rayjob-sample
```

要查看日志，首先使用 `kubectl get pods` 找到运行作业的 pod 的名称。

示例输出：

```bash
NAME                                        READY   STATUS      RESTARTS   AGE
kuberay-operator-8b86754c-r4rc2             1/1     Running     0          25h
rayjob-sample-raycluster-j6t8n-head-kx2gz   1/1     Running     0          35m
rayjob-sample-w98c7                         0/1     Completed   0          30m
```

ay 集群仍在运行，因为 `RayJob` 规范中未设置 `shutdownAfterJobFinishes` 。如果 `shutdownAfterJobFinishes` 设置为 `true`，则作业完成后集群将关闭。

接下来运行：

```text
kubetcl logs rayjob-sample-w98c7
```

获取 `RayJob` 中 `entrypoint` 命令的标准输出。示例输出：

```text
[...]
Running: 62.0/64.0 CPU, 4.0/4.0 GPU, 955.57 MiB/12.83 GiB object_store_memory:   0%|          | 0/200 [00:05<?, ?it/s]
Running: 61.0/64.0 CPU, 4.0/4.0 GPU, 999.41 MiB/12.83 GiB object_store_memory:   0%|          | 0/200 [00:05<?, ?it/s]
Running: 61.0/64.0 CPU, 4.0/4.0 GPU, 999.41 MiB/12.83 GiB object_store_memory:   0%|          | 1/200 [00:05<17:04,  5.15s/it]
Running: 61.0/64.0 CPU, 4.0/4.0 GPU, 1008.68 MiB/12.83 GiB object_store_memory:   0%|          | 1/200 [00:05<17:04,  5.15s/it]
Running: 61.0/64.0 CPU, 4.0/4.0 GPU, 1008.68 MiB/12.83 GiB object_store_memory: 100%|██████████| 1/1 [00:05<00:00,  5.15s/it]  
                                                                                                                             
2023-08-22 15:48:33,905 WARNING actor_pool_map_operator.py:267 -- To ensure full parallelization across an actor pool of size 4, the specified batch size should be at most 5. Your configured batch size for this operator was 16.
<PIL.Image.Image image mode=RGB size=500x375 at 0x7B37546CF7F0>
Label:  tench, Tinca tinca
<PIL.Image.Image image mode=RGB size=500x375 at 0x7B37546AE430>
Label:  tench, Tinca tinca
<PIL.Image.Image image mode=RGB size=500x375 at 0x7B37546CF430>
Label:  tench, Tinca tinca
<PIL.Image.Image image mode=RGB size=500x375 at 0x7B37546AE430>
Label:  tench, Tinca tinca
<PIL.Image.Image image mode=RGB size=500x375 at 0x7B37546CF7F0>
Label:  tench, Tinca tinca
2023-08-22 15:48:36,522 SUCC cli.py:33 -- -----------------------------------
2023-08-22 15:48:36,522 SUCC cli.py:34 -- Job 'rayjob-sample-ft8lh' succeeded
2023-08-22 15:48:36,522 SUCC cli.py:35 -- -----------------------------------
```
