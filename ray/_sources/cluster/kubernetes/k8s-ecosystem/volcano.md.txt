(kuberay-volcano)=
# KubeRay 与 Volcano 集成

[Volcano](https://github.com/volcano-sh/volcano) 是一个基于Kubernetes构建的批量调度系统。它提供了 Kubernetes 目前缺少的一套机制（组调度、作业队列、公平调度策略），而许多类别的批处理和弹性工作负载通常需要这些机制。 KubeRay 的 Volcano 集成可以在多租户 Kubernetes 环境中更有效地调度 Ray pod。

## 设置

### 步骤 1: 使用KinD创建Kubernetes集群
在终端中运行以下命令：

```shell
kind create cluster
```

### 步骤 2: 安装 Volcano

您需要在 Kubernetes 集群上成功安装 Volcano，然后才能启用 Volcano 与 KubeRay 的集成。
有关 Volcano 安装说明，请参阅 [快速入门指南](https://github.com/volcano-sh/volcano#quick-start-guide) 。

### 步骤 3: 安装具有批量调度功能的 KubeRay Operator 

部署 KubeRay Operator 使用 `--enable-batch-scheduler` 标志以启用 Volcano 批量调度支持。

使用 Helm 安装 KubeRay Operator 时，您应该使用以下两个选项之一：

* 在你的 [`values.yaml`](https://github.com/ray-project/kuberay/blob/753dc05dbed5f6fe61db3a43b34a1b350f26324c/helm-chart/kuberay-operator/values.yaml#L48)
文件设置 `batchScheduler.enabled` 为 `true` 
```shell
# values.yaml file
batchScheduler:
    enabled: true
```

* 在命令行上运行时传递 `--set batchScheduler.enabled=true` 标识：
```shell
# Install the Helm chart with --enable-batch-scheduler flag set to true 
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0 --set batchScheduler.enabled=true
```

### 步骤 4: 使用 Volcano 调度程序安装 RayCluster

RayCluster 自定义资源必须包含 `ray.io/scheduler-name: volcano` 标签来将集群 Pod 提交到 Volcano 进行调度的。

```shell
# Path: kuberay/ray-operator/config/samples
# Includes label `ray.io/scheduler-name: volcano` in the metadata.labels
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.volcano-scheduler.yaml
kubectl apply -f ray-cluster.volcano-scheduler.yaml

# Check the RayCluster
kubectl get pod -l ray.io/cluster=test-cluster-0
# NAME                                 READY   STATUS    RESTARTS   AGE
# test-cluster-0-head-jj9bg            1/1     Running   0          36s
```

您还可以在 RayCluster 元数据中提供以下标签：

- `ray.io/priority-class-name`: [Kubernetes](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/#priorityclass) 定义的集群优先级
  - 该标签仅在创建 `PriorityClass` 资源才有效
  - ```shell
    labels:
      ray.io/scheduler-name: volcano
      ray.io/priority-class-name: <replace with correct PriorityClass resource name>
    ```
- `volcano.sh/queue-name`: 集群提交到的 Volcano [queue](https://volcano.sh/en/docs/queue/) 名。
  - 该标签仅在创建 `Queue` 资源后才有效
  - ```shell
    labels:
      ray.io/scheduler-name: volcano
      volcano.sh/queue-name: <replace with correct Queue resource name>
    ```

如果启用了自动缩放， `minReplicas` 则用于组调度， 否则 `replicas` 使用所需的。

### 步骤 5: 使用Volcano进行批量调度

如需指导，请参阅 [示例](https://github.com/volcano-sh/volcano/tree/master/example)。

## 示例

在执行示例之前，请删除所有正在运行的 Ray Cluster，以确保成功运行下面的示例。
```shell
kubectl delete raycluster --all
```

### 组调度

此示例演示了群组调度如何与 Volcano 和 KubeRay 配合使用。

首先，创建一个容量为 4 个 CPU 和 6Gi RAM 的队列：

```shell
kubectl create -f - <<EOF
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: kuberay-test-queue
spec:
  weight: 1
  capability:
    cpu: 4
    memory: 6Gi
EOF
```

上述定义中的 **权重** 表示队列在集群资源划分中的相对权重。当集群中所有队列的总 **容量** 超过可用资源总量时，请使用此参数，从而强制队列在它们之间共享。权重较高的队列分配的总资源比例较大。

**容量** 是对队列在任何给定时间支持的最大资源的硬性约束。您可以根据需要更新它，以允许一次运行更多或更少的工作负载。

接下来，创建一个具有头节点（1 个 CPU + 2Gi RAM）和两个 worker 节点（每个 worker 节点 1 CPU + 1Gi RAM）的 RayCluster，总共 3 个 CPU 和 4Gi RAM：

```shell
# Path: kuberay/ray-operator/config/samples
# Includes  the `ray.io/scheduler-name: volcano` and `volcano.sh/queue-name: kuberay-test-queue` labels in the metadata.labels
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.volcano-scheduler-queue.yaml
kubectl apply -f ray-cluster.volcano-scheduler-queue.yaml
```

由于队列的容量为 4 个 CPU 和 6Gi RAM，因此该资源应该可以成功调度，不会出现任何问题。您可以通过检查集群的 Volcano PodGroup 的状态来验证这一点，以查看该阶段是 `Running` 以及最后一个状态是 `Scheduled`：

```shell
kubectl get podgroup ray-test-cluster-0-pg -o yaml

# apiVersion: scheduling.volcano.sh/v1beta1
# kind: PodGroup
# metadata:
#   creationTimestamp: "2022-12-01T04:43:30Z"
#   generation: 2
#   name: ray-test-cluster-0-pg
#   namespace: test
#   ownerReferences:
#   - apiVersion: ray.io/v1alpha1
#     blockOwnerDeletion: true
#     controller: true
#     kind: RayCluster
#     name: test-cluster-0
#     uid: 7979b169-f0b0-42b7-8031-daef522d25cf
#   resourceVersion: "4427347"
#   uid: 78902d3d-b490-47eb-ba12-d6f8b721a579
# spec:
#   minMember: 3
#   minResources:
#     cpu: "3"
#     memory: 4Gi
#   queue: kuberay-test-queue
# status:
#   conditions:
#   - lastTransitionTime: "2022-12-01T04:43:31Z"
#     reason: tasks in the gang are ready to be scheduled
#     status: "True"
#     transitionID: f89f3062-ebd7-486b-8763-18ccdba1d585
#     type: Scheduled
#   phase: Running
```

检查队列的状态以查看 1 个正在运行的作业：

```shell
kubectl get queue kuberay-test-queue -o yaml

# apiVersion: scheduling.volcano.sh/v1beta1
# kind: Queue
# metadata:
#   creationTimestamp: "2022-12-01T04:43:21Z"
#   generation: 1
#   name: kuberay-test-queue
#   resourceVersion: "4427348"
#   uid: a6c4f9df-d58c-4da8-8a58-e01c93eca45a
# spec:
#   capability:
#     cpu: 4
#     memory: 6Gi
#   reclaimable: true
#   weight: 1
# status:
#   reservation: {}
#   running: 1
#   state: Open
```

接下来，添加一个具有相同头节点和 worker 节点配置但名称不同的附加 RayCluster：

```shell
# Path: kuberay/ray-operator/config/samples
# Includes the `ray.io/scheduler-name: volcano` and `volcano.sh/queue-name: kuberay-test-queue` labels in the metadata.labels
# Replaces the name to test-cluster-1
sed 's/test-cluster-0/test-cluster-1/' ray-cluster.volcano-scheduler-queue.yaml | kubectl apply -f-
```

检查其 PodGroup 的状态以查看其阶段为 `Pending` ，最后状态为 `Unschedulable`:

```shell
kubectl get podgroup ray-test-cluster-1-pg -o yaml

# apiVersion: scheduling.volcano.sh/v1beta1
# kind: PodGroup
# metadata:
#   creationTimestamp: "2022-12-01T04:48:18Z"
#   generation: 2
#   name: ray-test-cluster-1-pg
#   namespace: test
#   ownerReferences:
#   - apiVersion: ray.io/v1alpha1
#     blockOwnerDeletion: true
#     controller: true
#     kind: RayCluster
#     name: test-cluster-1
#     uid: b3cf83dc-ef3a-4bb1-9c42-7d2a39c53358
#   resourceVersion: "4427976"
#   uid: 9087dd08-8f48-4592-a62e-21e9345b0872
# spec:
#   minMember: 3
#   minResources:
#     cpu: "3"
#     memory: 4Gi
#   queue: kuberay-test-queue
# status:
#   conditions:
#   - lastTransitionTime: "2022-12-01T04:48:19Z"
#     message: '3/3 tasks in gang unschedulable: pod group is not ready, 3 Pending,
#       3 minAvailable; Pending: 3 Undetermined'
#     reason: NotEnoughResources
#     status: "True"
#     transitionID: 3956b64f-fc52-4779-831e-d379648eecfc
#     type: Unschedulable
#   phase: Pending
```

由于新集群需要的 CPU 和 RAM 超过了队列允许的数量，因此即使其中一个 Pod 可以容纳剩余的 1 个 CPU 和 2Gi RAM，在有足够的空间容纳所有 Pod 之前，不会放置任何集群的 Pod。如果不以这种方式使用 Volcano 进行群组调度，通常会放置其中一个 Pod，从而导致集群被部分分配，并且某些作业（例如 [Horovod](https://github.com/horovod/horovod) 训练）陷入等待资源可用的状态。

查看这对为我们的新 RayCluster 调度 Pod 的影响，这些 Pod 列出如下 `Pending`:

```shell
kubectl get pods

# NAME                                            READY   STATUS         RESTARTS   AGE
# test-cluster-0-worker-worker-ddfbz              1/1     Running        0          7m
# test-cluster-0-head-vst5j                       1/1     Running        0          7m
# test-cluster-0-worker-worker-57pc7              1/1     Running        0          6m59s
# test-cluster-1-worker-worker-6tzf7              0/1     Pending        0          2m12s
# test-cluster-1-head-6668q                       0/1     Pending        0          2m12s
# test-cluster-1-worker-worker-n5g8k              0/1     Pending        0          2m12s
```

查看 Pod 详细信息，发现 Volcano 无法安排该组团：

```shell
kubectl describe pod test-cluster-1-head-6668q | tail -n 3

# Type     Reason            Age   From     Message
# ----     ------            ----  ----     -------
# Warning  FailedScheduling  4m5s  volcano  3/3 tasks in gang unschedulable: pod group is not ready, 3 Pending, 3 minAvailable; Pending: 3 Undetermined
```

删除第一个 RayCluster 以在队列中腾出空间：

```shell
kubectl delete raycluster test-cluster-0
```

第二个集群的 PodGroup 更改为以下 `Running` 状态，因为现在有足够的资源可用于调度整组 pod：

```shell
kubectl get podgroup ray-test-cluster-1-pg -o yaml

# apiVersion: scheduling.volcano.sh/v1beta1
# kind: PodGroup
# metadata:
#   creationTimestamp: "2022-12-01T04:48:18Z"
#   generation: 9
#   name: ray-test-cluster-1-pg
#   namespace: test
#   ownerReferences:
#   - apiVersion: ray.io/v1alpha1
#     blockOwnerDeletion: true
#     controller: true
#     kind: RayCluster
#     name: test-cluster-1
#     uid: b3cf83dc-ef3a-4bb1-9c42-7d2a39c53358
#   resourceVersion: "4428864"
#   uid: 9087dd08-8f48-4592-a62e-21e9345b0872
# spec:
#   minMember: 3
#   minResources:
#     cpu: "3"
#     memory: 4Gi
#   queue: kuberay-test-queue
# status:
#   conditions:
#   - lastTransitionTime: "2022-12-01T04:54:04Z"
#     message: '3/3 tasks in gang unschedulable: pod group is not ready, 3 Pending,
#       3 minAvailable; Pending: 3 Undetermined'
#     reason: NotEnoughResources
#     status: "True"
#     transitionID: db90bbf0-6845-441b-8992-d0e85f78db77
#     type: Unschedulable
#   - lastTransitionTime: "2022-12-01T04:55:10Z"
#     reason: tasks in the gang are ready to be scheduled
#     status: "True"
#     transitionID: 72bbf1b3-d501-4528-a59d-479504f3eaf5
#     type: Scheduled
#   phase: Running
#   running: 3
```

再次检查 Pod 以查看第二个集群现已启动并正在运行：

```shell
kubectl get pods

# NAME                                            READY   STATUS         RESTARTS   AGE
# test-cluster-1-worker-worker-n5g8k              1/1     Running        0          9m4s
# test-cluster-1-head-6668q                       1/1     Running        0          9m4s
# test-cluster-1-worker-worker-6tzf7              1/1     Running        0          9m4s
```

最后，清理剩余的集群和队列：

```shell
kubectl delete raycluster test-cluster-1
kubectl delete queue kuberay-test-queue
```
