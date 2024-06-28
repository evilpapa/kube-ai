(kuberay-gcs-ft)=
# KubeRay 中的 GCS 容错

全局控制服务 (GCS) 管理集群级元数据。
默认情况下，GCS 缺乏容错能力，因为它将所有数据存储在内存中，出现故障可能会导致整个 Ray 集群失败。
为了使GCS具有容错能力，您必须拥有高可用的Redis。
这样，当 GCS 重新启动时，它会从 Redis 实例中检索所有数据并恢复其常规功能。

## 先决条件

* Ray 2.0.0+
* KubeRay 0.6.0+
* Redis: 单个分片、一个或多个副本

## 快速开始

### 步骤 1: 使用 Kind 创建 Kubernetes 集群

```sh
kind create cluster --image=kindest/node:v1.23.0
```

### 步骤 2: 安装 KubeRay Operator

按照 [本文档](kuberay-operator-deploy) 过 Helm 存储库安装最新稳定的 KubeRay Operator 。

### 步骤 3: 安装启用 GCS FT 的 RayCluster

```sh
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml
kubectl apply -f ray-cluster.external-redis.yaml
```

### 步骤 4: 验证 Kubernetes 集群状态

```sh
# 步骤 4.1: List all Pods in the `default` namespace.
# The expected output should be 4 Pods: 1 head, 1 worker, 1 KubeRay, and 1 Redis.
kubectl get pods

# 步骤 4.2: List all ConfigMaps in the `default` namespace.
kubectl get configmaps

# [Example output]
# NAME                  DATA   AGE
# ray-example           2      5m45s
# redis-config          1      5m45s
# ...
```

[ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml) 文件定义 RayCluster、Redis 和 ConfigMap 的 Kubernetes 资源。
此示例中有两个 ConfigMap： `ray-example` 和 `redis-config`。
`ray-example` ConfigMap 包含两个 Python 脚本： `detached_actor.py` 和 `increment_counter.py`。

* `detached_actor.py` 是一个 Python 脚本，用于创建一个名为 `counter_actor` 的独立 Actor。
    ```python
    import ray

    @ray.remote(num_cpus=1)
    class Counter:
        def __init__(self):
            self.value = 0

        def increment(self):
            self.value += 1
            return self.value

    ray.init(namespace="default_namespace")
    Counter.options(name="counter_actor", lifetime="detached").remote()
    ```

* `increment_counter.py` 是一个递增计数器的 Python 脚本。
    ```python
    import ray

    ray.init(namespace="default_namespace")
    counter = ray.get_actor("counter_actor")
    print(ray.get(counter.increment.remote()))
    ```

### 步骤 5: 创建一个独立的 actor

```sh
# 步骤 4.1: Create a detached actor with the name, `counter_actor`.
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/detached_actor.py

# 步骤 4.2: Increment the counter.
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/increment_counter.py

# 2023-09-07 16:01:41,925 INFO worker.py:1313 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
# 2023-09-07 16:01:41,926 INFO worker.py:1431 -- Connecting to existing Ray cluster at address: 10.244.0.17:6379...
# 2023-09-07 16:01:42,000 INFO worker.py:1612 -- Connected to Ray cluster. View the dashboard at http://10.244.0.17:8265
# 1
```

(kuberay-external-storage-namespace-example)=
### 步骤 6: 检查Redis中的数据

```sh
# 步骤 6.1: Check the RayCluster's UID.
kubectl get rayclusters.ray.io raycluster-external-redis -o=jsonpath='{.metadata.uid}'
# [Example output]: 864b004c-6305-42e3-ac46-adfa8eb6f752

# 步骤 6.2: Check the head Pod's environment variable `RAY_external_storage_namespace`.
kubectl get pods $HEAD_POD -o=jsonpath='{.spec.containers[0].env}' | jq
# [Example output]:
# [
#   {
#     "name": "RAY_external_storage_namespace",
#     "value": "864b004c-6305-42e3-ac46-adfa8eb6f752"
#   },
#   ...
# ]

# 步骤 6.3: Log into the Redis Pod.
export REDIS_POD=$(kubectl get pods --selector=app=redis -o custom-columns=POD:metadata.name --no-headers)
# The password `5241590000000000` is defined in the `redis-config` ConfigMap.
kubectl exec -it $REDIS_POD -- redis-cli -a "5241590000000000"

# 步骤 6.4: Check the keys in Redis.
KEYS *
# [Example output]:
# 1) "864b004c-6305-42e3-ac46-adfa8eb6f752"

# 步骤 6.5: Check the value of the key.
HGETALL 864b004c-6305-42e3-ac46-adfa8eb6f752
```

在 [ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml) 中，`ray.io/external-storage-namespace` 注解未设置到 RayCluster。
因此，默认情况下，KubeRay 会自动将环境变量 `RAY_external_storage_namespace` 注入到 RayCluster 管理的所有以 RayCluster 的 UID 作为外部存储命名空间的 Ray Pod 中。

参考 [本节](kuberay-external-storage-namespace) 以了解有关注释的更多信息。

### 步骤 7: 杀死head Pod中的GCS进程

```sh
# 步骤 7.1: Check the `RAY_gcs_rpc_server_reconnect_timeout_s` environment variable in both the head Pod and worker Pod.
kubectl get pods $HEAD_POD -o=jsonpath='{.spec.containers[0].env}' | jq
# [Expected result]:
# No `RAY_gcs_rpc_server_reconnect_timeout_s` environment variable is set. Hence, the Ray head uses its default value of `60`.
kubectl get pods $YOUR_WORKER_POD -o=jsonpath='{.spec.containers[0].env}' | jq
# [Expected result]:
# KubeRay injects the `RAY_gcs_rpc_server_reconnect_timeout_s` environment variable with the value `600` to the worker Pod.
# [
#   {
#     "name": "RAY_gcs_rpc_server_reconnect_timeout_s",
#     "value": "600"
#   },
#   ...
# ]

# 步骤 7.2: Kill the GCS process in the head Pod.
kubectl exec -it $HEAD_POD -- pkill gcs_server

# 步骤 7.3: The head Pod fails and restarts after `RAY_gcs_rpc_server_reconnect_timeout_s` (60) seconds.
# In addition, the worker Pod isn't terminated by the new head after reconnecting because GCS fault
# tolerance is enabled.
kubectl get pods -l=ray.io/is-ray-node=yes
# [Example output]:
# NAME                                                 READY   STATUS    RESTARTS      AGE
# raycluster-external-redis-head-xxxxx                 1/1     Running   1 (64s ago)   xxm
# raycluster-external-redis-worker-small-group-yyyyy   1/1     Running   0             xxm
```

在 [ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml)， `RAY_gcs_rpc_server_reconnect_timeout_s` 环境变量未设置到 RayCluster 的头节点或者 worker 节点 Pod。
因此，KubeRay 自动将 `RAY_gcs_rpc_server_reconnect_timeout_s` 注入到 worker Pod 中并设置默认值 **600**，将 Head Pod 设置为默认值 **60**。
 worker Pod 的超时值必须长于 Head Pod 的超时值，以便 worker Pod 在 Head Pod 因故障重新启动之前不会终止。

### 步骤 8: 再次访问分离的 actor 

```sh
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/increment_counter.py
# 2023-09-07 17:31:17,793 INFO worker.py:1313 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
# 2023-09-07 17:31:17,793 INFO worker.py:1431 -- Connecting to existing Ray cluster at address: 10.244.0.21:6379...
# 2023-09-07 17:31:17,875 INFO worker.py:1612 -- Connected to Ray cluster. View the dashboard at http://10.244.0.21:8265
# 2
```

```{admonition} 在此示例中，分离的 Actor 始终位于 worker Pod 上。
 Head Pod 的 `rayStartParams` 设置了 `num-cpus: "0"`。
因此，不会在 Head Pod 上安排任何任务或 actor 。
```

启用 GCS 容错后，在 GCS 进程终止并重新启动后，您仍然可以访问分离的 Actor。
请注意，容错不会保留 actor 的状态。
结果是 2 而不是 1 的原因是 游离 actor 始终在 worker pod 上运行。
另一方面，如果 Head Pod 托管分离的 actor，则  `increment_counter.py` 脚本在此步骤中生成的结果为 1。

### 步骤 9: 删除Kubernetes集群

```sh
kind delete cluster
```

## KubeRay GCS 容错配置

快速入门示例中使用的 [ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml) 包含有关配置选项的详细注释。
***请结合 YAML 文件阅读本节。***

### 1. 启用GCS容错

* **`ray.io/ft-enabled`**: 添加 `ray.io/ft-enabled: "true"` 注解到 RayCluster 自定义资源，以启用GCS容错。
    ```yaml
    kind: RayCluster
    metadata:
    annotations:
        ray.io/ft-enabled: "true" # <- Add this annotation to enable GCS fault tolerance
    ```

### 2. 连接到外部 Redis

* **`redis-password`** 在头配置 `rayStartParams`:
使用此选项指定 Redis 服务的密码，从而允许 Ray head 连接到它。
在 [ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml)，RayCluster 自定义资源使用环境变量 `REDIS_PASSWORD` 来存储 Kubernetes 密钥中的密码。
    ```yaml
    rayStartParams:
      redis-password: $REDIS_PASSWORD
    template:
      spec:
        containers:
          - name: ray-head
            env:
              # This environment variable is used in the `rayStartParams` above.
              - name: REDIS_PASSWORD
                valueFrom:
                  secretKeyRef:
                    name: redis-password-secret
                    key: password
    ```

* **`RAY_REDIS_ADDRESS`** head Pod中的环境变量：
Ray读取 `RAY_REDIS_ADDRESS` 环境变量来与Redis服务器建立连接。
在 [ray-cluster.external-redis.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.external-redis.yaml)中，RayCluster 自定义资源使用 `redis` Kubernetes ClusterIP 服务名称作为与 Redis 服务器的连接点。 ClusterIP 服务也是由 YAML 文件创建的。
    ```yaml
    template:
      spec:
        containers:
          - name: ray-head
            env:
              - name: RAY_REDIS_ADDRESS
                value: redis:6379
    ```

(kuberay-external-storage-namespace)=
### 3. 使用外部存储命名空间

* **`ray.io/external-storage-namespace`** 注解 (**可选**):
大多数情况下， ***不需要设置 `ray.io/external-storage-namespace`*** 因为KubeRay会自动将其设置为 RayCluster 的 UID 。
仅当您完全了解 GCS 容错和 RayService 的行为时才修改此注释，以避免配置错误。
有关更多详细信息，请参阅前面的快速入门示例中的 [章节](kuberay-external-storage-namespace-example) 。
    ```yaml
    kind: RayCluster
    metadata:
    annotations:
        ray.io/external-storage-namespace: "my-raycluster-storage" # <- Add this annotation to specify a storage namespace
    ```

## 下一步

* 有关更多信息，请参阅 {ref}`Ray Serve 端到端容错文档 <serve-e2e-ft-guide-gcs>` 。
* 参阅 `Ray Core GCS 容错文档 <fault-tolerance-gcs>` 获取更多信息。