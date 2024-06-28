(kuberay-pod-command)=

# 为 Ray head/worker Pod 指定容器命令
您可以在两个时间点在 head/worker pod 上执行命令：

* (1) **`ray start` 之前**: 作为示例，您可以设置一些将由 `ray start` 的环境变量

* (2) **`ray start` 之后 (RayCluster 已就绪)**: 例如，当 RayCluster 就绪时，您可以启动 Ray serve 部署。

## 当前容器命令的 KubeRay operator 行为
* 容器命令的当前行为尚未最终确定， **将来可能会更新**。
* 查看 [code](https://github.com/ray-project/kuberay/blob/47148921c7d14813aea26a7974abda7cf22bbc52/ray-operator/controllers/ray/common/pod.go#L301-L326) 了解更多信息。

## Timing 1: `ray start` 之前
目前，对于 timing (1)，我们可以在RayCluster规范中设置容器的 `Command` 和 `Args` 来达到目标。

```yaml
# https://github.com/ray-project/kuberay/ray-operator/config/samples/ray-cluster.head-command.yaml
    rayStartParams:
        ...
    #pod template
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.5.0
          resources:
            ...
          ports:
            ...
          # `command` and `args` will become a part of `spec.containers.0.args` in the head Pod.
          command: ["echo 123"]
          args: ["456"]
```

* Ray  Head Pod
    * `spec.containers.0.command` 硬编码为 `["/bin/bash", "-lc", "--"]`.
    * `spec.containers.0.args` 包含两部分：
        * (第 1 部分) **用户指定命令**: 将来自 RayCluster 的 `headGroupSpec.template.spec.containers.0.command` 和  `headGroupSpec.template.spec.containers.0.args` 用字符串串联。
        * (第2部分) **ray start 命令**: 命令根据 RayCluster 中 `rayStartParams` 定义创建。命令看起来像 `ulimit -n 65536; ray start ...`。
        * 总而言之， `spec.containers.0.args` 将是 `$(user-specified command) && $(ray start command)`。

* 示例
    ```sh
    # Prerequisite: There is a KubeRay operator in the Kubernetes cluster.

    # Download `ray-cluster.head-command.yaml`
    curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.head-command.yaml

    # Create a RayCluster
    kubectl apply -f ray-cluster.head-command.yaml

    # Check ${RAYCLUSTER_HEAD_POD}
    kubectl get pod -l ray.io/node-type=head

    # Check `spec.containers.0.command` and `spec.containers.0.args`.
    kubectl describe pod ${RAYCLUSTER_HEAD_POD}

    # Command:
    #   /bin/bash
    #   -lc
    #   --
    # Args:
    #    echo 123  456  && ulimit -n 65536; ray start --head  --dashboard-host=0.0.0.0  --num-cpus=1  --block  --metrics-export-port=8080  --memory=2147483648
    ```


## Timing 2: `ray start` 之后 (RayCluster 已就绪)
我们有两种解决方案来在 RayCluster 准备就绪后执行命令。这两种解决方案之间的主要区别是解决方案 1 中用户可以通过 `kubectl logs` 检查日志。

### 解决方案 1: 容器命令 (推荐)
正如我们在“时序1：`ray start` 之前”一节中提到的，用户指定的命令将在 `ray start` 命令之前执行。因此，我们可以通过更新 `ray-cluster.head-command.yaml` 的 `headGroupSpec.template.spec.containers.0.command` 来在后台执行`ray_cluster_resources.sh`。

```yaml
# https://github.com/ray-project/kuberay/ray-operator/config/samples/ray-cluster.head-command.yaml
# Parentheses for the command is required.
command: ["(/home/ray/samples/ray_cluster_resources.sh&)"]

# ray_cluster_resources.sh
apiVersion: v1
kind: ConfigMap
metadata:
  name: ray-example
data:
  ray_cluster_resources.sh: |
    #!/bin/bash

    # wait for ray cluster to finish initialization
    while true; do
        ray health-check 2>/dev/null
        if [ "$?" = "0" ]; then
            break
        else
            echo "INFO: waiting for ray head to start"
            sleep 1
        fi
    done

    # Print the resources in the ray cluster after the cluster is ready.
    python -c "import ray; ray.init(); print(ray.cluster_resources())"

    echo "INFO: Print Ray cluster resources"
```

* 示例
    ```sh
    # (1) Update `command` to ["(/home/ray/samples/ray_cluster_resources.sh&)"]
    # (2) Comment out `postStart` and `args`.
    kubectl apply -f ray-cluster.head-command.yaml

    # Check ${RAYCLUSTER_HEAD_POD}
    kubectl get pod -l ray.io/node-type=head

    # Check the logs
    kubectl logs ${RAYCLUSTER_HEAD_POD}

    # INFO: waiting for ray head to start
    # .
    # . => Cluster initialization
    # .
    # 2023-02-16 18:44:43,724 INFO worker.py:1231 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
    # 2023-02-16 18:44:43,724 INFO worker.py:1352 -- Connecting to existing Ray cluster at address: 10.244.0.26:6379...
    # 2023-02-16 18:44:43,735 INFO worker.py:1535 -- Connected to Ray cluster. View the dashboard at http://10.244.0.26:8265
    # {'object_store_memory': 539679129.0, 'node:10.244.0.26': 1.0, 'CPU': 1.0, 'memory': 2147483648.0}
    # INFO: Print Ray cluster resources
    ```

### 解决方案 2：postStart 挂钩
```yaml
# https://github.com/ray-project/kuberay/ray-operator/config/samples/ray-cluster.head-command.yaml
lifecycle:
  postStart:
    exec:
      command: ["/bin/sh","-c","/home/ray/samples/ray_cluster_resources.sh"]
```

* 我们通过 postStart 挂钩执行 `ray_cluster_resources.sh` 脚本。根据 [本文档](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks)，无法保证挂钩将在容器 ENTRYPOINT 之前执行。因此，我们需要在  `ray_cluster_resources.sh` 中等待RayCluster完成初始化完成。

* 示例
    ```sh
    kubectl apply -f ray-cluster.head-command.yaml

    # Check ${RAYCLUSTER_HEAD_POD}
    kubectl get pod -l ray.io/node-type=head

    # Forward the port of Dashboard
    kubectl port-forward --address 0.0.0.0 ${RAYCLUSTER_HEAD_POD} 8265:8265

    # Open the browser and check the Dashboard (${YOUR_IP}:8265/#/job).
    # You shold see a SUCCEEDED job with the following Entrypoint:
    #
    # `python -c "import ray; ray.init(); print(ray.cluster_resources())"`
    ```
