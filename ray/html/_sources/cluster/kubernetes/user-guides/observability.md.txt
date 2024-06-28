(kuberay-observability)=

# KubeRay 可观测性

## Ray 仪表盘

* 要查看在 head Pod 上运行的 [Ray dashboard](observability-getting-started) ，参考 [以下说明](kuberay-port-forward-dashboard)。
* 要将 Ray 仪表板与 Prometheus 和 Grafana 集成，请参阅 [使用 Prometheus 和 Grafana](kuberay-prometheus-grafana) 了解更多详细信息。
* 要启用“CPU Flame Graph”和“Stack Trace”功能，请参阅 [使用 py-spy 进行分析](kuberay-pyspy-integration)。

## KubeRay 可观测性

方法 1 和 2 解决控制平面的可观察性，而方法 3、4 和 5 则重点关注数据平面的可观察性。

### 方法 1: 检查 KubeRay operator 的日志是否有错误

```bash
# Typically, the operator's Pod name is kuberay-operator-xxxxxxxxxx-yyyyy.
kubectl logs $KUBERAY_OPERATOR_POD -n $YOUR_NAMESPACE | tee operator-log
```

使用此命令将Operator 的日志重定向到名为 `operator-log`。然后在文件中查找错误。

### 方法 2: 检查自定义资源状态

```bash
kubectl describe [raycluster|rayjob|rayservice] $CUSTOM_RESOURCE_NAME -n $YOUR_NAMESPACE
```

运行此命令后，检查自定义资源的状态和事件是否有任何错误。

### 方法 3: 查看 Ray Pod 的日志

通过访问 Pod 上的日志文件直接检查 Ray 日志。有关更多详细信息，请参阅 [Ray 日志](configure-logging) 。

```bash
kubectl exec -it $RAY_POD -n $YOUR_NAMESPACE -- bash
# Check the logs under /tmp/ray/session_latest/logs/
```

(kuberay-port-forward-dashboard)=
### 方法 4: 检查仪表盘

```bash
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl port-forward $RAY_POD -n $YOUR_NAMESPACE --address 0.0.0.0 8265:8265
# Check $YOUR_IP:8265 in your browser to access the dashboard.
# For most cases, 127.0.0.1:8265 or localhost:8265 should work.
```

### 方法 5: Ray 状态 CLI

您可以使用 Head Pod 上的 [Ray State CLI](state-api-cli-ref) 检查 Ray Serve 应用程序的状态。

```bash
# Log into the head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- ray summary actors

# [Example output]:
# ======== Actors Summary: 2023-07-11 17:58:24.625032 ========
# Stats:
# ------------------------------------
# total_actors: 14


# Table (group by class):
# ------------------------------------
#     CLASS_NAME                          STATE_COUNTS
# 0   ...                                 ALIVE: 1
# 1   ...                                 ALIVE: 1
# 2   ...                                 ALIVE: 3
# 3   ...                                 ALIVE: 1
# 4   ...                                 ALIVE: 1
# 5   ...                                 ALIVE: 1
# 6   ...                                 ALIVE: 1
# 7   ...                                 ALIVE: 1
# 8   ...                                 ALIVE: 1
# 9   ...                                 ALIVE: 1
# 10  ...                                 ALIVE: 1
# 11  ...                                 ALIVE: 1
```