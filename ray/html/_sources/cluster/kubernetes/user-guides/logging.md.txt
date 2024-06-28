(kuberay-logging)=

# 日志持久化

日志（系统日志和应用程序日志）对于 Ray 应用程序和集群的故障排除非常有用。例如，如果节点意外终止，您可能想要访问系统日志。

与 Kubernetes 类似，Ray 不提供日志数据的原生存储解决方案。用户需要自行管理日志的生命周期。本页面提供有关如何从 Kubernetes 上运行的 Ray 集群收集日志的说明。

:::{tip}
跳至示例配置的 {ref}`部署说明 <kuberay-logging-tldr>`，
了解如何从 Ray pod 中提取日志。
:::

## Ray 日志目录
默认的，Ray 将日志文件写入到每个 Ray Pod 的文件系统路径 `/tmp/ray/session_*/logs`，包括应用程序日志和系统日志。在开始收集日志之前，请详细了解 {ref}`日志目录和日志文件 <logging-directory>` 和 {ref}`日志轮换配置 <log-rotation>`。

## 日志处理工具
Kubernetes 生态系统中有许多可用的开源日志处理工具。本页展示如何使用 [Fluent Bit][FluentBit] 提取 Ray 日志。
其他流行的工具包括 [Vector][Vector]、 [Fluentd][Fluentd]、 [Filebeat][Filebeat] 以及 [Promtail][Promtail]。

## 日志收集策略
使用以下两种日志记录策略之一收集写入 pod 文件系统的日志：
**sidecar 容器** 或 **daemonsets**。
在 [Kubernetes 文档][KubDoc] 中阅读有关这些日志记录模式的更多信息。

### Sidecar 容器
我们在本指南中提供了 sidecar 策略的 {ref}`示例 <kuberay-fluentbit>`。
您可以通过为每个 Ray pod 配置日志处理 sidecar 来处理日志。
 Ray 容器应配置为通过卷挂载与日志记录 sidecar 共享目录 `/tmp/ray`。

可以将 sidecar 配置为执行以下任一操作：
* 将 Ray 日志流式传输到 sidecar 的标准输出。
* 将日志导出到外部服务。

### 守护进程
或者，可以在 Kubernetes 节点级别收集日志。
为此，需要将日志处理守护进程集部署到 Kubernetes 集群的节点上。
使用此策略，关键是将 Ray 容器的 `/tmp/ray` 目录挂载到相关的 `hostPath`。

(kuberay-fluentbit)=
## 使用 Fluent Bit 设置日志记录 sidecar
在本节中，我们将举例说明如何为 Ray pod设置发送日志的 [Fluent Bit][FluentBit] sidecar 。

[这里][ConfigLink] 查看具有日志记录 sidecar 的单 Pod RayCluster 的完整配置。
我们现在讨论此配置并展示如何部署它。

### 配置日志处理
第一步是创建一个包含 Fluent Bit 配置的 ConfigMap。

这是一个最小的 ConfigMap，它告诉 Fluent Bit sidecar
* Tail Ray 日志。
* 将日志输出到容器的标准输出。
```{literalinclude} ../configs/ray-cluster.log.yaml
:language: yaml
:start-after: Fluent Bit ConfigMap
:end-before: ---
```
对上述配置的一些注意事项：
- 除了将日志流式传输到 stdout 之外，您还可以使用 [OUTPUT] 子句将日志导出
  到Fluent Bit 支持的任何[后端存储][FluentBitStorage]。
- 上面的 `Path_Key true` 行确保了文件名包含在 Fluent Bit 发出的日志记录中。
- `Refresh_Interval 5` 行要求 Fluent Bit 每 5 秒刷新一次日志目录中的文件列表，而不是默认的 60 次。
  原因是该 `/tmp/ray/session_latest/logs/` 目录最初不存在（Ray 必须先创建它）。
  设置 `Refresh_Interval` 较低的值可以让我们更快地看到 Fluent Bit 容器的标准输出中的日志。


### 将日志记录 sidecar 添加到 RayCluster 自定义资源 (CR) 

#### 添加日志和配置卷
对于 RayCluster CR 中的每个 pod 模板，
我们需要添加两个卷：一个用于 Ray 日志的卷，
另一个用于存储上面应用的 ConfigMap 中的 Fluent Bit 配置的卷。
```{literalinclude} ../configs/ray-cluster.log.yaml
:language: yaml
:start-after: Log and config volumes
```

#### 挂载 Ray 日志目录
将以下卷安装添加到 Ray 容器的配置中。
```{literalinclude} ../configs/ray-cluster.log.yaml
:language: yaml
:start-after: Share logs with Fluent Bit
:end-before: Fluent Bit sidecar
```

#### 添加 Fluent Bit sidecar 
最后，将 Fluent Bit sidecar 容器添加到 RayCluster CR 中的每个 Ray pod 配置中。
```{literalinclude} ../configs/ray-cluster.log.yaml
:language: yaml
:start-after: Fluent Bit sidecar
:end-before: Log and config volumes
```
安装该 `ray-logs` 卷使 sidecar 容器可以访问 Ray 的日志。
<nobr>`fluentbit-config`</nobr> 卷使 sidecar 能够访问日志记录配置。

#### 将所有内容放在一起
将上述所有元素放在一起，我们为单 Pod RayCluster 提供了以下 yaml 配置，
该配置将成为日志处理 sidecar。
```{literalinclude} ../configs/ray-cluster.log.yaml
:language: yaml
```

(kuberay-logging-tldr)=
### 部署带有日志记录 sidecar 的 RayCluster 


要部署上述配置，请部署 KubeRay Operator（如果尚未部署）：
参阅 {ref}`入门指南 <kuberay-operator-deploy>`
以获取有关此步骤的说明。

现在，运行以下命令来部署 Fluent Bit ConfigMap 和带有 Fluent Bit sidecar 的单 Pod RayCluster。
```shell
kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/releases/2.4.0/doc/source/cluster/kubernetes/configs/ray-cluster.log.yaml
```

确定 Ray pod 的名称
```shell
kubectl get pod | grep raycluster-complete-logs
```

检查 FluentBit sidecar 的 STDOUT 以查看 Ray 组件进程的日志。
```shell
# Substitute the name of your Ray pod.
kubectl logs raycluster-complete-logs-head-xxxxx -c fluentbit
```

[Vector]: https://vector.dev/
[FluentBit]: https://docs.fluentbit.io/manual
[FluentBitStorage]: https://docs.fluentbit.io/manual
[Filebeat]: https://www.elastic.co/guide/en/beats/filebeat/7.17/index.html
[Fluentd]: https://docs.fluentd.org/
[Promtail]: https://grafana.com/docs/loki/latest/clients/promtail/
[KubDoc]: https://kubernetes.io/docs/concepts/cluster-administration/logging/
[ConfigLink]: https://raw.githubusercontent.com/ray-project/ray/releases/2.4.0/doc/source/cluster/kubernetes/configs/ray-cluster.log.yaml


(redirect-to-stderr)=
## 将 Ray 日志重定向到 stderr

默认情况下，Ray 将日志写入该 ``/tmp/ray/session_*/logs`` 目录中的文件。如果您希望将日志重定向到主机 Pod 的 stderr，请在所有 Ray 节点上设置环境变量 ``RAY_LOG_TO_STDERR=1``。推荐这种做法，但如果您的日志处理工具仅捕获写入 stderr 的日志记录，这种做法可能会很有用。

```{admonition} Alert
:class: caution
此功能存在已知问题。例如，它可能会破坏 {ref}`Worker 日志重定向到 Driver <log-redirection-to-driver>`等功能。如果需要这些功能，请使用上面的 {ref}`Fluent Bit solution <kuberay-fluentbit>` 方案。

对于虚拟机上的集群，不要将日志重定向到 stderr。请按照 {ref}`本指南 <vm-logging>` 保存日志。
```

将日志记录重定向到 stderr 还会在每个日志记录消息前面添加一个 ``({component})`` 前缀，例如 ``(raylet)``。

```bash
[2022-01-24 19:42:02,978 I 1829336 1829336] (gcs_server) grpc_server.cc:103: GcsServer server started, listening on port 50009.
[2022-01-24 19:42:06,696 I 1829415 1829415] (raylet) grpc_server.cc:103: ObjectManager server started, listening on port 40545.
2022-01-24 19:42:05,087 INFO (dashboard) dashboard.py:95 -- Setup static dir for dashboard: /mnt/data/workspace/ray/python/ray/dashboard/client/build
2022-01-24 19:42:07,500 INFO (dashboard_agent) agent.py:105 -- Dashboard agent grpc address: 0.0.0.0:49228
```

这些前缀允许您将日志的 stderr 流过滤到感兴趣的组件。 请注意，多行日志记录在每行的开头 **没有** 此组件标记。

按照以下步骤在所有Ray节点上设置环境变量 ``RAY_LOG_TO_STDERR=1``

  ::::{tab-set}

  :::{tab-item} Single-node local cluster
  **Start the cluster explicitly with CLI** <br/>
  ```bash
  env RAY_LOG_TO_STDERR=1 ray start
  ```

  **Start the cluster implicitly with `ray.init`** <br/>
  ```python
  os.environ["RAY_LOG_TO_STDERR"] = "1"
  ray.init()
  ```
  :::

  :::{tab-item} KubeRay
  Set `RAY_LOG_TO_STDERR` to `1` in `spec.headGroupSpec.template.spec.containers.env` and `spec.workerGroupSpec.template.spec.containers.env`. Check out this [example YAML file](https://gist.github.com/scottsun94/da4afda045d6e1cc32f9ccd6c33281c2)

  :::


  ::::



