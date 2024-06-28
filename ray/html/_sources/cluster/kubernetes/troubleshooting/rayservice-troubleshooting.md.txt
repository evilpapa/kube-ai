(kuberay-raysvc-troubleshoot)=

# RayService 故障排除

RayService 是专为 Ray Serve 设计的自定义资源定义 (CRD)。在 KubeRay 中，创建 RayService 将首先创建 RayCluster，然后在 RayCluster 准备就绪后创建 Ray Serve 应用程序。如果问题与数据平面有关，特别是 Ray Serve 脚本或 Ray Serve 配置 (`serveConfigV2`)，则故障排除可能会很困难。本节提供一些提示来帮助您调试这些问题。

## 可观察性

### 方法 1: 检查 KubeRay operator 的日志是否有错误

```bash
kubectl logs $KUBERAY_OPERATOR_POD -n $YOUR_NAMESPACE | tee operator-log
```

上面的命令会将 operator 的日志重定向到名为 `operator-log` 的文件。然后您可以搜索文件中的错误。

### 方法 2: 检查 RayService CR 状态

```bash
kubectl describe rayservice $RAYSERVICE_NAME -n $YOUR_NAMESPACE
```

您可以检查RayService CR的状态和事件，看看是否有错误。

### 方法 3: 检查 Ray Pod 日志

您还可以通过访问 Pod 上的日志文件来直接检查 Ray Serve 日志。这些日志文件包含来自 Serve controller 和 HTTP 代理的系统级日志以及访问日志和用户级日志。有关更多详细信息，请参阅 [Ray Serve Logging](serve-logging) 和 [Ray Logging](configure-logging)。

```bash
kubectl exec -it $RAY_POD -n $YOUR_NAMESPACE -- bash
# Check the logs under /tmp/ray/session_latest/logs/serve/
```

### 方法 4: 检查 Dashboard

```bash
kubectl port-forward $RAY_POD -n $YOUR_NAMESPACE --address 0.0.0.0 8265:8265
# Check $YOUR_IP:8265 in your browser
```

有关仪表板上 Ray Serve 可观测性的更多详细信息，您可以参考 [文档](dash-serve-view) 和 [YouTube 视频](https://youtu.be/eqXfwM641a4)。

### 方法 5: Ray State CLI

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
# 0   ServeController                     ALIVE: 1
# 1   ServeReplica:fruit_app_OrangeStand  ALIVE: 1
# 2   HTTPProxyActor                      ALIVE: 3
# 3   ServeReplica:math_app_DAGDriver     ALIVE: 1
# 4   ServeReplica:math_app_Multiplier    ALIVE: 1
# 5   ServeReplica:math_app_create_order  ALIVE: 1
# 6   ServeReplica:fruit_app_DAGDriver    ALIVE: 1
# 7   ServeReplica:fruit_app_FruitMarket  ALIVE: 1
# 8   ServeReplica:math_app_Adder         ALIVE: 1
# 9   ServeReplica:math_app_Router        ALIVE: 1
# 10  ServeReplica:fruit_app_MangoStand   ALIVE: 1
# 11  ServeReplica:fruit_app_PearStand    ALIVE: 1
```

## 常见问题

* {ref}`kuberay-raysvc-issue1`
* {ref}`kuberay-raysvc-issue2`
* {ref}`kuberay-raysvc-issue3-1`
* {ref}`kuberay-raysvc-issue3-2`
* {ref}`kuberay-raysvc-issue4`
* {ref}`kuberay-raysvc-issue5`
* {ref}`kuberay-raysvc-issue6`
* {ref}`kuberay-raysvc-issue7`
* {ref}`kuberay-raysvc-issue8`
* {ref}`kuberay-raysvc-issue9`

(kuberay-raysvc-issue1)=
### 问题 1: Ray Serve 脚本不正确。

我们强烈建议您先在本地或 RayCluster 中测试 Ray Serve 脚本，然后再将其部署到 RayService。请参阅 [rayserve-dev-doc.md](kuberay-dev-serve) 。

(kuberay-raysvc-issue2)=
### 问题 2: `serveConfigV2` 不正确

为了灵活性，我们在 RayService CR 中设置 `serveConfigV2` 为多行 YAML 字符串。
这意味着 Ray Serve 中的 `serveConfigV2` 字段配置没有严格的类型检查。
以下是一些帮助您调试 `serveConfigV2` 字段的提示：

* 检查有关 Ray Serve 多应用程序 API 的架构 [文档](serve-api) 中的 `PUT "/api/serve/applications/"` 。
* 与 `serveConfig` 不同，`serveConfigV2` 遵循蛇形命名约定。例如， `numReplicas` 用于 `serveConfig`，而 `num_replicas` 用于 `serveConfigV2`。 

(kuberay-raysvc-issue3-1)=
### 问题 3-1: Ray 镜像不包含所需的依赖项。

您可以通过两种方式解决此问题：

* 使用所需的依赖项构建您自己的 Ray 镜像。
* 通过在 `runtime_env` 的 `serveConfigV2` 字段指定所需的依赖。
  * 例如，MobileNet 示例需要 `python-multipart`，而他没有包含在 `rayproject/ray-ml:2.5.0` 镜像中。
因此，运行时环境 YAML 文件包含 `python-multipart` 。更多详情，参考 [MobileNet 示例](kuberay-mobilenet-rayservice-example)。

(kuberay-raysvc-issue3-2)=
### 问题 3-2: 解决依赖性问题的示例。

> 注意: 我们强烈建议您在本地或 RayCluster 中测试您的 Ray Serve 脚本，然后再将其部署到 RayService。这有助于在早期阶段识别任何依赖关系问题。有关更多详细信息，请参阅 [rayserve-dev-doc.md](kuberay-dev-serve) 。

在 [MobileNet 示例](kuberay-mobilenet-rayservice-example)， [mobilenet.py](https://github.com/ray-project/serve_config_examples/blob/master/mobilenet/mobilenet.py) 由两个函数组成： `__init__()` 和 `__call__()`。
`__call__()` 函数仅在 Serve 应用程序收到请求时被调用。

* 示例 1: 在运行时环境 [MobileNet YAML](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-service.mobilenet.yaml) 文件移除 `python-multipart`
  * `python-multipart` 库只被 `__call__` 方法需要。因此，我们只能在向应用程序发送请求时才能观察到依赖关系问题。
  * 错误消息示例：
    ```bash
    Unexpected error, traceback: ray::ServeReplica:mobilenet_ImageClassifier.handle_request() (pid=226, ip=10.244.0.9)
      .
      .
      .
      File "...", line 24, in __call__
        request = await http_request.form()
      File "/home/ray/anaconda3/lib/python3.7/site-packages/starlette/requests.py", line 256, in _get_form
        ), "The `python-multipart` library must be installed to use form parsing."
    AssertionError: The `python-multipart` library must be installed to use form parsing..
    ```

* 示例 2: 在 [the MobileNet YAML](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-service.mobilenet.yaml) 更新 `rayproject/ray-ml:2.5.0` 为 `rayproject/ray:2.5.0`。 后者不包含 `tensorflow`。
  * `tensorflow` 类库在 [mobilenet.py](https://github.com/ray-project/serve_config_examples/blob/master/mobilenet/mobilenet.py) 中引入。
  * 错误消息示例：
    ```bash
    kubectl describe rayservices.ray.io rayservice-mobilenet

    # Example error message:
    Pending Service Status:
      Application Statuses:
        Mobilenet:
          ...
          Message:                  Deploying app 'mobilenet' failed:
            ray::deploy_serve_application() (pid=279, ip=10.244.0.12)
                ...
              File ".../mobilenet/mobilenet.py", line 4, in <module>
                from tensorflow.keras.preprocessing import image
            ModuleNotFoundError: No module named 'tensorflow'
    ```

(kuberay-raysvc-issue4)=
### 问题 4: 不正确的 `import_path`.

有关 `import_path` 格式的详细信息，参考 [本文档](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.schema.ServeApplicationSchema.html#ray.serve.schema.ServeApplicationSchema.import_path)。
拿 [MobileNet YAML 文件](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-service.mobilenet.yaml) 作为示例，
`import_path` 为 `mobilenet.mobilenet:app`。第一个 `mobilenet` 是 `working_dir` 目录的名字，
第二个 `mobilenet` 是目录 `mobilenet/` 中 Python 文件的名称， 
`app` 是 Python 文件内代表 Ray Serve 应用程序的变量名称。

```yaml
  serveConfigV2: |
    applications:
      - name: mobilenet
        import_path: mobilenet.mobilenet:app
        runtime_env:
          working_dir: "https://github.com/ray-project/serve_config_examples/archive/b393e77bbd6aba0881e3d94c05f968f05a387b96.zip"
          pip: ["python-multipart==0.0.6"]
```

(kuberay-raysvc-issue5)=
### 问题 5: 无法创建/更新服务应用程序。

当 KubeRay 尝试创建/更新 Serve 应用程序时，您可能会遇到以下错误消息：

#### 错误消息 1: `connect: connection refused`

```
Put "http://${HEAD_SVC_FQDN}:52365/api/serve/applications/": dial tcp $HEAD_IP:52365: connect: connection refused
```

对于 RayService，一旦头部 Pod 准备就绪，KubeRay 控制器就会向 RayCluster 提交创建 Serve 应用程序的请求。
需要注意的是，在头部 Pod 准备就绪后，Dashboard、Dashboard Agent 和 GCS 可能需要几秒钟才能启动。
因此，在必要组件完全运行之前，请求可能会最初失败几次。

如果等待 1 分钟后仍遇到此问题，则可能是仪表板或仪表板代理启动失败。
有关更多信息，您可以检查位于头部 Pod 上的位于 `/tmp/ray/session_latest/logs/` 路径的 `dashboard.log` 和 `dashboard_agent.log` 文件。

#### 错误消息 2: `i/o timeout`

```
Put "http://${HEAD_SVC_FQDN}:52365/api/serve/applications/": dial tcp $HEAD_IP:52365: i/o timeout"
```

导致此问题的一个可能原因可能是 Kubernetes NetworkPolicy 阻止了 Ray Pods 和仪表板代理端口（即 52365）之间的流量。

(kuberay-raysvc-issue6)=
### 问题 6: `runtime_env`

在 `serveConfigV2`，您可以通过 `runtime_env` 指定 Ray Serve 应用程序的运行时环境。
以下内容与 `runtime_env` 相关的一些常见问题 :

* `working_dir` 指向私有 AWS S3 存储桶，但 Ray Pods 没有访问该存储桶所需的权限。

* NetworkPolicy 阻止 Ray Pods 与 `runtime_env` 中指定的外部 URL 之间的流量。

(kuberay-raysvc-issue7)=
### 问题 7: 无法获取服务应用程序状态。

当 KubeRay 尝试获取 Serve 应用程序状态时，您可能会遇到以下错误消息：

```
Get "http://${HEAD_SVC_FQDN}:52365/api/serve/applications/": dial tcp $HEAD_IP:52365: connect: connection refused"
```

如 [Issue 5](#issue-5-fail-to-create--update-serve-applications) 提到，KubeRay 控制器当 head Pod 就绪时，提交一个 `Put` 请求到 RayCluster。
在 `Put` 请求 dashboard agent 成功后，会向仪表盘代理端口（即 52365）发送请求。
提交成功表明包括仪表盘代理在内的所有必要组件都已完全正常运行。
因此，与问题 5 不同，`Get` 请求失败是意料之中的。

如果您持续遇到此问题，则可能存在以下几种原因：

* 头 Pod 上的仪表板代理进程未运行。您可以检查头 Pod 上位于 `/tmp/ray/session_latest/logs/` 目录的 `dashboard_agent.log` 获取更多信息。 此外，您还可以通过手动终止头 Pod 上的仪表板代理进程来执行实验以重现此原因。
  ```bash
  # Step 1: Log in to the head Pod
  kubectl exec -it $HEAD_POD -n $YOUR_NAMESPACE -- bash

  # Step 2: Check the PID of the dashboard agent process
  ps aux
  # [Example output]
  # ray          156 ... 0:03 /.../python -u /.../ray/dashboard/agent.py --

  # Step 3: Kill the dashboard agent process
  kill 156

  # Step 4: Check the logs
  cat /tmp/ray/session_latest/logs/dashboard_agent.log

  # [Example output]
  # 2023-07-10 11:24:31,962 INFO web_log.py:206 -- 10.244.0.5 [10/Jul/2023:18:24:31 +0000] "GET /api/serve/applications/ HTTP/1.1" 200 13940 "-" "Go-http-client/1.1"
  # 2023-07-10 11:24:34,001 INFO web_log.py:206 -- 10.244.0.5 [10/Jul/2023:18:24:33 +0000] "GET /api/serve/applications/ HTTP/1.1" 200 13940 "-" "Go-http-client/1.1"
  # 2023-07-10 11:24:36,043 INFO web_log.py:206 -- 10.244.0.5 [10/Jul/2023:18:24:36 +0000] "GET /api/serve/applications/ HTTP/1.1" 200 13940 "-" "Go-http-client/1.1"
  # 2023-07-10 11:24:38,082 INFO web_log.py:206 -- 10.244.0.5 [10/Jul/2023:18:24:38 +0000] "GET /api/serve/applications/ HTTP/1.1" 200 13940 "-" "Go-http-client/1.1"
  # 2023-07-10 11:24:38,590 WARNING agent.py:531 -- Exiting with SIGTERM immediately...

  # Step 5: Open a new terminal and check the logs of the KubeRay operator
  kubectl logs $KUBERAY_OPERATOR_POD -n $YOUR_NAMESPACE | tee operator-log

  # [Example output]
  # Get \"http://rayservice-sample-raycluster-rqlsl-head-svc.default.svc.cluster.local:52365/api/serve/applications/\": dial tcp 10.96.7.154:52365: connect: connection refused
  ```

(kuberay-raysvc-issue8)=
### 问题 8: Kubernetes 集群资源耗尽时，循环重启 RayCluster。（KubeRay v0.6.1 及以下版本）

> 注意：目前 KubeRay 运营团队还没有明确的方案来应对 Kubernetes 集群资源耗尽的情况，
因此，我们建议确保 Kubernetes 集群有足够的资源来承载服务应用。

如果服务应用程序的状态持续非 `RUNNING` 状态超过 `serviceUnhealthySecondThreshold` 秒，
KubeRay operator 将认为 RayCluster 不健康并启动新的 RayCluster 的准备。
此问题的一个常见原因是 Kubernetes 集群没有足够的资源来容纳服务应用程序。
在这种情况下，KubeRay operator 可能会继续重启 RayCluster，从而导致重启循环。

我们也可以做一个实验来重现这种情况：

* 具有 8 个 CPU 节点的 Kubernetes 集群
* [ray-service.insufficient-resources.yaml](https://gist.github.com/kevin85421/6a7779308aa45b197db8015aca0c1faf)
  * RayCluster:
    * 该集群有 1 个带有 4 个物理 CPU 的头部 Pod，但 `rayStartParams` 中 `num-cpus` 设置为 0 以防止在头部 Pod 上安排任何服务副本。
    * 该集群还默认有 1 个工作 Pod，具有 1 个 CPU。
  * `serveConfigV2` 指定 5 个服务部署，每个部署有 1 个副本并需要 1 个 CPU。

```bash
# Step 1: Get the number of CPUs available on the node
kubectl get nodes -o custom-columns=NODE:.metadata.name,ALLOCATABLE_CPU:.status.allocatable.cpu

# [Example output]
# NODE                 ALLOCATABLE_CPU
# kind-control-plane   8

# Step 2: Install a KubeRay operator.

# Step 3: Create a RayService with autoscaling enabled.
kubectl apply -f ray-service.insufficient-resources.yaml

# Step 4: The Kubernetes cluster will not have enough resources to accommodate the serve application.
kubectl describe rayservices.ray.io rayservice-sample -n $YOUR_NAMESPACE

# [Example output]
# fruit_app_DAGDriver:
#   Health Last Update Time:  2023-07-11T02:10:02Z
#   Last Update Time:         2023-07-11T02:10:35Z
#   Message:                  Deployment "fruit_app_DAGDriver" has 1 replicas that have taken more than 30s to be scheduled. This may be caused by waiting for the cluster to auto-scale, or waiting for a runtime environment to install. Resources required for each replica: {"CPU": 1.0}, resources available: {}.
#   Status:                   UPDATING

# Step 5: A new RayCluster will be created after `serviceUnhealthySecondThreshold` (300s here) seconds.
# Check the logs of the KubeRay operator to find the reason for restarting the RayCluster.
kubectl logs $KUBERAY_OPERATOR_POD -n $YOUR_NAMESPACE | tee operator-log

# [Example output]
# 2023-07-11T02:14:58.109Z	INFO	controllers.RayService	Restart RayCluster	{"appName": "fruit_app", "restart reason": "The status of the serve application fruit_app has not been RUNNING for more than 300.000000 seconds. Hence, KubeRay operator labels the RayCluster unhealthy and will prepare a new RayCluster."}
# 2023-07-11T02:14:58.109Z	INFO	controllers.RayService	Restart RayCluster	{"deploymentName": "fruit_app_FruitMarket", "appName": "fruit_app", "restart reason": "The status of the serve deployment fruit_app_FruitMarket or the serve application fruit_app has not been HEALTHY/RUNNING for more than 300.000000 seconds. Hence, KubeRay operator labels the RayCluster unhealthy and will prepare a new RayCluster. The message of the serve deployment is: Deployment \"fruit_app_FruitMarket\" has 1 replicas that have taken more than 30s to be scheduled. This may be caused by waiting for the cluster to auto-scale, or waiting for a runtime environment to install. Resources required for each replica: {\"CPU\": 1.0}, resources available: {}."}
# .
# .
# .
# 2023-07-11T02:14:58.122Z	INFO	controllers.RayService	Restart RayCluster	{"ServiceName": "default/rayservice-sample", "AvailableWorkerReplicas": 1, "DesiredWorkerReplicas": 5, "restart reason": "The serve application is unhealthy, restarting the cluster. If the AvailableWorkerReplicas is not equal to DesiredWorkerReplicas, this may imply that the Autoscaler does not have enough resources to scale up the cluster. Hence, the serve application does not have enough resources to run. Please check https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayservice-troubleshooting.md for more details.", "RayCluster": {"apiVersion": "ray.io/v1alpha1", "kind": "RayCluster", "namespace": "default", "name": "rayservice-sample-raycluster-hvd9f"}}
```

(kuberay-raysvc-issue9)=
### Issue 9: 无需停机即可从 Ray Serve 的单应用程序 API 升级到其多应用程序 API

KubeRay v0.6.0 已开始通过在 RayService CRD 中  `serveConfigV2` 暴露的方式支持 Ray Serve API V2（多应用） 。
但 Ray Serve 不支持在集群中同时部署 API V1 和 API V2。
因此，如果用户想要通过替换 `serveConfig` 为 `serveConfigV2` 进行就地升级，可能会遇到以下错误信息：

```
ray.serve.exceptions.RayServeException: You are trying to deploy a multi-application config, however a single-application 
config has been deployed to the current Serve instance already. Mixing single-app and multi-app is not allowed. Please either
redeploy using the single-application config format `ServeApplicationSchema`, or shutdown and restart Serve to submit a
multi-app config of format `ServeDeploySchema`. If you are using the REST API, you can submit a multi-app config to the
the multi-app API endpoint `/api/serve/applications/`.
```

要解决此问题，您可以将 `serveConfig` 替换为 `serveConfigV2` 并修改 `rayVersion` ，当 Ray 版本为 2.0.0 或更高版本到 2.100.0 时，此修改无效。
这将触发新的 RayCluster 准备，而不是就地更新。

如果按照上述步骤操作后，仍然出现错误提示，且 GCS 容错功能已开启，则可能是由于新旧 RayClusters 的注解 `ray.io/external-storage-namespace` 相同导致的。
您可以移除注解，KubeRay 会自动为每个 RayCluster 自定义资源生成一个唯一的 key。
更多详情可参考 [kuberay#1297](https://github.com/ray-project/kuberay/issues/1297) 。