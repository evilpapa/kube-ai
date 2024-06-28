(observability-configure-manage-dashboard)=
# 配置管理 Ray 仪表盘
{ref}`Ray 仪表盘<observability-getting-started>` 是监控和调试 Ray 应用程序和集群的最重要工具之一。本页介绍如何在集群上配置 Ray 仪表盘。

仪表板配置可能因您启动 Ray 集群的方式而异（例如，本地 Ray 集群与 KubeRay）。与 Prometheus 和 Grafana 的集成是可选的，以增强仪表板体验。

:::{note}
Ray Dashboard 仅适用于交互式开发和调试，因为集群终止后，仪表板 UI 和底层数据将无法访问。对于生产监控和调试，用户应依赖 [日志持久化](../cluster/kubernetes/user-guides/logging.md)， [指标持久化](./metrics.md)， [Ray 状态持久化](../ray-observability/user-guides/cli-sdk.rst)，和其他可观察性工具。
:::

## 更改 Ray Dashboard 端口
Ray Dashboard 运行在头节点的 `8265` 端口。请按照以下说明自定义端口（如果需要）。

::::{tab-set}

:::{tab-item} 单节点本地集群
**通过 CLI 明确启动集群** <br/>
在命令行为 ``ray start`` 传递 ``--dashboard-port`` 参数。

**通过 `ray.init` 明确启动集群** <br/>
在调用 ``ray.init()`` 传递关键参数 ``dashboard_port``。
:::

:::{tab-item} VM 集群启动器
在 [集群启动 YAML 文件](https://github.com/ray-project/ray/blob/0574620d454952556fa1befc7694353d68c72049/python/ray/autoscaler/aws/example-full.yaml#L172) 的 `head_start_ray_commands` 节点包含  ``--dashboard-port``  参数。
```yaml
head_start_ray_commands: 
  - ray stop 
  # Replace ${YOUR_PORT} with the port number you need.
  - ulimit -n 65536; ray start --head --dashboard-port=${YOUR_PORT} --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml 

```
:::

:::{tab-item} KubeRay
参考 [指定非默认端口](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#specifying-non-default-ports) 页获取详细信息。
:::

::::

(dashboard-in-browser)=
## 在浏览器查看 Ray Dashboard
当你在笔记本电脑上启动单节点 Ray 集群时，你可以通过 Ray 初始化时打印的 URL 访问仪表板（默认 URL 为 `http://localhost:8265`）。


当你通过 {ref}`VM cluster launcher <vm-cluster-quick-start>`，{ref}`KubeRay operator <kuberay-quickstart>`，或手动配置启动一个远程 Ray 集群，Ray 仪表盘会在头节点启动，端口可能是非公开的。你需要额外的设置以从头节点之外进行访问。

:::{danger}
为了安全目的，在没有适当的身份验证的情况下，请勿公开 Ray Dashboard。
:::

::::{tab-set}

:::{tab-item} VM 集群启动器
**端口转发** <br/>
您可以使用 ``ray dashboard`` 命令 安全地将本地流量端口转发到仪表板。

```shell
$ ray dashboard [-p <port, 8265 by default>] <cluster config file>
```

仪表板现在可在 ``http://localhost:8265`` 查看。
:::

:::{tab-item} KubeRay

KubeRay 控制器通过指定 Ray 头 pod 名为 ``<RayCluster name>-head-svc`` 的 service 是仪表盘可用。
在 Kubernetes 集群内通过 ``http://<RayCluster name>-head-svc:8265`` 访问。

有两种方法可以在集群外部公开仪表板：

**1. 设置 ingress** <br/>
按照 [说明](kuberay-ingress) 设置入口以访问 Ray Dashboard。 **入口必须仅允许来自受信任来源的访问。**

**2. 端口转发** <br/>
您还可以使用端口转发从 Kubernetes 集群外部查看仪表板：

```shell
$ kubectl port-forward --address 0.0.0.0 service/${RAYCLUSTER_NAME}-head-svc 8265:8265 
# Visit ${YOUR_IP}:8265 for the Dashboard (e.g. 127.0.0.1:8265 or ${YOUR_VM_IP}:8265)
```

```{admonition} Note
:class: note
不要在生产环境中使用端口转发。按照上述说明使用 Ingress 公开仪表板。
```

有关在 Kubernetes 上配置 Ray 集群网络访问的更多信息，请参阅 {ref}`网络说明 <kuberay-networking>`。

:::

::::

## 在反向代理后运行

通过反向代理访问时，Ray Dashboard 应该可以立即使用。API 请求不需要单独代理。

始终在 URL 末尾以 ``/`` 结束访问仪表盘。
例如，如果您的代理设置为处理对 ``/ray/dashboard`` 的请求，则通过 ``www.my-website.com/ray/dashboard/`` 进行查看。

仪表板发送带有相对 URL 路径的 HTTP 请求。当 ``window.location.href`` 以反斜线 ``/`` 结束，浏览器会按预期处理这些请求。

尽管 [MDN](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL#examples_of_relative_urls) 定义了预期行为，但这是许多浏览器处理具有相对 URL 的请求的特性。

通过在反向代理中包含一条规则将用户的浏览器重定向到 ``/``，比如 ``/ray/dashboard`` --> ``/ray/dashboard/`` 来使仪表盘不以 ``/`` 结尾的方式可见。

下面是一个使用 [traefik](https://doc.traefik.io/traefik/getting-started/quick-start/) TOML 文件实现此目的的示例：

```yaml
[http]
  [http.routers]
    [http.routers.to-dashboard]
      rule = "PathPrefix(`/ray/dashboard`)"
      middlewares = ["test-redirectregex", "strip"]
      service = "dashboard"
  [http.middlewares]
    [http.middlewares.test-redirectregex.redirectRegex]
      regex = "^(.*)/ray/dashboard$"
      replacement = "${1}/ray/dashboard/"
    [http.middlewares.strip.stripPrefix]
      prefixes = ["/ray/dashboard"]
  [http.services]
    [http.services.dashboard.loadBalancer]
      [[http.services.dashboard.loadBalancer.servers]]
        url = "http://localhost:8265"
```

```{admonition} Warning
:class: warning
Ray Dashboard 提供对 Ray Cluster 的 **读写** 访问权限。反向代理必须提供身份验证或网络入口控制，以防止未经授权访问 Cluster。
```

## 禁用 Dashboard

如果您使用或 `ray[default]` 或 {ref}`其他安装命令 <installation>` 仪表盘会自动包含并启动。

要禁用仪表盘，适应以下参数 `--include-dashboard`。

::::{tab-set}

:::{tab-item} 单节点本地集群

**明确以 CLI 方式启动** <br/>

```bash
ray start --include-dashboard=False
```

**明确以 `ray.init` 方式启动集群** <br/>

```{testcode}
:hide:

import ray
ray.shutdown()
```

```{testcode}
import ray
ray.init(include_dashboard=False)
```

:::

:::{tab-item} VM 集群启动器
在 [集群启动器 YAML 文件](https://github.com/ray-project/ray/blob/0574620d454952556fa1befc7694353d68c72049/python/ray/autoscaler/aws/example-full.yaml#L172) 的 `head_start_ray_commands` 节点
包含 `ray start --head --include-dashboard=False` 参数。
:::

:::{tab-item} KubeRay

```{admonition} Warning
:class: warning
不建议禁用 Dashboard，因为 KubeRay 的一些特性如 `RayJob` 和 `RayService` 依赖它。
```

设置 `spec.headGroupSpec.rayStartParams.include-dashboard` 为 `False`。查看这个 [YAML 示例文件](https://gist.github.com/kevin85421/0e6a8dd02c056704327d949b9ec96ef9)。

:::
::::


(observability-visualization-setup)=
## 将 Grafana 可视化嵌入到 Ray 仪表盘

For the enhanced Ray Dashboard experience, like {ref}`viewing time-series metrics<dash-metrics-view>` together with logs, Job info, etc., set up Prometheus and Grafana and integrate them with Ray Dashboard.

### Setting up Prometheus
To render Grafana visualizations, you need Prometheus to scrape metrics from Ray Clusters. Follow {ref}`the instructions <prometheus-setup>` to set up your Prometheus server and start to scrape system and application metrics from Ray Clusters.


### Setting up Grafana
Grafana is a tool that supports advanced visualizations of Prometheus metrics and allows you to create custom dashboards with your favorite metrics. Follow {ref}`the instructions <grafana>` to set up Grafana.


(embed-grafana-in-dashboard)=
### Embedding Grafana visualizations into Ray Dashboard
To view embedded time-series visualizations in Ray Dashboard, the following must be set up:

1. The head node of the cluster is able to access Prometheus and Grafana.
2. The browser of the dashboard user is able to access Grafana. 

Configure these settings using the `RAY_GRAFANA_HOST`, `RAY_PROMETHEUS_HOST`, `RAY_PROMETHEUS_NAME`, and `RAY_GRAFANA_IFRAME_HOST` environment variables when you start the Ray Clusters.

* Set `RAY_GRAFANA_HOST` to an address that the head node can use to access Grafana. Head node does health checks on Grafana on the backend.
* Set `RAY_PROMETHEUS_HOST` to an address the head node can use to access Prometheus.
* Set `RAY_PROMETHEUS_NAME` to select a different data source to use for the Grafana dashboard panles to use. Default is "Prometheus".
* Set `RAY_GRAFANA_IFRAME_HOST` to an address that the user's browsers can use to access Grafana and embed visualizations. If `RAY_GRAFANA_IFRAME_HOST` is not set, Ray Dashboard uses the value of `RAY_GRAFANA_HOST`.

For example, if the IP of the head node is 55.66.77.88 and Grafana is hosted on port 3000. Set the value to `RAY_GRAFANA_HOST=http://55.66.77.88:3000`.
* If you start a single-node Ray Cluster manually, make sure these environment variables are set and accessible before you start the cluster or as a prefix to the `ray start ...` command, e.g., `RAY_GRAFANA_HOST=http://55.66.77.88:3000 ray start ...`
* If you start a Ray Cluster with {ref}`VM Cluster Launcher <cloud-vm-index>`, the environment variables should be set under `head_start_ray_commands` as a prefix to the `ray start ...` command.
* If you start a Ray Cluster with {ref}`KubeRay <kuberay-index>`, refer to this {ref}`tutorial <kuberay-prometheus-grafana>`.

If all the environment variables are set properly, you should see time-series metrics in {ref}`Ray Dashboard <observability-getting-started>`.

:::{note}
If you use a different Prometheus server for each Ray Cluster and use the same Grafana server for all Clusters, set the `RAY_PROMETHEUS_NAME` environment variable to different values for each Ray Cluster and add these datasources in Grafana. Follow {ref}`these instructions <grafana>` to set up Grafana.
:::

#### Alternate Prometheus host location
By default, Ray Dashboard assumes Prometheus is hosted at `localhost:9090`. You can choose to run Prometheus on a non-default port or on a different machine. In this case, make sure that Prometheus can scrape the metrics from your Ray nodes following instructions {ref}`here <scrape-metrics>`.

Then, configure `RAY_PROMETHEUS_HOST` environment variable properly as stated above. For example, if Prometheus is hosted at port 9000 on a node with ip 55.66.77.88, set `RAY_PROMETHEUS_HOST=http://55.66.77.88:9000`.


#### Alternate Grafana host location
By default, Ray Dashboard assumes Grafana is hosted at `localhost:3000` You can choose to run Grafana on a non-default port or on a different machine as long as the head node and the browsers of dashboard users can access it.

If Grafana is exposed with NGINX ingress on a Kubernetes cluster, the following line should be present in the Grafana ingress annotation:

```yaml
nginx.ingress.kubernetes.io/configuration-snippet: |
    add_header X-Frame-Options SAMEORIGIN always;
```

When both Grafana and the Ray Cluster are on the same Kubernetes cluster, set `RAY_GRAFANA_HOST` to the external URL of the Grafana ingress.



#### User authentication for Grafana
When the Grafana instance requires user authentication, the following settings have to be in its [configuration file](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/) to correctly embed in Ray Dashboard:

```ini
  [security]
  allow_embedding = true
  cookie_secure = true
  cookie_samesite = none
```

#### Troubleshooting

##### Dashboard message: either Prometheus or Grafana server is not deteced
If you have followed the instructions above to set up everything, run the connection checks below in your browser:
* check Head Node connection to Prometheus server: add `api/prometheus_health` to the end of Ray Dashboard URL (for example: http://127.0.0.1:8265/api/prometheus_health)and visit it.
* check Head Node connection to Grafana server: add `api/grafana_health` to the end of Ray Dashboard URL (for example: http://127.0.0.1:8265/api/grafana_health) and visit it.
* check browser connection to Grafana server: visit the URL used in `RAY_GRAFANA_IFRAME_HOST`.


##### Getting an error that says `RAY_GRAFANA_HOST` is not setup
If you have set up Grafana , check that:
* You've included the protocol in the URL (e.g., `http://your-grafana-url.com` instead of `your-grafana-url.com`).
* The URL doesn't have a trailing slash (e.g., `http://your-grafana-url.com` instead of `http://your-grafana-url.com/`).

##### Certificate Authority (CA error)
You may see a CA error if your Grafana instance is hosted behind HTTPS. Contact the Grafana service owner to properly enable HTTPS traffic.


## Viewing built-in Dashboard API metrics

Dashboard is powered by a server that serves both the UI code and the data about the cluster via API endpoints.
Ray emits basic Prometheus metrics for each API endpoint:

`ray_dashboard_api_requests_count_requests_total`: Collects the total count of requests. This is tagged by endpoint, method, and http_status.

`ray_dashboard_api_requests_duration_seconds_bucket`: Collects the duration of requests. This is tagged by endpoint and method.

For example, you can view the p95 duration of all requests with this query:

```text

histogram_quantile(0.95, sum(rate(ray_dashboard_api_requests_duration_seconds_bucket[5m])) by (le))
```

You can query these metrics from the Prometheus or Grafana UI. Find instructions above for how to set these tools up.
