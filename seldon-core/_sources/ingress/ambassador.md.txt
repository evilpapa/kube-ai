# 使用 Ingress Ambassador

Seldon Core 与 [Ambassador](https://www.getambassador.io/) 搭配良好，用于通过单入口暴露 Ambassador 以及 通过 Seldon 创建的 Ambassador 配置化的[运行中机器学习部署的动态暴露](https://kubernetes.io/blog/2018/06/07/dynamic-ingress-in-kubernetes/)。 Ambassador 是一个建立在 Envoy Proxy 基础上的 Kubernetes 原生 API 网关。完全通过 Kubernetes 自定义资源实现，Ambassador 提供强大的流量分发，身份验证，可观察性的管理能力。Ambassador 具有流行的服务网格原生实现，比如 Consul、Istio 以及 Linkerd。本文档中我们将讨论 Seldon Deployments 如何通过 Ambassador 暴露服务，以及如何同时使用这两种部署执行各种生产部署策略。

## 安装 Ambassador

安装 Ambassador 有[两部建议](https://www.getambassador.io/editions/)：

.. warning::
   Seldon Core 当前只支持 V1 Ambassador APIs. 你 **必须** 使用 ``datawire/ambassador`` helm chart 而不是使用 ``datawire/edge-stack`` 或者 V2 ``datawire/emissary`` charts。跟随 `安装指引 <../nav/installation.rst>`_ 可以正确工作。

### 建议 1：Ambassador 全栈

[Ambassador Edge Stack](https://www.getambassador.io/products/edge-stack/) 时最简单的开始 Ambassador 的方式，提供简单，[自动 TLS 配置](https://www.getambassador.io/docs/edge-stack/latest/topics/running/host-crd/#acme-support) 等其他特性，如 [身份验证](https://www.getambassador.io/docs/edge-stack/latest/topics/using/filters/)，[限流](https://www.getambassador.io/docs/edge-stack/latest/topics/using/rate-limits/)，和高级的路由行为，比如[自定义前缀](https://www.getambassador.io/docs/edge-stack/latest/topics/using/intro-mappings/)，[虚拟主机](https://www.getambassador.io/docs/edge-stack/latest/topics/using/headers/host/)，[method](https://www.getambassador.io/docs/edge-stack/latest/topics/using/method/)/[query-parameter](https://www.getambassador.io/docs/edge-stack/latest/topics/using/query_parameters/) 基础的路由等等。
查看 [AES 安装说明](https://www.getambassador.io/docs/edge-stack/latest/tutorials/getting-started/)将其安装到你的 Kubernetes 集群。

使用 `helm` 步骤可概况为：
```bash
kubectl create namespace ambassador || echo "namespace ambassador exists"

helm repo add datawire https://www.getambassador.io
helm install ambassador datawire/ambassador \
  --set image.repository=docker.io/datawire/ambassador \
  --set crds.keep=false \
  --namespace ambassador
```

### 建议 2: Ambassador API 网关 (现为 Emissary Ingress)

[Ambassador API Gateway](https://www.getambassador.io/products/api-gateway/) (现在为 Emissary Ingress) 是 Ambassador Edge Stack 的完全开源版本，并且提供传统控制器入口的所有功能。
查看 [Ambassador OSS 说明](https://www.getambassador.io/docs/latest/topics/install/) 将其安装到你的 Kubernetes 集群。

使用 `helm` 步骤可概况为：
```bash
kubectl create namespace ambassador || echo "namespace ambassador exists"

helm repo add datawire https://www.getambassador.io
helm install ambassador datawire/ambassador \
  --set image.repository=docker.io/datawire/ambassador \
  --set enableAES=false \
  --set crds.keep=false \
  --namespace ambassador
```

### TLS

强烈建议使用 TLS 加密控制进入 Ambassador 的流量。默认安装自带自签名证书，并且未使用任何其他 TLS 配置。查看 [说明](https://www.getambassador.io/docs/edge-stack/latest/howtos/tls-termination/) 来对集群进行 TLS 配置。

## Ambassador REST

假设 Ambassador 服务暴露于运行在 `namespace` 空间 `<deploymentName>` Seldon deployment 上的 `<ambassadorEndpoint>`：

注意，如果选择安装的是 Ambassador Edge Stack 需使用 https。即可使用[TLS 设置](https://www.getambassador.io/docs/edge-stack/latest/howtos/tls-termination/) 也可传递 `-k` 参数在 `curl` 中允许自签名证书。

对于限定了命名空间的 Seldon Core，`singleNamespace=true`，节点暴露为：

 * `http(s)://<ambassadorEndpoint>/seldon/<deploymentName>/api/v1.0/predictions`
 * `http(s)://<ambassadorEndpoint>/seldon/<namespace>/<deploymentName>/api/v1.0/predictions`

全局运行的 Seldon Core，`singleNamespace=false`，节点暴露为：

 * `http(s)://<ambassadorEndpoint>/seldon/<namespace>/<deploymentName>/api/v1.0/predictions`

## Curl示例

### Ambassador REST

如果安装的是 OSS Ambassador API 网关，假设基于 Ambassador 的 Seldon Deployment `mymodel` 服务暴露为 `0.0.0.0:8003` 可按下面发送请求：

```bash
curl -v 0.0.0.0:8003/seldon/mymodel/api/v1.0/predictions -d '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}' -H "Content-Type: application/json"
```

或者，安装的是 TLS 配置的 Ambassador Edge Stack，假设基于 Ambassador 的 Seldon Deployment `mymodel` 服务 hostname 为 `example-hostname.com`：

```bash
curl -v https://example-hostname.com/seldon/mymodel/api/v1.0/predictions -d '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}' -H "Content-Type: application/json"
```

如果都不是，可使用 IP 地址代替 `example-hostname` 并传递 -k 选项来方位不安全的 TLS。

```bash
curl -vk https://0.0.0.0/seldon/mymodel/api/v1.0/predictions -d '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}' -H "Content-Type: application/json"
```

## Ambassador 配置注解参考

| 注解 | 说明 |
|------------|-------------|
|`seldon.io/ambassador-config:<configuration>`| 自定义 Ambassador 配置 |
|`seldon.io/ambassador-header:<header>`| 添加到 Ambassador 配置的 Header |
|`seldon.io/ambassador-id:<instance id>`| 添加到 Ambassador `ambassador_id` 配置的实例 id |
|`seldon.io/ambassador-regex-header:<regex>`| T用于通过 header 头路由的正则表达式|
|`seldon.io/ambassador-retries:<number of retries>` | Tambassador 将在连接失败时重试请求的次数。默认 0。如果需要更多控制，请使用自定义配置。|
|`seldon.io/ambassador-service-name:<existing_deployment_name>`| 现有 Seldon Deployment 的名称，用于基于 shadow 或 header 的路由 |
|`seldon.io/grpc-timeout: <gRPC read timeout (msecs)>` | gRPC 读取超时时间 |
|`seldon.io/rest-timeout:<REST read timeout (msecs)>` | REST 读取超时时间 |
|`seldon.io/ambassador-circuit-breakers-max-connections:<maximum number of connections>` | Seldon 部署的最大连接数 |
|`seldon.io/ambassador-circuit-breakers-max-pending-requests:<maximum number of queued requests>` | 等待连接时将排队的最大请求数 |
|`seldon.io/ambassador-circuit-breakers-max-requests:<maximum number of parallel outstanding requests>` | Seldon 部署的最大并行未完成请求数 |
|`seldon.io/ambassador-circuit-breakers-max-retries:<maximum number of parallel retries>` | Seldon 部署允许的最大并行重试次数 |

所有注解需放在 `spec.annotations`。

下面查看明细。

### 金丝雀部署

如果您希望将一定比例的流量推送到新模型以测试它在生产中是否正常工作，则可以使用 Canary 部署。要将金丝雀添加到您的 SeldonDeployment，只需添加一个新的预测器部分并将主要和金丝雀的流量级别设置为所需的级别。例如：

```YAML
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: example
spec:
  name: canary-example
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.2.1
          name: classifier
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: main
    replicas: 1
    traffic: 75
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.2.2
          name: classifier
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: canary
    replicas: 1
    traffic: 25

```

以上示例包含 75% 流量分发到 「main」，25% 流量到 「canary」。

这里提供了一个[canary 发布](../examples/ambassador_canary.html)的示例。

### 影子部署

影子部署允许您向并行部署发送重复请求，但丢弃响应。这允许您在负载下测试机器学习模型并将结果与​​实时部署进行比较。

只需预测器中设置 `shadow` 布尔值。

提供了[影子部署](../examples/ambassador_shadow.html)工作示例。

要了解有关 ambassador 配置的更多信息，请参阅[他们的影子部署文档](https://www.getambassador.io/reference/shadowing/).

### 基于 Header 头的路由

基于标头的路由允许您根据传入请求中的标头将请求路由到特定的 Seldon 部署。

您只需要向您的 Seldon 部署资源添加一些注释。

  * `seldon.io/ambassador-header:<header>` : 加到 Ambassador 配置
     * 例:  `"seldon.io/ambassador-header":"location: london"	    `
  * `seldon.io/ambassador-regex-header:<header>` : 添加到配置的正则表达式标头
     * 例:  `"seldon.io/ambassador-header":"location: lond.*"	    `
  * `seldon.io/ambassador-service-name:<existing_deployment_name>` : 要附加到作为请求的替代映射的现有 Seldon 部署的名称。
     * 例: `"seldon.io/ambassador-service-name":"example"`

我们提供了一个[header头路由](../examples/ambassador_headers.html)工作示例。

要了解有关 ambassador 配置的更多信息，请参阅 [header 头路由](https://www.getambassador.io/reference/headers)文档。

### 限流

通过阻止对过载 Seldon 部署的额外连接或请求，限流器有助于提高系统的弹性。

您只需要向您的 Seldon 部署资源添加一些注释。

  * `seldon.io/ambassador-circuit-breakers-max-connections:<maximum number of connections>` : Seldon Deployment 的最大连接数
     * Example:  `"seldon.io/ambassador-circuit-breakers-max-connections":"200"`
  * `seldon.io/ambassador-circuit-breakers-max-pending-requests:<maximum number of queued requests>` : 等待连接时排队的最大请求数
     * Example:  `"seldon.io/ambassador-circuit-breakers-max-pending-requests":"100"`
  * `seldon.io/ambassador-circuit-breakers-max-requests:<maximum number of parallel outstanding requests>` : Seldon 部署的最大并行未完成请求数
     * Example: `"seldon.io/ambassador-circuit-breakers-max-requests":"200"`
  * `seldon.io/ambassador-circuit-breakers-max-retries:<maximum number of parallel retries>` : Seldon Deployment 允许的最大并行重试次数
     * Example: `"seldon.io/ambassador-circuit-breakers-max-retries":"3"`

提供了[限流器](../examples/ambassador_circuit_breakers.html)的工作示例。

要了解有关 ambassador 配置的更多信息，请参阅[限流器]](https://www.getambassador.io/docs/latest/topics/using/circuit-breakers/)文档。

## 同一集群中的多个 Ambassador

为了避免在运行多个 Ambassador 的集群中发生冲突，您可以将以下注释添加到您的 Seldon 部署资源中。

  * `seldon.io/ambassador-id:<instance id>`: 要添加到 Ambassador 的实例 id `ambassador_id` 配置

例如，

```YAML
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: multi-ambassadors
spec:
  annotations:
    seldon.io/ambassador-id: my_instance
  name: ambassadors-example
```

请注意，您的 Ambassador 实例必须配置为匹配的 `ambassador_id`。

查看 [AMBASSADOR_ID](https://www.getambassador.io/docs/latest/topics/running/running/#ambassador_id) 获取更多信息。

### 自定义 Amabassador 配置

上面讨论的配置应该涵盖大多数情况，但是自定义配置是必要的，以利用 Ambassador Edge Stack 的完整功能集作为流量管理和身份验证工具。根据您希望如何管理配置，有两种配置 Ambassador 的选项：自定义资源定义 (CRD) 和注释。

Ambassador 主要利用 [自定义资源](https://www.getambassador.io/docs/edge-stack/latest/topics/concepts/gitops-continuous-delivery/#policies-declarative-configuration-and-custom-resource-definitions) 来管理配置。这是  Kubernetes API 可读的自定义 `kind` 资源，用于以整个集群可观察的方式更新 Ambassador 的配置（例如，您可以使用 `kubectl get` 这些资源）。这些 CRD 可以独立于 Seldon Deploy 本身进行管理。

Ambassador 还支持基于注解的配置，可以使用 `seldon.io/ambassador-config` 注解键将其应用于 Seldon 部署。除了没有 `metadata` 和 `spec` 字段之外，基于注释的配置的整体格式与基于 CRD 的配置相同。以下片段演示了 CRD 和基于注释的配置之间的区别，并且在功能上是相同的。

```yaml
apiVersion: getambassador.io/v2
kind: Mapping
metadata:
  name: seldon_example_rest_mapping
spec:
  prefix: /mycompany/ml/
  service: production-model-example.seldon:8000
  timeout_ms: 3000
```

 * `seldon.io/ambassador-config:<configuration>` : 自定义 ambassador 配置
    * Example: `"seldon.io/ambassador-config":"apiVersion: ambassador/v2\nkind: Mapping\nname: seldon_example_rest_mapping\nprefix: /mycompany/ml/\nservice: production-model-example.seldon:8000\ntimeout_ms: 3000"`

提供了自定义 Ambassador 配置的[工作示例](../examples/ambassador_custom.html)。
