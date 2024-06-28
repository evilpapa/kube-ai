# Istio 入口

Seldon Core 可关联 [istio](https://istio.io/) 使用。Istio 提供了一个 [入口网关](https://istio.io/docs/tasks/traffic-management/ingress/)，Seldon Core 可以将新的部署自动关联到该网关。下面描述了使用 istio 的步骤。

## 安装 Seldon Core Operator

请确保通过 Helm 安装 seldon-core operator 时开启 istio 支持。例如：

```bash 
helm install seldon-core seldon-core-operator --set istio.enabled=true --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true
```

你需要将 istio 网关安装在 `istio-system` 命名空间。默认情况，我们假设网关叫 seldon-gateway。比如可通过以下 yaml 创建：

```
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
```

如果向创建 SSL 网关，请创建签名证书或者一个存在的签名证书（比如：fullchain.pem），key （privkey.pem）然后运行以下命令获取 SSL 网关。假设我们没有使用 [cert-manager](https://istio.io/latest/docs/ops/integrations/certmanager/)，然后我们自己进行签名证书创建：


```bash
openssl req -nodes -x509 -newkey rsa:4096 -keyout privkey.pem -out fullchain.pem -days 365 -subj "/C=GB/ST=GreaterLondon/L=London/O=SeldonSerra/OU=MLOps/CN=localhost"
```

导入 key 和证书到 istio-system 命名空间。

```bash
kubectl create -n istio-system secret tls seldon-ssl-cert --key=privkey.pem --cert=fullchain.pem
```

使用 YAML 文件创建 SSL Istio 网关。

```
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - hosts:
    - '*'
    port:
      name: https
      number: 443
      protocol: HTTPS
    tls:
      credentialName: seldon-ssl-cert
      mode: SIMPLE
```


如果有使用自己的网关，可在安装 seldon operator 时提供名称。比如你的网关名时 `mygateway` ，可通过以下命令安装 operator：

```bash 
helm install seldon-core seldon-core-operator --set istio.enabled=true --set istio.gateway=mygateway --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true
```

你可以为每个 Seldon Deploy 通过使用 `seldon.io/istio-gateway` 配置不同的网关。

## Istio 配置注释参考

| Annotation | 默认 | 描述 |
|------------|---------|------------|
|`seldon.io/istio-gateway:<gateway name>`| istio-system/seldon-gateway | 用此网关的部署。如果未提供应用名称空间，将指向 Seldon Delpoyment 的命名空间。 |
| `seldon.io/istio-retries` | None | istio 尝试次数 |
| `seldon.io/istio-retries-timeout` | None | 如果设置了 istio retries，每次请求多长事件超时 |
| `seldon.io/istio-host` | `*` | istio 虚拟服务的 Host |

所有注解均可放在 `spec.annotations` 或 `metadata.annotations`。将优先使用放在 `spec.annotations` 的配置。


## 流量路由

Istio 具备细化分配流量到部署中的能力。这将允许：

 * canary 更新
 * green-blue 发布
 * A/B 测试
 * 影子部署

更多信息请查看 [示例](../examples/istio_examples.html)，其中包括 [canary 更新](../examples/istio_canary.html)。

## 配置身份验证/授权
要强制客户端对 seldon 模型发布的访问进行验证/授权，可配置 Istio 的 `RequestAuthentication` 和 `AuthorizationPolicy`。
这将根据你的策略设计接受或拒绝对模型的请求。
更多信息请查看[这里](https://istio.io/latest/docs/reference/config/security/authorization-policy/)。

你可以设置指向特定命名空间的所有模型的策略，但必须使用 istio sidecar 代理，
并确保 seldon operator 按如下配置：
```
istio:
  enabled: true
  tlsMode: STRICT
```

如果设置了 `AuthorizationPolicy`，将破坏 Prometheus 指标抓取，
两种可能的选项可以解决此问题：
- 在 `AuthorizationPolicy` 中指定允许 GET 请求

示例：
```yaml
  - to:
    - operation:
        methods: ["GET"]
        paths: ["/prometheus"]
        ports: ["6000", "8000", "6001"]
```

- 也可在 Istio Operator 配置中 排除端口：
```yaml
  proxy:
        autoInject: enabled
        clusterDomain: cluster.local
        componentLogLevel: misc:error
        enableCoreDump: false
        excludeInboundPorts: ""
        excludeOutboundPorts: "15021"
```

## 问题解决
如果看到类似 `Failed to generate bootstrap config: mkdir ./etc/istio/proxy: permission denied` 问题，它可能时因为你的 istio 版本 <= 1.6。
Istio 代理 sidecar 默认以 root 运行（在版本 >= 1.7，默认时非 root）。
可通过在 helm chart 中修改 `defaultUserID=0` 修复，或者添加 `securityContext` 到 istio 的 sidecar 代理。

```
securityContext:
  runAsUser: 0
```


# 使用 Istio 服务网格
Istio 也可以直接用于集群内部的导流，而不是将其作为入口（导流集群外部流量）。

为此，虚拟服务 Seldon 会创建附加到名为 `mesh` 的「特殊」网关。这会将路由规则应用于网格内的流量，而无需通过网关进行路由。

由于 Istio 的限制（从 v1.11.3 开始），本地网格中的虚拟服务只能应用于一个主机。(在[此处](https://istio.io/latest/docs/ops/best-practices/traffic-management/#split-virtual-services)查看他们的文档)。因此，每个 Graph 都需要一个唯一的服务，这可以通过在主预测器中设置 `seldon.io/svc-name` 注解。

这是一个 `SeldonDeployment` 示例，它将利用内部网状网络在两个预测器之间分配流量，第一个预测器为 75%，第二个预测器为 25%：
``` yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  labels:
    app: seldon
  name: canary-example-1
  namespace: my-ns
spec:
  annotations:
    seldon.io/istio-gateway: mesh # NOTE
    seldon.io/istio-host: canary-example-1 # NOTE
  name: canary-example-1
  predictors:
  - annotations:
      seldon.io/svc-name: canary-example-1   # NOTE
    componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.11.0
          imagePullPolicy: IfNotPresent
          name: classifier
          securityContext:
            readOnlyRootFilesystem: false
        terminationGracePeriodSeconds: 1
    graph:
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    labels:
      sidecar.istio.io/inject: "true"
    name: main
    replicas: 1
    traffic: 75
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.11.0
          imagePullPolicy: IfNotPresent
          name: classifier
        terminationGracePeriodSeconds: 1
    graph:
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    labels:
      sidecar.istio.io/inject: "true"
    name: canary
    replicas: 1
    traffic: 25
```

需要指出的几个关键点：
1. 为主要的 (第一个) 预测器创建名为 `canary-example-1` 的唯一服务。此服务不能与命名空间中的任何其他服务发生冲突。该服务可以_不是_通过 SeldonDeployment 创建的服务，但也必须匹配必要的 Istio 路由规则。
2. 以上服务参考 `spec` 注解的 host 如下： `seldon.io/istio-host: canary-example-1`。浙江会设置 host 到 Istio Virutal Service 来创建新的服务。
3. 网关指定为 `seldon.io/istio-gateway: mesh` 以在 Istio Mesh 利用这个路由。注意：为了调用所有服务，并进行适当的路由，客户端_必须_同样_处在_网格中。这是通过将 Istio Sidecar 注入客户端的 pod 来实现的。

从集群内部和网格内部的 pod 内部，可以进行如下调用，并在两个预测器之间拆分流量：
``` shell
curl -X POST -H 'Content-Type: application/json' \
  -d '{"data": { "names": ["a", "b"], "ndarray": [[1,2]]}}' \
  http://mysvcname:8000/seldon/batest/canary-example-1/api/v1.0/predictions
```