# Openshift

## 使用 RedHat Openshift 服务网格运行

如果您使用 Openshift RedHat Service Mesh 运行，您可以按照以下步骤使用 Seldon。

### 创建网关

确保在 istio-system 中创建网关。如

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

### 激活 Istio

1. 更新 Seldon Core CSV 以激活 istio。添加：

```
  config:
    env:
    - name: ISTIO_ENABLED
      value: 'true'
```


### 命名空间 Seldon Core 安装

如果您在特定命名空间中安装 Seldon Core，您将需要：

 1. 添加 NetworkPolicy 以允许 webhook 运行。在你运行 operator 的名称空间运行：

```
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: seldon-webhook
  namespace: <namespace>
spec:
  ingress:
  - ports:
    - port: 8443
      protocol: TCP
  podSelector:
    matchLabels:
      control-plane: seldon-controller-manager
  policyTypes:
  - Ingress
```


## 删除 Seldon Core Operator

目前，在删除 Seldon 核心操作员时未清理 webhook 配置。您将需要删除 `MutatingWebhookConfiguration` 和 `ValidatingWebhookConfiguration`。

对于 Seldon Core 的命名空间安装，这些将被调用：

 * `seldon-mutating-webhook-configuration-<namespace>`
 * `seldon-validating-webhook-configuration-<namespace>`