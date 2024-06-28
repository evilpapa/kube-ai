# Seldon Core 与 Kubeflow 的原生集成

Seldon Core 团队是 Kubeflow 核心团队的一部分，这意味着我们不断确保 Seldon Core 与 Kubeflow Stack 完全集成。

Seldon Core 随 Kubeflow 一起安装。[Seldon Core 网站](https://docs.seldon.io/projects/seldon-core/en/latest/) 提供了完整的运行 Seldon Core 推理的文档。

如果你有保存在 PersistentVolume (PV), Google Cloud Storage bucket 或 Amazon S3 Storage 的模型，你可以使用 [Seldon Core 提供的模型服务器](https://docs.seldon.io/projects/seldon-core/en/latest/servers/overview.html)。

Seldon Core 也提供了 [特定模型语言封装](../wrappers/language_wrappers.html) 来封装你的推理代码，使其在 Seldon Core 中运行。

## Kubeflow 细节

您需要确保为您的模型提供服务的命名空间具有：

* 一个名为 kubeflow-gateway 的 Istio 网关
* 标签设置为 `serving.kubeflow.org/inferenceservice=enabled`

以下示例将标签应用于 `my-namespace` 命名空间以提供服务：

```console
kubectl label namespace my-namespace serving.kubeflow.org/inferenceservice=enabled
```

创建一个在命名空间 `my-namespace` 被调用的网关 `kubeflow-gateway`：

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: kubeflow-gateway
  namespace: my-namespace
spec:
  selector:
    istio: ingressgateway
  servers:
  - hosts:
    - '*'
    port:
      name: http
      number: 80
      protocol: HTTP
```

使用 `kubectl` 保存上述资源并应用。
