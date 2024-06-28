# 部署

一旦创建了 JSON 或 YAML 格式的 Seldon Deployment 部署资源，可使用`kubectl`，比如，你的部署打包为`my_ml_deployment.yaml`:

```bash
kubectl apply -f my_ml_deployment.yaml
```

## Helm Charts

您可以使用 Helm 来管理您的部署 [Helm 示例](../examples/helm_examples.html)。

我们有一系列 [helm charts 模板](../graph/helm_charts.html) ，您可以用作你的部署基础。

## Ksonnet Apps

您可以使用 Ksonnet 来管理您的部署 [Ksonnet 示例](../notebooks/ksonnet_examples.ipynb).

我们有一系列 [Ksonnet prototypes 模板](../graph/ksonnet_templates.html) ，您可以用作你的部署基础。


## 部署验证

您可以使用 `kubectl` 检查运行部署的状态

例如：

```bash
kubectl get sdep -o jsonpath='{.items[].status}'
```





