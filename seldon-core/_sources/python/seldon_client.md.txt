# Seldon Python 客户端

我们提供了 python 示例客户端用于内部微服务测试或者外部API的 REST 或者 gRPC 的 API 调用。

它的使用示例可以在大多数 [Seldon 核心示例](../examples/notebooks.html)中找到。

要使用客户端，只需为您的用例创建一个带有设置的实例，例如：

```python
from seldon_core.seldon_client import SeldonClient
sc = SeldonClient(deployment_name="mymodel",namespace="seldon",gateway_endpoint="localhost:8003",gateway="ambassador")
```

在上面我们将我们的 deployment_name 设置为「mymodel」，将命名空间设置为「seldon」。有关完整的参数集，请参见[此处](./api/seldon_core.html#seldon_core.seldon_client.SeldonClient)。

生成 REST 随机请求负载：

```python
r = sc.predict(transport="rest")
```

生成 gRPC 随机请求负载：

```python
r = sc.predict(transport="grpc")
```

默认返回类型是「SeldonMessage」类型的 protobuf，但您也可以选择返回一个 JSON 字典，这样可以更容易地与内部数据交互。您可以通过在初始化您的 Seldon 客户端或发送预测请求时传递 kwarg `client_return_type` 值 `dict`（默认值为 `proto`）来完成此操作。例如：

```python
sc = SeldonClient(..., client_return_type="dict")
```

或者，您可以在发送传递覆盖默认参数的请求：

```python
sc = SeldonClient(..., client_return_type="proto")

sc.predict(..., client_return_type="dict") # Here we override it
```

## 高级示例

 * [SSL 及认证](../examples/seldon_client.html)
