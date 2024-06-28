# 自定义预估服务器

开箱即用，Seldon 提供一些 [预封装预估
服务器](./overview.md)。
然而，但是，在某些情况下，推出您自己的可重复使用的
推论服务器是有意义的。
例如，您可能需要特定的依赖关系、特定版本、
自定义过程来下载模型权重等。

为了支持这些使用案例，Seldon 允许您轻松地构建自己的
推理服务器，并可以配置使用，
就像您进行预封装的一样。
即通过使用 `implementation` 选项并定义模型参数
（例如 `modelUri`）。

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: nlp-model
spec:
  predictors:
    - graph:
        children: []
        implementation: CUSTOM_INFERENCE_SERVER
        modelUri: s3://our-custom-models/nlp-model
        name: model
      name: default
      replicas: 1
```

## 构建一个新的预估服务器

要构建自定义服务器可
查看[如何封装模型](../wrappers/language_wrappers.md)说明。
唯一区别在于，服务器会接收一些模型参数，
如：`modelUri`。

作为灵感，您可以参考 [SKLearn 
server](https://github.com/SeldonIO/seldon-core/blob/d84b97431c49602d25f6f5397ba540769ec695d9/servers/sklearnserver/sklearnserver/SKLearnServer.py#L16-L23)
和 [其他预封装服务器](./overview.md)
的 `__init__()` 处理方法：

```python
def __init__(self, model_uri: str = None,  method: str = "predict_proba"):
        super().__init__()
        self.model_uri = model_uri
        self.method = method
        self.ready = False
        self.load()
```

## 添加一个服务器

可用预估服务器列表 Seldon Core 存放在
`seldon-config` 配置中，它同 Seldon 
Core operator 
在同一命名空间中。特别是，`predictor_servers` 关键字 为每个
预估服务器保存着 JSON 配置。

`predictor_servers` 关键字保存的 JSON 字典
如下：

```json
{
  "SKLEARN_SERVER": {
    "grpc": {
      "defaultImageVersion": "0.2",
      "image": "seldonio/sklearnserver_grpc"
    },
    "rest": {
      "defaultImageVersion": "0.2",
      "image": "seldonio/sklearnserver_rest"
    }
  }
}
```

添加新的预估服务器就像
上面一样添加一个新的配置到字典中。
如：

```json
{
  "CUSTOM_INFERENCE_SERVER": {
    "grpc": {
      "defaultImageVersion": "1.0",
      "image": "org/custom-server-grpc"
    },
    "rest": {
      "defaultImageVersion": "1.0",
      "image": "org/custom-server-rest"
    }
  },
  "SKLEARN_SERVER": {
    "grpc": {
      "defaultImageVersion": "0.2",
      "image": "seldonio/sklearnserver_grpc"
    },
    "rest": {
      "defaultImageVersion": "0.2",
      "image": "seldonio/sklearnserver_rest"
    }
  }
}
```

## 可用示例

一个可工作的构建 LighGBM 模型服务器可在[这里](../examples/custom_server.html)查看。