# XGBoost 服务

你可通过 Seldon 预封装 XGBoost 服务部署训练好的 XGBoost 模型。

## 要求

Seldon 预期训练模型是 `model.bst` 格式并使用 XGBoost 的
`bst.save_model()` 方法。
注意，这是 [推荐的序列化模型方法](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html)。

为了最大限度的提高模型序列化和服务运行时的兼容性，推荐在训练和预估时使用相同版本的工具集。
最新的 XGBoost 预封装服务以来版本如下：

| Package | Version |
| ------ | ----- |
| `xgboost` | `1.4.2` |

## 用法

使用 XGBoost 预封装服务，需在模型配置中设置 `XGBOOST_SERVER` 为 `implementation` 的值。
例如，针对保存的 Iris 模型，可参考如下配置：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: xgboost
spec:
  name: iris
  predictors:
  - graph:
      children: []
      implementation: XGBOOST_SERVER
      modelUri: gs://seldon-models/xgboost/iris
      name: classifier
    name: default
    replicas: 1
```

查看相似的 [可工作 notebook](../examples/server_examples.html) 示例。

## V2 KFServing 协议 [孵化中]

.. Warning:: 
  V2 KFServing 协议支持被考虑在孵化特性中。
  这意味着 Seldon Core 的某些特性仍未得到支持（比如：tracing, graphs等）。



XGBoost 服务也可用于暴露兼容 [V2
KFServing 协议](../graph/protocols.md#v2-kfserving-protocol)的 API。
注意，某些情况下，这会使用到 [Seldon
MLServer](https://github.com/SeldonIO/MLServer) 运行时。

为了支持 V2 KFServing 协议，设置`SeldonDeployment` 的 `protocol` 使用 `kfserving`，示例如下：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: xgboost
spec:
  name: iris
  protocol: kfserving # Activate the V2 protocol
  predictors:
  - graph:
      children: []
      implementation: XGBOOST_SERVER
      modelUri: gs://seldon-models/xgboost/iris
      name: classifier
    name: default
    replicas: 1
```

可参考此 [可工作 notebook](../examples/server_examples.html)示例。
