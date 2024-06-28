# 协议

Tensorflow 协议只在版本 >=1.1 可用。

Seldon Core 支持以下数据平面：

 * [REST 和 gRPC Seldon 协议](#rest-and-grpc-seldon-protocol)
 * [REST 和 gRPC Tensorflow Serving 协议](#rest-and-grpc-tensorflow-protocol)
 * [REST 和 gRPC V2 KFServing 协议](#v2-kfserving-protocol)

## REST 和 gRPC Seldon 协议

 * [REST Seldon Protocol](../reference/apis/index.html)

Seldon 是 SeldonDeployment 资源的默认协议。你可在 SeldonDeployment 资源设置 `transport: grpc` 指定 gRPC 协议，或者所有组件在图节点 endpoint.tranport 设置为 grpc。

查看 [notebook 示例](../examples/protocol_examples.html)。

## REST 和 gRPC Tensorflow 协议

   * [REST Tensorflow 协议定义](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md)。
   * [gRPC Tensorflow 协议定义](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)。

通过在 Seldon Deployment 定义 `protocol: tensorflow` 、 `transport: rest` 或 `transport: grpc` 来激活，参考 [notebook 示例](../examples/protocol_examples.html)。

在 Seldon 图定义中协议会按照预期的 Tensorflow Serving 服务运行单个模型图一样在图中运行单个模型。对于更复杂的图，可链式定义模型：

 * 将第一个响应结果作为请求发送到第二个。这将自动在 Seldon 图中定义中完成。用户应确保每个更改模型的响应可以向链中的下一个提供请求。
 * 只有多个链式模型才可处理预测。


一般考虑事项：

  * Seldon 组件标记为 MODELS、INPUT_TRANSFORMER 和 OUTPUT_TRANSFORMERS 才允许 PredictionService Predict 方法调用。
  * GetModelStatus 在所有模型图中可用。
  * GetModelMetadata 在所有模型图中可用。
  * Combining 和 Routing 当前在 Tensorflow 协议中不支持。
  * `status` 和 `metadata` 可在任何途中的任何模型调用。
  * 非标准 Seldon 扩展可在图上调用预估： `/v1/models/:predict`.
  * SeldonDeployment 定义中 `graph` 节点的模型名称必须和 Tensorflow 服务加载的模型名称相同。


## V2 协议 

Seldon 和 [NVIDIA Triton Server 
项目](https://github.com/triton-inference-server/server) 以及 [KFServing 
项目](https://github.com/kubeflow/kfserving)合作推出
新的机器学习预估协议。
这一共同努力的核心思想是，
这一新协议将成为标准推理协议，
并将用于多个推理服务。

在 Seldon Core，
可在 `SeldonDeployment` 将协议定义为 `protocol: kfserving`。
比如：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris-predict
  protocol: v2
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.14.0/sklearn/iris
      name: classifier
      parameters:
        - name: method
          type: STRING
          value: predict
    name: default
```

目前，
`v2` 协议只在预封装预估服务器子集中支持。
特别地，

| 预封装服务 | 支持 | 基础运行时 |
| -- | -- | -- |
| [TRITON_SERVER](../servers/triton.md) | ✅ | [NVIDIA Triton](https://github.com/triton-inference-server/server) |
| [SKLEARN_SERVER](../servers/sklearn.md) | ✅  | [Seldon MLServer](https://github.com/seldonio/mlserver) |
| [XGBOOST_SERVER](../servers/xgboost.md) | ✅  | [Seldon MLServer](https://github.com/seldonio/mlserver) |
| [MLFLOW_SERVER](../servers/mlflow.md) | ✅  | [Seldon MLServer](https://github.com/seldonio/mlserver) |

可在 [notebook 示例](../examples/protocol_examples.html)查看 `v2`。
