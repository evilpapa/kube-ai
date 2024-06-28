# Triton 推理服务

如果有可运行在 [NVIDIA Triton 推理服务](https://github.com/triton-inference-server/server) 的模型，也可使用 Seldon's Prepacked Triton 服务。

Triton 有多个后端支持，包括 TensorRT, Tensorflow, PyTorch 和 ONNX 模型。更多细节请参考 [Triton 支持文档](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_repository.html#section-framework-model-definition)。

## 示例

```
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: triton
spec:
  protocol: kfserving
  predictors:
  - graph:
      implementation: TRITON_SERVER
      modelUri: gs://seldon-models/trtis/simple-model
      name: simple
    name: simple
    replicas: 1
```

请尝试 [可工作 notebook](../examples/protocol_examples.html)
