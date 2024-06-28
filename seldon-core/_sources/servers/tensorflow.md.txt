# Tensorflow Serving

如果有基于 Tensorflow 的训练模型，可以直接发布未 REST or gRPC 微服务。

## 最小示例

### REST 最小示例

REST 需要指定参数：

 * signature_name
 * model_name

```
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: tfserving
spec:
  name: mnist
  predictors:
  - graph:
      children: []
      implementation: TENSORFLOW_SERVER
      modelUri: gs://seldon-models/tfserving/mnist-model
      name: mnist-model
      parameters:
        - name: signature_name
          type: STRING
          value: predict_images
        - name: model_name
          type: STRING
          value: mnist-model
    name: default
    replicas: 1

```

### gRPC 最小示例

gRPC 需要指定参数：

 * signature_name
 * model_name
 * model_input
 * model_output

```
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: tfserving
spec:
  name: mnist
  predictors:
  - graph:
      children: []
      implementation: TENSORFLOW_SERVER
      modelUri: gs://seldon-models/tfserving/mnist-model
      name: mnist-model
      endpoint:
        type: GRPC
      parameters:
        - name: signature_name
          type: STRING
          value: predict_images
        - name: model_name
          type: STRING
          value: mnist-model
        - name: model_input
          type: STRING
          value: images
        - name: model_output
          type: STRING
          value: scores          
    name: default
    replicas: 1

```


查看 [可工作 notebook](../examples/server_examples.html)


## 多模型服务

如 [示例 notebook](../examples/protocol_examples.html)所示，您可以利用 Tensorflow 服务的功能从存储库加载多个模型。请按照一下进行配置 [Tensorflow Serving 高级配置文档](https://www.tensorflow.org/tfx/serving/serving_config)。

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: example-tfserving
spec:
  protocol: tensorflow
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - args: 
          - --port=8500
          - --rest_api_port=8501
          - --model_config_file=/mnt/models/models.config
          image: tensorflow/serving
          name: multi
          ports:
          - containerPort: 8501
            name: http
            protocol: TCP
          - containerPort: 8500
            name: grpc
            protocol: TCP
    graph:
      name: multi
      type: MODEL
      implementation: TENSORFLOW_SERVER
      modelUri: gs://seldon-models/tfserving/multi-model
      endpoint:
        httpPort: 8501
        grpcPort: 8500
    name: model
    replicas: 1
```
