# 原生 Kafka 流处理（>=1.2.0, alpha）

Seldon 从 1.2 版开始提供本地 kafka 集成。当您在 SeldonDeployment 中指定时 `serverType: kafka`。

当指定 `serverType: kafka` 还需要在 `svcOrchSpec` 设置 KAFKA_BROKER、KAFKA_INPUT_TOPIC、KAFKA_OUTPUT_TOPIC 环境变量。下面显示了 Tensorflow CIFAR10 模型的示例：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: tfserving-cifar10
spec:
  protocol: tensorflow
  transport: rest
  serverType: kafka  
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - args: 
          - --port=8500
          - --rest_api_port=8501
          - --model_name=resnet32
          - --model_base_path=gs://seldon-models/tfserving/cifar10/resnet32
          - --enable_batching
          image: tensorflow/serving
          name: resnet32
          ports:
          - containerPort: 8501
            name: http
    svcOrchSpec:
      env:
      - name: KAFKA_BROKER
        value: 10.12.10.16:9094
      - name: KAFKA_INPUT_TOPIC
        value: cifar10-rest-input
      - name: KAFKA_OUTPUT_TOPIC
        value: cifar10-rest-output
    graph:
      name: resnet32
      type: MODEL
      endpoint:
        service_port: 8501
    name: model
    replicas: 1
```

以上使用 tensorflow 协议创建了一个 REST tensorflow 部署，并连接到输入和输出主题。

## 细节

对于 SeldonDeployment：

 1. 从任何 Seldon 推理图开始
 1. 设置 `spec.serverType` 为 `kafka`
 1. 添加 `spec.predictor[].svcOrchSpec.env` 为 KAFKA_BROKER, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC。

对于输入 kafka 主题：

为指定协议和传输的输入预测创建请求流。

 * 对于 REST: 给定预测请求协议中的 JSON 表示。
 * 对于 gRPC:给定协议的请求的 protobuffer 二进制序列化。 You should also add a metadata field called `proto-name` with the package name of the protobuffer so it can be decoded, for example `tensorflow.serving.PredictRequest`. We can only support proto buffers for native grpc protocols supported by Seldon.


## TLS 设置

要允许消费者和生产者与 Kafka 建立 TLS 连接，请在服务编排环节使用以下环境变量：

  * 设置  KAFKA_SECURITY_PROTOCOL 为 "ssl"
  * 如果您有密钥和证书的值，请使用：
     * KAFKA_SSL_CA_CERT
     * KAFKA_SSL_CLIENT_CERT
     * KAFKA_SSL_CLIENT_KEY
  * 如果您有证书的文件位置，请使用：
     * KAFKA_SSL_CA_CERT_FILE
     * KAFKA_SSL_CLIENT_CERT_FILE
     * KAFKA_SSL_CLIENT_KEY_FILE
  * 如果您的密钥受密码保护，则添加
     * KAFKA_SSL_CLIENT_KEY_PASS (可选)

下面显示了一个从 screts 获取值的示例规范，来自 [Kafka KEDA 演示](../examples/kafka_keda.html).

```
   svcOrchSpec:
      env:
      - name: KAFKA_BROKER
        value: <nodepot>:9093
      - name: KAFKA_INPUT_TOPIC
        value: cifar10-rest-input
      - name: KAFKA_OUTPUT_TOPIC
        value: cifar10-rest-output
      - name: KAFKA_SECURITY_PROTOCOL
        value: ssl
      - name: KAFKA_SSL_CA_CERT
        valueFrom:
          secretKeyRef:
            name: seldon-cluster-ca-cert
            key: ca.crt
      - name: KAFKA_SSL_CLIENT_CERT
        valueFrom:
          secretKeyRef:
            name: seldon-user
            key: user.crt
      - name: KAFKA_SSL_CLIENT_KEY
        valueFrom:
          secretKeyRef:
            name: seldon-user
            key: user.key
      - name: KAFKA_SSL_CLIENT_KEY_PASS
        valueFrom:
          secretKeyRef:
            name: seldon-user
            key: user.password
```

## KEDA 缩放

KEDA 可用于缩放 Kafka SeldonDeployments 通过查看消费者滞后来扩展。

```
      kedaSpec:
        pollingInterval: 15
        minReplicaCount: 1
        maxReplicaCount: 2
        triggers:
        - type: kafka
          metadata:
            bootstrapServers: <nodeport>:9093
            consumerGroup: model.tfserving-cifar10.kafka
            lagThreshold: "50"
            topic: cifar10-rest-input
            offsetResetPolicy: latest
            #authMode: sasl_ssl (for latest KEDA - not released yet)
          authenticationRef:
            name: seldon-kafka-auth
```

在上面我们有：

 * 定义启动服务通过 `bootstrapServer` 连接 KEDA
 * 通过 `consumerGroup` 定义消费组
 * 设置 `lagThreshold` 进行扩展
 * 通过 `topic` 监控特定主题
 * 通过 AuthenticanTrigger `authenticationRef` 设置 TLS 认证

我们使用的认证触发器从密钥中提取 TLS 明细，例如：

```
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: seldon-kafka-auth
  namespace: kafka
spec:
  secretTargetRef:
  - parameter: tls
    name: keda-enable-tls
    key: tls
  - parameter: ca
    name: seldon-cluster-ca-cert
    key: ca.crt
  - parameter: cert
    name: seldon-user
    key: user.crt
  - parameter: key
    name: seldon-user
    key: user.key
```

可工作示例在[此处](../examples/kafka_keda.html)。

## 示例

 * [可工作的 CIFAR10 图片分类](../examples/cifar10_kafka.html)。
 * [可工作的 Kafka 集成 KEDA 自动缩放使用 TLS 连接 Kafka](../examples/kafka_keda.html)。
