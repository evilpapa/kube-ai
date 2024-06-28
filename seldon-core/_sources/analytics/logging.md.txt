# 负载日志

为 Seldon Deployment 添加请求响应负载添加日志可在 graph 中的任意节点添加 logging 设置到图中。示例如下：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.3
          name: classifier
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
      logger:
        url: http://mylogging-endpoint
        mode: all
    name: example
    replicas: 1

```

最高级别的请求响应由以下提供：

```yaml
      logger:
        url: http://mylogging-endpoint
        mode: all
```

示例中，请求和响应负载由 `mode` 属性定义， CloudEvents 会发送到 `http://mylogging-endpoint`。

定义释义：

 * url: 任何 url。可选。如果未提供，默认为 Seldon Deployment 所在命名空间的原生 broker。
 * mode: `request`、`response` 或 `all`

## 日志直接落到 Kafka

您可以通过向 `svcOrchSpec` 添加适当的环境变量，将请求直接记录到 Kafka，以替代通过 CloudEvents 进行记录。示例如下：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: cifar10
  namespace: seldon
spec:
  name: resnet32
  predictors:
  - graph:
      implementation: TRITON_SERVER
      logger:
        mode: all
      modelUri: gs://seldon-models/triton/tf_cifar10
      name: cifar10
    name: default
    svcOrchSpec:
      env:
      - name: LOGGER_KAFKA_BROKER
        value: seldon-kafka-plain-0.kafka:9092
      - name: LOGGER_KAFKA_TOPIC
        value: seldon
    replicas: 1
  protocol: v2

```

需要的两个环境变量：

 * LOGGER_KAFKA_BROKER : Kafka Broker 服务端点。
 * LOGGER_KAFKA_TOPIC : 用于记录请求日志的 kafka Topic。

### 使用 SSL 记录到加密的 Kafka

您可以使用 SSL 将请求记录到加密的 Kafka。 SSL 使用在 SSL 握手过程中使用的私钥/证书对。

为了能够记录有效负载，客户端需要：
* 使用 SSL 进行身份验证
* 它自己的密钥库，由密钥对和签名证书组成
* 用于签署密钥-证书对的 CA 证书

CA 证书需要得到 broker 的认可，也可以用来验证 broker 的证书。

可在 [Confluent 文档](https://docs.confluent.io/platform/current/kafka/authentication_ssl.html) 和 [librdkafka 配置](https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md) 页阅读不同的配置项。

以下但是定义的发布示例：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: cifar10
  namespace: seldon
spec:
  name: resnet32
  predictors:
  - graph:
      implementation: TRITON_SERVER
      logger:
        mode: all
      modelUri: gs://seldon-models/triton/tf_cifar10
      name: cifar10
    name: default
    svcOrchSpec:
      env:
      - name: LOGGER_KAFKA_BROKER
        value: seldon-kafka-plain-0.kafka:9092
      - name: LOGGER_KAFKA_TOPIC
        value: seldon
      - name: KAFKA_SECURITY_PROTOCOL
        value: ssl
      - name: KAFKA_SSL_CA_CERT_FILE
        value: /path/to/ca.pem
      - name: KAFKA_SSL_CLIENT_CERT_FILE
        value: /path/to/access.cert
      - name: KAFKA_SSL_CLIENT_KEY_FILE
        value: /path/to/access.key
      - name: KAFKA_SSL_CLIENT_KEY_PASS
        valueFrom:
          secretKeyRef:
            name: my-kafka-secret
            key: ssl-password # Key password, if any (optional field)
    replicas: 1
  protocol: kfserving

```
Follow a [benchmarking notebook for CIFAR10 image payload logging showing 3K predictions per second with Triton Inference Server](../examples/kafka_logger.html).

## 设置全局默认值

如果不想每次设置自定义日志，可以通过 Helm Chart 变量的 `executor.requestLogger.defaultEndpoint` 进行设置，参考 [helm chart 高级设置章节](../reference/helm.rst)。

这里可以简单的定义个 URL 来调用。在通常的 kubernetes 方式中，如果提供了一个服务名称，除非指定 `.<namespace>` 将默认使用当前命名空间。

您仍希望确保模型的部署关于哪些请求被记录，所有的，请求的还是响应的（参上）。


