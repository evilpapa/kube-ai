# SKLearn 服务

如果你有已经训练好的保存为 pickle 的 SKLearn 模型，你可以非常简洁的使用 Seldon's prepackaged SKLearn server 进行发布。

## 要求

Seldon 预期模型使用 `joblib` 方式保存，并且命名未 `model.joblib`。
注意，这是 SKLearn 项目 [推荐的序列化模型方法](https://scikit-learn.org/stable/modules/model_persistence.html)。

请注意，当我们使用 `joblib`，在推理服务中，训练模型和框架版本的匹配非常重要。

在最新的 SKLearn 预封装服务中，预期的版本如下：

| Package | Version |
| ------ | ----- |
| `scikit-learn` | `0.24.2` |

要检测 Seldon Core 的版本兼容性，请常看[兼容列表](#version-compatibility).

## 使用方法

要使用 pre-packaged SKLearn server，需要在模型的配置文件中将 `implementation` 声明为 `SKLEARN_SERVER`。
例如，针对保存的 Iris 预估模型，可以这样配置：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      name: classifier
    name: default
    replicas: 1

```

可通过 [可工作
notebook](../examples/server_examples.html) 进行尝试。

### Sklearn 预估方法

默认情况下，服务会在加载的 model/pipeline 调用 `predict_proba`。
要想调用 `predict` 需设置参数 `method` 为 `predict`。
例如：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris-predict
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      name: classifier
      parameters:
        - name: method
          type: STRING
          value: predict
    name: default
    replicas: 1
```

`method` 可选参数值为：`predict`, `predict_proba`,
`decision_function`。


## V2 KFServing 协议 [孵化中]

.. Warning:: 
  V2 KFServing 协议支持被考虑在孵化特性中。
  这意味着 Seldon Core 的某些特性仍未得到支持（比如：tracing, graphs等）。

MLFlow 服务也可用于暴露兼容 [V2
KFServing 协议](../graph/protocols.md#v2-kfserving-protocol)的 API。
注意，某些情况下，这会使用到 [Seldon
MLServer](https://github.com/SeldonIO/MLServer) 运行时。

为了支持 V2 KFServing 协议，设置`SeldonDeployment` 的 `protocol` 使用 `kfserving`，示例如下：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris-predict
  protocol: kfserving # Activate the V2 protocol
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      name: classifier
      parameters:
        - name: method
          type: STRING
          value: predict
    name: default
```

可通过相似的 [已工作
notebook](../examples/server_examples.html)尝试。

## 版本兼容

SKLearn 使用的预编译包以来安装的 Seldon Core 版本。
特别的，

| Seldon Version | SKLearn Version |
| -------------- | --------------- |
| `>=1.3`          | `0.23.2`          |
| `<1.3` (latest `1.2.3`)          | `0.20.3`          |

注意，使用不同的版本训练和推会带来不可预期的问题。

### 使用老版本

若要使用老版本的 SKLearn inference server，可通过覆盖 `componentSpecs` 设置。
比如，使用 `1.2.3` 版本可以按照如下配置：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris
  predictors:
  - componentSpecs:
    - spec:
       containers:
       - name: classifier
         image: seldonio/sklearnserver_rest:1.2.3
    graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      name: classifier
    name: default
    replicas: 1
    svcOrchSpec: 
      env: 
      - name: SELDON_LOG_LEVEL
        value: DEBUG
```

### 使用不同版本的 SKLearn

如果需要使用未被支持的 SKLearn 版本，可以通过扩展现有的 SKLearn server 来创建你自己的服务。
特别的，可通过代码目录的
[`servers/sklearnserver`](https://github.com/SeldonIO/seldon-core/tree/master/servers/sklearnserver)来创建自己的镜像。
此镜像用于 `SKLEARN_SERVER` 实现，可通过覆盖 `componentSpecs` 自定义。

注意，也可通过修改全局配置[`seldon-config` configmap](custom.md) 来指定使用SKLearn server。
这个修改将影响集群中所有的 `SeldonDeployments` 的 `SKLEARN_SERVER` 实现。
比如，下面的 `configmap` 配置：

```yaml
  SKLEARN_SERVER:
    grpc:
      defaultImageVersion: "1.2.3"
      image: seldonio/sklearnserver_grpc
    rest:
      defaultImageVersion: "1.2.3"
      image: seldonio/sklearnserver_rest
```
