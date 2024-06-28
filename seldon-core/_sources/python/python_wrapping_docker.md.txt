# 使用 Docker 为 Seldon Core 打包 Python 模型


在本指南中，我们说明了将您自己的 Python 模型封装在一个 docker 镜像中所需的步骤，该镜像准备使用 Docker 与 Seldon Core 一起部署。

## 第 1 步 - 创建源代码

你需要

 * 一个运行模型的 python 类文件按
 * 带有 seldon-core 选项的 requirements.txt

我们将深入详细步骤：

### Python 文件
你的源代码应该包含一个类型与文件名相同的 python 文件。例如，我们的 python 模型脚手架 `wrappers/s2i/python/test/model-template-app/MyModel.py`：

```python
class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")

    def predict(self,X,features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called - will run identity function")
        return X
```

 * 该文件名为 MyModel.py，它定义了一个类 MyModel
 * 该类包含一个 predict 方法，该方法采用数组 (numpy) X 和 feature_names 并返回一个预测数组。
 * 您可以在类 init 方法中添加任何必需的初始化。
 * 您的返回数组应该至少是二维的。

### requirements.txt
填充您的代码所需的任何软件依赖项到 requirements.txt。该文件至少应包含：

```text
seldon-core
```

## 第 2 步 - 定义 Dockerfile

在与源代码和 requirements.txt 相同的目录中定义一个 Dockerfile。它将定义我们的 python 构建镜像所需的核心参数，以将您的模型封装为 env vars。一个例子是：

```dockerfile
FROM python:3.7-slim
WORKDIR /app

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME MyModel
ENV SERVICE_TYPE MODEL

# Changing folder to default user
RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
```


## 第 3 步 - 构建您的镜像
使用 `docker build . -t $ORG/$MODEL_NAME:$TAG` 创建 Docker 镜像。可以使用一个简单的名称，但约定是使用 ORG/IMAGE:TAG 格式。

## 与 Keras/Tensorflow 模型一起使用

为了确保带有 Tensorflow 后端的 Keras 模型正常工作，您可能需要在加载模型后调用 `_make_predict_function()`。这是因为 Flask 可能会在与初始化模型的进程不同的线程中调用预测请求。请参阅 [keras 问题](https://github.com/keras-team/keras/issues/6462)以进行进一步讨论。

## 环境变量
下面解释了构建器映像所需的环境变量。可在 Dockerfile 或  `docker run` 用 `-e` 指定参数。


### MODEL_NAME
包含模型的类的名称。还有将被导入的 python 文件的名称。

### SERVICE_TYPE

正在创建的服务类型。可用选项有：

 * MODEL
 * ROUTER
 * TRANSFORMER
 * COMBINER
 * OUTLIER_DETECTOR


### Flask 设置

查看 [Flask - Builtin Configuration Values](https://flask.palletsprojects.com/config/#builtin-configuration-values) 获取可能的配置；以下参数添加  `FLASK_` 字符串前缀时可设置（例如 `FLASK_JSON_SORT_KEYS` 在 Flask 会转换成 `JSON_SORT_KEYS` in Flask）：

 * DEBUG
 * EXPLAIN_TEMPLATE_LOADING
 * JSONIFY_PRETTYPRINT_REGULAR
 * JSON_SORT_KEYS
 * PROPAGATE_EXCEPTIONS
 * PRESERVE_CONTEXT_ON_EXCEPTION
 * SESSION_COOKIE_HTTPONLY
 * SESSION_COOKIE_SECURE
 * SESSION_REFRESH_EACH_REQUEST
 * TEMPLATES_AUTO_RELOAD
 * TESTING
 * TRAP_HTTP_EXCEPTIONS
 * TRAP_BAD_REQUEST_ERRORS

## 创建不同的服务类型

### MODEL

 * [最小化的模型源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/model-template-app)
 * [模型 notebooks 示例](../examples/notebooks.html)

### ROUTER
 * [Seldon Core 中路由定义](../analytics/routers.html)
 * [最小化的模型路由源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/router-template-app)

### TRANSFORMER

 * [小化的模型转换源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/transformer-template-app)
 * [转换器示例](https://github.com/SeldonIO/seldon-core/tree/master/examples/transformers)


## 高级用法

### 模型类参数
当发布镜像到 Kubernetes 时根据 SeldonDeloyment 中定义的内容进行填充 `parameters`。例如，我们的 [Python TFServing proxy](https://github.com/SeldonIO/seldon-core/tree/master/servers/tfserving_proxy) 具有如下定义的类 init 签名方法：

```python
class TfServingProxy(object):

    def __init__(self,rest_endpoint=None,grpc_endpoint=None,model_name=None,signature_name=None,model_input=None,model_output=None):
```

当部署 Seldon Deployment 以下参数可设置。可查看示例 [MNIST TFServing example](https://github.com/SeldonIO/seldon-core/blob/master/examples/models/tfserving-mnist/tfserving-mnist.ipynb) 中在 [SeldonDeployment](https://github.com/SeldonIO/seldon-core/blob/master/examples/models/tfserving-mnist/mnist_tfserving_deployment.json.template) 定义的部分参数示例：

```json
{
  "graph": {
    "name": "tfserving-proxy",
    "type": "MODEL",
    "children": [],
    "parameters": [
      {
        "name": "grpc_endpoint",
        "type": "STRING",
        "value": "localhost:8000"
      },
      {
        "name": "model_name",
        "type": "STRING",
        "value": "mnist-model"
      },
      {
        "name": "model_output",
        "type": "STRING",
        "value": "scores"
      },
      {
        "name": "model_input",
        "type": "STRING",
        "value": "images"
      },
      {
        "name": "signature_name",
        "type": "STRING",
        "value": "predict_images"
      }
    ]
  }
}
```


`type` 参数允许值在 [proto buffer 定义](https://github.com/SeldonIO/seldon-core/blob/44f7048efd0f6be80a857875058d23efc4221205/proto/seldon_deployment.proto#L117-L131)查看。


### 自定义指标
`from version 0.3`

要将自定义指标添加到您的响应中，您可以 类中定义一个可选方法 `metrics`，该方法返回一个指标字典列表。例如下所示：

```python
class MyModel(object):

    def predict(self, X, features_names):
        return X

    def metrics(self):
        return [{"type": "COUNTER", "key": "mycounter", "value": 1}]
```

有关自定义指标和指标 `dict` 格式的更多详细信息，请参见[此处](../analytics/analytics.html#custom-metrics)。

这是一个 [带有自定义指标 python 模型的 notebook 说明示例](../examples/custom_metrics.html)。

### Custom Request Tags
`from version 0.3`

要添加自定义请求标记数据，您可以添加一个可选方法 `tags`，该方法可以返回自定义元数据的字典，如下例所示：

```python
class MyModel(object):

    def predict(self, X, features_names):
        return X

    def tags(self):
        return {"mytag": 1}
```
