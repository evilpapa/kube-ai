# Seldon Python 组件

要创建在 Seldon 下运行的组件，您应该创建一个类来实现您正在创建的组件类型所需的签名。

## 模型

要封装您的机器学习模型，请创建一个具有以下签名的 predict 方法的类：

```python
    def predict(self, X: np.ndarray, names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
```

您的 predict 方法将接收一个 numpy 数组 `X` 其中包含一组可迭代的列名（如果它们存在于输入特征中）和可选的元数据字典。它应该将预测结果返回为：

- Numpy 数组
- 值列表
- 字符串或字节

一个简单的例子如下所示：

```python
class MyModel(object):
    """
    Model template. 你可以在运行时从 __init__ 从本地加载模型参数
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        添加任意初始化参数。在运行时将从kubernetes Sedlondeployment 资源定义的图中传入定义好的参数。
        """
        print("Initializing")

    def predict(self, X, features_names=None):
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

### 返回类名

您还可以提供一种方法来返回用于预测的列名称，`class_names` 其签名方法如下所示：

```python
    def class_names(self) -> Iterable[str]:
```

### 示例

你可以参考[各种笔记本示例](../examples/notebooks.html)。

## 转换器

Seldon Core 允许您创建组件以在输入请求方向（输入转换器）或输出响应方向（输出转换器）上转换特征。对于这些组件，使用以下签名创建方法：

```python
    def transform_input(self, X: np.ndarray, names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:

    def transform_output(self, X: np.ndarray, names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
```

## 合并器

Seldon Core 允许您创建将来自多个模型的响应组合成单个响应的组件。要为此创建一个类，请添加一个带有以下签名的方法：

```python
    def aggregate(self, features_list: List[Union[np.ndarray, str, bytes]], feature_names_list: List) -> Union[np.ndarray, List, str, bytes]:
```

下面显示了一个对一组响应求平均值的简单示例：

```python
import numpy as np
logger = logging.getLogger(__name__)

class ImageNetCombiner(object):

    def aggregate(self, Xs, features_names):
        return (np.reshape(Xs[0],(1,-1)) + np.reshape(Xs[1], (1,-1)))/2.0
```

## 路由器

路由器提供将请求定向到一组子组件之一的功能。为此，您应该创建一个带有签名的方法，如下所示，该方法返回请求需要路由到的子组件的 `id`。`id` 是连接子路由的索引。

```python
    def route(self, features: Union[np.ndarray, str, bytes], feature_names: Iterable[str]) -> int:
```

要查看这方面的示例，您可以遵循作为 Seldon Core 部分的各种[示例路由](https://github.com/SeldonIO/seldon-core/tree/master/components/routers)。

## 添加自定义指标

要返回与调用关联的指标，请创建一个带有签名的方法，如下所示：

```python
    def metrics(self) -> List[Dict]:
```

此方法应返回[自定义指标](../analytics/analytics.md#custom-metrics)文档中所述的指标字典。

下面是一个说明性示例：

```python
class ModelWithMetrics(object):

    def __init__(self):
        print("Initialising")

    def predict(self,X,features_names):
        print("Predict called")
        return X

    def metrics(self):
        return [
            {"type": "COUNTER", "key": "mycounter", "value": 1}, # a counter which will increase by the given value
            {"type": "GAUGE", "key": "mygauge", "value": 100},   # a gauge which will be set to given value
            {"type": "TIMER", "key": "mytimer", "value": 20.2},  # a timer which will add sum and count metrics - assumed millisecs
        ]
```

注意：在 Seldon Core 1.1 之前，自定义指标始终返回给客户端。从 SC 1.1 开始，您可以将 `INCLUDE_METRICS_IN_CLIENT_RESPONSE` 环境变量设置为 `true` 或 `false` 来控制此行为。尽管这个环境变量有价值，自定义指标将始终暴露给  Prometheus。

在 Seldon Core 1.1.0 之前，未实现在 info 等级的预估调用记录自定义指标日志信息。从 Seldon Core 1.1.0 开始，可在 debug 级别记录。要抑制此警告，请实现一个返回空列表的指标函数：

```python
def metrics(self):
    return []
```

## 返回标签

如果希望向返回的元数据添加任意标签，可以提供一个 tags 签名方法，如下所示：

```python
    def tags(self) -> Dict:
```

个简单的例子如下所示：

```python
class ModelWithTags(object):

    def predict(self,X,features_names):
        return X

    def tags(self,X):
        return {"system":"production"}
```


## 运行时指标和标签

从 SC 1.3 开始 `metrics` 和 `tags` 也可以定义 `predict`、`transform_input`、`transform_output`、`send_feedback`、`route` 和 `aggregate` 输出。

这是线程安全的。

```python
from seldon_core.user_model import SeldonResponse


class Model:
    def predict(self, features, names=[], meta={}):
        runtime_metrics = {"type": "COUNTER", "key": "instance_counter", "value": len(X)},
        runtime_tags = {"runtime": "tag", "shared": "right one"}
        return SeldonResponse(data=X, metrics=runtime_metrics, tags=runtime_tags)

    def metrics(self):
        return [{"type": "COUNTER", "key": "requests_counter", "value": 1}]

    def tags(self):
        return {"static": "tag", "shared": "not right one"}
```

主义 `tags` 和 `metrics` 有限通过 `SeldonResponse` 定义。
在上面的示例中，返回的标签将是：
```json
{"runtime":"tag", "shared":"right one", "static":"tag"}
```


## REST 健康点
如果您想添加 REST 健康点，您可以实现 `health_status` 签名方法，如下所示：
```python
    def health_status(self) -> Union[np.ndarray, List, str, bytes]:
```

您可以使用它来验证您的服务是否可以在构建 docker 镜像后正常响应 HTTP 调用，也可以用作 kubernetes 活跃度和就可读探测器来验证您的模型是否健康。

一个简单的例子如下所示：

```python
class ModelWithHealthEndpoint(object):
    def predict(self, X, features_names):
        return X

    def health_status(self):
        response = self.predict([1, 2], ["f1", "f2"])
        assert len(response) == 2, "health check returning bad predictions" # or some other simple validation
        return response
```

当您用于 `seldon-core-microservice` 启动 HTTP 服务器时，
您可以通过检查 `/health/status` 端点来验证模型是否已启动并正在运行：
```console
$ curl localhost:5000/health/status
{"data":{"names":[],"tensor":{"shape":[2],"values":[1,2]}},"meta":{}}
```

此外，你可以使用 `/health/ping` 端点
来实现一个检查 HTTP 服务已启动的轻量级调用：

```console
$ curl localhost:5000/health/ping
pong%
```

您还可以通过将 HTTP 运行状况端点添加到 
`SeldonDeployment` YAML 中来覆盖默认的活跃度和可读探测。
您可以修改探针的参数以满足您的可靠性需求，而不会对容器施加太大压力。在 
[kubernetes 文档](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)中阅读有关这些探测器的更多信息。
一个例子如下所示：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
spec:
  name: my-app
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: my-app-image:version
          name: classifier
          livenessProbe:
            failureThreshold: 3
            initialDelaySeconds: 60
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/status
              port: http
              scheme: HTTP
            timeoutSeconds: 1
          readinessProbe:
            failureThreshold: 3
            initialDelaySeconds: 20
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/status
              port: http
              scheme: HTTP
            timeoutSeconds: 1
```

However, note if `executor.fullHealthChecks` is set to `true` then the Seldon orchestrator will call your health status method to check the model is ready.

## 低级别方法

如果想要更多的控制，可以提供一个低级方法将原始 proto 缓冲区有效负载作为输入。这些签名如下所示在 `seldon_core>=0.2.6.1` 发布：

```python
    def predict_raw(self, msg: Union[Dict, prediction_pb2.SeldonMessage]) -> prediction_pb2.SeldonMessage:

    def send_feedback_raw(self, feedback: prediction_pb2.Feedback) -> prediction_pb2.SeldonMessage:

    def transform_input_raw(self, msg: Union[Dict, prediction_pb2.SeldonMessage]) -> prediction_pb2.SeldonMessage:

    def transform_output_raw(self, msg: Union[Dict, prediction_pb2.SeldonMessage]) -> prediction_pb2.SeldonMessage:

    def route_raw(self, msg: prediction_pb2.SeldonMessage) -> prediction_pb2.SeldonMessage:

    def aggregate_raw(self, msgs: prediction_pb2.SeldonMessageList) -> prediction_pb2.SeldonMessage:

    def health_status_raw(self) -> prediction_pb2.SeldonMessage:
```

## 用户定义的异常

如果要处理自定义异常，请定义如下所示 `model_error_handler` 方法：

```python
    model_error_handler = flask.Blueprint('error_handlers', __name__)
```

一个例子如下：

```python
"""
Model Template
"""
class MyModel(Object):

    """
    The field is used to register custom exceptions
    """
    model_error_handler = flask.Blueprint('error_handlers', __name__)

    """
    Register the handler for an exception
    """
    @model_error_handler.app_errorhandler(UserCustomException)
    def handleCustomError(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def __init__(self, metrics_ok=True, ret_nparray=False, ret_meta=False):
        pass

    def predict(self, X, features_names, **kwargs):
        raise UserCustomException('Test-Error-Msg',1402,402)
        return X
```

```python
"""
User Defined Exception
"""
class UserCustomException(Exception):

    status_code = 404

    def __init__(self, message, application_error_code,http_status_code):
        Exception.__init__(self)
        self.message = message
        if http_status_code is not None:
            self.status_code = http_status_code
        self.application_error_code = application_error_code

    def to_dict(self):
        rv = {"status": {"status": self.status_code, "message": self.message,
                         "app_code": self.application_error_code}}
        return rv

```

## 多值 numpy 数组

默认的，当使用 data ndarray 参数，转换为 ndarray（默认情况下）会将所有内部类型转换为相同类型。对于模型需要转换为不同类型值作为的输入数组时，可通过 `predict_raw` 覆盖函数来实现此目的，该函数可以访问原始请求，并按如下方式创建 numpy 数组：

```python
import numpy as np

class Model:
    def predict_raw(self, request):
        data = request.get("data", {}).get("ndarray")
        if data:
            mult_types_array = np.array(data, dtype=object)

        # Handle other data types as required + your logic
```

## Gunicorn 和负载

如果封装的 python 类在 [Gunicorn](./python_server) 下提供服务，
那么作为每个 gunicorn worker 初始化的一部分，`load` 方法
将在您的类上被调用（如果它有的话）。
您应该使用此方法来加载和初始化您的模型。
这对于需要在每个工作进程中创建会话的 Tensorflow 模型
很重要。
该 [Tensorflow MNIST 例子](../examples/deep_mnist.html)做到这一点。

```python
import tensorflow as tf
import numpy as np
import os

class DeepMnist(object):
    def __init__(self):
        self.loaded = False
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]

    def load(self):
        print("Loading model",os.getpid())
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph("model/deep_mnist_model.meta")
        saver.restore(self.sess,tf.train.latest_checkpoint("./model/"))
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.loaded = True
        print("Loaded model")

    def predict(self,X,feature_names):
        if not self.loaded:
            self.load()
        predictions = self.sess.run(self.y,feed_dict={self.x:X})
        return predictions.astype(np.float64)
```

## 整数

Python 中的 `json` 包将没有小数部分的数字解析为整数。
因此，
只包含没有小数部分的数字的张量将被解析为整数张量。

为了说明上述情况，我们可以考虑以下示例：

```json
{
  "data": {
    "ndarray": [0, 1, 2, 3]
  }
}
```

默认情况下，该 `json` 包会将 `data.ndarray` 
字段中的数组解析为 Python `Integer` 值数组。
由于没有浮点值，因此 `numpy` 
将创建一个带有 `dtype = np.int32` 的张量。

如果我们想强制不同的行为，我们可以使用底层 `predict_raw()` 
方法来控制输入有效负载的反序列化。
作为一个例子，使用上面的例子，
我们可以通过使用  `dtype = np.float64` 实现 `predict_raw()` 方法来强制结果张量 ：

```python
import numpy as np

class Model:
    def predict_raw(self, request):
        data = request.get("data", {}).get("ndarray")
        if data:
            float_array = np.array(data, dtype=np.float64)

        # Make predictions using float_array
```


## 孵化功能


### REST 元数据端点
python 封装器将自动公开一个 `/metadata` 端点以返回有关加载模型的元数据。
开发人员在他们的类中实现一个 `metadata` 方法来提供包含模型元数据的返回的 `dict`。

有关更多详细信息，请参阅[元数据文档](../reference/apis/metadata.md)。


#### 示例格式：
```python
class Model:
    ...

    def init_metadata(self):

        meta = {
            "name": "model-name",
            "versions": ["model-version"],
            "platform": "platform-name",
            "inputs": [{"name": "input", "datatype": "BYTES", "shape": [1]}],
            "outputs": [{"name": "output", "datatype": "BYTES", "shape": [1]}],
        }

        return meta
```

#### 验证
输出开发者定义的 `metadata` 方法将按照 [kfserving dataplane 提案](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#model-metadata) 协议进行验证，查看 [Github issue](https://github.com/SeldonIO/seldon-core/issues/1638) 详情：
```javascript
$metadata_model_response =
{
  "name" : $string,
  "versions" : [ $string, ... ], // optional
  "platform" : $string,
  "inputs" : [ $metadata_tensor, ... ],
  "outputs" : [ $metadata_tensor, ... ]
}
```
以及
```javascript
$metadata_tensor =
{
  "name" : $string,
  "datatype" : $string,
  "shape" : [ $number, ... ]
}
```

如果验证失败，服务器将在请求 `metadata` 时回复 `500` 响应 `MICROSERVICE_BAD_METADATA`。

#### 例子
- [带有元数据模型的基础示例](../examples/metadata.html)
- [使用 MinIO 的 SKLearn Server 示例](../examples/minio-sklearn.html)



## 下一步

创建组件之后，需要创建一个可被 Seldon Core 管理的 Docker 镜像。参考 [s2i](./python_wrapping_s2i.md) 或 [Docker](./python_wrapping_docker.md) 文档来完成。
