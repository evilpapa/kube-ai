# 使用 s2i 为 Seldon Core 打包 Python 模型

在本指南中，我们说明了将您自己的 Python 模型封装在一个 docker 镜像中所需的步骤，该镜像准备使用 [source-to-image 应用程序 s2i](https://github.com/openshift/source-to-image) 与 Seldon Core 一起部署。

如果您不熟悉 s2i，您可以阅读有关使用 [s2i 的一般说明](../wrappers/s2i.md)，然后按照以下步骤操作。


## 第 1 步 - 安装 s2i

 [下载安装 s2i](https://github.com/openshift/source-to-image#installation)

 * 使用 s2i 的先决条件是：
   * Docker
   * Git (如果从远程 git repo 构建)

要检查一切是否正常，您可以运行

```bash
s2i usage seldonio/seldon-core-s2i-python3:1.14.0
```


## 第 2 步 - 创建源代码

要使用我们的 s2i 映像构建器打包您的 Python 模型，您需要：

 * 带有运行模型的类的 python 文件
 * 您的模型的依赖项和环境，可以使用以下任一项来描述：
   - `requirements.txt`
   - `setup.py`
   - `environment.yml`
 * `.s2i/environment` - s2i 构建器用于正确封装模型的模型定义

我们将详细介绍每个步骤：

### Python 文件

你的源代码应该包含一个 python 文件，它定义了一个与文件同名的类。有关更多详细信息，请[参阅有关创建 Python 类的详细信息](python_component.md)。

### 依赖关系

您可以使用任一描述模型的依赖：
`requirements.txt`、`setup.py` 或 `environment.yml`。

#### requirements.txt

填充您的代码所需的任何软件依赖项到 `requirements.txt`。
这些将在创建镜像时通过 pip 安装。

#### setup.py

与 `requirements.txt` 相似，
你可以在 `setup.py` 文件描述模型的依赖项：

```python
from setuptools import setup

setup(
  name="my-model",
  # ...
  install_requires=[
    "scikit-learn",
  ]
)
```

#### environment.yml

使用 `environment.yml` 文件描述您的 Conda 环境：

```yaml
name: my-conda-environment
channels:
  - defaults
dependencies:
  - python=3.6
  - scikit-learn=0.19.1
```

在镜像创建期间，`s2i` 会创建你的 Conda 环境
获取所有需要的依赖。
在运行时，创建的 Conda 环境将在启动时激活。

### .s2i/environment

定义我们的 python 构建器构建镜像封装模型所需的核心参数来。例：

```bash
MODEL_NAME=MyModel
SERVICE_TYPE=MODEL
```

构建映像时，也可以在命令行上提供或覆盖这些值。

有关此文件可能的键和值，请参见下文。

## 第 3 步 - 构建您的映像
使用 `s2i build` 从源代码创建镜像。你需要本机安装 Docker 如果您的源代码在公共 git 存储库中，需要可选的安装 git。您可以从三个 python 构建镜像中中进行选择。

 * Python 3.6 : seldonio/seldon-core-s2i-python36:1.10.0-dev seldonio/seldon-core-s2i-python3:1.10.0-dev
   * 请注意[在 Python 3.7 运行 TensorFlow 存在问题](https://github.com/tensorflow/tensorflow/issues/20444)（2018 年 11 月），并且 TensorFlow（2018 年 12 月）未正式支持 Python 3.7。
 * Python 3.6 plus ONNX 通过 [Intel nGraph](https://github.com/NervanaSystems/ngraph) 支持 : seldonio/seldon-core-s2i-python3-ngraph-onnx:0.1

使用 s2i，您可以直接从 git 存储库或本地源文件夹构建。有关更多详细信息，请参阅 [s2i 文档](https://github.com/openshift/source-to-image/blob/master/docs/cli.md#s2i-build)。一般格式为：

```bash
s2i build <src-folder> seldonio/seldon-core-s2i-python3:1.14.0 <my-image-name>
```

如果使用 python 3 请更改为 seldonio/seldon-core-s2i-python3。

在 seldon-core 中使用测试模板模型的示例调用：

```bash
s2i build https://github.com/seldonio/seldon-core.git --context-dir=wrappers/s2i/python/test/model-template-app seldonio/seldon-core-s2i-python3:1.14.0 seldon-core-template-model
```

上面的 s2i 构建调用：

 * 使用 GitHub 存储库：https://github.com/seldonio/seldon-core.git 以及仓库中 `wrappers/s2i/python/test/model-template-app` 文件夹。
 * 使用构建镜像 `seldonio/seldon-core-s2i-python3`
 * 创建一个 docker 镜像 `seldon-core-template-model`


对于从本地源文件夹构建，我们克隆 seldon-core 存储库的示例：

```bash
git clone https://github.com/seldonio/seldon-core.git
cd seldon-core
s2i build wrappers/s2i/python/test/model-template-app seldonio/seldon-core-s2i-python3:1.14.0 seldon-core-template-model
```

如需更多帮助，请参阅：

```bash
s2i usage seldonio/seldon-core-s2i-python3:1.14.0
s2i build --help
```

## 与 Keras/Tensorflow 模型一起使用

为了确保带有 Tensorflow 后端的 Keras 模型正常工作，您可能需要在加载模型后调用 `_make_predict_function()`。这是因为 Flask 可能会在与初始化模型的进程不同的线程中调用预测请求。请参阅 [keras 问题](https://github.com/keras-team/keras/issues/6462)以进行进一步讨论。

## 环境变量

下面解释了构建器映像所需的环境变量。您可以在 `.s2i/environment` 文件中或在命令行 `s2i build` 中提供它们。


### MODEL_NAME

包含模型的类的名称。还有将被导入的 python 文件的名称。

### SERVICE_TYPE

正在创建的服务类型。可用选项有：

 * MODEL
 * ROUTER
 * TRANSFORMER
 * COMBINER
 * OUTLIER_DETECTOR


### EXTRA_INDEX_URL

.. Warning::
   ``EXTRA_INDEX_URL`` 建议作为参数传递给 ``s2i`` 
   命令，而不是 ``.s2i/environment`` 作为避免在代码中
   检查凭据的做法添加。

用于从 私有/自托管 PyPi 库构建。

### PIP_TRUSTED_HOST

用于通过将其添加到 pip 可信主机来添加 私有/自托管 的不安全 PyPi 注册库。

```bash
s2i build \
   -e EXTRA_INDEX_URL=https://<pypi-user>:<pypi-auth>@mypypi.example.com/simple \
   -e PIP_TRUSTED_HOST=mypypi.example.com \
   <src-folder> \
   seldonio/seldon-core-s2i-python3:1.10.0-dev \
   <my-image-name>
```

### PAYLOAD_PASSTHROUGH

如果启用，Python 服务器将不会尝试解编码请求或者响应负载。

这意味着 `SeldonComponent` 模型的 `predict()` 方法将按原样接收有效负载，并负责对其进行解码。同样的，`predict()` 返回值必须是序列化的响应。

默认情况下，此选项将被禁用。

## 创建不同的服务类型

### MODEL

 * [最小化的模型源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/model-template-app)
 * [模型 notebooks 示例](../examples/notebooks.html)
 * [SKLearn Iris 分类器示例](../examples/iris)

### ROUTER
 * [Seldon Core 中路由定义](../analytics/routers.html)
 * [最小化的模型路由源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/router-template-app)

### TRANSFORMER

 * [最小化的模型转换源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/transformer-template-app)
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
    "endpoint": { "type": "REST" },
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


### 本地 Python 依赖
`from version 0.5`

要使用私有存储库安装 Python 依赖项，请使用以下构建命令：

```bash
s2i build -i <python-wheel-folder>:/whl <src-folder> seldonio/seldon-core-s2i-python3:1.10.0-dev <my-image-name>
```

此命令将在 中查找本地 `<python-wheel-folder>` 文件夹查找 Python wheels 并在搜索 PyPI 之前使用这些 wheels。

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

### 自定义请求标签
`from version 0.3`

要添加自定义请求标记数据，您可以添加一个可选方法 `tags`，该方法可以返回自定义元数据的字典，如下例所示：

```python
class MyModel(object):

    def predict(self, X, features_names):
        return X

    def tags(self):
        return {"mytag": 1}
```
