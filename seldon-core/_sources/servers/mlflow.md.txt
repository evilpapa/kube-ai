# MLflow 服务器

如果您有一个 MLflow 模型你可以使用 Seldon 预封装服务进行一个（或多个）
版本服务发布。
在初始化过程中，内建的
可复用服务器会自动创建 [Conda
环境](https://www.mlflow.org/docs/latest/projects.html#project-environments)
并根据指定的 `conda.yaml` 文件来创建。

## 预处理

要使用内置的 MLflow 服务器，需要满足以下先决条件：

- 你的 [MLmodel artifact
  文件夹](https://www.mlflow.org/docs/latest/models.html) 可
  远程访问（如：`gs://seldon-models/mlflow/elasticnet_wine_1.8.0`）。
- 模型需要与 [python_function
  flavour](https://www.mlflow.org/docs/latest/models.html#python-function-python-function)兼容。
- `MLproject` 环境需要指定使用 Conda。

## Conda 环境创建

MLflow内置服务器初始化期间会根据
`MLmodel`的`conda.yaml`配置文件创建 Conda 环境。
注意此方法可能会减慢你的 Kubernetes `SeldonDeployment` 
启动时间。

某些情况下，可以考虑 [自定义创建可复用
服务器](./custom.md)。
比如，当 Conda 环境稳定时可使用
一组固定的依赖库创建自己的镜像。
此镜像可作为可复用预加载环境
用于不同的模型版本。

## 示例

保存了 Iris 预估模型的示例如下

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: mlflow
spec:
  name: wines
  predictors:
    - graph:
        children: []
        implementation: MLFLOW_SERVER
        modelUri: gs://seldon-models/mlflow/elasticnet_wine_1.8.0
        name: classifier
      name: default
      replicas: 1
```

## MLFlow xtype

默认情况下服务器会通过`numpy.ndarray` 调用模型的预估函数。如果希望通过 `pandas.DataFrame` 替代，你可通过 `xtype` 参数赋值 `DataFrame` 来使用，如：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: mlflow
spec:
  name: wines
  predictors:
    - graph:
        children: []
        implementation: MLFLOW_SERVER
        modelUri: gs://seldon-models/mlflow/elasticnet_wine_1.8.0
        name: classifier
        parameters:
        - name: xtype
          type: STRING
          value: DataFrame
      name: default
      replicas: 1
```

也可尝试 [可工作的
notebook](../examples/server_examples.html#Serve-MLflow-Elasticnet-Wines-Model)
或者查看 [talk at the Spark + AI Summit
2019](https://www.youtube.com/watch?v=D6eSfd9w9eA)。

## V2 KFServing 协议 [孵化中]

.. Warning:: 
  V2 KFServing 协议支持被考虑在孵化
  特性中。
  这意味着 Seldon Core 的某些特性仍未得到支持（比如：
  tracing, graphs等）。

MLFlow 服务也可用于暴露兼容 [V2
KFServing 协议](../graph/protocols.md#v2-kfserving-protocol)的 API。
注意，某些情况下，这会使用到 [Seldon
MLServer](https://github.com/SeldonIO/MLServer) 运行时。

### 创建使用 `mlflow` 的模型并使用 `seldon-core` 发布
作为示例，我们将使用 elasticnet wine 模型。

- 创建 `conda` 环境

```bash
$ conda -y create -n python3.8-mlflow-example python=3.8
$ conda activate python3.8-mlflow-example
```

- 安装 `mlflow`

```bash
$ pip install mlflow
```

- 训练 elasticnet wine 示例

```bash
$ git clone https://github.com/mlflow/mlflow
$ cd mlflow/examples
$ python sklearn_elasticnet_wine/train.py
```
脚本结束后，会有一个模型持久化存储在 `mlruns/0/<uuid>/artifacts/model`。这可以
通过 ui 获取（`mlflow ui`）

- 使用 [conda-pack](https://conda.github.io/conda-pack/) 安装部署和打包conda 所需的附加软件包

```bash
$ pip install conda-pack
$ pip install mlserver
$ pip install mlserver-mlflow
$ cd mlflow/examples/mlruns/0/<uuid>/artifacts/model
$ conda pack -o environment.tar.gz -f
```
这会将当前的 conda 环境打包到 `environment.tar.gz`，这也是 `mlserver` 创建服务模型时同训练所需要的一样的环境。

- 将模型目录复制到可由 seldon-core 访问的 Google Storage 存储桶

```bash
$ gsutil cp -r ../model gs://seldon-models/test/elasticnet_wine_<uuid>
```

- 发布模型到 seldon-core
为了支持 V2 KFServing 协议，需要
指定 `SeldonDeployment` 配置项  `protocol` 为 `kfserving`。
比如，

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: mlflow
spec:
  protocol: kfserving  # Activate the v2 protocol
  name: wines
  predictors:
    - graph:
        children: []
        implementation: MLFLOW_SERVER
        modelUri: gs://seldon-models/test/elasticnet_wine_<uuid>
        name: classifier
      name: default
      replicas: 1
```

- 使用 REST 从部署的模型中获取预测

```python
import json

import requests

inference_request = {
    "parameters": {
        "content_type": "pd"
    },
    "inputs": [
        {
          "name": "fixed acidity",
          "shape": [1],
          "datatype": "FP32",
          "data": [7.4],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "volatile acidity",
          "shape": [1],
          "datatype": "FP32",
          "data": [0.7000],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "citric acidity",
          "shape": [1],
          "datatype": "FP32",
          "data": [0],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "residual sugar",
          "shape": [1],
          "datatype": "FP32",
          "data": [1.9],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "chlorides",
          "shape": [1],
          "datatype": "FP32",
          "data": [0.076],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "free sulfur dioxide",
          "shape": [1],
          "datatype": "FP32",
          "data": [11],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "total sulfur dioxide",
          "shape": [1],
          "datatype": "FP32",
          "data": [34],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "density",
          "shape": [1],
          "datatype": "FP32",
          "data": [0.9978],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "pH",
          "shape": [1],
          "datatype": "FP32",
          "data": [3.51],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "sulphates",
          "shape": [1],
          "datatype": "FP32",
          "data": [0.56],
          "parameters": {
              "content_type": "np"
          }
        },
        {
          "name": "alcohol",
          "shape": [1],
          "datatype": "FP32",
          "data": [9.4],
          "parameters": {
              "content_type": "np"
          }
        },
    ]
}

endpoint = "http://localhost:8003/seldon/seldon/mlflow/v2/models/infer"
response = requests.post(endpoint, json=inference_request)

print(json.dumps(response.json(), indent=2))
```

### 注意事项
- conda 环境中安装的 `mlserver` 版本需与 `seldon-core` 匹配。我们正在开发工具以使其更加无缝。
- 检查使用 [`conda-pack`](https://conda.github.io/conda-pack/#caveats) 的注意事项