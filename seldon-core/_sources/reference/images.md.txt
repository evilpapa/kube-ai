# 最新 Seldon 镜像


## 核心镜像

| 描述 | 镜像 URL | 稳定版本 | 开发 |
|-------------|-----------|----------------|-------------|
| [Seldon Operator](../workflow/install.md) | [seldonio/seldon-core-operator](https://hub.docker.com/r/seldonio/seldon-core-operator/tags/) | 1.10.0 | 1.11.0-dev |
| [Seldon 服务编排 (Go)](../graph/svcorch.md)| [seldonio/seldon-core-executor](https://hub.docker.com/r/seldonio/executor/tags/) | 1.10.0 | 1.11.0-dev |

## 预包装服务


| 描述 | 镜像 URL | Version |
|-------------|-----------|---------|
| [MLFlow 服务 REST](../servers/mlflow.md) | [seldonio/mlflowserver_rest](https://hub.docker.com/r/seldonio/mlflowserver_rest/tags/) | 1.10.0 |
| [MLFlow 服务 GRPC](../servers/mlflow.md) | [seldonio/mlflowserver_grpc](https://hub.docker.com/r/seldonio/mlflowserver_grpc/tags/) | 1.10.0 |
| [SKLearn 服务 REST](../servers/sklearn.md) | [seldonio/sklearnserver_rest](https://hub.docker.com/r/seldonio/sklearnserver_rest/tags/) | 1.10.0 |
| [SKLearn 服务 GRPC](../servers/sklearn.md) | [seldonio/sklearnserver_grpc](https://hub.docker.com/r/seldonio/sklearnserver_grpc/tags/) | 1.10.0 |
| [XGBoost 服务 REST](../servers/xgboost.md) | [seldonio/xgboostserver_rest](https://hub.docker.com/r/seldonio/xgboostserver_rest/tags/) | 1.10.0 |
| [XGBoost 服务 GRPC](../servers/xgboost.md) | [seldonio/xgboostserver_grpc](https://hub.docker.com/r/seldonio/xgboostserver_grpc/tags/) | 1.10.0 |

## 语言封装

| 描述 | 镜像 URL | 稳定版本 | 开发 |
|-------------|-----------|----------------|-------------|
| [Seldon Python 3 (3.6) Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python3](https://hub.docker.com/r/seldonio/seldon-core-s2i-python3/tags/) | 1.10.0 | 1.11.0-dev |
| [Seldon Python 3.6 Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python36](https://hub.docker.com/r/seldonio/seldon-core-s2i-python36/tags/) | 1.10.0 | 1.11.0-dev |
| [Seldon Python 3.7 Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python37](https://hub.docker.com/r/seldonio/seldon-core-s2i-python37/tags/) | 1.10.0 | 1.11.0-dev |
| [Seldon Python 3.6 GPU Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python36-gpu](https://hub.docker.com/r/seldonio/seldon-core-s2i-python36-gpu/tags/) | 1.10.0 | 1.11.0-dev |
| [Seldon Python 3.7 GPU Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python37-gpu](https://hub.docker.com/r/seldonio/seldon-core-s2i-python37-gpu/tags/) | 1.10.0 | 1.11.0-dev |

## 服务端代理

| 描述 | 镜像 URL | 稳定版本 |
|-------------|-----------|----------------|
| [NVIDIA inference server proxy](integration_nvidia_link.rst) | [seldonio/nvidia-inference-server-proxy](https://hub.docker.com/r/seldonio/nvidia-inference-server-proxy/tags/) | 0.1 |
| [SageMaker proxy](https://github.com/SeldonIO/seldon-core/tree/master/integrations/sagemaker) | [seldonio/sagemaker-proxy](https://hub.docker.com/r/seldonio/sagemaker-proxy/tags/) | 0.1 |
| [Tensorflow Serving REST proxy](../servers/tensorflow.md) | [seldonio/tfserving-proxy_rest](https://hub.docker.com/r/seldonio/tfserving-proxy_rest/tags/) | 0.7 |
| [Tensorflow Serving GRPC proxy](../servers/tensorflow.md) | [seldonio/tfserving-proxy_grpc](https://hub.docker.com/r/seldonio/tfserving-proxy_grpc/tags/) | 0.7 |


## Python 模块

| 描述 | Python Version | Version |
|-------------|----------------|---------|
| [seldon-core](https://pypi.org/project/seldon-core/) | >3.4,<3.7 | 1.10.0 |
| [seldon-core](https://pypi.org/project/seldon-core/) | 2,>=3,<3.7 | 0.2.6 (废弃) |


## 孵化中

### 语言封装

| 描述 | 镜像 URL | 稳定版本 | 开发 |
|-------------|-----------|----------------|-------------|
| [Seldon Python ONNX Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python3-ngraph-onnx](https://hub.docker.com/r/seldonio/seldon-core-s2i-python3-ngraph-onnx/tags/) | 0.3  |   |
| [Seldon Java Build Wrapper for S2I](../java/README.md) | [seldonio/seldon-core-s2i-java-build](https://hub.docker.com/r/seldonio/seldon-core-s2i-java-build/tags/) | 0.1 | |
| [Seldon Java Runtime Wrapper for S2I](../java/README.md) | [seldonio/seldon-core-s2i-java-runtime](https://hub.docker.com/r/seldonio/seldon-core-s2i-java-runtime/tags/) | 0.1 | |
| [Seldon R Wrapper for S2I](../R/README.md) | [seldonio/seldon-core-s2i-r](https://hub.docker.com/r/seldonio/seldon-core-s2i-r/tags/) | 0.2 | |
| [Seldon NodeJS Wrapper for S2I](../nodejs/README.md) | [seldonio/seldon-core-s2i-nodejs](https://hub.docker.com/r/seldonio/seldon-core-s2i-nodejs/tags/) | 0.1 | 0.2-SNAPSHOT |


### Java 包

可在 Maven 仓库中找到这些包。

| 描述 | 包 | 版本 |
|-------------|---------|---------|
| [Seldon Core Wrapper](https://github.com/SeldonIO/seldon-java-wrapper) | seldon-core-wrapper | 0.1.5 |
| [Seldon Core JPMML](https://github.com/SeldonIO/JPMML-utils) | seldon-core-jpmml | 0.0.1 |



## 废弃

### 语言封装

| 描述 | 镜像 URL | 稳定版本 | 开发 |
|-------------|-----------|----------------|-------------|
| [Seldon Python 2 Wrapper for S2I](../python/python_wrapping_s2i.md) | [seldonio/seldon-core-s2i-python2](https://hub.docker.com/r/seldonio/seldon-core-s2i-python2/tags/) | 0.5.1 | 废弃 |
