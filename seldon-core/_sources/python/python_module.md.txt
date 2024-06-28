# Seldon Core Python Package

Seldon Core 在 PyPI 上提供可用的模块 `seldon-core`。该模块可以很好的配合 Seldon Core 工作，并且是 Python S2I 封装的基础。模块提供：

 * `seldon-core-microservice` 在 Seldon Core 提供可执行的微服务组件服务。为 Seldon Core 的 Python Wrapper 所用。
 * `seldon_core.seldon_client` 类库。 调用 Seldon 核心服务（内部微服务或外部 API）的核心参考 API 模块。用于测试可执行文件以及在 Python 中创建自己的 Seldon Core 客户端。

## 安装

从 PyPI 安装：

```bash
$ pip install seldon-core
```

### Tensorflow 支持

Seldon Core 添加了可选支持，以发送一个 `TFTensor` 作为预测输入。
然而，大多数用户喜欢发送 `numpy` 数组，字符串，二级制或者 JSON 输入。
因此，为了避免 `tensorflow` 在不需要 `TFTensor` 支持的情况下的依赖安装，默认情况下不会安装它。

要包含可选的 `TFTensor` 支持，可参考下面的 `seldon-core` 安装：

```bash
$ pip install seldon-core[tensorflow]
```

### Google 云存储支持

作为存储训练模型选项的一部分，Seldon Core 添加了可选的支持以
从 GCS（Google 云存储）中获取它们。
我们知道用户通常只需要一个存储后端。
因此，为了避免 `seldon-core` 包膨胀，
我们默认不安装 GCS 依赖项。

要包括可选的 GCS 支持，您可以安装 `seldon-core` 为：

```bash
$ pip install seldon-core[gcs]
```

我们目前正在寻找单个替代
多云存储库 `seldon-core` 多云存储库的替代。
此讨论目前在 [issue #1028](https://github.com/SeldonIO/seldon-core/issues/1028) 开放。
欢迎反馈和建议！

### Azure Blob 存储支持

作为存储训练模型的选项的一部分，Seldon Core 添加了可选支持
以从 Azure Blob 存储中获取它们。
我们知道用户通常只需要一个存储后端。
因此，为避免 `seldon-core` 包膨胀，
默认情况下不会安装 Azure Blob 存储依赖项。

要包含可选的 Azure 支持，可通过如下方式安装 `seldon-core`：

```bash
$ pip install seldon-core[azure]
```

### 安装所有额外依赖

如果想安装 `seldon-core` 所有的额外依赖，你可以
这么做：

```bash
$ pip install seldon-core[all]
```

要记住这包含一些可能用不到的依赖。
所以，除非必要，我们建议绝大部分用户参
考[上述文档](#install)安装默认的 `seldon-core`。

## 故障排除

如果您在安装 `seldon-core` 后遇到问题，这里有一些
诊断问题的提示。

### ImportError: cannot import name 'BlockBlobService'

我们用来支持 Azure Blob 存储的库[发布了一个
更新](https://github.com/Azure/azure-storage-python/issues/640)，其中包含对以前版本的重大更改。
此更新会破坏 `seldon-core` 低于或等于 `0.5.0` 的版本，
但不应影响版本 `0.5.0.2` 及以上的用户。
如果您遇到此问题，
您应该会看到类似于
下面的堆栈跟踪：

```pytb
.../seldon_core/storage.py in <module>
     23 import re
     24 from urllib.parse import urlparse
---> 25 from azure.storage.blob import BlockBlobService
     26 from minio import Minio
     27 from seldon_core.imports_helper import _GCS_PRESENT

ImportError: cannot import name 'BlockBlobService'
```

推荐的解决方法是更新 `seldon-core` 到版本 `0.5.0.2` 或
更高版本。
或者，如果您无法升级到更新的版本，
以下方法也适用：

```bash
$ pip install azure-storage-blob==2.1.0 seldon-core
```

## 下一步

[创建 python 预估类](python_component.md)


