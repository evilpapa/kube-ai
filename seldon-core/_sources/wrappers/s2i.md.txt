# 源码转镜像 (s2i)

[Source to image](https://github.com/openshift/source-to-image) 是一个 supported 支持的工具来为源码创建 docker 镜像。 我们提供映像构建器，让您可以轻松地封装数据模型，以便它们可以由 seldon-core 进行管理。

一般的工作流程是：

 1. [下载安装 s2i](https://github.com/openshift/source-to-image#installation)

 1. 选择最适合您的代码的构建器映像并获取使用说明，例如：

    ```bash
    s2i usage seldonio/seldon-core-s2i-python3
    ```

 1. 以镜像构建器可接受的形式创建源代码存储库，并从中构建您的 docker 容器。 下面我们展示了一个使用我们的 seldon-core git repo 的示例，其中包含一些 Python 模型的模板示例。

    ```bash
    s2i build https://github.com/seldonio/seldon-core.git \
        --context-dir=wrappers/s2i/python/test/model-template-app seldonio/seldon-core-s2i-python3 \
        seldon-core-template-model
    ```

目前我们有 s2i builder 镜像

 * [Python (Python3)](../python/README.md) : 为 Tensorflow，Keras，PyTorch 或 sklearn 模型使用此镜像。
 * [R](../R/README.md)
 * [Java](../java/README.md)
 * [NodeJS](../nodejs/README.md)

