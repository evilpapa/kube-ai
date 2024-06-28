# 使用 s2i 为 Seldon Core 打包 R 模型（孵化）

本章中，我们说名使用 [source-to-image app s2i](https://github.com/openshift/source-to-image) 为 R 模型构建使用 Seldon Core 部署的 docker 镜像，如果想要使用原生 Docker，查看[Docker 说明](r_wrapping_docker.md)。

如果您不熟悉 s2i，您可以阅读 [使用 s2i 的一般说明](../wrappers/s2i.md)，然后按照以下步骤。

## 步骤 1 - 安装 s2i

[下载安装 s2i](https://github.com/openshift/source-to-image#installation)

- 使用 s2i 准备工作
  - Docker
  - Git（如果使用远程 git 仓库）

所有工作就绪，可执行以下

```bash
s2i usage seldonio/seldon-core-s2i-r:0.1
```

## 步骤 2 - 创建源代码

要使用 s2i 构建封装 R 模型的镜像，你需要：

- 通过 `initialise_seldon` 方法提供一个 S3 类的 R 文件的模型，并为您的组件提供适当的泛型，例如模型预测。
- 运行的可选 install.R 以安装所需的任何库
- .s2i/environment - s2i 正确构建模型的配置

我们将详细介绍每个步骤：

### R 运行时模型文件

源代码需要包含定义了 S3 类的模型的 R 文件。比如，请查看我们的 skeleton R 模型文件在 `incubating/wrappers/s2i/R/test/model-template-app/MyModel.R`：

```R
library(methods)

predict.mymodel <- function(mymodel,newdata=list()) {
  write("MyModel predict called", stdout())
  newdata
}


new_mymodel <- function() {
  structure(list(), class = "mymodel")
}


initialise_seldon <- function(params) {
  new_mymodel()
}
```

- `seldon_initialise` 函数 通过 `new_mymodel` 构造器创建模型 S3 类。这将在启动时调用，您可以使用它来加载模型所需的任何参数。
- 通用 `predict` 函数是为我们的模型类创建。将使用 `newdata` 字段的 `data.frame` 来进行预估。

ROUTERS 和 TRANSFORMERS 有类似的模板。

### install.R

通过任意软件依赖放置 `install.R` 到你的代码：

```R
install.packages('rpart')
```

### .s2i/environment

定义我们的 R 构建器映像所需的核心参数来封装您的模型。一个例子是：

```bash
MODEL_NAME=MyModel.R
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0
```

构建映像时，也可以在命令行上提供或覆盖这些值。

## 步骤 3 - 构建您的映像

使用 `s2i build` 从代码构建 Docker 镜像。你需要本地安装 Docker，可选的如果时从公共 git 仓库构建需要安装 git。

使用 s2i 直接从远程仓库或者本地文件夹构建。查看 [s2i 文档](https://github.com/openshift/source-to-image/blob/master/docs/cli.md#s2i-build)获取更多信息，一般格式为：

```bash
s2i build <git-repo> seldonio/seldon-core-s2i-r:0.1 <my-image-name>
s2i build <src-folder> seldonio/seldon-core-s2i-r:0.1 <my-image-name>
```

在 seldon-core 中使用测试模板模型的示例调用：

```bash
s2i build https://github.com/seldonio/seldon-core --context-dir=incubating/wrappers/s2i/R/test/model-template-app seldonio/seldon-core-s2i-r:0.1 seldon-core-template-model
```

上面的 s2i 构建调用：

- 使用 GitHub repo: https://github.com/seldonio/seldon-core 及 repo 中文件夹 `incubating/wrappers/s2i/R/test/model-template-app`。
- 使用构建镜像 `seldonio/seldon-core-s2i-r`
- 创建 docker 镜像 `seldon-core-template-model`

对于从本地源文件夹构建，我们克隆 seldon-core 存储库的示例：

```bash
git clone https://github.com/seldonio/seldon-core
cd seldon-core
s2i build incubating/wrappers/s2i/R/test/model-template-app seldonio/seldon-core-s2i-r:0.1 seldon-core-template-model
```

如需更多帮助，请参阅：

```bash
s2i usage seldonio/seldon-core-s2i-r:0.1
s2i build --help
```

## 参考

### 环境变量

面解释了构建器映像理解的必需环境变量。您可以在 `.s2i/environment` 文件中或在命令行 `s2i build` 中查看它们。

#### MODEL_NAME

包含模型的 R 文件的名称。

#### API_TYPE

要创建的 API 类型。目前只能是REST。

#### SERVICE_TYPE

正在创建的服务类型。可用选项有：

- MODEL
- ROUTER
- TRANSFORMER

#### PERSISTENCE

目前只能由0。将来，将允许定期保存组件的状态。

### 创建不同的服务类型

#### MODEL

- [模型源代码的最小脚手架](https://github.com/SeldonIO/seldon-core/tree/master/incubating/wrappers/s2i/R/test/model-template-app)
- [models 示例](../examples/notebooks.html)

#### ROUTER
- [路由器源代码的最小框架](https://github.com/SeldonIO/seldon-core/tree/master/incubating/wrappers/s2i/R/test/router-template-app)

#### TRANSFORMER

- [transformer 源代码的最小骨架](https://github.com/SeldonIO/seldon-core/tree/master/incubating/wrappers/s2i/R/test/transformer-template-app)
