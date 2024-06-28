# 使用 Docker 为 Seldon Core 打包 R 模型

本导航，我们将说明使用 Docker 为 Seldon Core 开发好的 R 模型创建 docker 镜像。

## 第 1 步 - 创建源代码

您将需要一个提供 S3 类的 R 文件，该文件通过 `initialise_seldon` 函数为您的模型为您的组件提供适当的泛型，例如预测模型。您还需要声明代码所需的任何依赖项。

### R 运行时模型文件

您的源代码应包含一个 R 文件，该文件为您的模型定义了一个 S3 类。例如，查看我们的骨架 R 模型文件 `incubating/wrappers/s2i/R/test/model-template-app/MyModel.R`：

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

- 一个 `seldon_initialise` 函数通过构造函数 `new_mymodel` 为模型创建一个 S3 类。这将在启动时调用，您可以使用它来加载模型所需的任何参数。
- 为模型类创建一个 `predict` 函数。这将使用具有要预测的 `newdata` 字段来调用 `data.frame`。

ROUTERS 和 TRANSFORMERS 有类似的模板。

### 依赖

若在 docker 之外运行您的代码，您可以任何依赖控制软件填充 `install.R`。例如：

```R
install.packages('rpart')
```

这些相同的依赖项需要安装在 docker 镜像中，如下一节所述。

## 第 2 步 - 构建镜像

如何安装依赖取决于[你选择的基础镜像](https://www.r-bloggers.com/2019/01/docker-images-for-r-r-base-versus-r-apt/)，以及依赖的二进制版本是否可用。使用 `rocker/r-apt:bionic` 作为基础镜像，并安装二进制依赖，如果可能，	会生成更快、更小的构建。

docker file 示例可在 [seldon kubeflow 示例](https://github.com/kubeflow/example-seldon/blob/master/models/r_mnist/runtime/Dockerfile)查看：

```dockerfile
FROM rocker/r-apt:bionic

RUN apt-get update && \
    apt-get install -y -qq \
    	r-cran-plumber \
    	r-cran-jsonlite \
    	r-cran-optparse \
    	r-cran-stringr \
    	r-cran-urltools \
    	r-cran-caret \
    	r-cran-pls \
    	curl

ENV MODEL_NAME mnist.R
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

RUN mkdir microservice
COPY . /microservice
WORKDIR /microservice

RUN curl -OL https://raw.githubusercontent.com/SeldonIO/seldon-core/v0.5.0/incubating/wrappers/s2i/R/microservice.R > /microservice/microservice.R

EXPOSE 5000

CMD Rscript microservice.R --model $MODEL_NAME --api $API_TYPE --service $SERVICE_TYPE --persistence $PERSISTENCE
```

这里二进制版本的库安装在顶部。seldon 封装器所需依赖项 'plumber', 'jsonlite', 'optparse', 'stringr', 'urltools' 和 'caret' - 除了这些，你可以添加你自己的依赖项（这里的「pls」是示例，封装器不需要）。

然后设置环境变量，最后将作为参数传递给 CMD 中的 R 微服务。下面解释环境变量的含义。

创建一个目录并将本地源代码处理到该目录中，然后将其设置为工作目录。然后将 seldon 微服务封装器文件复制到此目录中。将模型封装为作为微服务运行。暴露命令将 5000 设置为服务的端口。

然后使用 `docker build . -t $ORG/$MODEL_NAME:$TAG` 命令从源代码创建 Docker 映像。可以使用一个简单的名称，但约定是使用 ORG/IMAGE:TAG 格式。

## 参考

### 环境变量

下面解释了构建器映像所需环境变量。您可以在 Dockerfile 中或作为`docker run` 的 `-e` 参数提供。

#### MODEL_NAME

包含模型的 R 文件的名称。

#### API_TYPE

要创建的 API 类型。目前只能是REST。

#### SERVICE_TYPE

创建的服务类型。可用选项有：

- MODEL
- ROUTER
- TRANSFORMER

#### PERSISTENCE

目前只能是 0。将来，将允许定期保存组件的状态。

### Creating different service types

#### MODEL

- [模型源代码的最小骨架](https://github.com/SeldonIO/seldon-core/tree/master/incubating/wrappers/s2i/R/test/model-template-app)
- [示例模型](../examples/notebooks.html)

#### ROUTER
- [路由器源代码的最小框架](https://github.com/seldonio/seldon-core/tree/master/incubating/wrappers/s2i/R/test/router-template-app)

#### TRANSFORMER

- [转换器源代码的最小骨架](https://github.com/seldonio/seldon-core/tree/master/incubating/wrappers/s2i/R/test/transformer-template-app)
