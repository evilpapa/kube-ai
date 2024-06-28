# 使用 s2i 为 Seldon Core 打包 NodeJS 模型

本章中，我们说明了使用[source-to-image app s2i](https://github.com/openshift/source-to-image)构建的镜像在 node 引擎运行封装的 JS 模型的步骤。

如果您不熟悉 s2i，您可以阅读 [使用 s2i 的一般说明](../wrappers/s2i.md)，然后按照以下步骤。

## 步骤 1 - Install s2i

[下载安装 s2i](https://github.com/openshift/source-to-image#installation)

 * 使用 s2i 准备工作
   * Docker
   * Git（如果使用远程 git 仓库）

所有工作就绪，可执行以下

```bash
s2i usage seldonio/seldon-core-s2i-nodejs:0.1
```

## 步骤 2 - 创建源代码

要使用 s2i 构建封装 NodeJS 模型的镜像，你需要：

- 一个 JS 文件，它为您的模型提供 ES5 函数对象或 ES6 类，并且具有适合您的组件的泛型，即模型的 `init` 和 `a `predict` 组件
- 包含所有依赖和模型元数据的 package.json
- .s2i/environment - s2i 构建器用于正确封装模型的模型定义

我们将详细介绍每个步骤：

### NodeJS 运行时模型文件

的源代码应该为您的模型提供 ES5 函数对象或 ES6 类。例如，查看我们的脚手架 JS 结构：

```js
let MyModel = function() {};

MyModel.prototype.init = async function() {
  // A mandatory init method for the class to load run-time dependencies
  this.model = "My Awesome model";
};

MyModel.prototype.predict = function(newdata, feature_names) {
  //A mandatory predict function for the model predictions
  console.log("Predicting ...");
  return newdata;
};

module.exports = MyModel;
```

该模型也可以是 ES6 类，如下所示

```js
class MyModel {
  async init() {
    // A mandatory init method for the class to load run-time dependencies
    this.model = "My Awesome ES6 model";
  }
  predict(newdata, feature_names) {
    //A mandatory predict function for the model predictions
    console.log("ES6 Predicting ...");
    return newdata;
  }
}
module.exports = MyModel;
```

- 一个用于模型对象的 `init` 方法。这将在启动时调用，您可以使用它来加载模型所需的任何参数。此函数也可能是异步的，例如，如果它必须从远程位置加载模型权重。
- 一个创建的模型类的通用 `predict` 方法。这将用于预测包含 `newdata` 字段的数据对象。

### package.json

`package.json` 使用命令 `npm init` 填充您的代码所需的任何软件依赖项，并将您的依赖项保存到文件中。

### .s2i/environment

定义我们的 node JS 构建器映像所需的核心参数来封装您的模型。一个例子是：

```bash
MODEL_NAME=MyModel.js
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0
```

构建映像时，也可以在命令行上提供或覆盖这些值。

## 步骤 3 - Build your image

使用 `s2i build` 从你的源码构建 Docker 镜像。如果您的源代码在公共 git 存储库中，您将需要在机器上安装 Docker 和可选的 git。

使用 s2i，您可以直接从 git 存储库或本地源文件夹构建。有关更多详细信息，请参阅 [s2i 文档](https://github.com/openshift/source-to-image/blob/master/docs/cli.md#s2i-build)。一般格式为：

```bash
s2i build <git-repo> seldonio/seldon-core-s2i-nodejs:0.1 <my-image-name>
s2i build <src-folder> seldonio/seldon-core-s2i-nodejs:0.1 <my-image-name>
```

在 seldon-core 中使用测试模板模型的示例调用：

```bash
s2i build https://github.com/seldonio/seldon-core.git --context-dir=incubating/wrappers/s2i/nodejs/test/model-template-app seldonio/seldon-core-s2i-nodejs:0.1 seldon-core-template-model
```

上面的 s2i 构建调用：

- 使用 GitHub 存储库: https://github.com/seldonio/seldon-core.git 和该存储库中的目录 `incubating/wrappers/s2i/nodejs/test/model-template-app`
- 使用构建器图像 `seldonio/seldon-core-s2i-nodejs`
- 使用 docker 图像 `seldon-core-template-model`

对于从本地源文件夹构建，我们克隆 seldon-core 存储库的示例：

```bash
git clone https://github.com/seldonio/seldon-core.git
cd seldon-core
s2i build incubating/wrappers/s2i/nodejs/test/model-template-app seldonio/seldon-core-s2i-nodejs:0.1 seldon-core-template-model
```

如需更多帮助，请参阅：

```bash
s2i usage seldonio/seldon-core-s2i-nodejs:0.1
s2i build --help
```

## 参考

### 环境变量

下面解释了构建器映像理解的必需环境变量。您可以在 `.s2i/environment` 文件中或在命令 `s2i build` 中提供。

### MODEL_NAME

包含模型的 JS 文件的名称。

### API_TYPE

要创建的 API 类型。可以是 REST 或 GRPC。

### SERVICE_TYPE

正在创建的服务类型。可用选项有：

- MODEL
- TRANSFORMER

### PERSISTENCE

目前只能由0。

## 创建不同的服务类型

### MODEL

- [模型示例](../examples/notebooks.html)
