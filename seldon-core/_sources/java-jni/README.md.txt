# 使用 s2i 为 Seldon Core 打包 Java 模型

本指南中说明了使用 docker 镜像工具 [source-to-image 程序 s2i](https://github.com/openshift/source-to-image) 
和 JNI 封装为发布 Seldon Core 封装 Java 模型所需的操作。

JNI 封装利用 [Python 预估服务](../python) 通过 JNI 和 JAVA 模型通信。
Python 封装通常比[原生 Java](../java/README.md)的更新更快，
因此能更快的体验 Seldon Core 特性。

如果你对 s2i 不熟悉，请参考 [s2i 使用说明](../wrappers/s2i.md)并跟随下面的步骤。

## 步骤 1 - 安装 s2i

 [下载安装 s2i](https://github.com/openshift/source-to-image#installation)

 * 使用 s2i 准备工作
   * Docker
   * Git（如果使用远程 git 仓库）

所有工作就绪，可执行以下

```bash
s2i usage seldonio/s2i-java-jni-build:0.5.1
```

## 步骤 2 - 创建源代码

要使用 s2i 构建封装 Java 模型的镜像，你需要：

 * 依赖 `io.seldon.wrapper` 库(>= `0.3.0`)的 Maven 项目
 * 创建为你组件所实现的 `io.seldon.wrapper.api.SeldonPredictionService` 的类
 * .s2i/environment - 构建器用于正确封装模型的模型定义

我们将详细介绍以下每个步骤：

###  Maven 项目
创建 Spring Boot Maven 项目并包含以下依赖：

```XML
<dependency>
	<groupId>io.seldon.wrapper</groupId>
	<artifactId>seldon-core-wrapper</artifactId>
	<version>0.3.0</version>
</dependency>
```

完成的示例可查看 `incubating/wrappers/s2i/java-jni/test/model-template-app/pom.xml`。

### 预估类
为了处理模型或者其他组件的请求，
你需要在 `io.seldon.wrapper.api.SeldonPredictionService` 实现以下一个或多个方法，
特别的：

```java
default public SeldonMessage predict(SeldonMessage request);
default public SeldonMessage route(SeldonMessage request);
default public SeldonMessage sendFeedback(Feedback request);
default public SeldonMessage transformInput(SeldonMessage request);
default public SeldonMessage transformOutput(SeldonMessage request);
default public SeldonMessage aggregate(SeldonMessageList request);
```

一个完整的 H2O 实现示例 
`examples/models/h2o_mojo/src/main/java/io/seldon/example/h2o/model`
如下：

```java
public class H2OModelHandler implements SeldonPredictionService {
	private static Logger logger = LoggerFactory.getLogger(H2OModelHandler.class.getName());
	EasyPredictModelWrapper model;

	public H2OModelHandler() throws IOException {
		MojoReaderBackend reader =
                MojoReaderBackendFactory.createReaderBackend(
                  getClass().getClassLoader().getResourceAsStream(
                     "model.zip"),
                      MojoReaderBackendFactory.CachingStrategy.MEMORY);
		MojoModel modelMojo = ModelMojoReader.readFrom(reader);
		model = new EasyPredictModelWrapper(modelMojo);
		logger.info("Loaded model");
	}

	@Override
	public SeldonMessage predict(SeldonMessage payload) {
		List<RowData> rows = H2OUtils.convertSeldonMessage(payload.getData());
		List<AbstractPrediction> predictions = new ArrayList<>();
		for(RowData row : rows)
		{
			try
			{
				BinomialModelPrediction p = model.predictBinomial(row);
				predictions.add(p);
			} catch (PredictException e) {
				logger.info("Error in prediction ",e);
			}
		}
        DefaultData res = H2OUtils.convertH2OPrediction(predictions, payload.getData());

		return SeldonMessage.newBuilder().setData(res).build();
	}

}

```

以上代码：

  * 在启动时加载本地文件夹模型资源
  * 使用提供的工具类转换 proto buffer 消息为 H2O RowData
  * 运行 BionomialModel 预估，转换并返回 `SeldonMessage` 结果

#### H2O 工具类

我们在 seldon-core-wrapper 提供 H2O 工具类 `io.seldon.wrapper.utils.H2OUtils` 用于
转换 seldon-core proto buffer 类型消息。

#### DL4J 工具类

我们在 seldon-core-wrapper 提供 DL4J 工具类 `io.seldon.wrapper.utils.DL4JUtils` 用于
转换 seldon-core proto buffer 类型消息。

### .s2i/environment

定义构建 JAVA S2I 模型封装镜像的核心参数。
如：

```bash
SERVICE_TYPE=MODEL
JAVA_IMPORT_PATH=io.seldon.example.model.ExampleModelHandler
```

他们的值可以在命令行构建镜像参数中被覆盖。

## 步骤 3 - 构建你自己的镜像

从源码中使用 `s2i build` 创建 Docker 镜像。
你需要将 Docker 安装在本机，
可选的如果代码在公有仓库需要安装 git。

使用 s2i 可以从本地文件夹或者 git 仓库直接构建。
参考 [s2i 文档](https://github.com/openshift/source-to-image/blob/master/docs/cli.md#s2i-build)获取更多信息。
通常格式为：

```bash
s2i build \
  <git-repo | src-folder> \
  seldonio/s2i-java-jni-build:0.5.1 \
  --runtime-image seldonio/s2i-java-jni-runtime:0.5.1 \
  <my-image-name> 
```

使用 `seldon-core` 内部的测试模板模型调用的示例：

```bash
s2i build \
  https://github.com/seldonio/seldon-core.git \
  --context-dir=incubating/wrappers/s2i/java-jni/test/model-template-app \
  seldonio/s2i-java-jni-build:0.5.1 \
  --runtime-image seldonio/s2i-java-jni-runtime:0.5.1 \
  jni-model-template:0.1.0
```

以上 s2i 生成调用：

 * 使用 `seldon-core` [GitHub 仓库](https://github.com/seldonio/seldon-core)
   和仓库中的 `incubating/wrappers/s2i/java-jni/test/model-template-app`
   路径
 * 使用构建镜像 `seldonio/s2i-java-jni-build`
 * 使用运行时镜像 `seldonio/s2i-java-jni-runtime`
 * 创建 docker 镜 `jni-template-model`


从本地文件夹构建镜像，可在本 seldon-core 仓库中查看示例

```bash
git clone https://github.com/seldonio/seldon-core.git
cd seldon-core
s2i build \
  incubating/wrappers/s2i/java-jni/test/model-template-app \
  seldonio/s2i-java-jni-build:0.5.1 \
  --runtime-image seldonio/s2i-java-jni-runtime:0.5.1 \
  jni-model-template:0.1.0
```

更多请查看：

```bash
s2i usage seldonio/s2i-java-jni-build:0.5.1
s2i build --help
```

## 参考

## 环境变量
镜像构建下所需的环境变量如下。
可在 
`.s2i/environment` 文件或者 `s2i build` 命令行
提供。

### JAVA_IMPORT_PATH

导入 Java 模型实现，
例如，在上面的例子中，这将是
`io.seldon.example.model.ExampleModelHandler`。


### SERVICE_TYPE

创建的 service 类型，
可用值为：

 * MODEL
 * ROUTER
 * TRANSFORMER
 * COMBINER


## 创建不同的 service types

### MODEL

 * [一个最小的模型源码脚手架](https://github.com/SeldonIO/seldon-core/tree/master/incubating/wrappers/s2i/java/test/model-template-app)
 * [H2O MOJO 示例](../examples/h2o_mojo.html)
