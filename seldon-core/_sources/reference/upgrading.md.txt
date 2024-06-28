 # 升级 Seldon Core

此页面提供了如何从以前的版本升级的说明。如果预期运行中的模型在不中断的情况下升级（即如果运行版本为 0.4.2，则首先必须升级到 0.5.2，然后升级到 1.1 等），则必须按顺序运行上述每个模型。

如果您运行我们的 Openshift 0.4.2 认证 operator，并希望升级到我们的 1.1 认证 operator，您还需要按照「从以前的版本升级到 0.5.2」部分中的「升级过程」步骤。


确保您已阅读 [CHANGELOG](./changelog.html) 来查看每个版本的特性改动和 bug 修复。

## 升级到 1.10

### 服务器更新

SKLearn 服务器升级到 sklearn 0.24.2

XGBoost 服务器升级到 xgboost 1.4.2

### Alibi 服务升级

Alibi 升级到 to 0.6.0

Alibi 服务器 python 升级到 3.7.10


## 升级到 1.8

### Rclone Storage Initailizer

在 1.8 版本 rclone-based [storage initalizer](https://github.com/SeldonIO/seldon-core/tree/master/components/rclone-storage-initializer) 变为默认使用

controller 通过 helm 值设置相关镜像：

```
storageInitializer:
  image: seldonio/rclone-storage-initializer:1.9.1
```



可通过设置图定义 `storageInitializerImage` 变量值来为每个发布的[Prepackaged Model Servers](../servers/overview.xhtml) 自定义存储加载器。

此过度需要为打包模型服务器 **创建新的 secrets** 来兼容 rclone 配置格式，[参见](../servers/overview.xhtml#handling-credentials)。

如果您现在不想配置这些 secrets 并想保留之前的存储加载实例，可设置 helm value 如下：

```
storageInitializer:
  image: gcr.io/kfserving/storage-initializer:v0.4.0
```


请参与[此处](../servers/kfserving-storage-initializer.xhtml)获取更多信息。

### 请求日志

Seldon Core 1.9 我们将[seldon-request-logger](https://github.com/SeldonIO/seldon-core/tree/master/components/seldon-request-logger) 移动到独立的仓库。

### 传统 Java Engine 编排

Seldon Core 1.9 最终废弃 Java Engine，并移除了相关仓库代码。

## 升级到 1.7

### Python 以来升级

各种 CVEs 通过 #2970 解决，其中包括几个包升级，这些升级可能会影响安装可能不兼容的包的应用程序。这也包括安装 pip=20.2，此版本的 pip 仍使用较旧的解析器。

## 升级到 1.6

### Webhook 移除

作为 1.6.0 版本的一部分，我们删除了 Seldon Core Mutating Webhook。这不会导致任何明显的更改，但建议您在升级到版本 1.6.0 后手动删除 webhook

## 升级到 1.5

### REST 和 gRPC

为了实现 REST 和 gRPC 的高级处理能力来发布 python 模型和镜像，需要升级到 1.5 python wrapper。如果不升级则只能使用原始封装的协议支持。

果您不打算升级部署，您可以使用并扩展[向后兼容 notebook](../examples/backwards_compatibility.html) 来检查部署是否有效。

## 升级到 1.3

### 重大变化

默认 sklearn 服务器使用的 sklearn 版本为 0.23.2。要使用不同的版本，您需要按照 [sklearn 服务器文档](../servers/sklearn.html)描述的步骤进行。

## 升级到 1.2.1

*\[NOTE\]* 1.2.0 有一个问题，所有 Seldon 部署都被标记为「NotReady」，因为有一个 [volumeName 更新 bug](https://github.com/SeldonIO/seldon-core/issues/2017) 造成。可通过 1.2.0 volume patch [示例概述](../examples/patch_1_2.html)来解决。建议直接升级到 1.2.1 版本。

作为此升级的一部分，所有 seldon-managed 的 Pod 都将进行滚动更新。

### 新特性 / 重大变化

- helm 值 `createResources` 更名为 `managerCreateResources`。
- 允许管理员创建 CRD。如果 `managerCreateResources` 为 true 时，针对 `create` CRDs 之前版本的 RBAC 会添加额外的 RBAC，它只能被获取和列出。
- 如果更新 analytics helm chart 需[先执行](https://github.com/SeldonIO/seldon-core/pull/1917)命令 `kubectl delete deployment -n seldon-system -l app=grafana`
- 所有预先打包的模型服务器现在都使用 RedHat UBI 映像创建。这样做的一个后果是，它们都将按照最佳实践以非 root 身份运行。

### 请求日志

seldon-core-operator helm chart 的 values.yaml 已更改。该字段 `defaultRequestLoggerEndpointPrefix` 被替换为：

```
requestLogger:
  defaultEndpoint: 'http://default-broker'
```

This default value will find a broker in the model’s namespace. To point to another namespace it would be `default-broker.anothernamespace`.

## 从老版本升级到 1.1

当我们迁移到 1.x+ 时，需要考虑几个重大变化。这些概述如下

### 新功能/重大变化

#### 部署命名和滚动更新

Seldon Core 创建的部署已更改为遵循固定方案。现在将是：

```
<seldondeployment name>-<predictor name>-<podspec idx>-<container names>
```

示例

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: rest-seldon
spec:
  name: restseldon
  protocol: seldon
  transport: rest
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier
    graph:
      name: classifier
      type: MODEL
    name: model
    replicas: 1
```

对于上述资源，将使用名称创建一个部署：

```
rest-seldon-model-0-classifier
```

这将改变滚动更新的完成方式。现在，如果容器的名称未更改，则对上述第一个 PodSpec 的任何更改都将按预期通过滚动更新进行更新。但是，如果您将「classifier」更改为「classifier2」，则会创建一个新部署，该部署将在运行时替换旧部署。

#### 新的服务编排

从 1.1 版开始，Seldon Core 带有一个用 Go 编写的新服务编排器，它取代了以前的 Java 引擎。存在一些重大变化：

- 不再添加 Seldon 协议中的元数据字段。任何自定义元数据都需要通过图中的各个组件添加并公开给 Prometheus 指标
- 图中的所有组件都必须是 REST 或 gRPC，并且只有给定的协议对外公开。
- Prometheus 中的指标名称已更改为包含 `executor` 名称而不是 `engine`：查看[分析文档](../analytics/analytics.html)

新的服务编排器具有多项优势，包括能够处理 Tensorflow REST 和 gRPC 协议以及对 REST 和 gRPC 的完整指标和跟踪支持。

对于那些希望使用已弃用的 Java 引擎服务编排器的人，请参阅[服务编排器](../graph/svcorch.xhtml)文档以了解详细信息。

#### Ambassador 重试

重试次数已从之前的硬编码值 3 中删除。现在可通过 [Ambassador 注解](../ingress/ambassador.html)设置。

### Python 封装标签更新

Python Wrapper 使用格式为 0.1 … 0.18 的命名约定。在此版本中，我们重命名了 Python Wrapper 标签的版本以匹配与 Executor、Operator 等相同的约定。这意味着此版本的 Python Wrapper 标签是 1.1，快照将是 1.1.1-SNAPSHOT。

### 过期的快照

每当一个新的 PR 被合并到 master 时，我们都会设置我们的 CI 来构建一个「SNAPSHOT」版本，该版本将包含该特定开发/主分支代码的 Docker 镜像。

以前，我们总是让 SNAPSHOT 标签被最新的标签覆盖。这并不能让我们知道在使用 master 时有人可能会尝试什么版本，所以我们想引入一种方法来为每个进入 master 的镜像获取唯一标签。

现在，每次将 PR 登陆到 master 时，都会创建一个新的「过时」SNAPSHOT 版本，该版本会推送带有标签 `"<next-version>-SNAPSHOT_<timestamp>"` 的镜像，它包含了各自的 helm charts，请依照特定版本（如版本中的 `version.txt` 所概述）进行安装。

可按照[安装页](../workflow/install.xhtml)说明来安装 snapshot 版本。

### 封装器兼容性表

为了验证 Seldon Core v1.0 和 v.1.1 是否与旧的 s2i 封装器版本兼容，我们使用单节点模型进行了一个简单的测试。该模型已通过 REST 和 GRPC API 以及新的编排器和已弃用的 Java 引擎（仅 v1.0 与 Java 引擎）一起部署。测试验证模型是否可以成功满足推理请求。

**注意:** 只有 Python 封装器版本 0.19 才能使用新的编排器完全支持自定义指标和标签。如果您需要使用旧版本的 Python 封装器，您可以继续使用上述 Java 引擎，直到下一个版本。

| 语言封装 | 版本 | API类型 | 新编排 | 废弃Java引擎 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Python | 0.19  | both | yes | yes | 完整的自定义指标和标签支持 |
| Python | 0.11 ... 0.18  | both | yes | yes | . |
| Python | 0.10  | REST | no | yes | . |
| Python | 0.10  | GRPC| yes | yes | . |


使用 REST API 部署的 Java 封装器的请求格式差异示例：

1. 使用新的编排器：
  
  ```
  curl -s -X POST \
   -d 'json={"data": {"names": ["a", "b"], "ndarray": [[1.0, 2.0]]}}' \
   localhost:8003/seldon/seldon/compat-rest-java-02-executor/api/v1.0/predictions
  ```
2. 使用已弃用的 Java 引擎：
  
  ```
  curl -s -X POST -H 'Content-Type: application/json' \
   -d '{"data": {"names": ["a", "b"], "ndarray": [[1.0, 2.0]]}}' \
   localhost:8003/seldon/seldon/compat-rest-java-02-engine/api/v1.0/predictions
  ```

## 升级到 0.5.2 

此版本包括重大改进和功能，包括添加预先打包的模型服务器，修复了几个关键错误。

这个版本也放弃了对 kubernetes 1.11 的支持，并在变异和验证 webhook 中添加了更改，您需要确保按照下面概述的方式进行转换。

### 升级过程

为了升级，主要要求是确保 Kubernetes 集群更新到 1.12 或更高版本。

完成此操作后，有必要删除旧的 webhook。这可以通过以下命令完成（您需要确保在安装了 seldon 核心的命名空间中执行这些命令）。

## 升级到 0.2.8 

### 升级过程

#### 安装流程现在替换为 Helm

通过 helm charts 安装 Seldon Core 已更改。现在只有一个简单的 Helm chart `seldon-core-operator` 来安装 CRD 和 controller。 Ingress 现已独立，可在选项中设置：

- Ambassador - 通过官方 Helm chart
- Istio

更多信息参考 [安装文档](../workflow/install.xhtml).

Helm chart `seldon-core-operator` 需要集群层面的 RBAC 并以集群管理员进行安装。

##### 放弃对 KSonnet 的支持

Ksonnet 现已启用。你需要通过 Helm 安装 Seldon Core.
