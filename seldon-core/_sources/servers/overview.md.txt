# 预备打包模型服务器

Seldon 提供几个预封装服务器，您可以使用这些服务器部署训练好的模型：

- [SKLearn 服务](./sklearn.html)
- [XGBoost 服务](./xgboost.html)
- [Tensorflow Serving 服务](./tensorflow.html)
- [MLflow 服务](./mlflow.html)
- [自定义服务](./custom.html)

针对这些打包好的服务只需要定义存储在本地、Google bucket、S3 bucket、azure 或 minio 对象存储的模型文件位置。使用 Sklearn 服务的示例如下：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn-iris
spec:
  predictors:
    - name: default
      replicas: 1
      graph:
        name: classifier
        implementation: SKLEARN_SERVER
        modelUri: gs://seldon-models/v1.14.0/sklearn/iris
```

默认情况下，只有发布到 Google 云存储的公共模型才能访问。 
有关如何配置 AWS S3、Minio 和其他存储解决方案的凭据，请参阅下文。


## 初始化容器

Seldon Core 使用 [Init Containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) 为预封装服务下载模型二进制文件。
我们使用 [rclone](https://rclone.org/) 实现方式的 [storage initailizer](https://github.com/SeldonIO/seldon-core/tree/master/components/rclone-storage-initializer) 来定义 [helm values](../charts/seldon-core-operator.html#values) 中的`Init Containers`。

```yaml
storageInitializer:
  image: seldonio/rclone-storage-initializer:1.14.0
```
可参考 [Dockerfile](https://github.com/SeldonIO/seldon-core/blob/master/components/rclone-storage-initializer/Dockerfile)。修改配置来指定为其他 `initContainer`，查看以下要求明细：

Kubernetes 将 `secrets` 密钥信息作为环境变量注入到 init 容器。默认的 secret 名称可通过 [helm value](../charts/seldon-core-operator.html#values) 设置。

```yaml
predictiveUnit:
  defaultEnvSecretRefName: ""
```

备注：Seldon Core 1.8 之前使用的时 `kfserving/storage-initializer`，想继续使用它，请参考[这里](kfserving-storage-initializer.xhtml)。

### 自定义 Init Containers

通过 helm values 自定义**全局** `initContainer` 镜像和默认 `secret`。

要明白预封装服务器如何使用 `initContainers`，请参考下面使用 Seldon Deployment 的配置文件，其中`volumes`, `volumeMounts` 和 `initContainers` 会被 `Seldon Core Operator` 注入到服务中

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: sklearn-iris
spec:
  predictors:
  - name: default
    replicas: 1
    graph:
      name: classifier
      type: MODEL

    componentSpecs:
    - spec:
        volumes:
        - name: classifier-provision-location
          emptyDir: {}

        initContainers:
        - name: classifier-model-initializer
          image: seldonio/rclone-storage-initializer:1.14.0
          imagePullPolicy: IfNotPresent
          args:
            - "s3://sklearn/iris"
            - "/mnt/models"

          volumeMounts:
          - mountPath: /mnt/models
            name: classifier-provision-location

          envFrom:
          - secretRef:
              name: seldon-init-container-secret

        containers:
        - name: classifier
          image: seldonio/sklearnserver:1.8.0-dev

          volumeMounts:
          - mountPath: /mnt/models
            name: classifier-provision-location
            readOnly: true

          env:
          - name: PREDICTIVE_UNIT_PARAMETERS
            value: '[{"name":"model_uri","value":"/mnt/models","type":"STRING"}]'
```

主要查看：
- 预封装计算服务依赖 模型二进制文件存储到 `/mnt/models` 路径
- `initContainers` 由 `{predictiveUnitName}-model-initializer` 构造
- `container` 的 `entrypoint` 必须接收两个参数
  - 一个存储模型的 URI
  - 一个为模型文件下载后的存储路径
- 如果用户要提供自己的名称匹配上述模式，它将按规定使用 initContainer

例如下面 `sklearn-iris` Seldon Deployment 的配置。正如我们所看到的，使用预封装模型服务器可以避免很多自定义模板，使用上更简洁：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-sklearn
spec:
  predictors:
  - name: default
    replicas: 1
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: s3://sklearn/iris
      storageInitializerImage: seldonio/rclone-storage-initializer:1.14.0  # 此处定义模型文件下载镜像
      envSecretRefName: seldon-init-container-secret                       # 自定义密钥信息
```

请注意，Storage Initializer 使用的镜像和秘密可以在每次部署时进行自定义。

查看[示例](../examples/custom_init_container.html) 了解初始化容器如何使用，以及如何自定义使用 [rclone](https://rclone.org/) 进行容器的云存储操作。

## 预封装模型服务的深度定制

如果想定义更多资源，可以在你的 podSpecs 中添加 `Container` 设置，如：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn-iris
spec:
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: classifier
          resources:
            requests:
              memory: 50Mi
    name: default
    replicas: 1
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/sklearn/iris
```

发布时，镜像名称和其他配置参数会被自动添加进来。

后续步骤：

- [可用示例](../examples/server_examples.html)
- [SKLearn 服务](./sklearn.html)
- [XGBoost 服务](./xgboost.html)
- [Tensorflow Serving服务](./tensorflow.html)
- [MLflow 服务](./mlflow.html)
- [基于MinIOn的SKLearn 服务](../examples/minio-sklearn.html)

你也可以创建自己的[自定义预估服务](custom.xhtml)，它也可以像官方封装的服务一样进行使用。

如果您的用例时不可重复使用的标准服务器类型，你也可以使用我们的语言封装器来创建自己的组件（相当于以 seldon 的标准开发自己的组件）。

## 处理凭据

### 一般说明

使用环境变量可配置 Rclone：


```yaml
RCLONE_CONFIG_<remote name>_<config variable>: <config value>
```

注意：可以同时配置多个远端

以上配置等同于 modelUri 与 `rclone` 方式的兼容。

```yaml
modelUri: <remote>:<bucket name>
```

例如 `modelUri: s3:sklearn/iris`.

注意: Rclone 将删除双斜杠，它等同于 `s3://sklearn/iris`

如下为一些配置示例，其他云解决方案请参考牛逼哄哄的 [rclone 项目文档](https://rclone.org/)。

### 公共 GCS 配置示例

注: 它默认被 `seldonio/rclone-storage-initializer` 镜像使用。

参考文献: [rclone 文档](https://rclone.org/googlecloudstorage/)。

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_GS_TYPE: google cloud storage
  RCLONE_CONFIG_GS_ANONYMOUS: "true"
```

部署

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: rclone-sklearn-gs
spec:
  predictors:
  - name: default
    replicas: 1
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: gs:seldon-models/sklearn/iris
      envSecretRefName: seldon-rclone-secret
```

### minio 配置示例

参考文献: [rclone 文档](https://rclone.org/s3/#minio)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: minio
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: minioadmin
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: minioadmin
  RCLONE_CONFIG_S3_ENDPOINT: http://minio.minio-system.svc.cluster.local:9000
```

### 基于 IAM 权限的 AWS S3 配置

参考文献: [rclone 文档](https://rclone.org/s3/#amazon-s3)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: ""
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: ""
  RCLONE_CONFIG_S3_ENV_AUTH: "true"
```

### PVC 方式示例

相较于对象存储，可以直接使用 PVC 方式。如果你有很多非常大文件，又想避免上传/下载，可通过 NFS 驱动器实现。

使用以下格式 `modelUri` 配置来定义 PVC，唯一需要确认的是，以 `runAsUser` 参数运行的容器是否具备文件读写权限。

```yaml
...
    modelUri: pvc://<pvc-name>/<path>
```