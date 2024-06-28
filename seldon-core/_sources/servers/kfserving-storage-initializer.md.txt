# KFserving Storage Initializer（已弃用）

Seldon Core 1.8 版本之前，预封装模型服务器默认使用 `kfserving/storage-initializer`。这些仍可以通过配置 helm 来使用：


```yaml
storageInitializer:
  image: kfserving/storage-initializer:v0.6.1
```

> :warning: **NOTE:** 当前默认存储加载器是 `seldonio/rclone-storage-initializer:1.10.0-dev` 可参考[这里](./overview.md).


当 `kfserving/storage-initializer` 使用 `modeluri` 时支持以下对象存储提供方：

- Google Cloud Storage (using `gs://`)
- S3-compatible (using `s3://`)
- Minio-compatible (using `s3://`)
- Azure Blob storage (using `https://(.+?).blob.core.windows.net/(.+)`)

Kubernetes PersistentVolume [也可被用来](../examples/pvc-tfjob.html) 替代存储桶 `pvc://`.


## 处理凭证

为了处理凭据，您必须将密钥作为环境变量添加到 `Init Containers`。您需要执行以下操作：

1. 了解需要设置哪些环境变量
2. 创建一个包含环境变量的密钥
3. 为 Seldon Core Controller 或 Seldon Core 提供密钥名称

### 1. 了解需要设置哪些环境变量

要了解所需的环境变量是什么，您可以直接查看我们在 `Init Containers` 使用的 [Storage.py 类库](https://github.com/SeldonIO/seldon-core/blob/master/python/seldon_core/storage.py)。

#### WS 必需变量

  RCLONE_CONFIG_S3_PROVIDER: aws
- RCLONE_CONFIG_S3_ACCESS_KEY_ID
- RCLONE_CONFIG_S3_SECRET_ACCESS_KEY
- RCLONE_CONFIG_S3_ENDPOINT

#### Minio 所需变量

  RCLONE_CONFIG_S3_PROVIDER: minio
- RCLONE_CONFIG_S3_ACCESS_KEY_ID
- RCLONE_CONFIG_S3_SECRET_ACCESS_KEY
- RCLONE_CONFIG_S3_ENDPOINT
- RCLONE_CONFIG_S3_ENV_AUTH

#### Google Cloud 所需变量

目前，对于 Google Cloud，它需要遵循一种稍微复杂的方法，因为它需要将密钥作为文件挂载。为此，请按照 Google Cloud 部分的示例进行操作。

如果未设置应用程序凭据，客户端将使用匿名客户端。

### 2. 创建一个包含环境变量的secret

您现在可以创建一个密钥，下面我们将展示 AWS 凭证的环境变量的样子。

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-init-container-secret
type: Opaque
data:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: aws
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: "<your AWS_ACCESS_KEY_ID here>"
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: "<your AWS_SECRET_ACCESS_KEY here>"
  RCLONE_CONFIG_S3_ENDPOINT: "<your S3 endpoint here>"
```

也可从名两行创建 `Secret` 对象：

```bash
kubectl create secret generic seldon-init-container-secret \
    --from-literal=RCLONE_CONFIG_S3_ENDPOINT='XXXX' \
    --from-literal=RCLONE_CONFIG_S3_ACCESS_KEY_ID='XXXX' \
    --from-literal=RCLONE_CONFIG_S3_SECRET_ACCESS_KEY='XXXX' \
    --from-literal=RCLONE_CONFIG_S3_PROVIDER='aws' \
    --from-literal=RCLONE_CONFIG_S3_TYPE='s3' \
    --from-literal=RCLONE_CONFIG_S3_ENV_AUTH=false
```

阅读 [Kubernetes 文档](https://kubernetes.io/docs/concepts/configuration/secret/)来学习 Kubernetes Secrets。

### 3. 确保您的 SeldonDeployment 可以访问密钥

为了让您的 SeldonDeployment 知道密钥的名称是什么，我们必须指定我们创建的密钥名称——上面示例中我们使用的是 `seldon-init-container-secret`。

#### 选项 1：默认 Seldon Core Manager Controller 值

在通过 Helm 图安装 Seldon Core 时可通过 `values.yaml` 变量 `executor.defaultEnvSecretRefName` 设置一个全局默认的。可在 [高级 Helm 安装页](../reference/helm.rst)查看所有选项。

```yaml
# ... 其他变量
predictiveUnit:
  defaultEnvSecretRefName: seldon-init-container-secret
# ... 其他变量
```

#### 选项 2：通过 SeldonDeployment 配置覆盖

当您使用 SeldonDeploymen YAML 部署模型时，也可以提供覆盖值。您可以通过以下 `envSecretRefName` 值执行此操作：

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: s3://seldon-models/sklearn/iris
      envSecretRefName: seldon-init-container-secret
      name: classifier
    name: default
    replicas: 1
```

### 示例

#### MinIO 在同一个 Kubernetes 集群中运行
假设您有 MinIO 实例运行在 `minio.minio-system.svc.cluster.local` 端口 `9000` 上你向关联 `mymodel` 桶你可以设置
```bash
RCLONE_CONFIG_S3_ENDPOINT=http://minio.minio-system.svc.cluster.local:9000
```
使用 `modelUri` 设置为 `s3://mymodel`。

完整示例查看 [笔记本](../examples/minio-sklearn.html)。

## 为 Google Cloud 添加凭据

目前，Google 凭据需要设置一个文件，因此所需的过程涉及创建一个服务帐户，如下所述。

你可以创建一个`ServiceAccount` 并附加一个不同格式的 `Secret` 的文件，类似于 kfserving 的做法。 在[这个主题](https://github.com/kubeflow/kfserving/blob/master/docs/samples/storage/s3/README.md)查看 tfserving 文档。持的注解前缀包括 `serving.kubeflow.org` 和 `machinelearning.seldon.io`。

对于 GCP/GKE，你可能需要创建一个 service-account 键并作为一个本地 `json` 文件。
首先确保 `[SA-NAME]@[PROJECT-ID].iam.gserviceaccount.com` 服务账号在 gcloud 终端创建，该帐户具有足够的权限来使用您的模型（即）访问存储桶 (例如 `Storage Object Admin`)。

现在，使用 `gcloud` 本地工具生成 `keys`
```bash
gcloud iam service-accounts keys create gcloud-application-credentials.json --iam-account [SA-NAME]@[PROJECT-ID].iam.gserviceaccount.com
```

一旦文件 `gcloud-application-credentials.json` 在本地创建完，使用下面的命令创建创建 k8s `secret`：
```bash
kubectl create secret generic user-gcp-sa --from-file=gcloud-application-credentials.json=<LOCALFILE JSON FILE>
```

在密钥中的文件需要为 `gcloud-application-credentials.json` (名称可在 seldon configmap 设置，`kubectl get cm -n seldon-system seldon-config -o yaml` 可见)。

然后创建一个服务账户关联到密钥：

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: user-gcp-sa
secrets:
  - name: user-gcp-sa
```

这些可在 SeldonDeployment 中设置 `serviceAccountName: user-gcp-sa` 进行引用，他和 `m̀odelUri` 同级例如

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  name: iris
  predictors:
  - graph:
      children: []
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.14.0/sklearn/iris
      serviceAccountName: user-gcp-sa
      name: classifier
    name: default
    replicas: 1
```
