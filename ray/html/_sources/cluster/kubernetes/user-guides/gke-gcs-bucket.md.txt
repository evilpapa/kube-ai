(kuberay-gke-bucket)=
# 配置 KubeRay 以在 GKE 中使用 Google Cloud Storage Bucket 

如果您已经熟悉 GKE 中的 Workload Identity，则可以跳过本文档。要点是，将 Kubernetes 服务帐户链接到 Google Cloud 服务帐户后，您需要在每个 Ray pod 中指定一个服务帐户。否则，请继续阅读。

此示例是 <https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity> 文档的删节版本。如果您对详细信息感兴趣，完整的文档值得阅读。

## 在 GKE 上创建 Kubernetes 集群

此示例使用 GKE 创建一个最小的 KubeRay 集群。

在本地计算机或 [Google Cloud Shell](https://cloud.google.com/shell)上运行此命令以及以下所有命令。如果从本地计算机运行，请安装 [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)。

```bash
gcloud container clusters create cloud-bucket-cluster \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-west1-b --machine-type e2-standard-8 \
    --workload-pool=my-project-id.svc.id.goog # Replace my-project-id with your GCP project ID
```


此命令在 `us-west1-b` 区域中的一个节点创建了一个 `cloud-bucket-cluster` Kubernetes 集群。示例使用 `e2-standard-8` 机型，具有 8 个 vCPU 和 32 GB 内存。

有关如何查找项目 ID 的详细信息，请参阅 <https://support.google.com/googleapi/answer/7014113?hl=en> or <https://cloud.google.com/resource-manager/docs/creating-managing-projects>。

现在获取集群的凭据以用于 `kubectl`:

```bash
gcloud container clusters get-credentials cloud-bucket-cluster --zone us-west1-b --project my-project-id
```

## 创建 IAM 服务帐户

```bash
gcloud iam service-accounts create my-iam-sa
```

## 创建 Kubernetes 服务帐户

```bash
kubectl create serviceaccount my-ksa
```

## 将 Kubernetes 服务帐户链接到 IAM 服务帐户，反之亦然

如果您不使用默认命名空间，请在以下两个命令中替换 `default` 为您的命名空间。

```bash
gcloud iam service-accounts add-iam-policy-binding my-iam-sa@my-project-id.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:my-project-id.svc.id.goog[default/my-ksa]"
```

```bash
kubectl annotate serviceaccount my-ksa \
    --namespace default \
    iam.gke.io/gcp-service-account=my-iam-sa@my-project-id.iam.gserviceaccount.com
```

## 创建一个 Google Cloud Storage Bucket 并允许 Google Cloud Service 帐户访问它

请按照 <https://cloud.google.com/storage/docs/creating-buckets>  上的文档使用 Google Cloud Console 或命令行工具 `gsutil` 创建存储桶。 

此示例授予主体 `my-iam-sa@my-project-id.iam.gserviceaccount.com` 对存储桶的“存储管理员”权限。 在 Google Cloud Console 中（“存储桶”>“存储桶详细信息”下的“权限”选项卡）或使用以下命令启用权限：

```bash
gsutil iam ch serviceAccount:my-iam-sa@my-project-id.iam.gserviceaccount.com:roles/storage.admin gs://my-bucket
```

## 创建最小的 RayCluster YAML 清单

您可以通过 `curl` 下载本教程的 RayCluster YAML 清单，如下：

```bash
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.gke-bucket.yaml
```

关键部分是以下几行：

```yaml
      spec:
        serviceAccountName: my-ksa
        nodeSelector:
          iam.gke.io/gke-metadata-server-enabled: "true"
```

将这些行包含在 Ray 集群的每个 pod 规范中。为了简单起见，本示例使用单节点集群（1 个头节点和 0 个 worker 节点）。

## 创建 RayCluster

```bash
kubectl apply -f ray-cluster.gke-bucket.yaml
```

## 测试来自 RayCluster 的 GCS 存储桶访问

使用 `kubectl get pod` 获取 Ray  Head Pod 的名称。然后运行以下命令以在 Ray head pod 中执行 shell：

```bash
kubectl exec -it raycluster-mini-head-xxxx -- /bin/bash
```

在 shell 中，运行 `pip install google-cloud-storage` 安装 Google Cloud Storage Python 客户端库。

（对于生产用例，您需要确保 `google-cloud-storage` 安装在集群的每个节点上，或者使用 `ray.init(runtime_env={"pip": ["google-cloud-storage"]})` 在运行时根据需要安装软件包 - 请参阅 <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments> 了解更多详细信息。）

然后运行以下Python代码来测试对存储桶的访问：

```python
import ray
import os
from google.cloud import storage

GCP_GCS_BUCKET = "my-bucket"
GCP_GCS_FILE = "test_file.txt"

ray.init(address="auto")

@ray.remote
def check_gcs_read_write():
    client = storage.Client()
    bucket = client.get_bucket(GCP_GCS_BUCKET)
    blob = bucket.blob(GCP_GCS_FILE)
    
    # Write to the bucket
    blob.upload_from_string("Hello, Ray on GKE!")
    
    # Read from the bucket
    content = blob.download_as_text()
    
    return content

result = ray.get(check_gcs_read_write.remote())
print(result)
```

你能看到以下输出：

```text
Hello, Ray on GKE!
```
