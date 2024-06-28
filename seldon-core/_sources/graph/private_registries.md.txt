# 从私有 Docker 镜像拉取

从私有 Docker 镜像拉取需要为 SeldonDeployment 资源设置 imagePullSecrets。比如，下面配置了私有镜像 ```private-docker-repo/my-image``` 的简单模型。你需要在资源应用到集群之前，在 Kubernetes 中创建 docker 镜像密钥 ```myreposecret```。

```json
{
  "apiVersion": "machinelearning.seldon.io/v1alpha2",
  "kind": "SeldonDeployment",
  "metadata": {
    "name": "private-model"
  },
  "spec": {
    "name": "private-model-example",
    "predictors": [
      {
        "componentSpecs": [{
          "spec": {
            "containers": [
              {
                "image": "private-docker-repo/my-image",
                "name": "private-model"
              }
            ],
        	"imagePullSecrets": [
              {
                "name": "myreposecret"
              }
            ]
          }
        }],
        "graph": {
          "children": [],
          "endpoint": {
            "type": "REST"
          },
          "name": "private-model",
          "type": "MODEL"
        },
        "name": "private-model",
        "replicas": "1"
      }
    ]
  }
}
```

创建 Docker 镜像密钥请查看 [Kubernetes 文档](https://kubernetes.io/docs/concepts/containers/images/#creating-a-secret-with-a-docker-config)。
