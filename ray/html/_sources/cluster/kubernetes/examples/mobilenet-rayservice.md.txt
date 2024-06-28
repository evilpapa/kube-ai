(kuberay-mobilenet-rayservice-example)=

# 在 Kubernetes 上提供 MobileNet 图像分类器

> **注意:** Ray Serve 应用程序及其客户端的 Python 文件位于 [ray-project/serve_config_examples](https://github.com/ray-project/serve_config_examples) 仓库。

## 步骤 1： 使用 Kind 创建 Kubernetes 集群

```sh
kind create cluster --image=kindest/node:v1.23.0
```

## 步骤 2：安装 KubeRay operator

按照 [本文档](kuberay-operator-deploy) 过 Helm 存储库安装最新稳定的 KubeRay Operator 。
请注意，本示例中的 YAML 文件使用了 `serveConfigV2`，它从 KubeRay v0.6.0 开始支持。

## 步骤 3： 安装 RayService

```sh
# Download `ray-service.mobilenet.yaml`
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-service.mobilenet.yaml

# Create a RayService
kubectl apply -f ray-service.mobilenet.yaml
```

* [mobilenet.py](https://github.com/ray-project/serve_config_examples/blob/master/mobilenet/mobilenet.py) 文件需要 `tensorflow` 作为依赖。因此，YAML 文件使用 `rayproject/ray-ml:2.5.0` 而不是 `rayproject/ray:2.5.0`。
* `python-multipart` 是请求解析函数 `starlette.requests.form()` 所必需的，因此在运行时环境的 YAML 文件中包含了 `python-multipart` 。

## 步骤 4： 转发 Ray Serve 端口

```sh
kubectl port-forward svc/rayservice-mobilenet-serve-svc 8000
```

请注意，Serve 服务是在 Ray Serve 应用程序准备就绪并运行后创建的。RayCluster 中的所有 Pod 运行后，此过程可能需要大约 1 分钟。

## 步骤 5： 发送请求到 ImageClassifier

* 步骤 5.1: 准备图像文件。
* 步骤 5.2: 更新 [mobilenet_req.py](https://github.com/ray-project/serve_config_examples/blob/master/mobilenet/mobilenet_req.py) 的 `image_path`
* 步骤 5.3: 请求到 `ImageClassifier`.
  ```sh
  python mobilenet_req.py
  # sample output: {"prediction":["n02099601","golden_retriever",0.17944198846817017]}
  ```
