(kuberay-text-summarizer-rayservice-example)=

# Kubernetes 上提供文本摘要模型服务

> **注意:** Ray Serve 应用程序及其客户端的 Python 文件位于 [ray-project/serve_config_examples](https://github.com/ray-project/serve_config_examples) 仓库。

## 步骤 1：创建带有 GPU 的 Kubernetes 集群

参考 [aws-eks-gpu-cluster.md](kuberay-eks-gpu-cluster-setup) 或 [gcp-gke-gpu-cluster.md](kuberay-gke-gpu-cluster-setup) 创建一个具有 1 个 CPU 节点和 1 个 GPU 节点的 Kubernetes 集群。

## 步骤 2：安装 KubeRay 控制器

按照 [本文档](kuberay-operator-deploy) 过 Helm 存储库安装最新稳定的 KubeRay Operator 。
请注意，本示例中的 YAML 文件使用了 `serveConfigV2`，它从 KubeRay v0.6.0 开始支持。

## 步骤 3： 安装 RayService

```sh
# Step 3.1: Download `ray-service.text-summarizer.yaml`
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-service.text-summarizer.yaml

# Step 3.2: Create a RayService
kubectl apply -f ray-service.text-summarizer.yaml
```

此 RayService 配置包含一些重要设置：

* `tolerations` 允许将 Worker 调度到没有任何污点的节点或具有特定污点的节点上。但是，由于我们在 Pod 的资源配置中设置了 `nvidia.com/gpu: 1` ，因此 Worker 只会被调度到 GPU 节点上。
    ```yaml
    # Please add the following taints to the GPU node.
    tolerations:
        - key: "ray.io/node-type"
        operator: "Equal"
        value: "worker"
        effect: "NoSchedule"
    ```

## 步骤 4： 转发 Serve 的端口

首先从此命令获取服务名称。

```sh
kubectl get services
```

然后，端口转发到服务。

```sh
kubectl port-forward svc/text-summarizer-serve-svc 8000
```

请注意，RayService 的 Kubernetes 服务将在 Serve 应用程序准备就绪并运行后创建。RayCluster 中的所有 Pod 运行后，此过程可能需要大约 1 分钟。

## Step 5: 发送请求到 text_summarizer 模型

```sh
# Step 5.1: Download `text_summarizer_req.py` 
curl -LO https://raw.githubusercontent.com/ray-project/serve_config_examples/master/text_summarizer/text_summarizer_req.py

# Step 5.2: Send a request to the Summarizer model.
python text_summarizer_req.py
# Check printed to console
```

## Step 6: 删除服务

```sh
# path: ray-operator/config/samples/
kubectl delete -f ray-service.text-summarizer.yaml
```

## Step 7: 卸载 kuberay operator

参考 [本文档](https://github.com/ray-project/kuberay/tree/master/helm-chart/kuberay-operator) 通过 Helm 仓库删除最新稳定的 KubeRay operator 。