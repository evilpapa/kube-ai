(kuberay-k8s-setup)=

# 托管 Kubernetes 服务

KubeRay Operator 和 Ray 可以在任何云或本地 Kubernetes 集群上运行。
配置远程 Kubernetes 集群的最简单方法是使用基于云的托管服务。
我们为开始使用托管 Kubernetes 服务的用户收集了一些有用的链接。

(gke-setup)=
# 设置 GKE 集群 (Google Cloud)

- {ref}`kuberay-gke-gpu-cluster-setup`

(eks-setup)=
# 设置 EKS 集群 (AWS)

- {ref}`kuberay-eks-gpu-cluster-setup`

(aks-setup)=
# 设置 AKS (Microsoft Azure) 
你可以在 [此处](https://azure.microsoft.com/en-us/services/kubernetes-service/) 找到 AKS 的登录页面。
如果您设置了帐户，则可以立即开始在提供商的控制台中尝试 Kubernetes 集群。
或者，查看 [文档](https://docs.microsoft.com/en-us/azure/aks/) 和
[快速入门指南](https://docs.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli)。要在 Kubernetes 上成功部署 Ray，
您需要配置 Kubernetes 节点池；
在 [这里](https://docs.microsoft.com/en-us/azure/aks/use-multiple-node-pools) 找到指导。
