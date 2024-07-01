(kuberay-ingress)=

# Ingress

三个示例展示了如何使用 ingress 访问 Ray 集群：

  * [AWS EKS 上的 AWS Application Load Balancer (ALB) 入口支持](kuberay-aws-alb)
  * [GKE Ingress 支持](kuberay-gke-ingress)
  * [在 Kind 上手动设置 NGINX Ingress](kuberay-nginx)

(kuberay-aws-alb)=
## AWS EKS 上的 AWS 应用程序负载均衡器 (ALB) 入口支持

### 先决条件
* 创建 EKS 集群。请参阅 [Amazon EKS 入门 – AWS 管理控制台和 AWS CLI](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#eks-configure-kubectl)。

* 设置 [AWS AWS 负载均衡器控制器](https://github.com/kubernetes-sigs/aws-load-balancer-controller)，参阅 [安装说明](https://kubernetes-sigs.github.io/aws-load-balancer-controller/latest/deploy/installation/)。请注意，存储库为每个版本维护一个网页。确认您使用的是最新的安装说明。

* (可选) 尝试 [aws-load-balancer-controller](https://github.com/kubernetes-sigs/aws-load-balancer-controller) 仓库的 [echo server 示例](https://github.com/kubernetes-sigs/aws-load-balancer-controller/blob/main/docs/examples/echo_server.md) 。

* (可选) 阅读 [how-it-works.md](https://github.com/kubernetes-sigs/aws-load-balancer-controller/blob/main/docs/how-it-works.md) 理解 [aws-load-balancer-controller](https://github.com/kubernetes-sigs/aws-load-balancer-controller) 机制。

### 指示
```sh
# 步骤 1: Install KubeRay operator and CRD
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# 步骤 2: Install a RayCluster
helm install raycluster kuberay/ray-cluster --version 1.0.0-rc.0

# 步骤 3: Edit the `ray-operator/config/samples/ray-cluster-alb-ingress.yaml`
#
# (1) Annotation `alb.ingress.kubernetes.io/subnets`
#   1. Please include at least two subnets.
#   2. One Availability Zone (ex: us-west-2a) can only have at most 1 subnet.
#   3. In this example, you need to select public subnets (subnets that "Auto-assign public IPv4 address" is Yes on AWS dashboard)
#
# (2) Set the name of head pod service to `spec...backend.service.name`
eksctl get cluster ${YOUR_EKS_CLUSTER} # Check subnets on the EKS cluster

# 步骤 4: Check ingress created by 步骤 4.
kubectl describe ingress ray-cluster-ingress

# [Example]
# Name:             ray-cluster-ingress
# Labels:           <none>
# Namespace:        default
# Address:          k8s-default-rayclust-....${REGION_CODE}.elb.amazonaws.com
# Default backend:  default-http-backend:80 (<error: endpoints "default-http-backend" not found>)
# Rules:
#  Host        Path  Backends
#  ----        ----  --------
#  *
#              /   ray-cluster-kuberay-head-svc:8265 (192.168.185.157:8265)
# Annotations: alb.ingress.kubernetes.io/scheme: internet-facing
#              alb.ingress.kubernetes.io/subnets: ${SUBNET_1},${SUBNET_2}
#              alb.ingress.kubernetes.io/tags: Environment=dev,Team=test
#              alb.ingress.kubernetes.io/target-type: ip
# Events:
#   Type    Reason                  Age   From     Message
#   ----    ------                  ----  ----     -------
#   Normal  SuccessfullyReconciled  39m   ingress  Successfully reconciled

# 步骤 6: Check ALB on AWS (EC2 -> Load Balancing -> Load Balancers)
#        The name of the ALB should be like "k8s-default-rayclust-......".

# 步骤 7: Check Ray Dashboard by ALB DNS Name. The name of the DNS Name should be like
#        "k8s-default-rayclust-.....us-west-2.elb.amazonaws.com"

# 步骤 8: Delete the ingress, and AWS Load Balancer controller will remove ALB.
#        Check ALB on AWS to make sure it is removed.
kubectl delete ingress ray-cluster-ingress
```

(kuberay-gke-ingress)=

## GKE Ingress 支持

### 先决条件

* 创建 GKE 集群并确保您已安装 kubectl 工具并经过身份验证以与 GKE 集群进行通信。请参阅 [教程](kuberay-gke-gpu-cluster-setup) ，了解如何使用 GPU 创建 GKE 集群的示例。 （本部分不需要 GPU。）

* 了解 <https://cloud.google.com/kubernetes-engine/docs/concepts/ingress> 中的概念可能会有所帮助。

### 指示
将以下文件另存为 `ray-cluster-gclb-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ray-cluster-ingress
  annotations:
    kubernetes.io/ingress.class: "gce"
spec:
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: raycluster-kuberay-head-svc # Update this line with your head service in 步骤 3 below.
                port:
                  number: 8265
```

现在运行以下命令：

```bash
# 步骤 1: Install KubeRay operator and CRD
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# 步骤 2: Install a RayCluster
helm install raycluster kuberay/ray-cluster --version 1.0.0-rc.0

# 步骤 3: Edit ray-cluster-gclb-ingress.yaml to replace the service name with the name of the head service from the RayCluster. (Output of `kubectl get svc`)

# 步骤 4: Apply the Ingress configuration
kubectl apply -f ray-cluster-gclb-ingress.yaml

# 步骤 5: Check ingress created by 步骤 4.
kubectl describe ingress ray-cluster-ingress

# 步骤 6: After a few minutes, GKE allocates an external IP for the ingress. Check it using:
kubectl get ingress ray-cluster-ingress

# Example output:
# NAME                  CLASS    HOSTS   ADDRESS         PORTS   AGE
# ray-cluster-ingress   <none>   *       34.160.82.156   80      54m

# 步骤 7: Check Ray Dashboard by visiting the allocated external IP in your browser. (In this example, it is 34.160.82.156)

# 步骤 8: Delete the ingress.
kubectl delete ingress ray-cluster-ingress
```

(kuberay-nginx)=
## 在 Kind 上手动设置 NGINX Ingress 

```sh
# 步骤 1: Create a Kind cluster with `extraPortMappings` and `node-labels`
# Reference for the setting up of Kind cluster: https://kind.sigs.k8s.io/docs/user/ingress/
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
EOF

# 步骤 2: Install NGINX ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 10 # Wait for the Kubernetes API Server to create the related resources
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s

# 步骤 3: Install KubeRay operator and CRD
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0-rc.0

# 步骤 4: Install RayCluster and create an ingress separately.
# More information about change of setting was documented in https://github.com/ray-project/kuberay/pull/699 
# and `ray-operator/config/samples/ray-cluster.separate-ingress.yaml`
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.separate-ingress.yaml
kubectl apply -f ray-operator/config/samples/ray-cluster.separate-ingress.yaml

# 步骤 5: Check the ingress created in 步骤 4.
kubectl describe ingress raycluster-ingress-head-ingress

# [Example]
# ...
# Rules:
# Host        Path  Backends
# ----        ----  --------
# *
#             /raycluster-ingress/(.*)   raycluster-ingress-head-svc:8265 (10.244.0.11:8265)
# Annotations:  nginx.ingress.kubernetes.io/rewrite-target: /$1

# 步骤 6: Check `<ip>/raycluster-ingress/` on your browser. You will see the Ray Dashboard.
#        [Note] The forward slash at the end of the address is necessary. `<ip>/raycluster-ingress`
#               will report "404 Not Found".
```
