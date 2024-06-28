(kuberay-pod-security)=

# Pod 安全

Kubernetes 定义了三种不同的 Pod 安全标准，包括 `privileged`、 `baseline` 和 `restricted`，以广泛覆盖安全范围。
`privileged` 标准允许用户进行已知的权限升级，因此对于安全关键型应用程序来说不够安全。

文档介绍如何配置 RayCluster YAML 文件以应用 `restricted` Pod 安全标准。
以下参考资料可以帮助您更好地理解本文档：

* [Kubernetes - Pod Pod 安全标准](https://kubernetes.io/docs/concepts/security/pod-security-standards/#restricted)
* [Kubernetes - Pod 安全准入](https://kubernetes.io/docs/concepts/security/pod-security-admission/)
* [Kubernetes - 审计](https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/)
* [Kind - 审计](https://kind.sigs.k8s.io/docs/user/auditing/)

# 准备

克隆 [KubeRay 仓库](https://github.com/ray-project/kuberay) 并签出 `master` 分支。
本教程需要存储库中的多个文件。

# 步骤 1: 创建 Kind 集群

```bash
# Path: kuberay/
kind create cluster --config ray-operator/config/security/kind-config.yaml --image=kindest/node:v1.24.0
```

`kind-config.yaml` 使用 `audit-policy.yaml` 中定义的审核策略启用审核日志记录。
`audit-policy.yaml` 定义了一个审计策略来监听 `pod-security` 命名空间中的 Pod 事件。
通过这个策略，我们可以检查我们的 Pod 是否违反了标准的 `restricted` 策略。

[Pod 安全准入](https://kubernetes.io/docs/concepts/security/pod-security-admission/) 功能首次在 Kubernetes v1.22（alpha）中引入，
并在 Kubernetes v1.25 中变得稳定。
此外，KubeRay目前支持Kubernetes v1.19到v1.24。（在撰写本文时，我们尚未使用 Kubernetes v1.25 测试 KubeRay）。 因此，我在此步骤中使用 **Kubernetes v1.24** 。

# 步骤 2: 检查审核日志

```bash
docker exec kind-control-plane cat /var/log/kubernetes/kube-apiserver-audit.log
```
日志应该为空，因为命名空间 `pod-security` 不存在。

# 步骤 3: 创建 `pod-security` 空间

```bash
kubectl create ns pod-security
kubectl label --overwrite ns pod-security \
  pod-security.kubernetes.io/warn=restricted \
  pod-security.kubernetes.io/warn-version=latest \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/audit-version=latest \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/enforce-version=latest
```

通过 `pod-security.kubernetes.io` 标签，内置的 Kubernetes Pod 安全准入控制器会将
`restricted` Pod 安全标准应用到 `pod-security` 命名空间中的所有 Pod。
`pod-security.kubernetes.io/enforce=restricted` 标签意味着如果 Pod 违反 `restricted` 安全标准中定义的策略，它将被拒绝。
请参阅 [Pod 安全准入](https://kubernetes.io/docs/concepts/security/pod-security-admission/) 了解更多详细信息。

# 步骤 4: 安装 KubeRay operator

```bash
# Update the field securityContext in helm-chart/kuberay-operator/values.yaml
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  runAsNonRoot: true
  seccompProfile:
    type: RuntimeDefault

# Path: kuberay/helm-chart/kuberay-operator
helm install -n pod-security kuberay-operator .
```

# 步骤 5: 创建 RayCluster (选择步骤 5.1 或 5.2)

* 如果选择步骤 5.1， `pod-security` 空间下不会创建任何 Pod
* 如果选择步骤 5.2， Pod 可以成功创建。

## 步骤 5.1: 在没有正确配置 `securityContext` 的情况下创建 RayCluster

```bash
# Path: kuberay/ray-operator/config/samples
kubectl apply -n pod-security -f ray-cluster.complete.yaml

# Wait 20 seconds and check audit logs for the error messages.
docker exec kind-control-plane cat /var/log/kubernetes/kube-apiserver-audit.log

# Example error messagess
# "pods \"raycluster-complete-head-fkbf5\" is forbidden: violates PodSecurity \"restricted:latest\": allowPrivilegeEscalation != false (container \"ray-head\" must set securityContext.allowPrivilegeEscalation=false) ...

kubectl get pod -n pod-security
# NAME                               READY   STATUS    RESTARTS   AGE
# kuberay-operator-8b6d55dbb-t8msf   1/1     Running   0          62s

# Clean up the RayCluster
kubectl delete rayclusters.ray.io -n pod-security raycluster-complete
# raycluster.ray.io "raycluster-complete" deleted
```

命名空间 `pod-security` 中不会创建任何 Pod，并检查审核日志中是否有错误消息。

## 步骤 5.2: 使用正确的 `securityContext` 配置创建 RayCluster 

```bash
# Path: kuberay/ray-operator/config/security
kubectl apply -n pod-security -f ray-cluster.pod-security.yaml

# Wait for the RayCluster convergence and check audit logs for the messages.
docker exec kind-control-plane cat /var/log/kubernetes/kube-apiserver-audit.log

# Forward the dashboard port
kubectl port-forward --address 0.0.0.0 svc/raycluster-pod-security-head-svc -n pod-security 8265:8265

# Log in to the head Pod
kubectl exec -it -n pod-security ${YOUR_HEAD_POD} -- bash

# (Head Pod) Run a sample job in the Pod
python3 samples/xgboost_example.py

# Check the job status in the dashboard on your browser.
# http://127.0.0.1:8265/#/job => The job status should be "SUCCEEDED".

# (Head Pod) Make sure Python dependencies can be installed under `restricted` security standard 
pip3 install jsonpatch
echo $? # Check the exit code of `pip3 install jsonpatch`. It should be 0.

# Clean up the RayCluster
kubectl delete -n pod-security -f ray-cluster.pod-security.yaml
# raycluster.ray.io "raycluster-pod-security" deleted
# configmap "xgboost-example" deleted
```

 Head Pod 和 worker Pod 将按照 `ray-cluster.pod-security.yaml` 创建。
首先，我们登录 head Pod，运行 XGBoost 示例脚本，并在仪表板中检查作业状态。
 接下来，我们用来 `pip` 安装 Python 依赖项（即 `jsonpatch`），`pip` 命令的退出代码应该为 0。
