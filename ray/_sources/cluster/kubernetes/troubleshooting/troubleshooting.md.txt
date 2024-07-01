(kuberay-troubleshootin-guides)=

# 故障排除指南

本文档解决了常见问题。如果您在此处找不到问题的答案，请随时通过我们的 [社区渠道](https://github.com/ray-project/kuberay#getting-involved) 联系我们。

# 内容

- [Worker 初始化容器](#worker-init-container)
- [Cluster 域](#cluster-domain)
- [RayService](#rayservice)
- [GPU 多租户](#gpu-multitenancy)
- [其他问题](#questions)

## Worker 初始化容器

KubeRay operator 会将默认的 [init container](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) 入到每个 worker Pod 中。
此 init 容器负责等待，直到 Head Pod 上的全局控制服务 (GCS) 准备好，然后再建立与头的连接。 init 容器将 `ray health-check` 用于持续检查 GCS 服务器状态。

默认的 worker 进程初始化容器可能不适用于所有用例，或者用户可能想要自定义初始化容器。

### 1. 初始化容器故障排除

worker 进程初始化容器陷入 `Init:0/1` 状态的一些常见原因是：

* head Pod 中的 GCS 服务器进程失败。请检查 head Pod 中 `/tmp/ray/session_latest/logs/` 日志目录是否有与 GCS 服务器相关的错误。
* `ray` 可执行文件不包含在 `$PATH` 镜像中，因此 init 容器将无法运行 `ray health-check`。
* `CLUSTER_DOMAIN` 环境变量设置不正确。参考 [cluster domain](#cluster-domain) 获取信息。
* Worker init 容器与 Worker Pod 模板共享相同的 ***ImagePullPolicy***、***SecurityContext***、***Env***、***VolumeMounts*** 以及 ***Resources*** 。共享这些设置可能会导致死锁。有关更多详细信息，请参阅 [#1130](https://github.com/ray-project/kuberay/issues/1130) 。

如果 init 容器停留在 `Init:0/1` 状态 2 分钟，我们将停止将输出消息重定向到 `/dev/null` 而是将它们打印到 worker Pod 日志中。要进一步排除故障，您可以使用 `kubectl logs`检查日志。

### 2. 禁用 init 容器注入

如果你想自定义worker init容器，你可以禁用注入并添加你自己的 init 容器。
要禁用注入，请将 KubeRay operator 中的环境变量 `ENABLE_INIT_CONTAINER_INJECTION` 设置为f `false` （适用于 KubeRay v0.5.2）。
请参阅 [#1069](https://github.com/ray-project/kuberay/pull/1069) 和 [KubeRay Helm chart](https://github.com/ray-project/kuberay/blob/ddb5e528c29c2e1fb80994f05b1bd162ecbaf9f2/helm-chart/kuberay-operator/values.yaml#L83-L87) ，了解如何设置环境变量的说明。禁用后，您可以将自定义 init 容器添加到 worker Pod 模板中。

## 集群域

在 KubeRay 中，我们使用完全限定域名（FQDN）来建立 worker 和 head 之间的连接。
head service 的 FQDN 是 `${HEAD_SVC}.${NAMESPACE}.svc.${CLUSTER_DOMAIN}`。
默认的 [集群域](https://kubernetes.io/docs/tasks/administer-cluster/dns-custom-nameservers/#introduction) 是 `cluster.local`，适用于大多数 Kubernetes 集群。
但是，请务必注意，某些集群可能具有不同的集群域。
您可以通过 Pod 中的 `/etc/resolv.conf` 检查集群域。

要设置自定义集群域，请调整 KubeRay operator 的 `CLUSTER_DOMAIN` 环境变量。
Helm chart 用户可在 [这里](https://github.com/ray-project/kuberay/blob/ddb5e528c29c2e1fb80994f05b1bd162ecbaf9f2/helm-chart/kuberay-operator/values.yaml#L88-L91) 修改。
欲了解更多信息，请参阅 [#951](https://github.com/ray-project/kuberay/pull/951) 和 [#938](https://github.com/ray-project/kuberay/pull/938) 。

## RayService

RayService 是专为 Ray Serve 设计的自定义资源定义 (CRD)。在 KubeRay 中，创建 RayService 将首先创建 RayCluster，然后在 RayCluster 准备就绪后创建 Ray Serve 应用程序。如果问题与数据平面有关，特别是 Ray Serve 脚本或 Ray Serve 配置 (`serveConfigV2`)，则故障排除可能会很困难。有关更多详细信息，请参阅 [rayservice-troubleshooting](kuberay-raysvc-troubleshoot) 。

## 问题

### 为什么对 RayCluster 或 RayJob CR 的更改未生效？

目前仅支持对 `RayCluster/RayJob` CR 中的 `replicas` 进行修改。对其他字段的更改可能不会生效或可能导致意外结果。
