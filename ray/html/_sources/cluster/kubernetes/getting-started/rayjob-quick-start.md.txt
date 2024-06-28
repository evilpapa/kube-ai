(kuberay-rayjob-quickstart)=

# RayJob 快速入门

:::{warning}
KubeRay v0.x 中的 RayJob 支持处于 alpha 阶段。
:::

## 先决条件

* Ray 1.10 或更高
* KubeRay v0.3.0+。 (推荐 v0.6.0+ )

## 什么是 RayJob？

RayJob 管理两个方面：

* **RayCluster**: 管理 Kubernetes 集群中的资源。
* **Job**: 运行 Kubernetes 作业 `ray job submit` 以将 Ray 作业提交到 RayCluster。

## RayJob 提供什么？

* **对 Ray Cluster和 Ray Job 的 Kubernetes 原生支持**: 您可以使用 Kubernetes 配置来定义 Ray 集群和作业，并用 `kubectl` 创建它们。作业完成后，集群可以自动删除。

## RayJob 配置

* `entrypoint` - 为此作业运行的 shell 命令。
* `rayClusterSpec` - 用于运行 Job 的 **RayCluster** 配置。
* `jobId` - _（可选）_ 为作业指定的作业 ID。如果未提供，则会生成一个。
* `metadata` - _（可选）_ 用户为作业提供的任意元数据。
* `runtimeEnvYAML` - _（可选）_ 以多行 YAML 字符串形式提供的运行时环境配置。 _（KubeRay 1.0 版中的新增功能。）_
* `shutdownAfterJobFinishes` - _（可选）_ 作业完成后是否回收集群。默认为 false。
* `ttlSecondsAfterFinished` - _（可选）_ 用于清理集群的 TTL。仅当`shutdownAfterJobFinishes`设置时才有效。
* `submitterPodTemplate` - _（可选）_ 针对 Ray 集群运行 `ray job submit` 的 Pod 的 Pod 模板规范。
* `entrypointNumCpus` - _（可选）_ 指定为入口点命令保留的 CPU 核心数量。 _（KubeRay 1.0 版中的新增功能。）_
* `entrypointNumGpus` - _（可选）_ 指定为入口点命令保留的 GPU 数量。 _（KubeRay 1.0 版中的新增功能。）_
* `entrypointResources` - _（可选）_ 一个 json 格式的字典，用于指定自定义资源及其数量。 _（KubeRay 1.0 版中的新增功能。）_
* `runtimeEnv` - [已弃用] _（可选）_ 运行时环境 json 字符串的 base64 编码字符串。

## 示例: 使用 RayJob 运行简单的 Ray 作业

## 步骤 1: 使用 Kind 创建 Kubernetes 集群

```sh
kind create cluster --image=kindest/node:v1.23.0
```

## 步骤 2: 安装 KubeRay Operator

按照 [本文档](kuberay-operator-deploy) 通过 Helm 存储库安装最新的稳定 KubeRay Operator。
请注意，本示例中的 YAML 文件使用 `serveConfigV2` 指定多应用程序 Serve 配置，从 KubeRay v0.6.0 开始支持该配置。

## 步骤 3: 安装 RayJob

```sh
# 步骤 3.1: 下载 `ray_v1alpha1_rayjob.yaml`
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray_v1alpha1_rayjob.yaml

# 步骤 3.2: 创建 RayJob
kubectl apply -f ray_v1alpha1_rayjob.yaml
```

## 步骤 4: 验证 Kubernetes 集群状态

```shell
# 步骤 4.1: List all RayJob custom resources in the `default` namespace.
kubectl get rayjob

# [Example output]
# NAME            AGE
# rayjob-sample   7s

# 步骤 4.2: List all RayCluster custom resources in the `default` namespace.
kubectl get raycluster

# [Example output]
# NAME                                 DESIRED WORKERS   AVAILABLE WORKERS   STATUS   AGE
# rayservice-sample-raycluster-6mj28   1                 1                   ready    2m27s

# 步骤 4.3: List all Pods in the `default` namespace.
# The Pod created by the Kubernetes Job will be terminated after the Kubernetes Job finishes.
kubectl get pods

# [Example output]
# kuberay-operator-7456c6b69b-rzv25                         1/1     Running     0          3m57s
# rayjob-sample-lk9jx                                       0/1     Completed   0          2m49s => Pod created by a Kubernetes Job
# rayjob-sample-raycluster-9c546-head-gdxkg                 1/1     Running     0          3m46s
# rayjob-sample-raycluster-9c546-worker-small-group-nfbxm   1/1     Running     0          3m46s

# 步骤 4.4: Check the status of the RayJob.
# The field `jobStatus` in the RayJob custom resource will be updated to `SUCCEEDED` once the job finishes.
kubectl get rayjobs.ray.io rayjob-sample -o json | jq '.status.jobStatus'

# [Example output]
# "SUCCEEDED"
```

KubeRay operator 将创建由自定义资源定义的 RayCluster `rayClusterSpec` 指定，以及用于向 RayCluster 提交 Ray 作业的 Kubernetes 作业。 
Ray job 在 RayJob 自定义资源的 `entrypoint` 字段定义。
在此示例中， `entrypoint` 是 `python /home/ray/samples/sample_code.py`，
`sample_code.py` 存储在 Kubernetes Config 并挂载在 RayCluster  Head Pod 的 Python 脚本。
由于默认  `shutdownAfterJobFinishes` 值为false，因此作业完成后不会删除 RayCluster。

## 步骤 5: 检查 Ray Job 的输出

```sh
kubectl logs -l=job-name=rayjob-sample

# [Example output]
# 2023-08-21 17:08:22,530 INFO cli.py:27 -- Job submission server address: http://rayjob-sample-raycluster-9c546-head-svc.default.svc.cluster.local:8265
# 2023-08-21 17:08:23,726 SUCC cli.py:33 -- ------------------------------------------------
# 2023-08-21 17:08:23,727 SUCC cli.py:34 -- Job 'rayjob-sample-5ntcr' submitted successfully
# 2023-08-21 17:08:23,727 SUCC cli.py:35 -- ------------------------------------------------
# 2023-08-21 17:08:23,727 INFO cli.py:226 -- Next steps
# 2023-08-21 17:08:23,727 INFO cli.py:227 -- Query the logs of the job:
# 2023-08-21 17:08:23,727 INFO cli.py:229 -- ray job logs rayjob-sample-5ntcr
# 2023-08-21 17:08:23,727 INFO cli.py:231 -- Query the status of the job:
# 2023-08-21 17:08:23,727 INFO cli.py:233 -- ray job status rayjob-sample-5ntcr
# 2023-08-21 17:08:23,727 INFO cli.py:235 -- Request the job to be stopped:
# 2023-08-21 17:08:23,728 INFO cli.py:237 -- ray job stop rayjob-sample-5ntcr
# 2023-08-21 17:08:23,739 INFO cli.py:245 -- Tailing logs until the job exits (disable with --no-wait):
# 2023-08-21 17:08:34,288 INFO worker.py:1335 -- Using address 10.244.0.6:6379 set in the environment variable RAY_ADDRESS
# 2023-08-21 17:08:34,288 INFO worker.py:1452 -- Connecting to existing Ray cluster at address: 10.244.0.6:6379...
# 2023-08-21 17:08:34,302 INFO worker.py:1633 -- Connected to Ray cluster. View the dashboard at http://10.244.0.6:8265
# test_counter got 1
# test_counter got 2
# test_counter got 3
# test_counter got 4
# test_counter got 5
# 2023-08-21 17:08:46,040 SUCC cli.py:33 -- -----------------------------------
# 2023-08-21 17:08:46,040 SUCC cli.py:34 -- Job 'rayjob-sample-5ntcr' succeeded
# 2023-08-21 17:08:46,040 SUCC cli.py:35 -- -----------------------------------
```

`entrypoint` 使用的 Python 脚本  `sample_code.py` 是一个简单的 Ray 脚本，它执行计数器的增量函数5次。


## 步骤 6: Cleanup

```sh
# 步骤 6.1: 删除 RayJob
kubectl delete -f ray_v1alpha1_rayjob.yaml

# 步骤 6.2: 删除 KubeRay operator
helm uninstall kuberay-operator

# 步骤 6.3: 删除 Kubernetes cluster
kind delete cluster
```
