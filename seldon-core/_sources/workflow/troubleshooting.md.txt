# 故障排查指南

如果 Seldon Deployment 未在此处运行，
则有一些提示来诊断问题。

## 我的模型似乎没有运行

检查 Seldon Deployment 是否运行：

```bash
kubectl get sdep
```

如果它存在，检查名为 `<name>` 的 Seldon deployment 的状态：

```bash
kubectl get sdep <name> -o jsonpath='{.status}'
```

看起来像：

```bash
>kubectl get sdep

NAME      AGE
mymodel   1m

>kubectl get sdep mymodel -o jsonpath='{.status}'
map[predictorStatus:[map[name:mymodel-mymodel-7cd068f replicas:1 replicasAvailable:1]] state:Available]
```

如果安装了 `jq` 可获得更好的输出：

```bash
>kubectl get sdep mymodel -o json | jq .status
{
  "predictorStatus": [
    {
      "name": "mymodel-mymodel-7cd068f",
      "replicas": 1,
      "replicasAvailable": 1
    }
  ],
  "state": "Available"
}
```

无效的 json/yaml 的模型示例：

```bash
>kubectl get sdep seldon-model -o json | jq .status
{
  "description": "Cannot find field: imagename in message k8s.io.api.core.v1.Container",
  "state": "Failed"
}
```

## 检查所有 SeldonDeployment 事件

```bash
kubectl describe sdep mysdep
```

这将显示 operator 所有的创建、更新、删除和错误时间的时间信息。

## Seldon Deployment 一直停留在「creating」状态

检查 pods 是否运行成功。

## 访问 Ambassador 节点返回404

模型运行中，
并使用 Ambassador 作为 ingress 有问题时，
请参阅 [Ambassador 文档](https://www.getambassador.io/docs/edge-stack/latest/topics/running/debugging/)。
你可以找到模型
执行的正确 URL 路径。

如果 ambassador 未运行，查看 pod 日志： `kubectl logs <pod_name>`。
注意 ambassador 是否以集群范围机型安装，
其 rbac 设置不被限定在特定命名空间，否则会出现权限错误。

## 通过 API 请求模型返回 500 错误

查看运行模型的 pod 日志

## Seldon Deployment 未列出

查看 Seldon Operator 日志。
他是处理 Seldon Deployment graphs 发送到 Kubernetes 集群的 pod。
默认安装下
你可以在 `seldon-system` 中找到 operator pod。
他被打成 `control-plane=seldon-controller-manager` 标签，
通过如下命令查看日志：

```bash
kubectl logs -n seldon-system -l control-plane=seldon-controller-manager
```

### 错误内存地址

有时候会看到 operator 
出现如下日志错误：

```
panic: runtime error: invalid memory address or nil pointer dereference
```

这种错误通常是有空或者
异常 `SeldonDeployment` spec 值导致。
通常主要为错误的 webhook 配置导致。
可尝试通过 [重新安装 Seldon Core](./install.md)
来修复他。

## 已经进行了上述尝试，但仍有问题

- 联系 [Slack 社区](https://join.slack.com/t/seldondev/shared_invite/enQtMzA2Mzk1Mzg0NjczLTJlNjQ1NTE5Y2MzMWIwMGUzYjNmZGFjZjUxODU5Y2EyMDY0M2U3ZmRiYTBkOTRjMzZhZjA4NjJkNDkxZTA2YmU) 
- 创建一个 [Seldon Core’s Github repo] 讨论(https://github.com/SeldonIO/seldon-core/issues)。
  请务必添加上述建议的任何诊断，
  以帮助我们诊断您的问题。