## Spartakus 使用报告

开发过程的一个重要部分是更好地了解应用程序将运行的真实用户环境。

我们提供了一个选项，使用 K8s 项目提供的一个名为 [Spartakus](https://github.com/kubernetes-incubator/spartakus) 匿名指标收集工具。

### 开启使用报告

为了帮助 seldon-core 的开发支持，任何时候都可通过「seldon-core-operator」 helm chart 设置 `–set usageMetrics.enabled=true` 选项来自愿开启使用报告回馈。

```bash
helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true
```

报告的信息是匿名的，只包含有关集群中每个节点的一些信息，包括操作系统版本、kubelet 版本、docker 版本以及 CPU 和内存容量信息。

一个上报信息的示例：
```json
{
    "clusterID": "846db7e9-c861-43d7-8d08-31578af59878",
    "extensions": [
        {
            "name": "seldon-core-version",
            "value": "0.1.5"
        }
    ],
    "masterVersion": "v1.9.3-gke.0",
    "nodes": [
        {
            "architecture": "amd64",
            "capacity": [
                {
                    "resource": "cpu",
                    "value": "4"
                },
                {
                    "resource": "memory",
                    "value": "15405960Ki"
                },
                {
                    "resource": "pods",
                    "value": "110"
                }
            ],
            "cloudProvider": "gce",
            "containerRuntimeVersion": "docker://17.3.2",
            "id": "33082e677f61a199c195553e52bbd65a",
            "kernelVersion": "4.4.111+",
            "kubeletVersion": "v1.9.3-gke.0",
            "operatingSystem": "linux",
            "osImage": "Container-Optimized OS from Google"
        }
    ],
    "timestamp": "1522059083",
    "version": "v1.0.0-5d3377f6946c3ce9159cc9e7589cfbf1de26e0df"
}
```

### 关闭使用报告

正常「seldon-core-operator」使用时，数据报告默认是关闭的。

```bash
helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts
```