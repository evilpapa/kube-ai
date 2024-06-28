# Seldon-core 基准和负载测试

此页面正在提供基准和负载测试。

这项工作正在进行中，我们欢迎您的反馈。

## 工具

 * 针对 REST 测试我们使用 [vegeta](https://github.com/tsenart/vegeta)
 * 针对 gRPC 测试我们使用 [ghz](https://ghz.sh/)

## 服务编排

这些基准测试旨在评估包括服务编排器在内的额外延迟。

 * [服务编排基准测试](../examples/bench_svcOrch.html)

### 结果

在一个3节点的 DigitalOcean 集群 24vCPUs 96 GB，运行着 Tensorflow 花朵图像分类器。

| 测试 | 额外延迟 |
| ---  | ------------------ |
| REST | 9ms |
| gRPC | 4ms |

进一步工作：

 * 统计置信测试


## Tensorflow

测试最大吞吐量和 HPA 使用情况。

 * [Tensorflow 基准测试](../examples/bench_tensorflow.html)

### 结果

在一个3节点的 DigitalOcean 集群 24vCPUs 96 GB, 使用 HPA 运行着 Tensorflow Flowers 花朵图像分类器并以单个型号的最大吞吐量运行。 No ramp up, as vegeta does not support this. 查看 notebook 明细。

```
Latencies:

mean: 259.990239 ms
50th: 131.917169 ms
90th: 310.053255 ms
95th: 916.684759 ms
99th: 2775.05271 ms

Throughput: 23.997572337989126/s
Errors: False
```

## 基于 Argo 工作流的灵活的基准测试

我们还有一个示例，说明如何利用我们在示例中展示的批量处理工作流，但如何利用 Seldon Core 模型执行基准测试。

 * [Seldon deployment 基准测试](../examples/vegeta_bench_argo_workflows.html)

