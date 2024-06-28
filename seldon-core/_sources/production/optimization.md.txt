# 推理优化

推理优化是一个复杂的主题，取决于您的模型和用例。本页提供了各种建议。


## 自定义 Python 代码优化

使用 Seldon python 封装器，需要查看各种优化领域。

### 带有 REST 和 gRPC 的 Seldon 协议有效负载类型

据您是要使用 REST 还是 gRPC 并要发送张量数据，请求的格式将在 python 封装器中具有反序列化/序列化成本。这在 [python 序列化笔记](../examples/python_serialization.html)提供了调查。

结论是：

  * gRPC 比 REST 快
  * tftensor 最适合大批量
  * 带有 gRPC 的 ndarray 不适合大批量
  * 更简单的 tensor/ndarray 更适合小批量

### KMP_AFFINITY

如果您在具有兼容库的 Intel CPU 上运行推理，那么正确使用 KMP 和 OMP 的环境变量会很有用。大多数关于这些主题的建议通常讨论单个推理请求以及如何优化低延迟。
当您希望处理并行推理请求时，在使用 KMP_AFFINITY 时必须小心，因为如果使用 CPU Affinity，它们可能会以意想不到的方式阻塞。我们提供了一个[示例基准测试笔记本](../examples/python_kmp_affinity.html)。

有许多资源可以为您的模型案例进行更深入的循环。我们发现的一些是：

   * [最大限度地提高 CPU 上的 TensorFlow 性能：推理工作负载的注意事项和建议](https://software.intel.com/content/www/us/en/develop/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html)
   * [KMP_AFFINITY 上的 Tensorflow 问题](https://github.com/tensorflow/tensorflow/issues/29354)
   * [使用基于英特尔® 至强® 处理器的高性能计算基础设施的 TensorFlow* 扩展深度学习训练和推理的最佳实践](https://indico.cern.ch/event/762142/sessions/290684/attachments/1752969/2841011/TensorFlow_Best_Practices_Intel_Xeon_AI-HPC_v1.0_Q3_2018.pdf)
   * [使用 ONNX 运行时默认执行提供程序为 Intel CPU 内核优化 BERT 模型](https://cloudblogs.microsoft.com/opensource/2021/03/01/optimizing-bert-model-for-intel-cpu-cores-using-onnx-runtime-default-execution-provider/)
   * [使用英特尔 OpenMP 线程亲和性进行固定](https://www.nas.nasa.gov/hecc/support/kb/using-intel-openmp-thread-affinity-for-pinning_285.html)
   * [考虑为容器化部署调整 OMP_NUM_THREADS 环境变量](https://thoth-station.ninja/j/mkl_threads.html)
   * [AWS 深度学习容器](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-guide.pdf.pdf)
   * [面向 TensorFlow 的英特尔® 优化的一般最佳实践](https://github.com/IntelAI/models/blob/master/docs/general/tensorflow/GeneralBestPractices.md)

### gRPC 多处理

从 Seldon Core 1.10.0 版本开始，python 封装器 gRPC 服务器也将尊重 GUNICORN_NUM_WORKERS 并能够处理并行 gRPC 请求。

## 基准

我们提供各种[基准测试笔记本](../examples/notebooks.html#benchmarking-and-load-tests)的链接。

