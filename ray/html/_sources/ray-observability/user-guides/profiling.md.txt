(profiling)=
# 分析
分析是诊断性能、内存不足、挂起或其他应用程序问题的最重要的调试工具之一。
以下是调试 Ray 应用程序时可以使用的常见分析工具的列表。
- CPU 分析
    - py-spy
- 内存分析
    - memray
- GPU 分析
    - PyTorch 分析器
- Ray Task / Actor 时间线

如果 Ray 无法与某些分析工具配合使用，请尝试在没有 Ray 的情况下运行它们来调试问题。

(profiling-cpu)=
## CPU 分析
分析驱动程序和工作进程的 CPU 使用情况。这可以帮助您了解不同进程的 CPU 使用情况并调试意外的高或低使用率。

(profiling-pyspy)=
### py-spy
[py-spy](https://github.com/benfred/py-spy/tree/master) 是 Python 程序的采样分析器。 Ray Dashboard 与 pyspy 具有原生集成：

- 可以让您直观地看到 Python 程序花费时间的情况，而无需重新启动程序或以任何方式修改代码。
- 它转储正在运行的进程的堆栈跟踪，以便您可以看到该进程在某个时间正在做什么。当程序挂起时它很有用。

:::{note}
在 docker 容器中使用 py-spy 时，您可能会遇到权限错误。要解决该问题：

- 如果在Docker容器中手动启动Ray，请按照 `py-spy 文档`_ 解决。
- 如果您是 KubeRay 用户，请按照 :ref:`KubeRay 配置指引 <kuberay-pyspy-integration>` 解决。
:::

以下是 {ref}`将 py-spy 与 Ray 和 Ray Dashboard 一起使用的步骤 <observability-debug-hangs>`。

(profiling-cprofile)=
### cProfile
cProfile 是 Python 的原生分析模块，用于分析 Ray 应用程序的性能。

以下是 {ref}`cProfile 的使用步骤 <dashboard-cprofile>`。

(profiling-memory)=
## 内存分析
分析驱动程序和工作进程的内存使用情况。这可以帮助您分析应用程序中的内存分配、跟踪内存泄漏以及调试高/低内存或内存不足问题。

(profiling-memray)=
### memray
memray 是 Python 的内存分析器。它可以跟踪 Python 代码、原生扩展模块和 Python 解释器本身中的内存分配。

以下是 {ref}`分析 Ray Tasks 和 Actors 内存使用情况的步骤 <memray-profiling>`。

(profiling-gpu)=
## GPU 分析
针对 GPU 工作负载（例如分布式训练）的 GPU 和 GRAM 分析。这可以帮助您分析性能和调试内存问题。
- 与 Ray Train 一起使用时，PyTorch 分析器受到开箱即用的支持
- NVIDIA Nsight 系统尚未原生支持。在此功能请求中留下您的评论 [以获取 Nisght 系统支持](https://github.com/ray-project/ray/issues/19631)。

(profiling-pytoch-profiler)=
### PyTorch 分析器
PyTorch Profiler 是一个允许在训练和推理期间收集性能指标（尤其是 GPU 指标）的工具。

以下是 {ref}`将 PyTorch Profiler 与 Ray Train 或 Ray Data 结合使用的步骤 <performance-debugging-gpu-profiling>`。

(profiling-timeline)=
## Ray Task / Actor 时间线
Ray 时间线描述了 Ray 任务和 Actor 的执行时间。这可以帮助您分析性能、识别掉队者并了解工作负载的分布。

在 Ray Dashboard 中打开 Ray 作业，然后按照 {ref}`下载并可视化 <dashboard-timeline>` Ray Timeline 生成的跟踪文件。
