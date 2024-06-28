.. _observability-key-concepts:

关键概念
============

本节介绍 Ray 中监视和调试工具和功能的关键概念。

仪表盘 （网络用户界面）
------------------
Ray 提供基于 Web 的仪表板来帮助用户监控和调试 Ray 应用程序和集群。

请参阅 :ref:`入门 <observability-getting-started>` 查看有关仪表板的更多详细信息。


Ray 状态
----------
Ray 状态是指各种 Ray 实体（例如，Actor、Task、Object 等）的状态。 Ray 2.0及以后版本支持 :ref:`通过通过 CLI 和 Python API 查询实体的状态 <observability-programmatic>`

以下命令列出集群中的所有 Actor：

.. code-block:: bash

    ray list actors

.. code-block:: text

    ======== List: 2022-07-23 21:29:39.323925 ========
    Stats:
    ------------------------------
    Total: 2

    Table:
    ------------------------------
        ACTOR_ID                          CLASS_NAME    NAME      PID  STATE
    0  31405554844820381c2f0f8501000000  Actor                 96956  ALIVE
    1  f36758a9f8871a9ca993b1d201000000  Actor                 96955  ALIVE

查看 :ref:`使用 CLI 或 SDK 进行监控 <state-api-overview-ref>` 以了解更多详细信息。

指标
-------
Ray 收集并公开物理统计信息（例如，每个节点的 CPU、内存、GRAM、磁盘和网络使用情况）、
内部统计信息（例如，集群中的 Actor 数量、集群中的 Worker 故障数量）和自定义应用程序指标（例如，用户定义的指标）。所有统计数据都可以导出为时间序列数据（默认导出到 Prometheus），并用于随着时间的推移监控集群。

查看 :ref:`指标视图 <dash-metrics-view>` 了解在 Ray Dashboard 中查看指标的位置。 查看 :ref:`指标收集 <collect-metrics>` ，了解如何从 Ray Cluster 收集的指标。

异常
----------
创建新任务或提交 Actor 任务会生成对象引用。当 ``ray.get`` 在对象引用上调用 时，如果相关任务、Actor 或对象出现任何问题，API 会引发异常。例如，

- :class:`RayTaskError <ray.exceptions.RayTaskError>` 当用户代码的错误引发异常时引发。 
- :class:`RayActorError <ray.exceptions.RayActorError>` 当 Actor 死亡时引发（由于系统故障，例如节点故障，或用户级故障，例如 ``__init__`` 方法引起的异常）。
- :class:`RuntimeEnvSetupError <ray.exceptions.RuntimeEnvSetupError>` r当 Actor 或任务因:ref:`运行时环境 <runtime-environments>` 创建失败而无法启动时引发。

有关更多详细信息，请参阅 :ref:`异常参考 <ray-core-exceptions>` 。

调试器
--------
Ray 有一个内置的调试器，用于调试分布式应用程序。
在 Ray Tasks 和 Actors 中设置断点，
当遇到断点时，进入 PDB 会话以：

- 检查该上下文中的变量
- 进入任务或 Actor
- 堆栈中向上或向下移动

查看 :ref:`Ray Debugger <ray-debugger>` 了解更多详细信息。

.. _profiling-concept:

分析
---------
分析是通过对应用程序的资源使用情况进行采样来分析应用程序性能的方法。 Ray 支持各种分析工具：

- 驱动程序和工作进程的 CPU 分析，包括与 :ref:`py-spy <profiling-pyspy>` 和 :ref:`cProfile <profiling-cprofile>`
- 使用 :ref:`memray <profiling-memray>` 对驱动程序和工作进程进行内存分析
- 使用 :ref:`Pytorch Profiler <profiling-pytoch-profiler>` 进行 GPU 分析
- 内置 Task 和 Actor 分析工具 :ref:`Ray Timeline <profiling-timeline>`

查看 :ref:`分析 <profiling>` 以了解更多详细信息。请注意，此列表并不全面，如果您发现其他有用的工具，请随时为其做出贡献。


追踪
-------
为了帮助调试和监控 Ray 应用程序，Ray 支持跨任务和 Actor 的分布式跟踪（与 OpenTelemetry 集成）。

有关更多详细信息，请参阅 :ref:`Ray 追踪 <ray-tracing>`。

应用程序日志
----------------
日志对于一般监控和调试很重要。对于分布式 Ray 应用程序来说，日志更加重要，但同时也更加复杂。 Ray应用程序同时运行在Driver和Worker进程上（甚至跨多台机器），这些进程的日志是应用程序日志的主要来源。

.. image:: ./images/application-logging.png
    :alt: Application logging

驱动程序日志
~~~~~~~~~~~
Ray 应用程序调用 ``ray.init()`` 的入口点成为 **Driver**。
所有 driver 程序日志的处理方式与普通 Python 程序相同。

.. _ray-worker-logs:

Worker l日志（stdout 和 stderr）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray 在 Ray 的工作进程中远程执行任务或 actor 。任务和 Actor 日志在 Worker stdout 和 stderr 中捕获。

Ray 有专门的支持来提高 Worker 进程产生的 stdout 和 stderr 的可见性，使 Ray 程序看起来像一个非分布式程序，也称为“Worker 日志重定向到驱动程序”。

- Ray 将所有任务和 Actor 中的 stdout 和 stderr 定向到 Worker 日志文件，包括 Worker 生成的任何日志消息。请参阅 :ref:`日志记录目录和文件结构 <logging-directory-structure>` 以了解 Ray 日志记录结构。
- Driver 读取 Worker 日志文件（所有任务和 Actor 的 stdout 和 stderr 所在的位置）并将日志记录发送到其自己的 stdout 和 stderr（也称为“将 Worker 日志重定向到 Driver 输出”）。

对于以下代码：

.. testcode::

    import ray
    # Initiate a driver.
    ray.init()

    @ray.remote
    def task_foo():
        print("task!")

    ray.get(task_foo.remote())

.. testoutput::
    :options: +MOCK

    (task_foo pid=12854) task!

#. Ray 任务 ``task_foo`` 在 Ray Worker 进程上运行。字符串 ``task!`` 被保存到相应的Workers ``stdout`` 日志文件中。
#. 驱动程序读取 Worker 日志文件并将其发送到其 ``stdout``（终端），您应该能够在其中看到字符串 ``task!``。

打印日志时，会同时打印进程 ID (pid) 和执行任务或 Actor 的节点的 IP 地址。这是输出：

.. code-block:: bash

    (pid=45601) task!

默认情况下，Actor 日志消息如下所示：

.. code-block:: bash

    (MyActor pid=480956) actor log message


默认情况下，Tasks 和 Actors 的所有 stdout 和 stderr 都被重定向到 Driver 输出。查看 :ref:`配置日志记录 <log-redirection-to-driver>` 以了解如何禁用此功能。



Job 日志
~~~~~~~~
Ray 应用程序通常作为 Ray 作业运行。 Ray 作业的工作日志始终捕获在:ref:`ay 日志记录目录 <logging-directory-structure>` ，而驱动程序日志则不然。

只有通过 :ref:`Jobs API <jobs-quickstart>` 提交的日志才能捕获 Driver 日志。.使用仪表板 UI、CLI（使用 ``ray job logs`` :ref:`CLI command <ray-job-logs-doc>`）或 the :ref:`Python SDK <ray-job-submission-sdk-ref>` （``JobSubmissionClient.get_logs()`` 或 ``JobSubmissionClient.tail_job_logs()`` 查找捕获的驱动程序日志。

.. note::
   如果您通过直接在头节点上执行 Ray 驱动程序或通过 Ray 客户端连接来运行 Ray 作业，请查看终端或 Jupyter Notebook 中的驱动程序日志。
