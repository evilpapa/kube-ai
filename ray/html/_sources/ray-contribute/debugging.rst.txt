Ray 开发者调试
============================

本调试指引是专为 Ray 项目的贡献者准备的。

开始进程调试
--------------------------------
当进程崩溃时，通常有必要在调试器中启动它们。
Ray 目前允许在以下调试器中启动进程：

- valgrind
- valgrind 分析器
- perftools 分析器
- gdb
- tmux

要使用以上工具，请确保您的机器上已经安装了这些
工具（已知 MacOS 上的 gdb 和 valgrind 存在问题）。
之后，您可以通过添加以下环境
变量 ``RAY_{PROCESS_NAME}_{DEBUGGER}=1`` 启动 ray 进程：
例如，如果你想在 ``valgrind`` 中启动 raylet，
你只需要设置环境变量 ``RAY_RAYLET_VALGRIND=1``。

要在 ``gdb`` 中启动进程，进程也必须在 ``tmux`` 中启动。
所以你想在 ``gdb`` 中启动 raylet，你需要
在你的 Python 脚本中添加以下内容：

.. code-block:: bash

 RAY_RAYLET_GDB=1 RAY_RAYLET_TMUX=1 python

你可以使用 ``tmux ls`` 列出 ``tmux`` 会话，然后附加到适当的会话。

你也可以获取 ``raylet`` 进程的核心转储，
它在提交 `issues`_ 时特别有用。
获取核心转储的过程是特定于操作系统的，
但通常涉及在启动 Ray 之前运行 ``ulimit -c unlimited`` 以允许写入核心转储文件。

.. _backend-logging:

后端日志
---------------
``raylet`` 进程日志记录有关事件的详细信息，
例如任务执行和节点之间的对象传输。
要在运行时设置日志级别，您可以在启动 Ray 之前设置 ``RAY_BACKEND_LOG_LEVEL`` 环境变量。
例如，您可以执行：

.. code-block:: shell

 export RAY_BACKEND_LOG_LEVEL=debug
 ray start

这将把源代码中的任何 ``RAY_LOG(DEBUG)`` 行打印到 ``raylet.err`` 文件中，
您可以在 :ref:`temp-dir-log-files` 中找到。

.. code-block:: shell

  logging.cc:270: Set ray log level from environment variable RAY_BACKEND_LOG_LEVEL to -1

(-1 is defined as RayLogLevel::DEBUG in logging.h.)

.. literalinclude:: /../../src/ray/util/logging.h
  :language: C
  :lines: 52,54

后端事件统计
-------------------
如果设置了 ``RAY_event_stats=1`` 环境变量，``raylet`` 进程还会定期将事件统计信息转储到 ``debug_state.txt`` 日志文件中。
要定期将统计信息打印到日志文件中，您还可以设置 ``RAY_event_stats_print_interval_ms=1000``。

事件统计包括 ASIO 事件处理程序、周期性计时器和 RPC 处理程序。
以下是事件统计的示例：

.. code-block:: shell

  Event stats:
  Global stats: 739128 total (27 active)
  Queueing time: mean = 47.402 ms, max = 1372.219 s, min = -0.000 s, total = 35035.892 s
  Execution time:  mean = 36.943 us, total = 27.306 s
  Handler stats:
    ClientConnection.async_read.ReadBufferAsync - 241173 total (19 active), CPU time: mean = 9.999 us, total = 2.411 s
    ObjectManager.ObjectAdded - 61215 total (0 active), CPU time: mean = 43.953 us, total = 2.691 s
    CoreWorkerService.grpc_client.AddObjectLocationOwner - 61204 total (0 active), CPU time: mean = 3.860 us, total = 236.231 ms
    CoreWorkerService.grpc_client.GetObjectLocationsOwner - 51333 total (0 active), CPU time: mean = 25.166 us, total = 1.292 s
    ObjectManager.ObjectDeleted - 43188 total (0 active), CPU time: mean = 26.017 us, total = 1.124 s
    CoreWorkerService.grpc_client.RemoveObjectLocationOwner - 43177 total (0 active), CPU time: mean = 2.368 us, total = 102.252 ms
    NodeManagerService.grpc_server.PinObjectIDs - 40000 total (0 active), CPU time: mean = 194.860 us, total = 7.794 s

回调延迟注入
--------------------------
有时，错误是由 RPC 问题引起的，例如，由于某些请求的延迟，系统陷入了死锁。
要调试和重现这种问题，我们需要一种方法来为 RPC 请求注入延迟。为了启用这个功能，引入了 ``RAY_testing_asio_delay_us``。
如果你想让某些 RPC 请求的回调在一段时间后执行，你可以使用这个变量。例如：

.. code-block:: shell

  RAY_testing_asio_delay_us="NodeManagerService.grpc_client.PrepareBundleResources=2000000:2000000" ray start --head


该语法是 ``RAY_testing_asio_delay_us="method1=min_us:max_us,method2=min_us:max_us"``。条目是逗号分隔的。
有一个特殊的方法 ``*``，表示所有方法。它与其他条目相比具有较低的优先级。

.. _`issues`: https://github.com/ray-project/ray/issues
