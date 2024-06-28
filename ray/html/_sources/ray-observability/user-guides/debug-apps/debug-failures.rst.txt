.. _observability-debug-failures:

调试失败
==================

Ray 存在哪些类型的故障？
-----------------------------------

Ray 由两个主要 API 组成。 ``.remote()`` 来创建任务或 Actor，以及 :func:`ray.get <ray.get>` 获取结果。
调试 Ray 意味着识别并修复运行通过 ``.remote`` API 创建的函数和类（任务和 Actor）的远程进程的故障

Ray API 是未来的 API（事实上，可以 :ref:`将 Ray 对象引用转换为标准的 Python 未来 API <async-ref-to-futures>`），
并且错误处理模型是相同的。 当任何远程任务或 Actors失败时，返回的对象引用包含异常。
当您调用对象引用的 ``get`` API 时，它会引发异常。

.. testcode::

  import ray
  @ray.remote
  def f():
      raise ValueError("it's an application error")

  # Raises a ValueError.
  try:
    ray.get(f.remote())
  except ValueError as e:
    print(e)

.. testoutput::

  ...
  ValueError: it's an application error

在 Ray 中，失败分为三种类型。有关更多详细信息，请参阅异常 API。

- **应用程序失败**: 这意味着用户代码的远程 task/actor 失败。这种情况下， ``get`` API 会引发 :func:`RayTaskError <ray.exceptions.RayTaskError>` 包括从远程进程引发的异常。
- **内部系统故障**: 这是 Ray 发生了故障，但故障是故意的。例如，当你调用取消 API 如 ``ray.cancel`` （针对任务）或 ``ray.kill`` （针对 actor ），系统会使远程 任务 / actor 失败，但是是故意的。
- **意外系统故障**:这意味着远程任务 / actor 由于意外的系统故障而失败如进程崩溃（像是内存不足错误）或节点失败。

  1. `Linux OOM killer <https://www.kernel.org/doc/gorman/html/understand/understand016.html>`_ 或 :ref:`Ray Memory Monitor <ray-oom-monitor>` 会杀死内存使用率高的进程以防止 OOM。
  2. 机器关机（如，现场实例终止）或 :term:`raylet <raylet>` 崩溃（如，意外故障）。
  3. 系统高负载或高压力（机器或者系统组件如 Raylet 或 :term:`GCS <GCS / Global Control Service>`），这让系统不稳定或发生故障。

调试应用故障。
------------------------------

Ray 将用户的代码分发到多台机器上的多个进程。应用程序故障意味着用户代码中的错误。
Ray 提供了类似于调试单进程 Python 程序的调试体验。

print
~~~~~

``print`` 调试是调试Python程序最常用的方法之一。
:ref:`Ray 的 Task 和 Actor 日志默认打印到 Ray Driver <ray-worker-logs>` 上，
这使您可以简单地使用 ``print`` 来调试应用程序故障。

调试器
~~~~~~~~

许多Python开发人员使用调试器来调试Python程序，而 `Python pdb <https://docs.python.org/3/library/pdb.html>`_) 是流行的选择之一。
Ray 原生支持 ``pdb``。你可以简单的添加 ``breakpoint()`` 到 Actor 和 Task 代码来启用 ``pdb``。参考 :ref:`Ray Debugger <ray-debugger>` 了解更多信息。


文件描述符用尽 (``Too may open files``)
--------------------------------------------------------

在 Ray 集群中，任意两个系统组件可以相互通信并建立 1 个或多个连接。
例如，某些 worker 可能需要与 GCS 通信来调度 Actor（worker <-> GCS连接）。
您的驱动程序可以调用 Actor 方法（worker <->worker 连接）。

Ray 可以支持数千个 raylet 和数千个工作进程。当 Ray 集群变得更大时，‘
每个组件可以拥有越来越多的网络连接，这需要文件描述符。

Linux 通常将每个进程的默认文件描述符限制为 1024。
当与组件的连接超过 1024 个时，它可能会引发以下错误消息。

.. code-block:: bash

  Too may open files

对于头节点 GCS 进程来说尤其常见，因为它是 Ray 中许多其他组件与之通信的集中组件。
当您看到此错误消息时，我们建议您通过 ``ulimit`` 命令调整每个进程的最大文件描述符限制

我们建议您应用 ``ulimit -n 65536`` 到您的主机配置。但是，您也可以有选择地将其应用于 Ray 组件（查看下面的示例）。
通常，每个 worker 有 2~3 个与 GCS 的连接。每个 raylet 有 1~2 个到 GCS 的连接。
65536 个文件描述符可以处理 10000~15000 个 worker 和 1000~2000 个节点。
如果您有更多 worker ，则应考虑使用高于 65536 的数字。

.. code-block:: bash

  # Start head node components with higher ulimit.
  ulimit -n 65536 ray start --head

  # Start worker node components with higher ulimit.
  ulimit -n 65536 ray start --address <head_node>

  # Start a Ray driver with higher ulimit.
  ulimit -n 65536 <python script>

如果失败，请通过运行 ``ulimit -Hn`` 来仔细检查硬限制是否足够大。
如果太小，您可以按如下方式增加硬限制（这些说明适用于 EC2）。

* 通过运行以下命令增加系统范围内打开文件描述符的硬 ulimit。

  .. code-block:: bash

    sudo bash -c "echo $USER hard nofile 65536 >> /etc/security/limits.conf"

* 注销并重新登录。


内存导致的失败问题
--------------------------------
惨绝 :ref:`调试内存问题 <ray-core-mem-profiling>` 获取信息。


本文档讨论了人们在使用 Ray 时遇到的一些常见问题以及一些已知问题。如果您遇到其他问题， `请告诉我们`_ 。

.. _`let us know`: https://github.com/ray-project/ray/issues