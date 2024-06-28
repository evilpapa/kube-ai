.. _observability-debug-hangs:

调试挂起
===============
在 Ray Dashboard 中查看堆栈跟踪
-----------------------------------
:ref:`Ray dashboard <observability-getting-started>`  可让您通过单击活动工作进程、任务、参与者和作业驱动进程的“CPU 分析”或“堆栈跟踪”操作来分析 Ray 驱动程序或工作进程。

.. image:: /images/profile.png
   :align: center
   :width: 80%

单击“Stack Trace”将使用 ``py-spy``返回当前堆栈跟踪示例. 。默认情况下，仅显示 Python 堆栈跟踪。要显示本机代码帧，请设置 URL 参数 ``native=1`` （仅在 Linux 上支持）。

.. image:: /images/stack.png
   :align: center
   :width: 60%

.. note::
   在 docker 容器中使用 py-spy 时，您可能会遇到权限错误。要解决此问题：
   
   * 如果您在 Docker 容器中手动启动 Ray，请按照 `py-spy 文档`_ 来解决。
   * 如果您是 KubeRay 用户，请按照 :ref:`KubeRay 配置指南 <kuberay-pyspy-integration>` 解决该问题。
   
.. _`py-spy 文档`: https://github.com/benfred/py-spy#how-do-i-run-py-spy-in-docker


使用 ``ray stack`` CLI 命令
------------------------------

安装完 ``py-spy`` 后（当 :ref:`安装 Ray <installation>` 的 “Ray Dashboard” 组件之后，也会自动的安装进来），你可以执行 ``ray stack`` 转储当前节点上所有 Ray Worker 进程的堆栈跟踪。

本文档讨论了人们在使用 Ray 时遇到的一些常见问题以及一些已知问题。如果您遇到其他问题，请 `告知我们`_ 。

.. _`let us know`: https://github.com/ray-project/ray/issues
