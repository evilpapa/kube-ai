.. _ref-usage-stats:

使用统计收集
======================

从 Ray 1.13 开始，Ray 默认收集使用统计数据（由选择退出提示保护）。
开源 Ray 工程团队将使用这些数据来更好地了解如何改进我们的库和核心 API，以及如何确定错误修复和增强功能的优先级。

以下是我们收集政策的指导原则：

- **不会有意外** — 我们开始收集数据之前会通知您。如果收集的数据或数据使用方式有任何变化，您都会收到通知。
- **轻松选择退出:** 您可以轻松选择退出数据收集
- **透明度** — 您将能够查看发送给我们的所有数据
- **控制** — 您将控制您的数据，并且我们将尊重您删除数据的请求。
- 我们 **不会** 收集任何个人身份信息或专有代码/数据
- 我们 **不会** 出售或购买您的数据。

您将始终能够 :ref:`禁用使用情况统计信息收集 <usage-disable>`。

有关更多背景信息，请参阅此 `RFC <https://github.com/ray-project/ray/issues/20857>`_ 。

收集哪些数据？
-----------------------

我们收集非敏感数据，以帮助我们了解 Ray 的使用方式（例如，使用了哪些 Ray 库）。
**个人身份数据绝不会收集。** 请查看 UsageStatsToReport 类以查看我们收集的数据。

.. _usage-disable:

如何禁用
-----------------
在启动集群之前，有多种方法可以禁用使用情况统计信息收集：

#. Add ``--disable-usage-stats`` option to the command that starts the Ray cluster (e.g., ``ray start --head --disable-usage-stats`` :ref:`command <ray-start-doc>`).

#. Run :ref:`ray disable-usage-stats <ray-disable-usage-stats-doc>` to disable collection for all future clusters. This won't affect currently running clusters. Under the hood, this command writes ``{"usage_stats": true}`` to the global config file ``~/.ray/config.json``.

#. Set the environment variable ``RAY_USAGE_STATS_ENABLED`` to 0 (e.g., ``RAY_USAGE_STATS_ENABLED=0 ray start --head`` :ref:`command <ray-start-doc>`).

#. If you're using `KubeRay <https://github.com/ray-project/kuberay/>`_, you can add ``disable-usage-stats: 'true'`` to ``.spec.[headGroupSpec|workerGroupSpecs].rayStartParams.``.

Currently there is no way to enable or disable collection for a running cluster; you have to stop and restart the cluster.


它是如何工作的？
-----------------

When a Ray cluster is started via :ref:`ray start --head <ray-start-doc>`, :ref:`ray up <ray-up-doc>`, :ref:`ray submit --start <ray-submit-doc>` or :ref:`ray exec --start <ray-exec-doc>`,
Ray will decide whether usage stats collection should be enabled or not by considering the following factors in order:

#. It checks whether the environment variable ``RAY_USAGE_STATS_ENABLED`` is set: 1 means enabled and 0 means disabled.

#. If the environment variable is not set, it reads the value of key ``usage_stats`` in the global config file ``~/.ray/config.json``: true means enabled and false means disabled.

#. If neither is set and the console is interactive, then the user will be prompted to enable or disable the collection. If the console is non-interactive, usage stats collection will be enabled by default. The decision will be saved to ``~/.ray/config.json``, so the prompt is only shown once.

Note: usage stats collection is not enabled when using local dev clusters started via ``ray.init()`` unless it's a nightly wheel. This means that Ray will never collect data from third-party library users not using Ray directly.

If usage stats collection is enabled, a background process on the head node will collect the usage stats
and report to ``https://usage-stats.ray.io/`` every hour. The reported usage stats will also be saved to
``/tmp/ray/session_xxx/usage_stats.json`` on the head node for inspection. You can check the existence of this file to see if collection is enabled.

Usage stats collection is very lightweight and should have no impact on your workload in any way.

请求删除收集的数据
------------------------------------

To request removal of collected data, please email us at ``usage_stats@ray.io`` with the ``session_id`` that you can find in ``/tmp/ray/session_xxx/usage_stats.json``.

常见问题 (FAQ)
--------------------------------

**Does the session_id map to personal data?**

No, the uuid will be a Ray session/job-specific random ID that cannot be used to identify a specific person nor machine. It will not live beyond the lifetime of your Ray session; and is primarily captured to enable us to honor deletion requests.

The session_id is logged so that deletion requests can be honored.

**Could an enterprise easily configure an additional endpoint or substitute a different endpoint?**

We definitely see this use case and would love to chat with you to make this work -- email ``usage_stats@ray.io``.


联系我们
----------
如果您对使用情况统计收集有任何反馈，请发送电子邮件至 ``usage_stats@ray.io``。
