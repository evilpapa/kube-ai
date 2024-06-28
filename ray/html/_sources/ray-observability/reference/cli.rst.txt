.. _state-api-cli-ref:

状态 CLI
=========

状态
-----
本节包含用于访问 :ref:`在线 Ray 资源 (actor, task, object, 等) <state-api-overview-ref>` 的命令。

.. note::

    API 是 :ref:`alpha 版本 <api-stability-alpha>`。 这些特性需要使用 ``pip install "ray[default]"`` 完整安装。此功能还需要仪表板组件可用。启动 ray 集群时需要包含仪表板组件，这是 ``ray start`` 和 ``ray.init()`` 的默认行为。要进行更深入的调试，您可以在仪表盘日志 ``<RAY_LOG_DIR>/dashboard.log`` 检查，通常为 ``/tmp/ray/session_latest/logs/dashboard.log``。

状态 CLI 允许用户访问各种资源（例如 actor、任务、对象）的状态。

.. click:: ray.util.state.state_cli:task_summary
   :prog: ray summary tasks

.. click:: ray.util.state.state_cli:actor_summary
   :prog: ray summary actors

.. click:: ray.util.state.state_cli:object_summary
   :prog: ray summary objects

.. click:: ray.util.state.state_cli:ray_list
   :prog: ray list

.. click:: ray.util.state.state_cli:ray_get
   :prog: ray get

.. _ray-logs-api-cli-ref:

日志
---
本节包含 Ray 集群 to :ref:`日志接入 <state-api-log-doc>` 的相关命令。

.. note::

    API 是 :ref:`alpha 版本 <api-stability-alpha>`。特性需要 Ray 使用 ``pip install "ray[default]"`` 进行完整安装。

日志 CLI 允许用户访问集群中的日志。
注意，只有活动节点的日志可以通过此 API 访问。

.. click:: ray.util.state.state_cli:logs_state_cli_group
   :prog: ray logs