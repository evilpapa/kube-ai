.. _state-api-ref:

状态 API
=========

.. note::

    API 是 :ref:`alpha <api-stability-alpha>` 版本。 此功能需要使用 ``pip install "ray[default]"`` 完整安装 Ray 。

有关示例的概述，请参阅 :ref:`监控 Ray 状态 <state-api-overview-ref>`。

有关 CLI 参考，请参阅 :ref:`Ray State CLI 参考 <state-api-cli-ref>` 或者 :ref:`Ray 日志 CLI 参考 <ray-logs-api-cli-ref>`。

状态 Python SDK
-----------------

状态 API 也导出为函数。

API 摘要
~~~~~~~~~~~~

.. autosummary::
   :toctree: doc/

    ray.util.state.summarize_actors
    ray.util.state.summarize_objects
    ray.util.state.summarize_tasks

API 列表
~~~~~~~~~~

.. autosummary::
   :toctree: doc/

    ray.util.state.list_actors
    ray.util.state.list_placement_groups
    ray.util.state.list_nodes
    ray.util.state.list_jobs
    ray.util.state.list_workers
    ray.util.state.list_tasks
    ray.util.state.list_objects
    ray.util.state.list_runtime_envs

获取 API
~~~~~~~~~

.. autosummary::
   :toctree: doc/

    ray.util.state.get_actor
    ray.util.state.get_placement_group
    ray.util.state.get_node
    ray.util.state.get_worker
    ray.util.state.get_task
    ray.util.state.get_objects

日志 API
~~~~~~~~

.. autosummary::
   :toctree: doc/

    ray.util.state.list_logs
    ray.util.state.get_log

.. _state-api-schema:

状态 APIs 架构
-----------------

.. autosummary::
   :toctree: doc/
   :template: autosummary/class_without_autosummary.rst

    ray.util.state.common.ActorState
    ray.util.state.common.TaskState
    ray.util.state.common.NodeState
    ray.util.state.common.PlacementGroupState
    ray.util.state.common.WorkerState
    ray.util.state.common.ObjectState
    ray.util.state.common.RuntimeEnvState
    ray.util.state.common.JobState
    ray.util.state.common.StateSummary
    ray.util.state.common.TaskSummaries
    ray.util.state.common.TaskSummaryPerFuncOrClassName
    ray.util.state.common.ActorSummaries
    ray.util.state.common.ActorSummaryPerClass
    ray.util.state.common.ObjectSummaries
    ray.util.state.common.ObjectSummaryPerKey

状态异常 API 
---------------------

.. autosummary::
   :toctree: doc/

    ray.util.state.exception.RayStateApiException
