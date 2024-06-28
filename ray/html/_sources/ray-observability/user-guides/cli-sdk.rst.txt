.. _observability-programmatic:

使用 CLI 或 SDK 进行监控
===============================

Ray 中的监控和调试功能可通过 CLI 或 SDK 获得。


CLI 命令 ``ray status`` 
----------------------------
您可以通过在头节点上运行 CLI 命令 ``ray status``来监控节点状态和资源使用情况。它显示了

- **节点状态**: 正在运行并自动向上或向下扩展的节点。运行节点的地址。有关待处理节点和失败节点的信息。
- **资源使用情况**: Ray 集群的资源使用情况。例如，从所有 Ray 任务和 Actor 请求的 CPU。使用的 GPU 数量。

以下是示例输出：

.. code-block:: shell

   $ ray status
   ======== Autoscaler status: 2021-10-12 13:10:21.035674 ========
   Node status
   ---------------------------------------------------------------
   Healthy:
    1 ray.head.default
    2 ray.worker.cpu
   Pending:
    (no pending nodes)
   Recent failures:
    (no failures)

   Resources
   ---------------------------------------------------------------
   Usage:
    0.0/10.0 CPU
    0.00/70.437 GiB memory
    0.00/10.306 GiB object_store_memory

   Demands:
    (no resource demands)

当您需要有关每个节点的更多详细信息时，请运行 ``ray status -v``。 当您需要调查为什么特定节点不自动缩小规模时，这非常有用。


.. _state-api-overview-ref:

Ray 状态 CLI 和 SDK
----------------------------

.. tip:: 提供有关使用 Ray state API 的反馈 - `反馈表 <https://forms.gle/gh77mwjEskjhN8G46>`_!

使用 Ray State API 通过 CLI 或 Python SDK（开发人员 API）访问 Ray 的当前状态（快照）。

.. note::

    此功能需要使用 ``pip install "ray[default]"``. 完整安装 Ray 。此功能还要求仪表板组件可用。启动 Ray Cluster 时需要包含仪表板组件，这是 `ray start`` 和 ``ray.init()`` 的默认行为。

.. note::

    State API CLI 命令是 :ref:`stable <api-stability-stable>`，而 Python SDK 是 :ref:`DeveloperAPI <developer-api-def>`。建议使用 CLI 而不是 Python SDK。


开始
~~~~~~~~~~~

此示例使用以下脚本运行两个任务并创建两个 Actor。

.. testcode::
    :hide:

    import ray

    ray.shutdown()

.. testcode::

    import ray
    import time

    ray.init(num_cpus=4)

    @ray.remote
    def task_running_300_seconds():
        time.sleep(300)

    @ray.remote
    class Actor:
        def __init__(self):
            pass

    # Create 2 tasks
    tasks = [task_running_300_seconds.remote() for _ in range(2)]

    # Create 2 actors
    actors = [Actor.remote() for _ in range(2)]

.. testcode::
    :hide:

    # Wait for the tasks to be submitted.
    time.sleep(2)

查看任务的汇总状态。如果它没有立即返回输出，请重试该命令。

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray summary tasks

        .. code-block:: text

            ======== Tasks Summary: 2022-07-22 08:54:38.332537 ========
            Stats:
            ------------------------------------
            total_actor_scheduled: 2
            total_actor_tasks: 0
            total_tasks: 2


            Table (group by func_name):
            ------------------------------------
                FUNC_OR_CLASS_NAME        STATE_COUNTS    TYPE
            0   task_running_300_seconds  RUNNING: 2      NORMAL_TASK
            1   Actor.__init__            FINISHED: 2     ACTOR_CREATION_TASK

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import summarize_tasks
            print(summarize_tasks())

        .. testoutput::

            {'cluster': {'summary': {'task_running_300_seconds': {'func_or_class_name': 'task_running_300_seconds', 'type': 'NORMAL_TASK', 'state_counts': {'RUNNING': 2}}, 'Actor.__init__': {'func_or_class_name': 'Actor.__init__', 'type': 'ACTOR_CREATION_TASK', 'state_counts': {'FINISHED': 2}}}, 'total_tasks': 2, 'total_actor_tasks': 0, 'total_actor_scheduled': 2, 'summary_by': 'func_name'}}

列出所有 Actors。

.. tabs::

    .. group-tab:: CLI （推荐）

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

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_actors
            print(list_actors())

        .. testoutput::

            [ActorState(actor_id='...', class_name='Actor', state='ALIVE', job_id='01000000', name='', node_id='...', pid=..., ray_namespace='...', serialized_runtime_env=None, required_resources=None, death_cause=None, is_detached=None, placement_group_id=None, repr_name=None), ActorState(actor_id='...', class_name='Actor', state='ALIVE', job_id='01000000', name='', node_id='...', pid=..., ray_namespace='...', serialized_runtime_env=None, required_resources=None, death_cause=None, is_detached=None, placement_group_id=None, repr_name=None)]


使用 get API 获取单个任务的状态。

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            # In this case, 31405554844820381c2f0f8501000000
            ray get actors <ACTOR_ID>

        .. code-block:: text

            ---
            actor_id: 31405554844820381c2f0f8501000000
            class_name: Actor
            death_cause: null
            is_detached: false
            name: ''
            pid: 96956
            resource_mapping: []
            serialized_runtime_env: '{}'
            state: ALIVE

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_actor
            # In this case, 31405554844820381c2f0f8501000000
            print(get_actor(id=<ACTOR_ID>))

通过 ``ray logs``  API 访问日志。

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list actors
            # In this case, ACTOR_ID is 31405554844820381c2f0f8501000000
            ray logs actor --id <ACTOR_ID>

        .. code-block:: text

            --- Log has been truncated to last 1000 lines. Use `--tail` flag to toggle. ---

            :actor_name:Actor
            Actor created

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_log

            # In this case, ACTOR_ID is 31405554844820381c2f0f8501000000
            for line in get_log(actor_id=<ACTOR_ID>):
                print(line)

关键概念
~~~~~~~~~~~~~
Ray State API 允许您通过 **summary**、 **list** 和 **get** API访问 **资源** 的 **状态**。 它还支持 **logs** API 来访问日志。

- **states**: 对应资源的集群状态。状态由不可变元数据（例如，Actor 的名称）和可变状态（例如，Actor 的调度状态或 pid）组成。
- **resources**: Ray 创建的资源。例如，actor、任务、对象、占位组等。
- **summary**: 返回资源汇总视图的 API。
- **list**: 返回每个资源实体的 API。
- **get**: 返回单个资源实体详细信息的 API。
- **logs**: 用于访问 Actor、任务、Workers 日志或系统日志文件的 API。



用户指南
~~~~~~~~~~~~~

按类型获取实体状态的摘要
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
返回给定 Ray 实体（对象、Actor、任务）的汇总信息。
建议首先通过摘要 API 开始监控状态。
当您发现异常情况时（例如，长时间运行的Actor、长时间未调度的任务），
您可以使用 ``list`` 或 ``get`` API 来获取单个异常实体的更多详细信息。

**汇总所有 actor**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray summary actors

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import summarize_actors
            print(summarize_actors())

        .. testoutput::

            {'cluster': {'summary': {'Actor': {'class_name': 'Actor', 'state_counts': {'ALIVE': 2}}}, 'total_actors': 2, 'summary_by': 'class'}}

**汇总所有 task**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray summary tasks

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import summarize_tasks
            print(summarize_tasks())

        .. testoutput::

            {'cluster': {'summary': {'task_running_300_seconds': {'func_or_class_name': 'task_running_300_seconds', 'type': 'NORMAL_TASK', 'state_counts': {'RUNNING': 2}}, 'Actor.__init__': {'func_or_class_name': 'Actor.__init__', 'type': 'ACTOR_CREATION_TASK', 'state_counts': {'FINISHED': 2}}}, 'total_tasks': 2, 'total_actor_tasks': 0, 'total_actor_scheduled': 2, 'summary_by': 'func_name'}}

**汇总所有对象**

.. note::

    默认情况下，对象按调用点进行汇总。但是，Ray 默认情况下不记录调用点。要获取调用点信息，请在启动 Ray 集群时设置环境变量 `RAY_record_ref_creation_sites=1` 
    RAY_record_ref_creation_sites=1 ray start --head

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray summary objects

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import summarize_objects
            print(summarize_objects())

        .. testoutput::

            {'cluster': {'summary': {'disabled': {'total_objects': 6, 'total_size_mb': 0.0, 'total_num_workers': 3, 'total_num_nodes': 1, 'task_state_counts': {'SUBMITTED_TO_WORKER': 2, 'FINISHED': 2, 'NIL': 2}, 'ref_type_counts': {'LOCAL_REFERENCE': 2, 'ACTOR_HANDLE': 4}}}, 'total_objects': 6, 'total_size_mb': 0.0, 'callsite_enabled': False, 'summary_by': 'callsite'}}

有关命令的更多详细信息，请参阅 :ref:`state CLI 参考 <state-api-cli-ref>` 关于 ``ray summary`` 命令。


列出某种类型的所有实体的状态
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

获取资源列表。可能的资源包括：

- :ref:`Actors <actor-guide>` 例如 Actor ID、状态、PID、death_cause ( :class:`output schema <ray.util.state.common.ActorState>`）
- :ref:`Tasks <ray-remote-functions>` 例如名称、调度状态、类型、运行时环境信息 (:class:`output schema <ray.util.state.common.TaskState>`)
- :ref:`Objects <objects-in-ray>`, 例如对象 ID、调用点、引用类型 ( (:class:`output schema <ray.util.state.common.ObjectState>`)
- :ref:`Jobs <jobs-overview>`,例如开始/结束时间、入口点、状态 (:class:`output schema <ray.util.state.common.JobState>`)
- :ref:`Placement Groups <ray-placement-group-doc-ref>`, 例如名称、捆绑包、统计信息 (:class:`output schema <ray.util.state.common.PlacementGroupState>`)
- 节点（Ray 工作节点），例如节点 ID、节点 IP、节点状态 (:class:`output schema <ray.util.state.common.NodeState>`)
- Workers（Ray 工作进程），例如工作 ID、类型、退出类型和详细信息 (:class:`output schema <ray.util.state.common.WorkerState>`)
- :ref:`运行时环境 <runtime-environments>`, 例如运行时环境、创建时间、节点 (:class:`output schema <ray.util.state.common.RuntimeEnvState>`)

**列出所有节点**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list nodes

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_nodes
            list_nodes()

**列出所有归置组**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list placement-groups

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_placement_groups
            list_placement_groups()


**列出进程创建的本地引用对象**

.. tip:: 您可以使用一个或多个过滤器列出资源：使用 `--filter` 或 `-f`

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list objects -f pid=<PID> -f reference_type=LOCAL_REFERENCE

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_objects
            list_objects(filters=[("pid", "=", 1234), ("reference_type", "=", "LOCAL_REFERENCE")])

**列出存活的 actor**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list actors -f state=ALIVE

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_actors
            list_actors(filters=[("state", "=", "ALIVE")])

**列出运行中的 task**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list tasks -f state=RUNNING

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_tasks
            list_tasks(filters=[("state", "=", "RUNNING")])

**列出非运行 task**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list tasks -f state!=RUNNING

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_tasks
            list_tasks(filters=[("state", "!=", "RUNNING")])

**列出具有名称的正在运行的任务**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list tasks -f state=RUNNING -f name="task_running_300_seconds()"

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_tasks
            list_tasks(filters=[("state", "=", "RUNNING"), ("name", "=", "task_running_300_seconds()")])

**列出包含更多详细信息的任务**

.. tip:: 当指定 ``--detail``时，API可以查询更多数据源以获取详细的状态信息。

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray list tasks --detail

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::

            from ray.util.state import list_tasks
            list_tasks(detail=True)

有关命令的更多详细信息，请参阅 :ref:`state CLI 参考 <state-api-cli-ref>` 的 ``ray list`` 命令。


获取特定实体（任务、actor 等）的状态
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**获取任务的状态**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray get tasks <TASK_ID>

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_task
            get_task(id=<TASK_ID>)

**获取节点的状态**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray get nodes <NODE_ID>

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_node
            get_node(id=<NODE_ID>)

有关命令的更多详细信息，请参阅 :ref:`状态 CLI 参考 <state-api-cli-ref>` 的 ``ray get`` 命令。


获取特定实体（任务、actor 等）的日志
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _state-api-log-doc:

State API 还允许您访问 Ray 日志。请注意，您无法从死节点访问日志。默认情况下，API 从头节点打印日志。

**从集群中的头节点获取所有可检索的日志文件名**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray logs cluster

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            # You could get the node ID / node IP from `ray list nodes`
            from ray.util.state import list_logs
            # `ray logs` by default print logs from a head node.
            # To list the same logs, you should provide the head node ID.
            # Get the node ID / node IP from `ray list nodes`
            list_logs(node_id=<HEAD_NODE_ID>)

**从节点获取特定日志文件**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            # Get the node ID / node IP from `ray list nodes`
            ray logs cluster gcs_server.out --node-id <NODE_ID>
            # `ray logs cluster` is alias to `ray logs` when querying with globs.
            ray logs gcs_server.out --node-id <NODE_ID>

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_log

            # Node IP can be retrieved from list_nodes() or ray.nodes()
            for line in get_log(filename="gcs_server.out", node_id=<NODE_ID>):
                print(line)

**从节点流式传输日志文件**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            # Get the node ID / node IP from `ray list nodes`
            ray logs raylet.out --node-ip <NODE_IP> --follow
            # Or,
            ray logs cluster raylet.out --node-ip <NODE_IP> --follow


    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_log

            # Retrieve the Node IP from list_nodes() or ray.nodes()
            # The loop blocks with `follow=True`
            for line in get_log(filename="raylet.out", node_ip=<NODE_IP>, follow=True):
                print(line)

**来自具有 actor id 的 actor 的流日志**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray logs actor --id=<ACTOR_ID> --follow

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_log

            # Get the Actor's ID from the output of `ray list actors`.
            # The loop blocks with `follow=True`
            for line in get_log(actor_id=<ACTOR_ID>, follow=True):
                print(line)

**来自 pid 的日志流**

.. tabs::

    .. group-tab:: CLI （推荐）

        .. code-block:: bash

            ray logs worker --pid=<PID> --follow

    .. group-tab:: Python SDK （内部开发 API）

        .. testcode::
            :skipif: True

            from ray.util.state import get_log

            # Retrieve the node IP from list_nodes() or ray.nodes()
            # get the PID of the worker running the Actor easily when output
            # of worker is directed to the driver (default)
            # The loop blocks with `follow=True`
            for line in get_log(pid=<PID>, node_ip=<NODE_IP>, follow=True):
                print(line)

有关命令的更多详细信息，请参阅 :ref:`状态 CLI 参考<state-api-cli-ref>` 的 ``ray logs`` 命令。


失败语义
^^^^^^^^^^^^^^^^^^^^^^^^^

State API 不保证始终返回一致或完整的集群快照。
默认情况下，当 API 缺少输出时，所有 Python SDK 都会引发异常。 
CLI 返回部分结果并提供警告消息。在以下情况下，API 可能会丢失输出。

**查询失败**

状态 API 查询“数据源”（例如，GCS、raylet 等）以获取并构建集群的快照。
然而，数据源有时不可用（例如，源已关闭或过载）。
在这种情况下，API 返回集群的部分（不完整）快照，并通过警告消息通知用户输出不完整。
所有警告都通过 Python 的 ``warnings`` 库打印，可以抑制它们。

**数据截断**

当返回的实体数（行数）太大（> 100K）时，状态 API 会截断输出数据以确保系统稳定性（发生这种情况时，无法选择截断的数据）。
当发生截断时， Python 的 ``warnings`` 模块会通知它。

**垃圾收集资源**

根据资源的生命周期，某些“已完成”资源无法通过 API 访问，因为它们已被垃圾收集。

.. note::

    不要依赖此 API 来获取有关已完成资源的正确信息。
    例如，Ray 定期垃圾收集 DEAD 状态 Actor 数据以减少内存使用。
    或者，当其沿袭超出范围时，它会清除任务的 FINISHED 状态。

API 参考
~~~~~~~~~~~~~~~~~~~~~~~~~~

- 有关 CLI 参考，请参阅 :ref:`状态 CLI 参考 <state-api-cli-ref>`。
- 有关 SDK 参考，请参阅 :ref:`状态 API 参考 <state-api-ref>`。
- 有关日志 CLI 参考，请参阅 :ref:`日志 CLI 参考 <ray-logs-api-cli-ref>`。




从集群外部使用 Ray CLI 工具
--------------------------------------------------------
这些 CLI 命令必须在 Ray Cluster 中的节点上运行。下面提供了从 Ray Cluster 外部的机器执行这些命令的示例。

.. tab-set::

    .. tab-item:: VM Cluster Launcher

        使用以下命令 ``ray exec`` 在集群上执行命令：

        .. code-block:: shell

            $ ray exec <cluster config file> "ray status"

    .. tab-item:: KubeRay

        使用 ``kubectl exec`` 和配置的 RayCluster 名称在集群上执行命令。
        Ray 使用定位到 Ray head pod 的 Service 在集群上执行 CLI 命令。

        .. code-block:: shell

            # First, find the name of the Ray head service.
            $ kubectl get pod | grep <RayCluster name>-head
            # NAME                                             READY   STATUS    RESTARTS   AGE
            # <RayCluster name>-head-xxxxx                     2/2     Running   0          XXs

            # Then, use the name of the Ray head service to run `ray status`.
            $ kubectl exec <RayCluster name>-head-xxxxx -- ray status
