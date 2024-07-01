.. _system-metrics:

系统指标
--------------
Ray 导出许多系统指标，这些指标可以对 Ray 工作负载的状态以及硬件利用率统计数据进行内省。下表描述了官方支持的指标：

.. note::

些标签在所有指标中都是通用的，例如 `SessionName`（唯一标识 Ray 集群实例）、`instance` （Prometheus 应用的每个节点标签，以及 `JobId`（Ray 作业 ID，如适用）。

.. list-table:: Ray 系统指标
   :header-rows: 1

   * - Prometheus 指标
     - 标签
     - 描述
   * - `ray_tasks`
     - `Name`, `State`, `IsRetry`
     - 按状态划分的当前任务数量（远程函数和 actor 调用）。 状态标签 (如 RUNNING, FINISHED, FAILED) 描述任务的状态。 有关详细信息，参考 `rpc::TaskState <https://github.com/ray-project/ray/blob/e85355b9b593742b4f5cb72cab92051980fa73d3/src/ray/protobuf/common.proto#L583>`_ 。 function/method 名称可作为名称标签使用。 如果任务由于失败或重建而重试，则 IsRetry  标签将设置为“1”，否则设置为“0”。
   * - `ray_actors`
     - `Name`, `State`
     - 特定状态下的当前 actor 数量。 状态标签在 gcs.proto 的 `rpc::ActorTableData <https://github.com/ray-project/ray/blob/e85355b9b593742b4f5cb72cab92051980fa73d3/src/ray/protobuf/gcs.proto#L85>`_ 协议。 The actor class name is available in the Name label.
   * - `ray_resources`
     - `Name`, `State`, `InstanceId`
     - Logical resource usage for each node of the cluster. Each resource has some quantity that is `in either <https://github.com/ray-project/ray/blob/9eab65ed77bdd9907989ecc3e241045954a09cb4/src/ray/stats/metric_defs.cc#L188>`_ USED state vs AVAILABLE state. The Name label defines the resource name (e.g., CPU, GPU).
   * - `ray_object_store_memory`
     - `Location`, `ObjectState`, `InstanceId`
     - Object store memory usage in bytes, `broken down <https://github.com/ray-project/ray/blob/9eab65ed77bdd9907989ecc3e241045954a09cb4/src/ray/stats/metric_defs.cc#L231>`_ by logical Location (SPILLED, IN_MEMORY, etc.), and ObjectState (UNSEALED, SEALED).
   * - `ray_placement_groups`
     - `State`
     - Current number of placement groups by state. The State label (e.g., PENDING, CREATED, REMOVED) describes the state of the placement group. See `rpc::PlacementGroupTable <https://github.com/ray-project/ray/blob/e85355b9b593742b4f5cb72cab92051980fa73d3/src/ray/protobuf/gcs.proto#L517>`_ for more information.
   * - `ray_memory_manager_worker_eviction_total`
     - `Type`, `Name`
     - The number of tasks and actors killed by the Ray Out of Memory killer (https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html) broken down by types (whether it is tasks or actors) and names (name of tasks and actors).
   * - `ray_node_cpu_utilization`
     - `InstanceId`
     - The CPU utilization per node as a percentage quantity (0..100). This should be scaled by the number of cores per node to convert the units into cores.
   * - `ray_node_cpu_count`
     - `InstanceId`
     - The number of CPU cores per node.
   * - `ray_node_gpus_utilization`
     - `InstanceId`, `GpuDeviceName`, `GpuIndex`
     - The GPU utilization per GPU as a percentage quantity (0..NGPU*100). `GpuDeviceName` is a name of a GPU device (e.g., Nvidia A10G) and `GpuIndex` is the index of the GPU.
   * - `ray_node_disk_usage`
     - `InstanceId`
     - The amount of disk space used per node, in bytes.
   * - `ray_node_disk_free`
     - `InstanceId`
     - The amount of disk space available per node, in bytes.
   * - `ray_node_disk_io_write_speed`
     - `InstanceId`
     - The disk write throughput per node, in bytes per second.
   * - `ray_node_disk_io_read_speed`
     - `InstanceId`
     - The disk read throughput per node, in bytes per second.
   * - `ray_node_mem_used`
     - `InstanceId`
     - The amount of physical memory used per node, in bytes.
   * - `ray_node_mem_total`
     - `InstanceId`
     - The amount of physical memory available per node, in bytes.
   * - `ray_component_uss_mb`
     - `Component`, `InstanceId`
     - The measured unique set size in megabytes, broken down by logical Ray component. Ray components consist of system components (e.g., raylet, gcs, dashboard, or agent) and the method names of running tasks/actors.
   * - `ray_component_cpu_percentage`
     - `Component`, `InstanceId`
     - The measured CPU percentage, broken down by logical Ray component. Ray components consist of system components (e.g., raylet, gcs, dashboard, or agent) and the method names of running tasks/actors.
   * - `ray_node_gram_used`
     - `InstanceId`, `GpuDeviceName`, `GpuIndex`
     - The amount of GPU memory used per GPU, in bytes.
   * - `ray_node_network_receive_speed`
     - `InstanceId`
     - 每个节点的网络发送吞吐量（以字节/秒为单位）。
   * - `ray_node_network_send_speed`
     - `InstanceId`
     - 集群中健康节点的数量，按自动缩放程序节点类型细分。
   * - `ray_cluster_active_nodes`
     - `node_type`
     - 自动缩放程序报告的失败节点数，按节点类型细分。
   * - `ray_cluster_failed_nodes`
     - `node_type`
     - The number of failed nodes reported by the autoscaler, broken down by node type.
   * - `ray_cluster_pending_nodes`
     - `node_type`
     - 自动缩放程序报告的待处理节点数，按节点类型细分。

指标语义和一致性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray guarantees all its internal state metrics are *eventually* consistent even in the presence of failures--- should any worker fail, eventually the right state will be reflected in the Prometheus time-series output. However, any particular metrics query is not guaranteed to reflect an exact snapshot of the cluster state.

For the `ray_tasks` and `ray_actors` metrics, you should use sum queries to plot their outputs (e.g., ``sum(ray_tasks) by (Name, State)``). The reason for this is that Ray's task metrics are emitted from multiple distributed components. Hence, there are multiple metric points, including negative metric points, emitted from different processes that must be summed to produce the correct logical view of the distributed system. For example, for a single task submitted and executed, Ray may emit  ``(submitter) SUBMITTED_TO_WORKER: 1, (executor) SUBMITTED_TO_WORKER: -1, (executor) RUNNING: 1``, which reduces to ``SUBMITTED_TO_WORKER: 0, RUNNING: 1`` after summation.
