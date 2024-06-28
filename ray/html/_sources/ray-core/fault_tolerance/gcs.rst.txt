.. _fault-tolerance-gcs:

GCS 容错
===================

全局控制服务 (GCS) 是管理集群级元数据的服务器。
它还提供一些集群级操作，包括 :ref:`actor <ray-remote-classes>`，:ref:`placement groups <ray-placement-group-doc-ref>` 和节点管理。
。默认情况下，GCS 不具备容错能力，因为所有数据都存储在内存中，它的故障意味着整个 Ray 集群都发生故障。为了使 GCS 具有容错能力，需要 HA Redis。
然后，当 GCS 重新启动时，它会从 Redis 实例加载所有数据并恢复常规功能。

恢复期间，以下功能不可用：

- Actor 的创建、删除和重建。
- Placement group 的创建、删除和重建。
- 资源管理。
-  worker 节点注册。
- 工作进程创建。

但是，正在运行的 Ray task 和 actor仍然有效，并且任何现有对象将继续可用。

设置 Redis
----------------

.. tab-set::

    .. tab-item:: KubeRay (officially supported)

        如果你正在使用 :ref:`KubeRay <kuberay-index>`，参考 :ref:`KubeRay 文档关于 GCS Fault Tolerance <kuberay-gcs-ft>`。

    .. tab-item:: ray start

        如果你使用 :ref:`ray start <ray-start-doc>` 启动 Ray 头节点，
        设置系统环境变量 ``RAY_REDIS_ADDRESS`` 为 Redis 地址，
        并且在 ``ray start`` 使用 ``--redis-password`` 指定密码：

        .. code-block:: shell

          RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD

    .. tab-item:: ray up

        如果使用 :ref:`ray up <ray-up-doc>` 启动 Ray 集群，修改 ``ray start`` 命令的 :ref:`head_start_ray_commands <cluster-configuration-head-start-ray-commands>` 字段并添加 ``RAY_REDIS_ADDRESS`` 和 ``--redis-password``：

        .. code-block:: yaml

          head_start_ray_commands:
            - ray stop
            - ulimit -n 65536; RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml --dashboard-host=0.0.0.0

    .. tab-item:: Kubernetes

        如果运行在非 :ref:`KubeRay <kuberay-index>` 方式的 Kubernetes，请参考 :ref:`文档 <deploy-a-static-ray-cluster-without-kuberay>`。


一旦 GCS 由 Redis 支持，当它重新启动时，它将通过从 Redis 读取来恢复状态。
当 GCS 从故障状态恢复时，raylet 将尝试重新连接到 GCS。
如果 raylet 超过 60 秒无法重新连接到 GCS，则 raylet 将退出并且相应的节点将失败。此超时阈值可以通过 OS 环境变量 ``RAY_gcs_rpc_server_reconnect_timeout_s`` 进行调整。

您还可以设置 OS 环境变量 ``RAY_external_storage_namespace`` 来隔离存储在 Redis 中的数据。
这可确保如果多个 Ray 集群共享同一个 Redis 实例，则不会发生数据冲突。

如果 GCS 的 IP 地址在重启后会发生变化，最好使用合格的域名并在启动时将其传递给所有 raylet。
Raylet 将解析域名并连接到正确的 GCS。
您需要确保在任何时候都只有一个 GCS 处于活动状态。

.. note::

  仅当您使用 :ref:`KubeRay <kuberay-index>` 实现 :ref:`Ray 服务容错 <serve-e2e-ft>` 时，
  官方才支持使用外部 Redis 的 GCS 容错。
  对于其他情况，您可以自行承担风险，并且需要实施其他机制来检测 GCS 或头节点的故障并重新启动它。
