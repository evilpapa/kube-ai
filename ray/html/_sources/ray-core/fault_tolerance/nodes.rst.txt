.. _fault-tolerance-nodes:

节点容错
====================

Ray 集群由一个或多个 worker 节点组成，
每个 worker 节点由工作进程和系统进程（例如 raylet）组成。
其中一个 worker 节点被指定为头节点，并具有额外的进程，例如 GCS。

在这里，我们描述节点故障及其对任务、actor 和对象的影响。

Worker 容错
-------------------

当一个 worker 节点失败，所有正在运行的任务和 actor 都会失败，所有由该节点的 worker 进程拥有的对象都将丢失。在这种情况下，:ref:`tasks <fault-tolerance-tasks>`、:ref:`actors <fault-tolerance-actors>`、:ref:`objects <fault-tolerance-objects>` 容错机制会启动，尝试使用其他 worker 节点来恢复失败。

头节点容错
-----------------

当头节点失败，Ray 集群将无法继续工作。
为了容忍头节点故障，我们需要使 :ref:`GCS 容错 <fault-tolerance-gcs>`，这样当我们启动一个新的头节点时，我们仍然拥有所有集群级别的数据。

Raylet 失败
--------------

当 raylet 进程失败时，相应的节点将被标记为死亡，并被视为与节点故障相同。
每个 raylet 都与一个唯一 ID 相关联，因此即使 raylet 在同一台物理机器上重新启动，它也会被视为 Ray 集群的新 raylet/节点。
