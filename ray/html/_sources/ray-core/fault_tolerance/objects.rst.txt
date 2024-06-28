.. _fault-tolerance-objects:
.. _object-fault-tolerance:

对象容错
======================

Ray 对象既有数据（调用 ``ray.get`` 时返回的值）也有元数据（例如，值的位置）。
数据存储在 Ray 对象存储中，而元数据存储在对象的 **所有者** 处。
对象的所有者是创建原始 ``ObjectRef`` 的 worker 进程，
例如，通过调用 ``f.remote()`` 或 ``ray.put()``。
请注意，这个 worker 通常是一个与创建对象 **值** 的 worker 进程不同的进程，除了 ``ray.put`` 的情况。

.. literalinclude:: ../doc_code/owners.py
  :language: python
  :start-after: __owners_begin__
  :end-before: __owners_end__


Ray 可以自动恢复数据丢失，但不能恢复所有者失败。

.. _fault-tolerance-objects-reconstruction:

从数据丢失中恢复
-------------------------

当对象值从对象存储中丢失时（例如在节点故障期间），Ray 将使用 *lineage reconstruction* 来恢复对象。
Ray 将首先通过其他节点上的相同对象的副本来自动尝试恢复值。 如果没有找到，
则 Ray 将通过 :ref:`重新执行 <fault-tolerance-tasks>` 先前创建值的任务来自动恢复值。
任务的参数通过相同的机制递归重建。

谱系重建目前有以下限制：

* 该对象以及其传递依赖项必须有任务（actor 或 非 actor）生成。
  这意味着 **ray.put 创建的对象不可恢复**。
* 任务假定为幂等的。因此， **默认情况下，actor 任务生成的对象不可重建**。
  要允许重建 actor 任务的结果，
  请将 ``max_task_retries`` 参数设置为非零值（有关更多详细信息，请参见 :ref:`actor 容错 <fault-tolerance-actors>`）。
* 任务只会被重新执行到其最大重试次数。默认情况下，非 actor 任务最多可以重试 3 次，actor 任务不可重试。
  可以通过 ``max_retries`` 参数（用于 :ref:`远程函数 <fault-tolerance-tasks>`）和 ``max_task_retries`` 
  参数（用于 :ref:`actors <fault-tolerance-actors>`）来覆盖这一点。
* 该对象的所有者必须仍然存活（请参见 :ref:`下文 <fault-tolerance-ownership>`）。

Lineage 重建可能会导致驱动程序内存使用量高于正常水平，因为驱动程序会保留在发生故障时可能重新执行的任何任务的描述。
要限制 Lineage 使用的内存量，请将环境变量 ``RAY_max_lineage_bytes``（默认为 1GB）设置为在超过阈值时驱逐 Lineage。

要完全禁用 Lineage 重建，请在 ``ray start`` 或 ``ray.init`` 期间设置环境变量 ``RAY_TASK_MAX_RETRIES=0``。
通过此设置，如果没有副本剩余，则会引发 ``ObjectLostError``。

.. _fault-tolerance-ownership:

从所有者故障中恢复
-----------------------------

对象的所有者会因节点或 worker 进程故障而死亡。
目前，**Ray 不支持从所有者故障中恢复**。
在这种情况下，Ray 将清理对象值的任何剩余副本，以防止内存泄漏。
随后尝试获取对象值的任何 worker 将收到 ``OwnerDiedError`` 异常，可以手动处理。

理解 `ObjectLostErrors`
--------------------------------

Ray 抛出 ``ObjectLostError`` 异常给应用程序，当对象由于应用程序或系统错误而无法检索时。
这可能发生在 ``ray.get()`` 调用期间或在获取任务参数时，并且可能发生多种原因。
这里是一个指南，用于理解不同错误类型的根本原因：

- ``OwnerDiedError``: 对象的所有者，即通过 ``.remote()`` 或 ``ray.put()`` 创建 ``ObjectRef`` 的 Python worker 已经死亡。
  所有者存储关键对象元数据，如果此进程丢失，则无法检索对象。
- ``ObjectReconstructionFailedError``: 如果一个对象，或者另一个对象依赖于此对象，
  由于 :ref:`上述限制 <fault-tolerance-objects-reconstruction>` 之一而无法重建，则会抛出此错误。
- ``ReferenceCountingAssertionError``: 该对象已被删除，因此无法检索。Ray 通过分布式引用计数实现自动内存管理，因此一般不会发生此错误。
  但是，有一种 `已知的极端情况 <https://github.com/ray-project/ray/issues/18456>`_ 可能会产生此错误。
- ``ObjectFetchTimedOutError``: 尝试从远程节点检索对象副本时节点超时。
  此错误通常表示系统级错误。
  可以使用环境 ``RAY_fetch_fail_timeout_milliseconds`` 变量配置超时时间（默认为 10 分钟）。
- ``ObjectLostError``: 对象已成功创建，但无法访问任何副本。
   这是在禁用沿袭重建且对象的所有副本从集群中丢失时抛出的一般错误。
