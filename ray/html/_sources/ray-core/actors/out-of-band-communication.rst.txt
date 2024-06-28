带外通信
=========================

通常，Ray actor 通信是通过 actor 方法调用完成的，并且数据是通过分布式对象存储共享的。
然而，在某些情况下，带外通信可能很有用。

封装库进程
--------------------------
许多库已经有成熟的、高性能的内部通信栈，
并且它们利用 Ray 作为语言集成的 actor 调度器。
实际的 actor 之间的通信主要是通过现有的通信栈完成的。
比如，Horovod-on-Ray 使用 NCCL 或 基于 MPI 的集体通信，RayDP 使用 Spark 的内部 RPC 和对象管理器。
参考 `Ray 分布式库模式 <https://www.anyscale.com/blog/ray-distributed-library-patterns>`_ 获取更多信息。

Ray 集体通信库
--------------
Ray 的集体通信库 (\ ``ray.util.collective``\ ) 允许分布式 CPU 或 GPU 之间进行高效的带外集体和点对点通信。
有关更多详细信息，请参阅 :ref:`Ray Collective <ray-collective>`。

HTTP 服务器
-----------
你可以在 actor 中启动一个 http 服务器，并向客户端公开 http 端点，
以便集群外的用户可以与 actor 通信。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/actor-http-server.py

类似地，您也可以公开其他类型的服务器（例如，gRPC 服务器）。

限制
-----------

当使用带外通信与 Ray Actor 进行通信时，请记住 Ray 不会管理 Actor 之间的调用。这意味着分布式引用计数等功能不适用于带外通信，因此您应注意不要以这种方式传递对象引用。
