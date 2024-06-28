.. _train-gbdt-guide:

开始使用 XGBoost 和 LightGBM
=====================================

Ray Train 内置对 XGBoost 和 LightGBM 的支持。


快速开始
-----------
.. tab-set::

    .. tab-item:: XGBoost

        .. literalinclude:: doc_code/gbdt_user_guide.py
           :language: python
           :start-after: __xgboost_start__
           :end-before: __xgboost_end__

    .. tab-item:: LightGBM

        .. literalinclude:: doc_code/gbdt_user_guide.py
           :language: python
           :start-after: __lightgbm_start__
           :end-before: __lightgbm_end__


在 Train 中使用基于树的模型进行基础训练
----------------------------------------------

就像在原始的 `xgboost.train() <https://xgboost.readthedocs.io/en/stable/parameter.html>`__ 和
`lightgbm.train() <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`__ 函数中一样，
训练参数以字典的形式传递给 ``params`` 。

.. tab-set::

    .. tab-item:: XGBoost

        Run ``pip install -U xgboost_ray``.

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __xgboost_start__
            :end-before: __xgboost_end__

    .. tab-item:: LightGBM

        Run ``pip install -U lightgbm_ray``.

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __lightgbm_start__
            :end-before: __lightgbm_end__


训练器构造函数传递 Ray 特定的参数。


.. _train-gbdt-checkpoints:

保存并加载 XGBoost 和 LightGBM 检查点
----------------------------------------------

在每一轮提升训练新树时，您可以保存检查点
以快照迄今为止的训练进度。
:class:`~ray.train.xgboost.XGBoostTrainer` 和 :class:`~ray.train.lightgbm.LightGBMTrainer`
都实现了开箱即用的检查点。 可以使用
静态方法 :meth:`XGBoostTrainer.get_model <ray.train.xgboost.XGBoostTrainer.get_model>` 和
:meth:`LightGBMTrainer.get_model <ray.train.lightgbm.LightGBMTrainer.get_model>` 将这些检查点加载到内存中。

唯一需要更改的是配置 :class:`~ray.train.CheckpointConfig` 以
设置检查点频率。例如，以下配置
在每个提升轮次上保存一个检查点，并且仅保留最新的检查点：

.. literalinclude:: doc_code/key_concepts.py
    :language: python
    :start-after: __checkpoint_config_ckpt_freq_start__
    :end-before: __checkpoint_config_ckpt_freq_end__

.. tip::

    一旦启用检查点，您可以按照 :ref:`本指南 <train-fault-tolerance>`
    启用容错功能。


如何扩展训练？
--------------------------

使用 Ray Train 的好处是，您可以通过调整 :class:`ScalingConfig <ray.train.ScalingConfig>` 来无缝扩展您的训练。

.. note::
    Ray Train 不会修改或以其他方式改变
    底层 XGBoost 或 LightGBM 分布式训练算法的工作原理。
    Ray 仅提供编排、数据提取和容错功能。
    有关 GBDT 分布式训练的更多信息，请参阅
    `XGBoost documentation <https://xgboost.readthedocs.io>`__ 和
    `LightGBM documentation <https://lightgbm.readthedocs.io/>`__。


以下是一些常见用例的示例：

.. tab-set::

    .. tab-item:: 多节点 CPU

        设置: 4 个节点，每个节点有 8 个 CPU。

        用例：在多节点训练中利用所有资源。

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_cpu_start__
            :end-before: __scaling_cpu_end__

        请注意，我们为训练器资源传递了 0 个 CPU，以便
        所有资源都可以分配给实际的分布式训练 worker 。


    .. tab-item:: 单节点多 GPU

        设置：1 个节点，配备 8 个 CPU 和 4 个 GPU。

        用例：如果您有一个具有多个 GPU 的单个节点，则需要
        使用分布式训练来利用所有 GPU。

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_gpu_start__
            :end-before: __scaling_gpu_end__

    .. tab-item:: 多节点多 GPU

        设置：4 个节点，每个节点有 8 个 CPU 和 4 个 GPU。

        用例：如果您有多个具有多个 GPU 的节点，则需要为每个 GPU 分配一个 worker。

        .. literalinclude:: doc_code/gbdt_user_guide.py
            :language: python
            :start-after: __scaling_gpumulti_start__
            :end-before: __scaling_gpumulti_end__

        请注意，您只需调整工人数量。Ray 会自动处理其他所有事情。


你应该使用多少个远程 actor ？
--------------------------------------

这取决于您的工作负载和集群设置。
通常，对于仅使用 CPU 的训练，每个节点运行多个远程 actor 并没有什么固有的好处。
这是因为 XGBoost 已经可以利用多线程 CPU。

但是，在某些情况下，你应该考虑每个节点启动多个 actor ：

* 对于 **多 GPU 训练**，每个 GPU 应有一个单独的
  远程 actor。因此，如果您的机器有 24 个 CPU 和 4 个 GPU，您需要启动 4 个远程 actor，
  每个 actor 有 6 个 CPU 和 1 个 GPU
* 在 **异构集群** , ，您可能希望找到 CPU 数量的
  `最大公约数 <https://en.wikipedia.org/wiki/Greatest_common_divisor>`_。
  例如，对于具有三个节点（分别为 4、8 和 12 个 CPU）的集群，您应该将 actor 数量设置为 6，
  将每个 actor 的 CPU 数量设置为 4。

如何使用 GPU 进行训练？
-----------------------------

Ray Train 支持 XGBoost 和 LightGBM 的多 GPU 训练。 
核心后端自动利用 NCCL2 进行跨设备通信。
您所要做的就是为每个 GPU 启动一个 actor 并设置与 GPU 兼容的参数。
例如， XGBoost 的 ``tree_method`` 为 ``gpu_hist``。参考 XGBoost
文档获取更多详细信息。

例如，如果您有 2 台机器，每台机器有 4 个 GPU，您想要启动 8 个工作器，并设置 ``use_gpu=True``。
为每个 actor 分配少于（例如 0.5）或更多 GPU 通常没有好处。

您应该将 CPU 均匀地分配给每台机器的各个 actor ，
因此，如果您的机器除了 4 个 GPU 之外还有 16 个 CPU，
则每个 actor 应该有 4 个 CPU 可供使用。


.. literalinclude:: doc_code/gbdt_user_guide.py
    :language: python
    :start-after: __gpu_xgboost_start__
    :end-before: __gpu_xgboost_end__


.. _data-ingest-gbdt:

如何预处理训练数据？
------------------------------------

特别是对于表格数据，Ray Data 附带了开箱即用的 :ref:`预处理器 <air-preprocessors>` ，可实现常见的特征预处理操作。
您可以在将数据集传递到 Trainer 之前将其应用于数据集，从而将其与 Ray Train Trainers 一起使用。例如：


.. literalinclude:: ../data/doc_code/preprocessors.py
    :language: python
    :start-after: __trainer_start__
    :end-before: __trainer_end__


如何优化XGBoost的内存使用？
-------------------------------------

XGBoost 使用计算优化的数据结构 ``DMatrix``，
来保存训练数据。 将数据集转换为 ``DMatrix`` 时，
XGBoost 会创建中间副本，最终保存完整数据的完整副本。
XGBoost  将数据转换为本地数据格式。
在 64 位系统中，格式为 64 位浮点数。
根据系统和原始数据集的数据类型，此矩阵可能比原始数据集占用更多的内存。

基于 CPU 训练， **内存用量** 至少为数据集大小的 **3倍**，
假设如在 64 位系统上的 ``float32`` 数据类型，
再加上大约 **400,000 KiB** 用于其他资源，如操作系统需求和存储中间结果。

**示例**


* 机器类型: AWS m5.xlarge (4 vCPUs, 16 GiB RAM)
* 内存用量: ~15,350,000 KiB
* 数据集: 1,250,000 行 1024 个特征, dtype float32.
  总大小: 5,000,000 KiB
* XGBoost DMatrix 大小: ~10,000,000 KiB

该数据集恰好适合在此节点进行训练。

请注意，在 32 位系统上，DMatrix 大小可能会更低。

**GPUs**

通常，基于 GPU 的训练有相同的内存要求。
此外，GPU 必须有足够的内存来容纳数据集。

在上面的示例中，GPU 必须至少具有 10,000,000 KiB（约 9.6 GiB）内存。
但是，经验数据表明，使用 ``DeviceQuantileDMatrix``
似乎导致更多的峰值 GPU 内存使用量，可能用于加载数据时的中间存储（约 10%）。

**最佳实践**

为了减少峰值内存使用量，请考虑以下建议：


* 将数据存储为 ``float32`` 或更少。 您通常不需要更高的精度，
  并且以较小的格式保存数据有助于减少初始数据加载的峰值内存使用量。
* 从 CSV 加载数据时传递 ``dtype`` 。
  否则，将按默认方式 ``np.float64`` 加载浮点值，
  这会导致峰值内存使用量增加 33%。
