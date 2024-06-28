.. _train-checkpointing:

保存加载检查点
==============================

Ray Train 提供了一种快照训练进度的方法 :class:`Checkpoints <ray.train.Checkpoint>` 。

这对于以下情况有用：

1. **存储性能最佳的模型权重：** 将模型保存到持久存储中，并将其用于下游服务/推理。
2. **容错：** 处理在集群上可抢占机器 / pod 上长期运行的训练作业中的节点故障。
3. **分布式检查点：** 在进行 *模型并行训练*时，Ray Train 检查点提供了一种简单的方法，可以
   :ref:`从每个 worker 并行上传模型碎片 <train-distributed-checkpointing>`，
   而无需将完整模型收集到单个节点。
4. **与 Ray Tune 集成：** 检查点的保存和加载被默写 :ref:`Ray Tune 调度程序 <tune-schedulers>` 所需。


.. _train-dl-saving-checkpoints:

训练期间保存检查点
----------------------------------

:class:`Checkpoint <ray.train.Checkpoint>` 是 Ray Train 提供的轻量级接口，表示
存在于本地或远程存储中的 *目录* 。

例如，检查点可以指向云存储中的目录：
``s3://my-bucket/my-checkpoint-dir``。
本地可用的检查点指向本地文件系统上的位置：
``/tmp/my-checkpoint-dir``。

以下是在训练循环中保存检查点的方法：

1. 将您的模型检查点写入本地目录。

   - 由于 :class:`Checkpoint <ray.train.Checkpoint>` 只是指向一个目录，因此其内容完全由您决定。
   - 这意味着您可以使用任何您想要的序列化格式。
   - 这使得 **使用训练框架提供的熟悉的检查点实用程序变得容易**，例如
     ``torch.save``， ``pl.Trainer.save_checkpoint``，Accelerate 的 ``accelerator.save_model``，
     Transformer 的 ``save_pretrained``，``tf.keras.Model.save`` 等等。

2. 使用 :meth:`Checkpoint.from_directory <ray.train.Checkpoint.from_directory>` 从目录创建一个 :class:`Checkpoint <ray.train.Checkpoint>`。

3. 使用 :func:`ray.train.report(metrics, checkpoint=...) <ray.train.report>` 向 Ray Train 报告检查点。

   - 与检查点一起报告的指标用于 :ref:`跟踪性能最佳的检查点 <train-dl-configure-checkpoints>`。
   - 如果已配置，这会将 **检查点上传到持久存储** 。参阅 :ref:`persistent-storage-guide`。


.. figure:: ../images/checkpoint_lifecycle.png

    :class:`~ray.train.Checkpoint` 的声明周期，从本地
    保存到磁盘到通过 ``train.report`` 上传到持久存储。

如上图所示，保存检查点的最佳实践是首先将检查点转储到本地临时目录。
然后，调用 ``train.report``
将检查点上传到其最终持久存储位置。
然后，可以安全地清理本地临时目录以释放磁盘空间
（例如，从存在的 ``tempfile.TemporaryDirectory``）。

.. tip::

    在标准 DDP 训练中，每个 worker 都有完整模型的副本，您应该
    只保存并报告来自单个 worker 的检查点，以防止重复上传。

    这通常看起来像：

    .. literalinclude::  ../doc_code/checkpoints.py
        :language: python
        :start-after: __checkpoint_from_single_worker_start__
        :end-before: __checkpoint_from_single_worker_end__

    如果使用 DeepSpeed Zero-3 和 FSDP 等并行训练策略，其中
    每个 worker 仅具有完整模型的一个分片，则应保存并报告每个工作器的检查点。
    参阅 :ref:`train-distributed-checkpointing` 作为示例。


以下是使用不同训练框架保存检查点的几个示例：

.. tab-set::

    .. tab-item:: Native PyTorch

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __pytorch_save_start__
            :end-before: __pytorch_save_end__

        .. tip::

            您很可能希望在将 DDP 模型保存到检查点之前将其解开。
            ``model.module.state_dict()`` 是状态字典，其中每个键都没有前缀 ``"module."`` 前缀。


    .. tab-item:: PyTorch Lightning

        Ray Train 利用 PyTorch Lightning 的 ``Callback`` 接口来报告指标和检查点。
        我们提供了一个简单的回调 ``on_train_epoch_end`` 实现来报告。

        具体来说，在每个训练周期结束时，它

        - 从 ``trainer.callback_metrics`` 收集所有记录的指标
        - 通过 ``trainer.save_checkpoint`` 保存检查点
        - 通过 :func:`ray.train.report(metrics, checkpoint) <ray.train.report>` 向 Ray Train 报告

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_save_example_start__
            :end-before: __lightning_save_example_end__

        您始终可以从 :attr:`result.checkpoint <ray.train.Result.checkpoint>` 和
        :attr:`result.best_checkpoints <ray.train.Result.best_checkpoints>` 路径获取检查点。

        对于更高级的用法（例如以不同的频率报告、报告自定义检查点文件），您可以实现自己的自定义回调。
        这是一个每 3 个 epoch 报告一次检查点的简单示例：

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_custom_save_example_start__
            :end-before: __lightning_custom_save_example_end__


    .. tab-item:: Hugging Face Transformers

        Ray Train 利用 HuggingFace Transformers Trainer 的 ``Callback`` 接口
        来报告指标和检查点。

        **选项 1: 使用 Ray Train 的默认回调报告**

        我们提供了一个简单的回调实现 :class:`~ray.train.huggingface.transformers.RayTrainReportCallback`
        来报告检查点保存情况。您可以通过 ``save_strategy`` 和 ``save_steps`` 更改检查点频率。
        它会收集最新记录的指标并将其与最新保存的检查点一起报告。

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __transformers_save_example_start__
            :end-before: __transformers_save_example_end__

        请注意， :class:`~ray.train.huggingface.transformers.RayTrainReportCallback`
        将最新的指标和检查点绑定在一起，
        用户可以正确配置 ``logging_strategy``， ``save_strategy`` 和 ``evaluation_strategy``
        来确保监控指标与检查点保存在同一步骤记录。

        例如，评估指标（在本例中的 ``eval_loss``）在评估期间被记录。
        如果用户希望根据 ``eval_loss`` 保留最佳的 3 个检查点，
        他们应该调整保存和评估频率。以下是两个有效配置的示例：

        .. code-block:: python

            args = TrainingArguments(
                ...,
                evaluation_strategy="epoch",
                save_strategy="epoch",
            )

            args = TrainingArguments(
                ...,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=50,
                save_steps=100,
            )

            # And more ...


        **选项 2: 实现自定义回调**

        如果您觉得 Ray Train 的默认 :class:`~ray.train.huggingface.transformers.RayTrainReportCallback`
        不足以满足您的用例，您也可以自己实现回调！
        下面是一个收集最新指标并报告保存检查点的示例实现。

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __transformers_custom_save_example_start__
            :end-before: __transformers_custom_save_example_end__


        您可以通过实现自己的 Transformers Trainer 回调来确定何时报告（``on_save``， ``on_epoch_end``， ``on_evaluate``）
        以及报告什么（自定义指标和检查点文件）


.. _train-distributed-checkpointing:

保存来自多个 worker 的检查点（分布式检查点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个 worker 仅具有完整模型的一个分片的模型并行训练策略中，
您可以从每个 worker 并行保存和报告检查点分片。

.. figure:: ../images/persistent_storage_checkpoint.png

    Ray Train 的分布式检查点。每个 worker 将自己的检查点分片独立上传到持久存储。

分布式检查点是进行
模型并行训练（例如 DeepSpeed、FSDP、Megatron-LM）时保存检查点的最佳实践。

有两个主要好处：

1. **速度更快，从而减少空闲时间。** 更快的检查点会激励更频繁的检查点！

   每个 worker 都可以并行上传其检查点分片，从而最大限度地提高集群的网络带宽。
   集群不再由单个节点上传大小为“M”的完整模型，
   而是将负载分散到“N”个节点上，
   每个节点上传大小为“M / N”的分片。

2. **分布式检查点避免需要将完整模型收集到单个工作者的 CPU 内存上。**

   此收集操作对执行检查点的 worker 提出了很大的 CPU 内存要求，并且是 OOM 错误的常见来源。


以下是使用 PyTorch 进行分布式检查点的示例：

.. literalinclude:: ../doc_code/checkpoints.py
    :language: python
    :start-after: __distributed_checkpointing_start__
    :end-before: __distributed_checkpointing_end__


.. note::

    具有相同名称的检查点文件将在 worker 之间发生冲突。
    您可以通过向检查点文件添加特定于等级的后缀来解决此问题。

    请注意，文件名冲突不会出错，
    但会导致最后上传的版本被保留。
    如果文件内容在所有 Worker 中都相同，则这没有问题。

    DeepSpeed 等框架提供的模型分片保存实用程序将创建
    特定于等级的文件名，因此您通常不需要担心这一点。


.. _train-dl-configure-checkpoints:

配置检查点
-----------------------

Ray Train 通过 :class:`~ray.train.CheckpointConfig` 提供了一些检查点配置选项。
主要配置是仅保留 top ``K`` 与指标相关的顶级检查点。
性能较差的检查点将被删除以节省存储空间。默认情况下，所有检查点都会保留。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __checkpoint_config_start__
    :end-before: __checkpoint_config_end__


.. note::

    如果您想通过 :py:class:`~ray.train.CheckpointConfig` 保存与某个指标相关的 top ``num_to_keep`` 检查点，
    请确保该指标始终与检查点一起报告。



训练后使用检查点
--------------------------------

最近的检查点可通过 :attr:`Result.checkpoint <ray.train.Result.checkpoint>` 访问。

完成的持久化检查点列表可通过 :attr:`Result.best_checkpoints <ray.train.Result.best_checkpoints>` 访问。
如果设置了 :class:`CheckpointConfig(num_to_keep) <ray.train.CheckpointConfig>`， 此列表将包含最佳的 ``num_to_keep`` 个保存点。

参阅 :ref:`train-inspect-results` ，了解检查训练结果的完整指南。

:meth:`Checkpoint.as_directory <ray.train.Checkpoint.as_directory>`
和 :meth:`Checkpoint.to_directory <ray.train.Checkpoint.to_directory>`
是与 Train 检查点交互的两个主要 API：

.. literalinclude:: ../doc_code/checkpoints.py
    :language: python
    :start-after: __inspect_checkpoint_example_start__
    :end-before: __inspect_checkpoint_example_end__


.. _train-dl-loading-checkpoints:

从检查点恢复训练状态
----------------------------------------

为了实现容错功能，您应该修改训练循环以从 :class:`~ray.train.Checkpoint` 中恢复训练状态。

可以使用 :func:`ray.train.get_checkpoint <ray.train.get_checkpoint>` 来在训练功能中
访问要恢复的 :class:`Checkpoint <ray.train.Checkpoint>`。

:func:`ray.train.get_checkpoint <ray.train.get_checkpoint>` 返回的检查点以两种方式填充：

1. 它可以自动填充为最新报告的检查点，例如在 :ref:`自动错误恢复 <train-fault-tolerance>` 或 :ref:`手动恢复 <train-restore-guide>`。
2. 可以通过将检查点传递给 Ray :class:`Trainer <ray.train.trainer.BaseTrainer>` 的参数 ``resume_from_checkpoint`` 来手动填充它。
   这对于使用上一次运行的检查点初始化新的训练运行很有用。


.. tab-set::

    .. tab-item:: Native PyTorch

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __pytorch_restore_start__
            :end-before: __pytorch_restore_end__


    .. tab-item:: PyTorch Lightning

        .. literalinclude:: ../doc_code/checkpoints.py
            :language: python
            :start-after: __lightning_restore_example_start__
            :end-before: __lightning_restore_example_end__


.. note::

    在这些示例中，:meth:`Checkpoint.as_directory <ray.train.Checkpoint.as_directory>`
    用于将检查点内容作为本地目录查看。

    *如果检查点指向本地目录* ，则此方法仅返回本地目录路径而不进行复制。

    *如果检查点指向远程目录*，此方法将检查点下载
    到本地临时目录并返回临时目录的路径。

    **如果同一节点上的多个进程同时调用此方法，**
    则只有一个进程会执行下载，而其他进程则等待下载完成。
    下载完成后，所有进程都会收到相同的本地（临时）目录以供读取。

    一旦所有进程完成检查点工作，临时目录就会被清理。
