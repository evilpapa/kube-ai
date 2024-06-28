.. _:: ../doc_code:

.. _train-fault-tolerance:

处理故障和节点抢占
=====================================

自动从 Train Worker 故障中恢复
------------------------------------------------

Ray Train 具有内置容错功能，可以从 Worker 故障中恢复（即 ``RayActorError``）。
当检测到故障时，将关闭 Worker 并添加新的 Worker。

训练函数会重启，但是可以通过检查点恢复上一次执行的进度。

.. tip::
    为了保留进度，您的训练函数 **必须** 实现
    :ref:`保存 <train-dl-saving-checkpoints>` *和* :ref:`加载检查点 <train-dl-loading-checkpoints>` 的逻辑。

每次从工作程序故障中恢复都被视为重试。
重试次数通过传递给 ``Trainer`` 设置在 :class:`~ray.train.RunConfig` 
中 :class:`~ray.train.FailureConfig` 的 ``max_failures`` 属性进行配置。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __failure_config_start__
    :end-before: __failure_config_end__

哪个检查点将被恢复？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Train 会自动从最新的
可用 :ref:`Ray Train 报告的检查点<train-checkpointing>` 恢复训练。

这将是传递给 :func:`train.report() <ray.train.report>` 的最后一个检查点。

.. _train-restore-guide:

恢复 Ray Train 实现
------------------------------

在实验层面，Trainer 恢复功能
允许您从上次中断的地方恢复之前中断的实验。

一个 Train 实验可能由于以下原因之一而中断：

- 实验被手动中断（例如，Ctrl+C，或预占头节点实例）。
- 头节点崩溃（例如，OOM 或其他运行时错误）。
- 整个集群发生故障（例如，网络错误影响所有节点）。

所有 Ray Train 的内置 Trainer 都支持 Trainer 恢复，
但我们在示例中使用 ``TorchTrainer`` 进行演示。
我们还使用 ``<Framework>Trainer`` 来引用所有内置 Trainer 共享的方法。

假设您的初始训练实验配置如下。
实际的训练循环仅用于演示目的：
重要的细节是已经实现了 
:ref:`保存 <train-dl-saving-checkpoints>` *和* :ref:`加载检查点 <train-dl-loading-checkpoints>` 的逻辑。

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_initial_run_start__
    :end-before: __ft_initial_run_end__

实验的结果和检查点保存到 :class:`~ray.train.RunConfig` 配置的路径。
如果实验由于上述原因之一而中断，请使用此路径恢复：

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_restored_run_start__
    :end-before: __ft_restored_run_end__

.. tip::

    您还可以从远程路径恢复（例如，从存储在 s3 存储桶中的实验目录）。

    .. literalinclude:: ../doc_code/dl_guide.py
        :language: python
        :dedent:
        :start-after: __ft_restore_from_cloud_initial_start__
        :end-before: __ft_restore_from_cloud_initial_end__

    .. literalinclude:: ../doc_code/dl_guide.py
        :language: python
        :dedent:
        :start-after: __ft_restore_from_cloud_restored_start__
        :end-before: __ft_restore_from_cloud_restored_end__

.. note::

    不通的 Trainer 可能允许在恢复时重新指定更多参数。
    只有 **datasets** 在恢复时需要重新指定，如果它们最初被提供。

    `TorchTrainer.restore`， `TensorflowTrainer.restore` 以及 `HorovodTrainer.restore` 可以接受
    与其父类 :meth:`DataParallelTrainer.restore <ray.train.data_parallel_trainer.DataParallelTrainer.restore>` 相同的参数。

    除非另有说明，其他 Trainer 将接受
    与 :meth:`BaseTrainer.restore <ray.train.trainer.BaseTrainer.restore>` 相同的参数。


自动恢复
~~~~~~~~~~~

Adding the branching logic below will allow you to run the same script after the interrupt,
picking up training from where you left on the previous run. 
添加下面的分支逻辑将允许您在中断后运行相同的脚本，
从上次运行中断的地方继续训练。
注意，我们使用 
:meth:`<Framework>Trainer.can_restore <ray.train.trainer.BaseTrainer.can_restore>` 实用程序方法
来确定给定实验目录的存在性和有效性。

.. literalinclude:: ../doc_code/dl_guide.py
    :language: python
    :start-after: __ft_autoresume_start__
    :end-before: __ft_autoresume_end__

.. seealso::

    参考 :meth:`BaseTrainer.restore <ray.train.trainer.BaseTrainer.restore>` 文档获取完整示例。

.. note::

    `<Framework>Trainer.restore` 不同于
    :class:`<Framework>Trainer(..., resume_from_checkpoint=...) <ray.train.trainer.BaseTrainer>`。
    `resume_from_checkpoint` 用于启动一个 *新* Train 实验，将结果写入新目录并从迭代 0 开始。

    `<Framework>Trainer.restore` 用于继续现有实验，
    新结果将继续追加到现有日志中。
