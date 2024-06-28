.. _train-monitoring-and-logging:

监控和记录指标
==============================

Ray Train 提供了一个 API，用于通过
调用 ``train.report(metrics)`` 从分布式工作节点上的训练函数（运行）
到 ``Trainer``（执行您的 python 脚本）报告中间结果和检查点。
结果将从分布式 worker 收集并传递给驱动程序进行记录和显示。

.. warning::

    仅使用来自等级 0 的 worker 的结果。但是，为了确保一致性，
    必须在每个 worker 上调用 ``train.report()``。
    如果要聚合来自多个 worker 的结果，请参见 :ref:`train-aggregating-results`。

报告的主要用例是每个训练阶段结束时的指标（准确性、损失等）。

.. tab-set::

    .. tab-item:: PyTorch

        .. code-block:: python

            from ray import train

            def train_func():
                ...
                for i in range(num_epochs):
                    result = model.train(...)
                    train.report({"result": result})

    .. tab-item:: PyTorch Lightning

        在 PyTorch Lightning，我们使用回调调用 ``train.report()``。

        .. code-block:: python

            from ray import train
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import Callback

            class MyRayTrainReportCallback(Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    metrics = {k: v.item() for k, v in metrics.items()}

                    train.report(metrics=metrics)

            def train_func_per_worker():
                ...
                trainer = pl.Trainer(
                    # ...
                    callbacks=[MyRayTrainReportCallback()]
                )
                trainer.fit()

.. _train-aggregating-results:

如何获取并汇总来自不同 worker 的结果？
-----------------------------------------------------------

实际应用中，您可能希望计算除准确性和损失之外的优化指标：召回率、精确度、Fbeta 等。
你可能还想从多个 worker 收集指标。
虽然 Ray Train 目前仅从等级 0 的 worker 报告指标，
但您可以使用第三方库或机器学习框架的分布式原语来报告来自多个 worker 的指标。


.. tab-set::

    .. tab-item:: Native PyTorch

        Ray Train 原生支持 `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_，它为分布式、可扩展的 PyTorch 模型提供了一组机器学习指标。

        下面是报告所有 worker 的汇总 R2 分数和平均训练和验证损失的示例。

        .. literalinclude:: ../doc_code/torchmetrics_example.py
            :language: python
            :start-after: __start__
