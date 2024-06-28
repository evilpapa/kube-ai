.. _train-inspect-results:

检查训练结果
===========================

调用 :meth:`Trainer.fit() <ray.train.trainer.BaseTrainer.fit>` 的返回结果是
一个 :class:`~ray.train.Result` 对象。

:class:`~ray.train.Result` 对象包含以下信息：

- 最后报告的指标（例如损失）
- 最后报告的检查点（加载模型）
- 如果发生任何错误，则显示错误消息

查看指标
---------------
您可以从 :class:`~ray.train.Result` 对象检索报告给 Ray Train 的指标。

常见指标包括训练或验证损失，或预测准确性。

从 :class:`~ray.train.Result` 对象检索到的指标与
您作为 :ref:`在您的训练函数中 <train-monitoring-and-logging>` 的参数
传递给 :func:`train.report <ray.train.report>` 的指标相对应，。


最近报告的指标
~~~~~~~~~~~~~~~~~~~~~

使用 :attr:`Result.metrics <ray.train.Result.metrics>` 检索
最新报告的指标。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_metrics_start__
    :end-before: __result_metrics_end__

所有报告指标的数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 :attr:`Result.metrics_dataframe <ray.train.Result.metrics_dataframe>`检索所有报告指标的 pandas DataFrame。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_dataframe_start__
    :end-before: __result_dataframe_end__


检索检查点
----------------------
您可以从 :class:`~ray.train.Result` 对象检索报告给 Ray Train 的检查点。

:ref:`Checkpoints <train-checkpointing>` 包含恢复训练状态所需的所有信息。、
这通常包括训练后的模型。

您可以将检查点用于常见的下游任务，例如
:ref:`使用 Ray Data 进行离线批量推理 <batch_inference_ray_train>`,
或 :doc:`使用 Ray Serve 进行在线模型服务 </serve/index>`。

:class:`~ray.train.Result` 对象检索到的检查点与您
:ref:`在训练函数中 <train-monitoring-and-logging>` 作为参数
传递给 :func:`train.report <ray.train.report>` 的检查点相对应。

最后保存的检查点
~~~~~~~~~~~~~~~~~~~~~
使用 :attr:`Result.checkpoint <ray.train.Result.checkpoint>`检索最后一个检查点。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_checkpoint_start__
    :end-before: __result_checkpoint_end__


其他检查点
~~~~~~~~~~~~~~~~~
有时，您希望访问较早的检查点。例如，如果
由于过度拟合导致损失在更多训练后增加，
您可能希望检索损失最低的检查点。

您可以使用以下方式 :attr:`Result.best_checkpoints <ray.train.Result.best_checkpoints>` 检索
所有可用检查点及其指标的列表

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_best_checkpoint_start__
    :end-before: __result_best_checkpoint_end__

访问存储位置
---------------------------
如果您稍后需要检索结果，可以使用 :attr:`Result.path <ray.train.Result.path>`
获取训练运行的存储位置。

此路径将对应于您在 :class:`~ray.train.RunConfig` 中
配置的 :ref:`storage_path <train-log-dir>` 。
它将是该路径内的（嵌套）子目录，通常为
`TrainerName_date-string/TrainerName_id_00000_0_...` 的形式。


.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_path_start__
    :end-before: __result_path_end__


你可以使用 :meth:`Result.from_path <ray.train.Result.from_path>` 恢复结果：

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_restore_start__
    :end-before: __result_restore_end__



查看错误
--------------
如果训练期间发生错误，
:attr:`Result.error <ray.train.Result.error>` 
将会设置并包含引发的异常。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __result_error_start__
    :end-before: __result_error_end__


在持久存储中查找结果
-------------------------------------
有训练结果（包括报告的指标、检查点和错误文件）都存储
在配置的 :ref:`持久化存储 <train-log-dir>` 中。

参考 :ref:`我们的持久化存储指南 <train-log-dir>`，为您的
训练运行位置配置。
