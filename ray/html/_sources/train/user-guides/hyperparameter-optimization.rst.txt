.. _train-tune:

使用 Ray Tune 进行超参数优化
===================================

Ray Train 原生支持使用  :ref:`Ray Tune <tune-main>` 进行超参数优化。


.. https://docs.google.com/drawings/d/1yMd12iMkyo6DGrFoET1TIlKfFnXX9dfh2u3GSdTz6W4/edit

.. figure:: ../images/train-tuner.svg
    :align: center

    `Tuner` 接受 `Trainer` 并执行多次训练运行，每次都有不同的超参数配置。

关键概念
------------

使用 :class:`~ray.tune.Tuner` 进行超参数优化时，有许多关键概念：

* 你想在 *搜索空间* 中调整的一组超参数。
* 一种 *搜索算法* 可以有效地优化您的参数，
  并且可以选择使用 *scheduler* 来提前停止搜索并加快实验速度。
* *搜索空间*、*搜索算法*、*scheduler* 和 *Trainer* 都传递给 Tuner，
 Tuner 会并行评估多个超参数。
* 每次单独的超参数评估运行称为 *trial*。
* Tuner 会将结果作为 :class:`~ray.tune.ResultGrid` 返回。

.. note::
   Tuner 还可以用于在不使用 Ray Train 的情况下启动超参数调优。
   请参阅 :ref:`Ray Tune 文档 <tune-main>` 以获取更多指南和示例。

基本用法
-----------

你可以使用以下 :class:`Trainer <ray.train.base_trainer.BaseTrainer>` 并
将其传递给 :class:`~ray.tune.Tuner`。

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __basic_start__
    :end-before: __basic_end__



如何配置 Tuner？
-------------------------

有两个主要配置对象可以传递到 Tuner：:class:`TuneConfig <ray.tune.tune_config.TuneConfig>` 和 :class:`RunConfig <ray.train.RunConfig>`。

:class:`TuneConfig <ray.tune.TuneConfig>` 包含调整特定设置，包括：

- 要使用的调整算法
- 对结果进行排序的指标和模式
- 要使用的并行量

以下是 `TuneConfig` 一些常见的配置示例：

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __tune_config_start__
    :end-before: __tune_config_end__

请参阅 :class:`TuneConfig API 参考 <ray.tune.tune_config.TuneConfig>` 以了解更多详细信息。

:class:`RunConfig <ray.train.RunConfig>` 包含比调整特定设置更通用的配置。
其中包括：

- 失败/重试配置
- 详细程度
- 实验名称
- 日志目录
- 检查点配置
- 自定义回调
- 与云存储集成

下面我们展示一些常见的 :class:`RunConfig <ray.train.RunConfig>` 配置。

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __run_config_start__
    :end-before: __run_config_end__

参考 :class:`RunConfig API 参考 <ray.train.RunConfig>` 以了解更多详细信息。


搜索空间配置
--------------------------

`Tuner` 接受 `param_space` 参数，您可以在其中定义超参数配置将
从中进行采样的搜索空间。

根据模型和数据集，您可能需要调整：

- 训练批次大小
- 深度学习训练的学习率（例如图像分类）
- 基于树的模型的最大深度（例如 XGBoost）

您可以使用 Tuner 调整 Ray Train 的大多数参数和配置，
包括但不限于：

- Ray :class:`Datasets <ray.data.Dataset>`
- :class:`~ray.train.ScalingConfig`
- 和其他超参数。


在此处阅读有关 :ref:`Tune 搜索空间 <tune-search-space-tutorial>` 的更多信息。

Train - Tune 陷
--------------------

在将 Tuner 与 Trainer 结合使用时，存在一些有关参数规范的陷阱：

- 默认情况下，配置字典和配置对象将深度合并。
- Tuner 中 ``param_space`` 会覆盖 Tuner 和 Trainer 中的重复参数。
- **例外:** :class:`RunConfig <ray.train.RunConfig>` 和 :class:`TuneConfig <ray.tune.tune_config.TuneConfig>` 的所有参数本质上都是不可调的。

参考 :doc:`/tune/tutorials/tune_get_data_in_and_out` 示例。

高级调整
---------------

Tuner 还提供调整不同数据预处理步骤和不同训练/验证数据集的能力，
如下面的代码片段所示。

.. literalinclude:: ../doc_code/tuner.py
    :language: python
    :start-after: __tune_dataset_start__
    :end-before: __tune_dataset_end__
