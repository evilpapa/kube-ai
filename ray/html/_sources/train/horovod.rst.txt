
.. _train-horovod:

Horovod 指南
========================

Ray Train 为你配置 Horovod 环境 和 Rendezvous
服务。让您可以运行你的 ``DistributedOptimizer`` 训练脚本。
有关更多信息，参考 `Horovod 文档 <https://horovod.readthedocs.io/en/stable/index.html>`_ 。

快速开始
-----------
.. literalinclude:: ./doc_code/hvd_trainer.py
  :language: python



更新你的训练函数
-----------------------------

首先，更新 :ref:`训练函数 <train-overview-training-function>` 以支持分布式训练。

如果你的训练函数已经可以在 `Horovod Ray Executor <https://horovod.readthedocs.io/en/stable/ray_include.html#horovod-ray-executor>`_ 上运行，
你不需要做任何额外的更改。

要加入 Horovod，请访问 `Horovod 指南 
<https://horovod.readthedocs.io/en/stable/index.html#get-started>`_。


创建 HorovodTrainer
-----------------------

``Trainer``\s 是主要的 Ray Train 类用于管理状态和执行训练。
对于 Horovod 使用 :class:`~ray.train.horovod.HorovodTrainer`，
你可以这样设置：

.. code-block:: python

    from ray.train import ScalingConfig
    from ray.train.horovod import HorovodTrainer
    # For GPU Training, set `use_gpu` to True.
    use_gpu = False
    trainer = HorovodTrainer(
        train_func,
        scaling_config=ScalingConfig(use_gpu=use_gpu, num_workers=2)
    )

使用 Horovod 进行训练时，始终使用 HorovodTrainer，
无论训练框架是什么，例如 PyTorch 或 TensorFlow。

要自定义后端设置，可以传递一个 
:class:`~ray.train.horovod.HorovodConfig`：

.. code-block:: python

    from ray.train import ScalingConfig
    from ray.train.horovod import HorovodTrainer, HorovodConfig

    trainer = HorovodTrainer(
        train_func,
        tensorflow_backend=HorovodConfig(...),
        scaling_config=ScalingConfig(num_workers=2),
    )

更多配置信息，参考 :py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` API。

运行训练函数
-----------------------

借助分布式训练函数和 Ray Train ``Trainer``，
你现在可以开始训练了。

.. code-block:: python

    trainer.fit()


进一步阅读
---------------

Ray Train 的 :class:`~ray.train.horovod.HorovodTrainer` 用自己的实现
替换了本机库的分布式通信后端。
因此，其余的集成点保持不变。 如果您将 Horovod 与
:ref:`PyTorch <train-pytorch>` 或 :ref:`Tensorflow <train-tensorflow-overview>` 集成，
请参阅相应的指南以获取进一步的配置和信息。

如果您正在实施自己的基于 Horovod 的训练程序而
不使用任何训练库， 请阅读
:ref:`用户指南 <train-user-guides>`，因为您可以将大部分内容应用于
通用用例并轻松地对其进行调整。


