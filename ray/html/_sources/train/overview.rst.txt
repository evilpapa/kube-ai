.. _train-key-concepts:

.. _train-overview:

Ray Train 概览
==================

        
要有效使用 Ray Train，您需要了解四个主要概念：

#. :ref:`训练函数 <train-overview-training-function>`: 包含模型训练逻辑的 Python 函数。
#. :ref:`Worker <train-overview-worker>`: 运行训练功能的进程。
#. :ref:`扩展配置: <train-overview-scaling-config>` worker 数量和计算资源（例如 CPU 或 GPU）的配置。
#. :ref:`训练器 <train-overview-trainers>`: 一个 Python 类，将训练函数、工作者和扩展配置结合在一起，以执行分布式训练作业。

.. figure:: images/overview.png
    :align: center

.. _train-overview-training-function:

训练函数
-----------------

训练函数是用户定义的 Python 函数，包含端到端模型训练循环逻辑。
启动分布式训练作业时，每个 worker 都会执行此训练函数。

Ray Train 文档使用以下约定：

#. `train_func` 是一个包含训练代码的用户定义函数。
#. `train_func` 被传递到 Trainer 的 `train_loop_per_worker` 参数中。

.. code-block:: python

    def train_func():
        """User-defined training function that runs on each distributed worker process.
        
        This function typically contains logic for loading the model, 
        loading the dataset, training the model, saving checkpoints, 
        and logging metrics.
        """
        ...

.. _train-overview-worker:

Worker
------

Ray Train 将模型训练计算分发到集群中的各个工作进程。
每个工作进程都是执行 `train_func` 的进程。
工作进程的数量决定了训练作业的并行性，并在 `ScalingConfig` 中配置。

.. _train-overview-scaling-config:

扩展配置
---------------------

:class:`~ray.train.ScalingConfig` 是定义训练作业规模的机制。
指定并行度和计算资源的两个基本参数：

* `num_workers`: 为分布式训练作业启动的 worker 数量。
* `use_gpu`: 每个 worker 是否应该使用 GPU 还是 CPU。

.. code-block:: python

    from ray.train import ScalingConfig

    # Single worker with a CPU
    scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

    # Single worker with a GPU
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

    # Multiple workers, each with a GPU
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True)

.. _train-overview-trainers:

训练器
-------

Trainer 将前三个概念结合在一起，以启动分布式训练作业。
Ray Train 为不通框架提供 :ref:`Trainer 类 <train-api>` 。
通过调用 `fit()` 方法执行训练作业：

#. 按照 `scaling_config` 的定义启动 worker。
#. 在所有 worker 上设置框架的分布式环境。
#. 在所有 worker 上运行 `train_func` 。

.. code-block:: python

    from ray.train.torch import TorchTrainer
    
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    trainer.fit()