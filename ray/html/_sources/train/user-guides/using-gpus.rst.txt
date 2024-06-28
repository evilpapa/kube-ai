.. _train_scaling_config:

配置 Scale 和 GPU
==========================
增加 Ray Train 训练运行的规模很简单，只需几行代码即可完成。
此操作的主要接口时 :class:`~ray.train.ScalingConfig`，
它配置了工作器的数量及其应使用的资源。

在本指南中， *worker* 是指 Ray Train 分布式训练 worker，
它是运行训练功能的 :ref:`Ray Actor <actor-key-concept>` 。

增加 worker 数量
--------------------------------
控制训练代码中并行性的主要接口是设置工作器的数量。
这可以通过将 ``num_workers`` 属性传递给 :class:`~ray.train.ScalingConfig` 来完成：

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8
    )


使用 GPU
----------
要使用 GPU，请传递 ``use_gpu=True`` 给 :class:`~ray.train.ScalingConfig`。
这将为每个训练 worker 请求一个 GPU。在下面的示例中，
训练将在 8 个 GPU 上运行（8 个工作器，每个工作器使用一个 GPU）。

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        use_gpu=True
    )


在训练函数中使用 GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
设置 ``use_gpu=True`` 后，Ray Train 将自动在您的训练函数中设置环境变量，
以便可以检测和使用 GPU
（例如 ``CUDA_VISIBLE_DEVICES``）。

您可以通过 :meth:`ray.train.torch.get_device` 获取关联的设备。

.. code-block:: python

    import torch
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer, get_device


    def train_func(config):
        assert torch.cuda.is_available()

        device = get_device()
        assert device == torch.device("cuda:0")


    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True
        )
    )
    trainer.fit()


为每个 worker 设置资源
--------------------------------
如果你想为每个训练 worker 分配多个 CPU 或 GPU，或者如果你
定义了 :ref:`自定义集群资源 <cluster-resources>`，请
设置 ``resources_per_worker`` 属性：

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        resources_per_worker={
            "CPU": 4,
            "GPU": 2,
        }
        use_gpu=True,
    )


.. note::
    如果您在 ``resources_per_worker``中指定 GPU ，则还需要设置
    ``use_gpu=True``。

您还可以指示 Ray Train 使用部分 GPU。在这种情况下，多个 worker 将被
分配相同的 CUDA 设备。

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        resources_per_worker={
            "CPU": 4,
            "GPU": 0.5,
        }
        use_gpu=True,
    )


设置通信后端（PyTorch）
-------------------------------------------

.. note::

    这是一个高级设置。大多数情况下，您无需更改此设置。

你可以通过将 :class:`~ray.train.torch.TorchConfig` 传递给 :class:`~ray.train.torch.TorchTrainer`来
设置 PyTorch 分布式通信后端（例如 GLOO 或 NCCL）。

参阅 `PyTorch API 参考 <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__
以了解有效选项。

.. code-block:: python

    from ray.train.torch import TorchConfig, TorchTrainer

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=num_training_workers,
            use_gpu=True,
        ),
        torch_config=TorchConfig(backend="gloo"),
    )


.. _train_trainer_resources:

Trainer 资源
-----------------
到目前为止，我们已经为每个训练 worker 配置了资源。从技术上讲，每个
训练 worker 都是一个 :ref:`Ray Actor <actor-guide>`。当您调用
:meth:`Trainer.fit() <ray.train.trainer.BaseTrainer.fit>` 时，Ray Train 还会为
:class:`Trainer <ray.train.trainer.BaseTrainer>` 对象创建一个 actor。

此对象通常仅管理训练 worker 之间的轻量级通信。
您仍然可以指定其资源，
如果您实施了执行更繁重处理的自己的训练器，这将很有用。

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=8,
        trainer_resources={
            "CPU": 4,
            "GPU": 1,
        }
    )

默认情况下，训练器使用 1 个 CPU。如果您有一个包含 8 个 CPU 的集群，
并且想要启动 4 个训练工作器和 2 个 CPU，那么这将行不通，
因为所需的 CPU 总数将为 9 (4 * 2 + 1)。
在这种情况下，您可以指定训练器资源使用 0 个 CPU：

.. code-block:: python

    from ray.train import ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=4,
        resources_per_worker={
            "CPU": 2,
        },
        trainer_resources={
            "CPU": 0,
        }
    )

