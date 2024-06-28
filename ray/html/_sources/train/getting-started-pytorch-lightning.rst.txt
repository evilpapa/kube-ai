.. _train-pytorch-lightning:

开始使用 PyTorch Lightning
==================================

本教程介绍将现有 PyTorch Lightning 脚本转换为使用 Ray Train 的过程。

了解如何：

1. 配置 Lightning Trainer，使其与 Ray 一起分布在正确的 CPU 或 GPU 设备上运行。
2. 配置 :ref:`训练函数 <train-overview-training-function>` 以报告指标并保存检查点。
3. 为训练作业配置 :ref:`扩展 <train-overview-scaling-config>` 和 CPU 或 GPU 资源要求。
4. 使用 :class:`~ray.train.torch.TorchTrainer` 启动分布式训练作业。

快速开始
----------

作为参考，最终代码如下：

.. code-block:: python

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func(config):
        # Your PyTorch Lightning training code here.
    
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. 您的 `train_func` 是每个分布式训练 :ref:`worker <train-overview-worker>` 执行的 Python 代码。
2. 您的 `ScalingConfig` 定义分布式训练 worker 的数量以及是否使用 GPU。
3. 您的 `TorchTrainer` 启动了分布式训练作业。

比较有和没有 Ray Train 的 PyTorch Lightning 训练脚本。

.. tabs::

    .. group-tab:: PyTorch Lightning

        .. code-block:: python

            import torch
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            from torch.utils.data import DataLoader
            import pytorch_lightning as pl

            # Model, Loss, Optimizer
            class ImageClassifier(pl.LightningModule):
                def __init__(self):
                    super(ImageClassifier, self).__init__()
                    self.model = resnet18(num_classes=10)
                    self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    self.criterion = torch.nn.CrossEntropyLoss()
                
                def forward(self, x):
                    return self.model(x)
                
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    outputs = self.forward(x)
                    loss = self.criterion(outputs, y)
                    self.log("loss", loss, on_step=True, prog_bar=True)
                    return loss
                    
                def configure_optimizers(self):
                    return torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Data
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
            train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

            # Training
            model = ImageClassifier()
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(model, train_dataloaders=train_dataloader)

                

    .. group-tab:: PyTorch Lightning + Ray Train

        .. code-block:: python

            import torch
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            from torch.utils.data import DataLoader
            import pytorch_lightning as pl

            from ray.train.torch import TorchTrainer
            from ray.train import ScalingConfig
            import ray.train.lightning

            # Model, Loss, Optimizer
            class ImageClassifier(pl.LightningModule):
                def __init__(self):
                    super(ImageClassifier, self).__init__()
                    self.model = resnet18(num_classes=10)
                    self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    self.criterion = torch.nn.CrossEntropyLoss()
                
                def forward(self, x):
                    return self.model(x)
                
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    outputs = self.forward(x)
                    loss = self.criterion(outputs, y)
                    self.log("loss", loss, on_step=True, prog_bar=True)
                    return loss
                    
                def configure_optimizers(self):
                    return torch.optim.Adam(self.model.parameters(), lr=0.001)
       

            def train_func(config):

                # Data
                transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
                train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

                # Training
                model = ImageClassifier()
                # [1] Configure PyTorch Lightning Trainer.
                trainer = pl.Trainer(
                    max_epochs=10,
                    devices="auto",
                    accelerator="auto",
                    strategy=ray.train.lightning.RayDDPStrategy(),
                    plugins=[ray.train.lightning.RayLightningEnvironment()],
                    callbacks=[ray.train.lightning.RayTrainReportCallback()],
                )
                trainer = ray.train.lightning.prepare_trainer(trainer)
                trainer.fit(model, train_dataloaders=train_dataloader)

            # [2] Configure scaling and resource requirements.
            scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

            # [3] Launch distributed training job.
            trainer = TorchTrainer(train_func, scaling_config=scaling_config)
            result = trainer.fit()            


设置训练函数
--------------------------

首先，更新您的训练代码以支持分布式训练。
首先将您的代码包装在 :ref:`训练函数 <train-overview-training-function>`：

.. code-block:: python

    def train_func(config):
        # Your PyTorch Lightning training code here.

每个分布式训练 worker 都执行此功能。


ay Train 在每个 worker 上设置分布式进程组。
您只需对 Lightning Trainer 定义进行一些更改。

.. code-block:: diff

     import pytorch_lightning as pl
    -from pl.strategies import DDPStrategy
    -from pl.plugins.environments import LightningEnvironment
    +import ray.train.lightning 

     def train_func(config):
         ...
         model = MyLightningModule(...)
         datamodule = MyLightningDataModule(...)
        
         trainer = pl.Trainer(
    -        devices=[0,1,2,3],
    -        strategy=DDPStrategy(),
    -        plugins=[LightningEnvironment()],
    +        devices="auto",
    +        accelerator="auto",
    +        strategy=ray.train.lightning.RayDDPStrategy(),
    +        plugins=[ray.train.lightning.RayLightningEnvironment()]
         )
    +    trainer = ray.train.lightning.prepare_trainer(trainer)
        
         trainer.fit(model, datamodule=datamodule)

以下各节讨论了每个变化。

配置分布式策略
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray Train 为 Lightning 提供了几种子类分布式策略。
这些策略保留与其基本策略类相同的参数列表。
在内部，它们配置根设备和分布式采样器参数。
    
- :class:`~ray.train.lightning.RayDDPStrategy` 
- :class:`~ray.train.lightning.RayFSDPStrategy` 
- :class:`~ray.train.lightning.RayDeepSpeedStrategy` 


.. code-block:: diff

     import pytorch_lightning as pl
    -from pl.strategies import DDPStrategy
    +import ray.train.lightning

     def train_func(config):
         ...
         trainer = pl.Trainer(
             ...
    -        strategy=DDPStrategy(),
    +        strategy=ray.train.lightning.RayDDPStrategy(),
             ...
         )
         ...

配置 Ray 集群环境插件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray Train 还提供了一个 :class:`~ray.train.lightning.RayLightningEnvironment` 类
作为 Ray Cluster 的规范。此实用程序类配置了 worker 的本地、全局和节点等级以及世界大小。


.. code-block:: diff

     import pytorch_lightning as pl
    -from pl.plugins.environments import LightningEnvironment
    +import ray.train.lightning

     def train_func(config):
         ...
         trainer = pl.Trainer(
             ...
    -        plugins=[LightningEnvironment()],
    +        plugins=[ray.train.lightning.RayLightningEnvironment()],
             ...
         )
         ...


配置并行设备
^^^^^^^^^^^^^^^^^^^^^^^^^^

此外，Ray TorchTrainer 已经为您配置了正确的
``CUDA_VISIBLE_DEVICES``。应始终通过设置
``devices="auto"`` 和 ``acelerator="auto"`` 来使用所有可用的 GPU。


.. code-block:: diff

     import pytorch_lightning as pl

     def train_func(config):
         ...
         trainer = pl.Trainer(
             ...
    -        devices=[0,1,2,3],
    +        devices="auto",
    +        accelerator="auto",
             ...
         )
         ...



报告检查点和指标
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了保留您的检查点并监控训练进度，请添加
:class:`ray.train.lightning.RayTrainReportCallback` 实用回调程序到你的 Trainer 中。

                    
.. code-block:: diff

     import pytorch_lightning as pl
     from ray.train.lightning import RayTrainReportCallback

     def train_func(config):
         ...
         trainer = pl.Trainer(
             ...
    -        callbacks=[...],
    +        callbacks=[..., RayTrainReportCallback()],
         )
         ...


向 Ray Train 报告指标和检查点使您能够支持 :ref:`容错训练 <train-fault-tolerance>` 和 :ref:`超参数优化 <train-tune>`.。
请注意，该 :class:`ray.train.lightning.RayTrainReportCallback` 类仅提供简单的实现，并且可以 :ref:`进一步定制 <train-dl-saving-checkpoints>`。

准备你的 Lightning Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后，将您的 Lightning Trainer 传入
:meth:`~ray.train.lightning.prepare_trainer` 以
验证您的配置。


.. code-block:: diff

     import pytorch_lightning as pl
     import ray.train.lightning

     def train_func(config):
         ...
         trainer = pl.Trainer(...)
    +    trainer = ray.train.lightning.prepare_trainer(trainer)
         ...


配置规模和 GPU
------------------------

在你的训练功能之外，创建一个 :class:`~ray.train.ScalingConfig` 对象来配置：

1. `num_workers` - 分布式训练 worker 的数量。
2. `use_gpu` - 每个 worker 是否应该使用 GPU（或 CPU）。

.. code-block:: python

    from ray.train import ScalingConfig
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)


有关更多详细信息，请参阅 :ref:`train_scaling_config`。

启动训练任务
---------------------

将所有这些结合在一起，您现在可以使用
:class:`~ray.train.torch.TorchTrainer` 启动分布式训练工作。

.. code-block:: python

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

有关更多 `TorchTrainer` 的配置选项，请参阅 :ref:`train-run-config` 。

访问训练结果
-----------------------

训练完成后，Ray Train 返回一个 :class:`~ray.train.Result` 对象，
其中包含有关训练运行的信息，包括训练期间报告的指标和检查点。

.. code-block:: python

    result.metrics     # The metrics reported during training.
    result.checkpoint  # The latest checkpoint reported during training.
    result.path     # The path where logs are stored.
    result.error       # The exception that was raised, if training failed.

.. TODO: Add results guide

下一步
---------- 

将 PyTorch Lightning 训练脚本转换为使用 Ray Train 后：

* 请参阅 :ref:`用户指南 <train-user-guides>` 以了解有关如何执行特定任务的更多信息。
* 浏览 :ref:`示例 <train-examples>` ，了解如何使用 Ray Train 的端到端示例。
* 有关本教程中的类和方法的更多详细信息，请参阅 :ref:`API 参考 <train-api>`。

版本兼容性
---------------------

Ray Train 已使用 `pytorch_lightning` 版本 `1.6.5` 和 `2.0.4` 进行了测试。 要获得完全兼容性，请使用 ``pytorch_lightning>=1.6.5`` 。 
早期版本不被禁止，但可能会导致意外问题。 
如果遇到任何兼容性问题，请考虑升级您的 PyTorch Lightning 版本或 `提交问题 <<https://github.com/ray-project/ray/issues>` _。

.. _lightning-trainer-migration-guide:

LightningTrainer 迁移指南
--------------------------------

Ray 2.4 引入了 `LightningTrainer`，并公开了
`LightningConfigBuilder` 配置来定义 `pl.LightningModule` 
和 `pl.Trainer`。

然后，它实例化模型和训练器对象并在黑盒中
运行预定义的训练函数。

此版本的 LightningTrainer API 具有约束力，
限制了您管理训练功能的能力。

Ray 2.7 引入了新统一的 :class:`~ray.train.torch.TorchTrainer` API，
它提供了增强的透明度、灵活性和简单性。
此 API 与标准 PyTorch Lightning 脚本更加一致，
确保用户更好地控制其原生 Lightning 代码。


.. tabs::

    .. group-tab:: (弃用) LightningTrainer


        .. code-block:: python
            
            from ray.train.lightning import LightningConfigBuilder, LightningTrainer

            config_builder = LightningConfigBuilder()
            # [1] Collect model configs
            config_builder.module(cls=MNISTClassifier, lr=1e-3, feature_dim=128)

            # [2] Collect checkpointing configs
            config_builder.checkpointing(monitor="val_accuracy", mode="max", save_top_k=3)

            # [3] Collect pl.Trainer configs
            config_builder.trainer(
                max_epochs=10,
                accelerator="gpu",
                log_every_n_steps=100,
                logger=CSVLogger("./logs"),
            )

            # [4] Build datasets on the head node
            datamodule = MNISTDataModule(batch_size=32)
            config_builder.fit_params(datamodule=datamodule)

            # [5] Execute the internal training function in a black box
            ray_trainer = LightningTrainer(
                lightning_config=config_builder.build(),
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
                run_config=RunConfig(
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=3,
                        checkpoint_score_attribute="val_accuracy",
                        checkpoint_score_order="max",
                    ),
                )
            )
            ray_trainer.fit()

                

    .. group-tab:: (新 API) TorchTrainer

        .. code-block:: python
            
            import pytorch_lightning as pl
            from ray.train.torch import TorchTrainer
            from ray.train.lightning import (
                RayDDPStrategy, 
                RayLightningEnvironment,
                RayTrainReportCallback,
                prepare_trainer
            ) 

            def train_func(config):
                # [1] Create a Lightning model
                model = MNISTClassifier(lr=1e-3, feature_dim=128)

                # [2] Report Checkpoint with callback
                ckpt_report_callback = RayTrainReportCallback()
                
                # [3] Create a Lighting Trainer
                datamodule = MNISTDataModule(batch_size=32)

                trainer = pl.Trainer(
                    max_epochs=10,
                    log_every_n_steps=100,
                    logger=CSVLogger("./logs"),
                    # New configurations below
                    devices="auto",
                    accelerator="auto",
                    strategy=RayDDPStrategy(),
                    plugins=[RayLightningEnvironment()],
                    callbacks=[ckpt_report_callback],
                )

                # Validate your Lightning trainer configuration
                trainer = prepare_trainer(trainer)

                # [4] Build your datasets on each worker
                datamodule = MNISTDataModule(batch_size=32)
                trainer.fit(model, datamodule=datamodule)

            # [5] Explicitly define and run the training function
            ray_trainer = TorchTrainer(
                train_func_per_worker,
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
                run_config=RunConfig(
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=3,
                        checkpoint_score_attribute="val_accuracy",
                        checkpoint_score_order="max",
                    ),
                )
            )
            ray_trainer.fit()
