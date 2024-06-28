.. _train-pytorch:

PyTorch 入门
========================

本教程介绍将现有 PyTorch 脚本转换为使用 Ray Train 的过程。

了解如何：

1. 配置模型以在正确的 CPU/GPU 设备上运行分布式模型。
2. 配置数据加载器以在各个 :ref:`workers <train-overview-worker>` 之间分片数据，并将数据放置在正确的 CPU 或 GPU 设备上。
3. 配置 :ref:`训练功能 <train-overview-training-function>` 以报告指标并保存检查点。
4. 为训练作业配置 :ref:`扩展 <train-overview-scaling-config>` 及 CPU 或 GPU 资源需求。
5. 启动一个带有 :class:`~ray.train.torch.TorchTrainer` 类的分部署训练作业。

快速开始
----------

作为参考，最终代码如下：

.. code-block:: python

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func(config):
        # Your PyTorch training code here.
    
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. `train_func` 是在每个分布式训练 worker 上执行的 Python 代码。
2. `ScalingConfig` 定义分布式训练 worker 的数量以及是否使用 GPU。
3. `TorchTrainer` 启动分布式训练作业。

比较有和没有 Ray Train 的 PyTorch 训练脚本。

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python

            import tempfile
            import torch
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            from torch.utils.data import DataLoader
            from torch.optim import Adam
            from torch.nn import CrossEntropyLoss

            # Model, Loss, Optimizer
            model = resnet18(num_classes=10)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            criterion = CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.001)

            # Data
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

            # Training
            for epoch in range(10):
                for images, labels in train_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                checkpoint_dir = tempfile.gettempdir() 
                checkpoint_path = checkpoint_dir + "/model.checkpoint"
                torch.save(model.state_dict(), checkpoint_path)

                

    .. group-tab:: PyTorch + Ray Train

        .. code-block:: python
       
            import tempfile
            import torch
            from torchvision.models import resnet18
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor, Normalize, Compose
            from torch.utils.data import DataLoader
            from torch.optim import Adam
            from torch.nn import CrossEntropyLoss
            from ray.train.torch import TorchTrainer
            from ray.train import ScalingConfig, Checkpoint

            def train_func(config):

                # Model, Loss, Optimizer
                model = resnet18(num_classes=10)
                model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                # [1] Prepare model.
                model = ray.train.torch.prepare_model(model)
                criterion = CrossEntropyLoss()
                optimizer = Adam(model.parameters(), lr=0.001)

                # Data
                transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
                train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
                # [2] Prepare dataloader.
                train_loader = ray.train.torch.prepare_data_loader(train_loader)

                # Training
                for epoch in range(10):
                    for images, labels in train_loader:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    checkpoint_dir = tempfile.gettempdir() 
                    checkpoint_path = checkpoint_dir + "/model.checkpoint"
                    torch.save(model.state_dict(), checkpoint_path)
                    # [3] Report metrics and checkpoint.
                    ray.train.report({"loss": loss.item()}, checkpoint=Checkpoint.from_directory(checkpoint_dir))
            
            # [4] Configure scaling and resource requirements.
            scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

            # [5] Launch distributed training job.
            trainer = TorchTrainer(train_func, scaling_config=scaling_config)
            result = trainer.fit()

设置训练函数
--------------------------

首先，更新您的训练代码以支持分布式训练。
首先将您的代码包装在 :ref:`训练函数 <train-overview-training-function>`：

.. code-block:: python

    def train_func(config):
        # Your PyTorch training code here.

每个分布式训练 worker 都执行此功能。

设置模型
^^^^^^^^^^^^^^

使用 :func:`ray.train.torch.prepare_model` 实用函数可以：

1. 将您的模型移动到正确的设备上。
2. 在 ``DistributedDataParallel`` 中进行包装。

.. code-block:: diff

    -from torch.nn.parallel import DistributedDataParallel
    +import ray.train.torch

     def train_func(config): 

         ...

         # Create model.
         model = ...

         # Set up distributed training and device placement.
    -    device_id = ... # Your logic to get the right device.
    -    model = model.to(device_id or "cpu")
    -    model = DistributedDataParallel(model, device_ids=[device_id])
    +    model = ray.train.torch.prepare_model(model)
         
         ...

设置数据集
^^^^^^^^^^^^^^^^

.. TODO: Update this to use Ray Data.

使用 :func:`ray.train.torch.prepare_data_loader` 效用函数，其会：

1. 添加 ``DistributedSampler`` 到你的 ``DataLoader``。
2. 将批次移动到正确的设备。

请注意，如果您将 Ray Data 传递给 Trainer，则此步骤不是必需的。
请参阅 :ref:`data-ingest-torch`。

.. code-block:: diff

     from torch.utils.data import DataLoader
    -from torch.utils.data import DistributedSampler
    +import ray.train.torch

     def train_func(config):

         ...

         dataset = ...
         
         data_loader = DataLoader(dataset, batch_size=worker_batch_size)
    -    data_loader = DataLoader(dataset, batch_size=worker_batch_size, sampler=DistributedSampler(dataset)) 
    +    data_loader = ray.train.torch.prepare_data_loader(data_loader)

         for X, y in data_loader:
    -        X = X.to_device(device)
    -        y = y.to_device(device)

         ...

.. tip::
    请记住， ``DataLoader`` 其中 ``batch_size``是每个 worker 的批处理大小。
    全局批处理大小可以通过以下公式根据工作器批处理大小计算得出（反之亦然）：

    .. code-block:: python

        global_batch_size = worker_batch_size * ray.train.get_context().get_world_size()


报告检查点和指标
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了监控进度，您可以使用 :func:`ray.train.report` 实用功能报告中间指标和检查点。

.. code-block:: diff

    +import ray.train
    +from ray.train import Checkpoint

     def train_func(config):

         ...
         torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth"))
    +    metrics = {"loss": loss.item()} # Training/validation metrics.
    +    checkpoint = Checkpoint.from_directory(checkpoint_dir) # Build a Ray Train checkpoint from a directory
    +    ray.train.report(metrics=metrics, checkpoint=checkpoint)

         ...

有关更多详细信息，请参阅 :ref:`train-monitoring-and-logging` 和 :ref:`train-checkpointing`。


配置比例和 GPU
------------------------

在你的训练功能之外，创建一个 :class:`~ray.train.ScalingConfig` 对象来配置：

1. `num_workers` - 分布式训练 worker 的数量。
2. `use_gpu` - 每个 worker 至少使用一个 GPU （或 CPU）。

.. code-block:: python

    from ray.train import ScalingConfig
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)


更多详细信息，请参阅 :ref:`train_scaling_config`。

启动训练任务
---------------------

将所有这些结合在一起，您现在可以使用 :class:`~ray.train.torch.TorchTrainer` 启动分布式训练工作。

.. code-block:: python

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

访问训练结果
-----------------------

训练完成后， :class:`~ray.train.Result` 将返回一个对象，
其中包含有关训练运行的信息，包括训练期间报告的指标和检查点。

.. code-block:: python

    result.metrics     # The metrics reported during training.
    result.checkpoint  # The latest checkpoint reported during training.
    result.path     # The path where logs are stored.
    result.error       # The exception that was raised, if training failed.

.. TODO: Add results guide

下一步
----------

将 PyTorch 训练脚本转换为使用 Ray Train 后：

* 参阅 :ref:`用户指南 <train-user-guides>` 以了解有关如何执行特定任务的更多信息。
* 浏览 :ref:`示例 <train-examples>` ，了解如何使用 Ray Train 的端到端示例。
* 深入 :ref:`API 参考 <train-api>` ，了解本教程中使用的类和方法的更多详细信息。