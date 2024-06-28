.. _train-deepspeed:

DeepSpeed 入门
==========================

:class:`~ray.train.torch.TorchTrainer` 帮助您轻松地在分布式 Ray 集群中启动 `DeepSpeed <https://www.deepspeed.ai/>`_  训练。

代码示例
------------

您只需要使用 TorchTrainer 运行现有的训练代码。最终代码如下所示：

.. code-block:: python

    import deepspeed
    from deepspeed.accelerator import get_accelerator

    def train_func(config):
        # Instantiate your model and dataset
        model = ...
        train_dataset = ...
        eval_dataset = ...
        deepspeed_config = {...} # Your Deepspeed config

        # Prepare everything for distributed training
        model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=tokenized_datasets["train"],
            collate_fn=collate_fn,
            config=deepspeed_config,
        )

        # Define the GPU device for the current worker
        device = get_accelerator().device_name(model.local_rank)

        # Start training
        ...
    
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(...),
        ...
    )
    trainer.fit()


下面是仅使用 DeepSpeed 进行 ZeRO-3 训练的一个简单示例。

.. tabs::

    .. group-tab:: Example with Ray Data

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer.py
                :language: python
                :start-after: __deepspeed_torch_basic_example_start__
                :end-before: __deepspeed_torch_basic_example_end__

    .. group-tab:: Example with PyTorch DataLoader

        .. dropdown:: Show Code

            .. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer_no_raydata.py
                :language: python
                :start-after: __deepspeed_torch_basic_example_no_raydata_start__
                :end-before: __deepspeed_torch_basic_example_no_raydata_end__

.. tip::

    要使用纯 PyTorch 运行 DeepSpeed，您 **无需** 在训练函数中提供任何其他 Ray Train 实用程序（例
    如 :meth:`~ray.train.torch.prepare_model` 或 :meth:`~ray.train.torch.prepare_data_loader` ）。
    相反，继续像往常一样使用 `deepspeed.initialize() <https://deepspeed.readthedocs.io/en/latest/initialize.html>`_ 为分布式训练做好一切准备。

使用其他框架运行 DeepSpeed
-----------------------------------

许多深度学习框架已与 DeepSpeed 集成，包括 Lightning、Transformers、Accelerate 等。您可以在 Ray Train 中运行所有这些组合。

查看以下示例以了解更多详细信息：

.. list-table::
   :header-rows: 1

   * - 框架
     - 例子
   * - Accelelate (:ref:`用户指南 <train-hf-accelerate>`)
     - `使用 Deepspeed、Accelerate 和 Ray Train 微调 Llama-2 系列模型。 <https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed>`_
   * - Transformers (:ref:`用户指南 <train-pytorch-transformers>`)
     - :ref:`使用 DeepSpeed 和 Hugging Face Transformers 对 GPT-J-6b 进行微调 <gptj_deepspeed_finetune>`
   * - Lightning (:ref:`用户指南 <train-pytorch-lightning>`)
     - :ref:`使用 DeepSpeed 和 PyTorch Lightning 对 vicuna-13b 进行微调 <vicuna_lightning_deepspeed_finetuning>`
