.. _train-hf-accelerate:

开始使用 Hugging Face Accelerate
========================================

:class:`~ray.train.torch.TorchTrainer` 可以帮助您轻松地在分布式 Ray 集群中启动 `Accelerate <https://huggingface.co/docs/accelerate>`_  训练。

您只需要使用 TorchTrainer 运行现有的训练代码。最终代码如下所示：

.. code-block:: python

    from accelerate import Accelerator

    def train_func(config):
        # Instantiate the accelerator
        accelerator = Accelerator(...)

        model = ...
        optimizer = ...
        train_dataloader = ...
        eval_dataloader = ...
        lr_scheduler = ...

        # Prepare everything for distributed training
        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

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

.. tip::

    分布式训练的模型和数据准备完全由 `Accelerator <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator>`_ 
    对象及其 `Accelerator.prepare() <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.prepare>`_  方法处理。
    
    与原生 PyTorch、PyTorch Lightning 或 Hugging Face Transformers 不同， **不要** 在训练函数中
    调用任何其他 Ray Train 实用程序（如 :meth:`~ray.train.torch.prepare_model` 或 :meth:`~ray.train.torch.prepare_data_loader` ）。

配置加速
--------------------

在 Ray Train 中，您可以通过训练函数中的 `accelerate.Accelerator <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator>`_ 
对象设置配置。以下是配置 Accelerate 的入门示例。

.. tabs::

    .. group-tab:: DeepSpeed

        例如，要使用 Accelerate 运行 DeepSpeed，请创建 `DeepSpeedPlugin <https://huggingface.co/docs/accelerate/main/en/package_reference/deepspeed>`_ 字典：

        .. code-block:: python

            from accelerate import Accelerator, DeepSpeedPlugin

            DEEPSPEED_CONFIG = {
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": False
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "gather_16bit_weights_on_model_save": True,
                    "round_robin_gradients": True
                },
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "steps_per_print": 10,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False
            }

            def train_func(config):
                # Create a DeepSpeedPlugin from config dict   
                ds_plugin = DeepSpeedPlugin(hf_ds_config=DEEPSPEED_CONFIG)

                # Initialize Accelerator
                accelerator = Accelerator(
                    ...,
                    deepspeed_plugin=ds_plugin,
                )
                
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

    .. group-tab:: FSDP

        对于 PyTorch FSDP，创建一个 `FullyShardedDataParallelPlugin <https://huggingface.co/docs/accelerate/main/en/package_reference/fsdp>`_ 
        并将其传递给加速器。

        .. code-block:: python

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
            from accelerate import Accelerator, FullyShardedDataParallelPlugin

            def train_func(config):
                fsdp_plugin = FullyShardedDataParallelPlugin(
                    state_dict_config=FullStateDictConfig(
                        offload_to_cpu=False, 
                        rank0_only=False
                    ),
                    optim_state_dict_config=FullOptimStateDictConfig(
                        offload_to_cpu=False, 
                        rank0_only=False
                    )
                )

                # Initialize accelerator
                accelerator = Accelerator(
                    ...,
                    fsdp_plugin=fsdp_plugin,
                )

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


请注意，Accelerate 还提供了一个 CLI 工具，`"accelerate config"` ，用于生成配置并使用  `"accelerate launch"` 启动训练作业。
但是，这里不需要它，因为 Ray 的 `TorchTrainer` 已经设置了 Torch 分布式环境并在所有 worker 上启动了训练功能。


接下来，请参阅下面这些端到端示例以了解更多详细信息：

.. tabs::

    .. group-tab:: Ray Data 使用示例

        .. dropdown:: 显示代码

            .. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer.py
                :language: python
                :start-after: __accelerate_torch_basic_example_start__
                :end-before: __accelerate_torch_basic_example_end__

    .. group-tab:: PyTorch DataLoader 使用示例

        .. dropdown:: 显示代码

            .. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer_no_raydata.py
                :language: python
                :start-after: __accelerate_torch_basic_example_no_raydata_start__
                :end-before: __accelerate_torch_basic_example_no_raydata_end__

.. seealso::

    如果你正在寻找更高级的用例，请查看此 Llama-2 微调示例：
    
    - `使用 Deepspeed、Accelerate 和 Ray Train 对 Llama-2 系列模型进行微调。 <https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed>`_

您可能还会发现这些用户指南很有帮助：

- :ref:`配置 Scale 和 GPU <train_scaling_config>`
- :ref:`配置和持久存储 <train-run-config>`
- :ref:`保存和加载检查点 <train-checkpointing>`
- :ref:`如何将 Ray Data 与 Ray Train 结合使用 <data-ingest-torch>`


AccelerateTrainer 迁移指南
---------------------------------

在 Ray 2.7 之前，Ray Train 的 :class:`AccelerateTrainer <ray.train.huggingface.AccelerateTrainer>` 是运行 Accelerate 代码的推荐方式。
作为 :class:`TorchTrainer <ray.train.torch.TorchTrainer>` 的子类，
AccelerateTrainer 接收由 ``accelerate config`` 生成的配置文件并将其应用于所有 worker 。
除此之外， ``AccelerateTrainer`` 的功能与 ``TorchTrainer`` 相同。

然而，这引起了人们的困惑，即这是否是运行 Accelerate 代码的 *唯一* 方式。
由于您可以使用和 ``Accelerator`` 和 ``TorchTrainer`` 组合来表达完整的 Accelerate 功能， 因此计划在 Ray 2.8 中弃用 ``AccelerateTrainer`` ，
并且建议直接使用 ``TorchTrainer`` 来运行加速代码。


