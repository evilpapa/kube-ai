.. _train-pytorch-transformers:

开始使用 Hugging Face Transformers
==========================================

本教程介绍将现有 Hugging Face Transformers 脚本转换为使用 Ray Train 的过程。

了解如何：

1. 配置 a :ref:`训练函数 <train-overview-training-function>` 以报告指标并保存检查点。
2. 为您的训练作业配置 :ref:`扩展 <train-overview-scaling-config>` 和 CPU 或 GPU 资源要求。
3. 使用 :class:`~ray.train.torch.TorchTrainer` 启动您的分布式训练任务。

快速开始
----------

作为参考，最终代码如下：

.. code-block:: python

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func(config):
        # Your Transformers training code here.
    
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

1. `train_func` 是在每个分布式训练 :ref:`worker <train-overview-worker>` 上执行的 Python 代码。
2. :class:`~ray.train.ScalingConfig` 定义分布式训练 worker 和计算资源（例如 GPU）的数量。
3. :class:`~ray.train.torch.TorchTrainer` 启动分布式训练作业。

比较有和没有 Ray Train 的 Hugging Face Transformers 训练脚本。

.. tabs::

    .. group-tab:: Hugging Face Transformers

        .. code-block:: python

            # Adapted from Hugging Face tutorial: https://huggingface.co/docs/transformers/training

            import numpy as np
            import evaluate
            from datasets import load_dataset
            from transformers import (
                Trainer,
                TrainingArguments,
                AutoTokenizer, 
                AutoModelForSequenceClassification,
            )

            # Datasets
            dataset = load_dataset("yelp_review_full")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)

            small_train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
            small_eval_dataset = dataset["test"].select(range(1000)).map(tokenize_function, batched=True)

            # Model
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-cased", num_labels=5
            )

            # Metrics
            metric = evaluate.load("accuracy")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)

            # Hugging Face Trainer
            training_args = TrainingArguments(
                output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=small_train_dataset,
                eval_dataset=small_eval_dataset,
                compute_metrics=compute_metrics,
            )

            # Start Training
            trainer.train()

                

    .. group-tab:: Hugging Face Transformers + Ray Train

        .. code-block:: python

            import numpy as np
            import evaluate
            from datasets import load_dataset
            from transformers import (
                Trainer,
                TrainingArguments,
                AutoTokenizer, 
                AutoModelForSequenceClassification,
            )

            import ray.train.huggingface.transformers
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer

            # [1] Encapsulate data preprocessing, training, and evaluation 
            # logic in a training function
            # ============================================================
            def train_func(config):
                # Datasets
                dataset = load_dataset("yelp_review_full")
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

                def tokenize_function(examples):
                    return tokenizer(examples["text"], padding="max_length", truncation=True)

                small_train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
                small_eval_dataset = dataset["test"].select(range(1000)).map(tokenize_function, batched=True)

                # Model
                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-cased", num_labels=5
                )

                # Evaluation Metrics
                metric = evaluate.load("accuracy")

                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    return metric.compute(predictions=predictions, references=labels)

                # Hugging Face Trainer
                training_args = TrainingArguments(
                    output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset,
                    compute_metrics=compute_metrics,
                )

                # [2] Report Metrics and Checkpoints to Ray Train
                # ===============================================
                callback = ray.train.huggingface.transformers.RayTrainReportCallback()
                trainer.add_callback(callback)

                # [3] Prepare Transformers Trainer
                # ================================
                trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

                # Start Training
                trainer.train()

            # [4] Define a Ray TorchTrainer to launch `train_func` on all workers
            # ===================================================================
            ray_trainer = TorchTrainer(
                train_func, scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
            )
            ray_trainer.fit()


设置训练函数
--------------------------

首先，更新您的训练代码以支持分布式训练。
您可以先将代码包装在 :ref:`训练函数 <train-overview-training-function>` 中：

.. code-block:: python

    def train_func(config):
        # Your Transformers training code here.

此函数在每个分布式训练 worker 上执行。
Ray Train 在进入此函数之前在每个 worker 上设置分布式进程组。

将所有逻辑放入此函数中，包括数据集构建和预处理、模型初始化、
变压器训练器定义等。

.. note::

    如果您正在使用 Hugging Face Datasets 或 Evaluate，请确保在训练函数内部调用 ``datasets.load_dataset`` 和 ``evaluate.load`` 
    不要从训练函数外部传递加载的数据集和指标，
    因为这可能会导致在将对象传输给工作器时出现序列化错误。


报告检查点和指标
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了保留您的检查点并监控训练进度，请想您的训练器添加
:class:`ray.train.huggingface.transformers.RayTrainReportCallback` 使用回调程序。


.. code-block:: diff

     import transformers
     from ray.train.huggingface.transformers import RayTrainReportCallback

     def train_func(config):
         ...
         trainer = transformers.Trainer(...)
    +    trainer.add_callback(RayTrainReportCallback())
         ...


向 Ray Train 报告指标和检查点可确保您可以使用 Ray Tune 和 :ref:`容错训练 <train-fault-tolerance>`。
请注意， :class:`ray.train.huggingface.transformers.RayTrainReportCallback` 仅提供了一个简单的实现，您可以 :ref:`进一步定义 <train-dl-saving-checkpoints>` 它。


准备一个 Transformers 训练器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最后，将您的 Transformers Trainer 传入
:meth:`~ray.train.huggingface.transformers.prepare_trainer` 以验证您的配置
并启用 Ray Data Integration。


.. code-block:: diff

     import transformers
     import ray.train.huggingface.transformers

     def train_func(config):
         ...
         trainer = transformers.Trainer(...)
    +    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
         trainer.train()
         ...


配置规模和 GPU
------------------------

在你的训练功能之外，创建一个 :class:`~ray.train.ScalingConfig` 对象来配置：

1. `num_workers` - T分布式训练 worker 的数量。
2. `use_gpu` - 每个 worker 是否应该使用 GPU（或 CPU）。

.. code-block:: python

    from ray.train import ScalingConfig
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)


有关更多详细信息，请参阅 :ref:`train_scaling_config`。

启动训练任务
---------------------

将所有这些结合在一起，您现在可以使用 :class:`~ray.train.torch.TorchTrainer` 启动分布式训练任务。

.. code-block:: python

    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

有关更多配置选项，请参阅 `TorchTrainer` 中的 :ref:`train-run-config` 。

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

Next steps
---------- 

将 Hugging Face Transformers 训练脚本转换为使用 Ray Train 后：

* 请参阅 :ref:`用户指南 <train-user-guides>` 以了解有关如何执行特定任务的更多信息。
* 浏览 :ref:`示例 <train-examples>` ，了解如何使用 Ray Train 的端到端示例。
* 有关本教程中的类和方法的更多详细信息，请参阅 :ref:`API 参考 <train-api>`。


.. _transformers-trainer-migration-guide:

TransformersTrainer 迁移指南
-----------------------------------

Ray 2.1 引入了 `TransformersTrainer`，它公开一个 `trainer_init_per_worker` 接口
来定义 `transformers.Trainer`，然后在黑盒中运行预定义的训练函数。

Ray 2.7 引入了全新统一的 :class:`~ray.train.torch.TorchTrainer` API，
它提供了增强的透明度、灵活性和简单性。
此 API 与标准 Hugging Face Transformers 脚本更加一致，
确保您能够更好地控制原生 Transformers 训练代码。


.. tabs::

    .. group-tab:: (弃用) TransformersTrainer


        .. code-block:: python
            
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM
            from datasets import load_dataset

            import ray
            from ray.train.huggingface import TransformersTrainer
            from ray.train import ScalingConfig

            # Dataset
            def preprocess(examples):
                ...

            hf_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
            processed_ds = hf_datasets.map(preprocess, ...)

            ray_train_ds = ray.data.from_huggingface(processed_ds["train"])
            ray_eval_ds = ray.data.from_huggingface(processed_ds["validation"])

            # Define the Trainer generation function
            def trainer_init_per_worker(train_dataset, eval_dataset, **config):
                MODEL_NAME = "gpt2"
                model_config = AutoConfig.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_config(model_config)
                args = transformers.TrainingArguments(
                    output_dir=f"{MODEL_NAME}-wikitext2",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="epoch",
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    max_steps=100,
                )
                return transformers.Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                )

            # Build a Ray TransformersTrainer
            scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
            ray_trainer = TransformersTrainer(
                trainer_init_per_worker=trainer_init_per_worker,
                scaling_config=scaling_config,
                datasets={"train": ray_train_ds, "evaluation": ray_eval_ds},
            )
            result = ray_trainer.fit()
                

    .. group-tab:: (新 API) TorchTrainer

        .. code-block:: python
            
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM
            from datasets import load_dataset

            import ray
            from ray.train.huggingface.transformers import (
                RayTrainReportCallback,
                prepare_trainer,
            )
            from ray.train import ScalingConfig

            # Dataset
            def preprocess(examples):
                ...

            hf_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
            processed_ds = hf_datasets.map(preprocess, ...)

            ray_train_ds = ray.data.from_huggingface(processed_ds["train"])
            ray_eval_ds = ray.data.from_huggingface(processed_ds["evaluation"])

            # [1] Define the full training function
            # =====================================
            def train_func(config):
                MODEL_NAME = "gpt2"
                model_config = AutoConfig.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_config(model_config)

                # [2] Build Ray Data iterables
                # ============================
                train_dataset = ray.train.get_dataset_shard("train")
                eval_dataset = ray.train.get_dataset_shard("evaluation")

                train_iterable_ds = train_dataset.iter_torch_batches(batch_size=8)
                eval_iterable_ds = eval_dataset.iter_torch_batches(batch_size=8)

                args = transformers.TrainingArguments(
                    output_dir=f"{MODEL_NAME}-wikitext2",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="epoch",
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    max_steps=100,
                )
                
                trainer = transformers.Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_iterable_ds,
                    eval_dataset=eval_iterable_ds,
                )

                # [3] Inject Ray Train Report Callback
                # ====================================
                trainer.add_callback(RayTrainReportCallback())

                # [4] Prepare your trainer
                # ========================
                trainer = prepare_trainer(trainer)
                trainer.train()

            # Build a Ray TorchTrainer
            scaling_config = ScalingConfig(num_workers=4, use_gpu=True)
            ray_trainer = TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                datasets={"train": ray_train_ds, "evaluation": ray_eval_ds},
            )
            result = ray_trainer.fit()
