.. _data-ingest-torch:

数据加载和预处理
==============================

Ray Train 与 :ref:`Ray Data <data>` 集成，为加载和预处理大型数据集提供高效的流式解决方案。
们推荐使用 Ray Data，因为它能够支持大规模分布式训练工作负载，具有高效的性能 - 有关优势和与其他替代方案的比较，请参见 :ref:`Ray Data 概述 <data_overview>`。

本指引中，我们将介绍如何将 Ray Data 集成到 Ray Train 脚本中，以及不同的方式来定制数据摄入管道。

.. TODO: Replace this image with a better one.

.. figure:: ../images/train_ingest.png
    :align: center
    :width: 300px

快速开始
----------
安装 Ray Data 和 Ray Train：

.. code-block:: bash

    pip install -U "ray[data,train]"

Data ingestion can be set up with four basic steps: 
数据提取可以通过以下四个基本步骤设置：

1. 创建一个 Ray 数据集。
2. 预处理 Ray 数据集。
3. 将预处理后的数据集输入到 Ray Train Trainer 中。
4. 在训练函数中使用 Ray 数据集。

.. tabs::

    .. group-tab:: PyTorch

        .. testcode::

            import torch
            import ray
            from ray import train
            from ray.train import Checkpoint, ScalingConfig
            from ray.train.torch import TorchTrainer

            # Set this to True to use GPU.
            # If False, do CPU training instead of GPU training.
            use_gpu = False

            # Step 1: Create a Ray Dataset from in-memory Python lists.
            # You can also create a Ray Dataset from many other sources and file
            # formats.
            train_dataset = ray.data.from_items([{"x": [x], "y": [2 * x]} for x in range(200)])

            # Step 2: Preprocess your Ray Dataset.
            def increment(batch):
                batch["y"] = batch["y"] + 1
                return batch

            train_dataset = train_dataset.map_batches(increment)


            def train_func(config):
                batch_size = 16

                # Step 4: Access the dataset shard for the training worker via
                # ``get_dataset_shard``.
                train_data_shard = train.get_dataset_shard("train")
                train_dataloader = train_data_shard.iter_torch_batches(
                    batch_size=batch_size, dtypes=torch.float32
                )

                for epoch_idx in range(1):
                    for batch in train_dataloader:
                        inputs, labels = batch["x"], batch["y"]
                        assert type(inputs) == torch.Tensor
                        assert type(labels) == torch.Tensor
                        assert inputs.shape[0] == batch_size
                        assert labels.shape[0] == batch_size
                        break # Only check one batch. Last batch can be partial.

            # Step 3: Create a TorchTrainer. Specify the number of training workers and
            # pass in your Ray Dataset.
            # The Ray Dataset is automatically split across all training workers.
            trainer = TorchTrainer(
                train_func,
                datasets={"train": train_dataset},
                scaling_config=ScalingConfig(num_workers=2, use_gpu=use_gpu)
            )
            result = trainer.fit()

        .. testoutput::
            :hide:

            ...

    .. group-tab:: PyTorch Lightning

        .. code-block:: python
            :emphasize-lines: 9,10,13,14,25,26

            from ray import train
         
            train_data = ray.data.read_csv("./train.csv")
            val_data = ray.data.read_csv("./validation.csv")

            def train_func_per_worker():
                # Access Ray datsets in your train_func via ``get_dataset_shard``.
                # The "train" dataset gets sharded across workers by default
                train_ds = train.get_dataset_shard("train")
                val_ds = train.get_dataset_shard("validation")

                # Create Ray dataset iterables via ``iter_torch_batches``.
                train_dataloader = train_ds.iter_torch_batches(batch_size=16)
                val_dataloader = val_ds.iter_torch_batches(batch_size=16)

                ...

                trainer = pl.Trainer(
                    # ...
                )

                # Feed the Ray dataset iterables to ``pl.Trainer.fit``.
                trainer.fit(
                    model, 
                    train_dataloaders=train_dataloader, 
                    val_dataloaders=val_dataloader
                )

            trainer = TorchTrainer(
                train_func,
                datasets={"train": train_data, "validation": val_data},
                scaling_config=ScalingConfig(num_workers=4),
            )
            trainer.fit()
    
    .. group-tab:: HuggingFace Transformers

        .. code-block:: python
            :emphasize-lines: 12,13,16,17,24,25

            import ray
            import ray.train
         
            ...

            train_data = ray.data.from_huggingface(hf_train_ds)
            eval_data = ray.data.from_huggingface(hf_eval_ds)

            def train_func(config):
                # Access Ray datsets in your train_func via ``get_dataset_shard``.
                # The "train" dataset gets sharded across workers by default
                train_ds = ray.train.get_dataset_shard("train")
                eval_ds = ray.train.get_dataset_shard("evaluation")

                # Create Ray dataset iterables via ``iter_torch_batches``.
                train_iterable_ds = train_ds.iter_torch_batches(batch_size=16)
                eval_iterable_ds = eval_ds.iter_torch_batches(batch_size=16)

                ...

                args = transformers.TrainingArguments(
                    ...,
                    max_steps=max_steps # Required for iterable datasets
                )

                trainer = transformers.Trainer(
                    ...,
                    model=model,
                    train_dataset=train_iterable_ds,
                    eval_dataset=eval_iterable_ds,
                )

                # Prepare your Transformers Trainer
                trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
                trainer.train()

            trainer = TorchTrainer(
                train_func,
                datasets={"train": train_data, "evaluation": val_data},
                scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
            )
            trainer.fit()


.. _train-datasets-load:

加载数据
~~~~~~~~~~~~

Ray 数据集可以从许多不同的数据源和格式中创建。有关更多详细信息，请参见 :ref:`Loading Data <loading_data>`。

.. _train-datasets-preprocess:

预处理数据
~~~~~~~~~~~~~~~~~~

Ray 数据支持广泛的预处理操作，可用于在训练之前转换数据。

- 对于常规预处理，请参阅 :ref:`转换数据 <transforming_data>`.
- 对于表格数据，请参见 :ref:`结构化数据预处理 <preprocessing_structured_data>`。
- 对于 PyTorch 张量，请参见 :ref:`使用 torch 张量进行转换 <transform_pytorch>`。
- 对于优化昂贵的预处理操作，请参见 :ref:`缓存预处理数据集 <dataset_cache_performance>`。

.. _train-datasets-input:

输入及分割数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你的预处理数据集可以通过 ``datasets`` 参数传递给 Ray Train Trainer（例如：:class:`~ray.train.torch.TorchTrainer`）。

传入 Trainer ``datasets`` 的数据集可以通过在每个分布式训练工作节点上运行的 ``train_loop_per_worker`` 中调用 :meth:`ray.train.get_dataset_shard` 来访问。

所有数据集默认都会被分割（即分片）到训练工作节点上。 :meth:`~ray.train.get_dataset_shard` 将返回数据集的 ``1/n`` 部分，其中 ``n`` 是训练工作节点的数量。

.. note::

    请注意，由于评估数据集被分割，用户必须在工作节点上聚合评估结果。
    你可以考虑使用 `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ (:ref:`example <deepspeed_example>`) 或
    其他框架中提供的实用工具。

可通过传递 ``dataset_config`` 参数来覆盖此行为。有关配置拆分逻辑的更多信息，请参见 :ref:`Splitting datasets <train-datasets-split>`。

.. _train-datasets-consume:

消费数据
~~~~~~~~~~~~~~

在 ``train_loop_per_worker`` 中，每个工作节点都可以通过 :meth:`ray.train.get_dataset_shard` 访问其数据集的分片。

这些数据可以通过多种方式使用：

- 要创建批量的 Iterable，可以调用 :meth:`~ray.data.DataIterator.iter_batches`。
- 要创建一个 PyTorch DataLoader 的替代品，可以调用 :meth:`~ray.data.DataIterator.iter_torch_batches`。

更多如何迭代数据的详细信息，请参见 :ref:`Iterating over data <iterating-over-data>`。

.. _train-datasets-pytorch:

从 PyTorch 数据开始
--------------------------

一些框架提供了自己的数据集和数据加载工具。例如：

- **PyTorch:** `Dataset & DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_
- **Hugging Face:** `Dataset <https://huggingface.co/docs/datasets/index>`_
- **PyTorch Lightning:** `LightningDataModule <https://lightning.ai/docs/pytorch/stable/data/datamodule.html>`_

这些实用程序仍可直接与 Ray Train 一起使用。特别是，如果您已经设置了数据提取管道，则可能需要这样做。
但是，为了实现更高性能的大规模数据提取，我们建议迁移到 Ray Data。

从高层次来看，您可以按如下方式比较这些概念：

.. list-table::
   :header-rows: 1

   * - PyTorch API
     - HuggingFace API
     - Ray Data API
   * - `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
     - `datasets.Dataset <https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset>`_
     - :class:`ray.data.Dataset`
   * - `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
     - n/a
     - :meth:`ray.data.Dataset.iter_torch_batches`


有关更多详细信息，请参阅以下每个框架的部分。

.. tabs::

    .. tab:: PyTorch Dataset 和 DataLoader

        **选项 1 (使用 Ray Data):** 将 PyTorch Dataset 转换为 Ray Dataset 并通过 ``datasets`` 参数传递给 Trainer。在 ``train_loop_per_worker`` 中，
        您可以通过 :meth:`ray.train.get_dataset_shard` 访问数据集。
        您可以通过 :meth:`ray.data.DataIterator.iter_torch_batches` 将其转换为替换 PyTorch DataLoader。

        更多详细信息，请参见 :ref:`从 PyTorch 数据集和 DataLoader 迁移 <migrate_pytorch>`。

        **选项 2 (不用 Ray Data):** 直接在 ``train_loop_per_worker`` 中实例化 Torch Dataset 和 DataLoader。
        你可以使用 :meth:`ray.train.torch.prepare_data_loader` 实用工具来设置 DataLoader 以进行分布式训练。
    
    .. tab:: LightningDataModule

        ``LightningDataModule`` 是使用 PyTorch ``Dataset`` 和 ``DataLoader`` 创建的。您可以在这里应用相同的逻辑。

    .. tab:: Hugging Face Dataset

        **选项 1 (使用 Ray Data):** 转换你的 Hugging Face Dataset 为 Ray Dataset 并通过 ``datasets`` 参数传递给 Trainer。
        在 ``train_loop_per_worker`` 中，你可以通过 :meth:`ray.train.get_dataset_shard` 访问数据集。

        For instructions, see :ref:`Ray Data for Hugging Face <loading_datasets_from_ml_libraries>`.

        **选项 2 (不用 Ray Data):** 直接在 ``train_loop_per_worker`` 中实例化 Hugging Face Dataset。

    .. tip:: 

        当直接使用 Torch 或 Hugging Face 数据集而不使用 Ray Data 时，请确保在 ``train_loop_per_worker`` 中实例化您的数据集。
        在 ``train_loop_per_worker`` 之外实例化数据集并通过全局范围传递可能会导致错误，因为数据集不可序列化。

.. _train-datasets-split:

分割数据集
------------------
默认，Ray Train 使用 :meth:`Dataset.streaming_split <ray.data.Dataset.streaming_split>` 将所有数据集分割到工作节点上。每个 worker 看到数据的一个不相交子集，而不是迭代整个数据集。除非随机洗牌，否则每次迭代数据集时都会使用相同的拆分。

如果要自定义哪些数据集被拆分，请在 Trainer 构造函数中传递 :class:`DataConfig <ray.train.DataConfig>`。

例如，要仅拆分训练数据集，请执行以下操作：

.. testcode::

    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )
    train_ds, val_ds = ds.train_test_split(0.3)

    def train_loop_per_worker():
        # Get the sharded training dataset
        train_ds = train.get_dataset_shard("train")
        for _ in range(2):
            for batch in train_ds.iter_batches(batch_size=128):
                print("Do some training on batch", batch)
        
        # Get the unsharded full validation dataset
        val_ds = train.get_dataset_shard("val")
        for _ in range(2):
            for batch in val_ds.iter_batches(batch_size=128):
                print("Do some evaluation on batch", batch)

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=ray.train.DataConfig(
            datasets_to_split=["train"],
        ),
    )
    my_trainer.fit()


完全自定义（高级）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
对于默认配置类未覆盖的用例，您还可以完全自定义输入数据集的拆分方式。定义一个自定义 :class:`DataConfig <ray.train.DataConfig>` 类（DeveloperAPI）。:class:`DataConfig <ray.train.DataConfig>` 类负责在节点之间共享设置和拆分数据。

.. testcode::

    # Note that this example class is doing the same thing as the basic DataConfig
    # implementation included with Ray Train.
    from typing import Optional, Dict, List

    import ray
    from ray import train
    from ray.train.torch import TorchTrainer
    from ray.train import DataConfig, ScalingConfig
    from ray.data import Dataset, DataIterator, NodeIdStr
    from ray.actor import ActorHandle

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    def train_loop_per_worker():
        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            for batch in it.iter_batches(batch_size=128):
                print("Do some training on batch", batch)


    class MyCustomDataConfig(DataConfig):
        def configure(
            self,
            datasets: Dict[str, Dataset],
            world_size: int,
            worker_handles: Optional[List[ActorHandle]],
            worker_node_ids: Optional[List[NodeIdStr]],
            **kwargs,
        ) -> List[Dict[str, DataIterator]]:
            assert len(datasets) == 1, "This example only handles the simple case"

            # Configure Ray Data for ingest.
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options = DataConfig.default_ingest_options()

            # Split the stream into shards.
            iterator_shards = datasets["train"].streaming_split(
                world_size, equal=True, locality_hints=worker_node_ids
            )

            # Return the assigned iterators for each worker.
            return [{"train": it} for it in iterator_shards]


    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": ds},
        dataset_config=MyCustomDataConfig(),
    )
    my_trainer.fit()


子类必须是可序列化的，因为 Ray Train 会将其从驱动脚本复制到 Trainer 的驱动 actor。Ray Train 在 Trainer 组的主 actor 上调用其 :meth:`configure <ray.train.DataConfig.configure>` 方法，以为每个 worker 创建数据迭代器。

通常，你可以使用 :class:`DataConfig <ray.train.DataConfig>` 来设置任何必须在 worker 开始迭代数据之前提前发生的共享设置。设置在每次 Trainer 运行开始时运行。


随机洗牌
----------------
根据您正在训练的模型，随机打乱每个时期的数据对于模型质量非常重要

Ray Data 有两种随机改组方法：

1. 在每个训练 worker 上对数据块进行随机化。这需要更少的通信，但牺牲了一些随机性（即，出现在同一数据块中的行更有可能在迭代顺序中靠近彼此）。
2. 全局洗牌，这更昂贵。这将完全使行迭代顺序与原始数据集顺序解耦，但代价是更多的计算、I/O 和通信。

对于大多数情况来说，选项 1 就足够了。

首先，通过 :meth:`randomize_block_order` 对数据集的每个 :ref:`block <dataset_concept>` 进行随机化。然后，在训练期间迭代数据集时，通过为 :meth:`iter_batches` 或 :meth:`iter_torch_batches` 指定 ``local_shuffle_buffer_size`` 来启用本地洗牌。

.. testcode::
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    # Randomize the blocks of this dataset.
    ds = ds.randomize_block_order()

    def train_loop_per_worker():
        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            # Use a shuffle buffer size of 10k rows.
            for batch in it.iter_batches(
                local_shuffle_buffer_size=10000, batch_size=128):
                print("Do some training on batch", batch)

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": ds},
    )
    my_trainer.fit()


如果您的模型对洗牌质量很敏感，请调 :meth:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>` 以执行全局洗牌。

.. testcode::

    import ray

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    # Do a global shuffle of all rows in this dataset.
    # The dataset will be shuffled on each iteration, unless `.materialize()`
    # is called after the `.random_shuffle()`
    ds = ds.random_shuffle()

有关如何优化 shuffing 以及选择哪种方法的更多信息，请参阅 :ref:`shuffling 优化指南 <optimizing_shuffles>`。


启用可重现性
------------------------
当开发或超参数调整模型时，数据提取的可重现性很重要，以确保数据提取不会影响模型质量。遵循以下三个步骤以启用可重现性：

**步骤 1:** 通过在 :class:`DataContext <ray.data.context.DataContext>` 中设置 `preserve_order` 标志来启用 Ray 数据集中的确定性执行。

.. testcode::

    import ray

    # Preserve ordering in Ray Datasets for reproducibility.
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

**步骤 2:** 设置任何洗牌操作的种子：

* :meth:`random_shuffle <ray.data.Dataset.random_shuffle>` 的 `seed` 参数
* :meth:`randomize_block_order <ray.data.Dataset.randomize_block_order>` 的 `seed` 参数
* :meth:`iter_batches <ray.data.DataIterator.iter_batches>` 的 `local_shuffle_seed` 参数

**步骤 3:** 遵循最佳实践，以确保您的训练框架在可重现性方面的设置正确。更多信息，请参见 `Pytorch 可重现性指南 <https://pytorch.org/docs/stable/notes/randomness.html>`_。



.. _preprocessing_structured_data:

预处理结构化数据
-----------------------------

.. note::
    本节适用于表格/结构化数据。预处理非结构化数据的推荐方式是使用 Ray Data 操作，例如 `map_batches`。
    有关更多详细信息，请参见 :ref:`Ray Data Working with Pytorch guide <working_with_pytorch>`。

针对表格数据，我们建议使用 Ray Data :ref:`preprocessors <air-preprocessors>`，这些预处理器实现了常见的数据预处理操作。
你可使用这些预处理器在将数据集传递到 Trainer 之前对数据集进行处理。例如：

.. testcode::

    import numpy as np
    from tempfile import TemporaryDirectory

    import ray
    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.data.preprocessors import Concatenator, StandardScaler

    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

    # Create preprocessors to scale some columns and concatenate the results.
    scaler = StandardScaler(columns=["mean radius", "mean texture"])
    concatenator = Concatenator(exclude=["target"], dtype=np.float32)

    # Compute dataset statistics and get transformed datasets. Note that the
    # fit call is executed immediately, but the transformation is lazy.
    dataset = scaler.fit_transform(dataset)
    dataset = concatenator.fit_transform(dataset)

    def train_loop_per_worker():
        context = train.get_context()
        print(context.get_metadata())  # prints {"preprocessor_pkl": ...}

        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            # Prefetch 10 batches at a time.
            for batch in it.iter_batches(batch_size=128, prefetch_batches=10):
                print("Do some training on batch", batch)

        # Save a checkpoint.
        with TemporaryDirectory() as temp_dir:
            train.report(
                {"score": 2.0},
                checkpoint=Checkpoint.from_directory(temp_dir),
            )

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": dataset},
        metadata={"preprocessor_pkl": scaler.serialize()},
    )

    # Get the fitted preprocessor back from the result metadata.
    metadata = my_trainer.fit().checkpoint.get_metadata()
    print(StandardScaler.deserialize(metadata["preprocessor_pkl"]))


在本示例中，我们使用 ``Trainer(metadata={...})`` 构造函数参数持久化了拟合的预处理器。此参数指定一个字典，将在 ``TrainContext.get_metadata()`` 和 ``checkpoint.get_metadata()`` 中可用，用于从 Trainer 保存的检查点中重新创建拟合的预处理器以用于推理。

性能技巧
----------------

预取批次
~~~~~~~~~~~~~~~~~~~
在迭代数据集进行训练时，您可以增加 ``prefetch_batches`` 参数以进一步提高性能。在训练当前批次时，这会启动 N 个后台线程来获取和处理下一个 N 个批次。

如果训练在跨节点数据传输或在最后一英里预处理上受到瓶颈，这种方法可以帮助。例如，将批次转换为张量或执行 ``collate_fn``。但是，增加 ``prefetch_batches`` 会导致更多数据需要保存在堆内存中。默认情况下，``prefetch_batches`` 设置为 1。

比如，以下代码每次预取 10 个批次给每个训练 worker：

.. testcode::

    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ds = ray.data.read_text(
        "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
    )

    def train_loop_per_worker():
        # Get an iterator to the dataset we passed in below.
        it = train.get_dataset_shard("train")
        for _ in range(2):
            # Prefetch 10 batches at a time.
            for batch in it.iter_batches(batch_size=128, prefetch_batches=10):
                print("Do some training on batch", batch)

    my_trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": ds},
    )
    my_trainer.fit()


.. _dataset_cache_performance:

缓存预处理数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如果你在 GPU 上训练并且有一个昂贵的 CPU 预处理操作，这种方法可能会成为训练吞吐量的瓶颈。

如果你的预处理数据集小到可以放入 Ray 对象存储内存中（默认情况下这是总集群 RAM 的 30%），通过在预处理数据集上调用 :meth:`materialize() <ray.data.Dataset.materialize>` 来在 Ray 的内置对象存储中 *materialize* 预处理数据集。此方法告诉 Ray Data 计算整个预处理数据并将其固定在 Ray 对象存储内存中。结果是，当重复迭代数据集时，不需要重新运行预处理操作。但是，如果预处理数据太大而无法放入 Ray 对象存储内存中，这种方法会大大降低性能，因为数据需要溢出到磁盘并从磁盘读回。

你希望在每个 epoch 运行的转换，例如随机化，应该在 materialize 调用之后。

.. testcode::

    from typing import Dict
    import numpy as np
    import ray

    # Load the data.
    train_ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

    # Define a preprocessing function.
    def normalize_length(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new_col = batch["sepal.length"] / np.max(batch["sepal.length"])
        batch["normalized.sepal.length"] = new_col
        del batch["sepal.length"]
        return batch

    # Preprocess the data. Transformations that are made before the materialize call
    # below are only run once.
    train_ds = train_ds.map_batches(normalize_length)

    # Materialize the dataset in object store memory.
    # Only do this if train_ds is small enough to fit in object store memory.
    train_ds = train_ds.materialize()

    # Dummy augmentation transform.
    def augment_data(batch):
        return batch

    # Add per-epoch preprocessing. Transformations that you want to run per-epoch, such
    # as data augmentation or randomization, should go after the materialize call.
    train_ds = train_ds.map_batches(augment_data)

    # Pass train_ds to the Trainer


向集群添加 CPU-only 节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如果你在昂贵的 CPU 预处理上受到瓶颈，并且预处理数据集太大而无法放入对象存储内存中，那么材料化数据集就行不通。在这种情况下，由于 Ray 支持异构集群，您可以向集群添加更多的 CPU-only 节点。

对于受对象存储内存瓶颈的情况，向集群添加更多的 CPU-only 节点会增加总集群对象存储内存，从而允许在预处理和训练阶段之间缓冲更多数据。

对于预处理计算时间受限的情况，添加更多的 CPU-only 节点会向集群添加更多的 CPU 核心，进一步并行化预处理。如果您的预处理仍然无法快速饱和 GPU，则添加足够的 CPU-only 节点以 :ref:`缓存预处理数据集 <dataset_cache_performance>`。
