.. _working_with_pytorch:

使用 PyTorch
====================

Ray Data 与 PyTorch 生态系统集成。

本指南介绍了如何：

* :ref:`将数据集作为 Torch 张量进行迭代以进行模型训练 <iterating_pytorch>`
* :ref:`编写处理 Torch 张量的转换 <transform_pytorch>`
* :ref:`使用 Torch 模型执行批量推理 <batch_inference_pytorch>`
* :ref:`保存包含 Torch 张量的数据集 <saving_pytorch>`
* :ref:`从 PyTorch 数据集迁移到 Ray 数据 <migrate_pytorch>`

.. _iterating_pytorch:

迭代 Torch 张量进行训练
-----------------------------------------
要迭代 Torch 格式的批量数据，请调用 :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`。每个批次表示为 `Dict[str, torch.Tensor]`，数据集中每列一个张量。

这对于使用数据集中的批次训练 Torch 模型非常有用。有关提供 ``collate_fn`` 进行自定义转换的配置详细信息，请参阅 `API 参考 <ray.data.Dataset.iter_torch_batches>`。

.. testcode::

    import ray
    import torch

    ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

    for batch in ds.iter_torch_batches(batch_size=2):
        print(batch)

.. testoutput::
    :options: +MOCK

    {'image': tensor([[[[...]]]], dtype=torch.uint8)}
    ...
    {'image': tensor([[[[...]]]], dtype=torch.uint8)}

与 Ray Train 集成
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray Data 与 :ref:`Ray Train <train-docs>` 集成，可轻松摄取数据以进行数据并行训练，并支持 PyTorch、PyTorch Lightning 或 Hugging Face 训练。

.. testcode::

    import torch
    from torch import nn
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    def train_func(config):
        model = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Datasets can be accessed in your train_func via ``get_dataset_shard``.
        train_data_shard = train.get_dataset_shard("train")

        for epoch_idx in range(2):
            for batch in train_data_shard.iter_torch_batches(batch_size=128, dtypes=torch.float32):
                features = torch.stack([batch[col_name] for col_name in batch.keys() if col_name != "target"], axis=1)
                predictions = model(features)
                train_loss = loss_fn(predictions, batch["target"].unsqueeze(1))
                train_loss.backward()
                optimizer.step()


    train_dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

    trainer = TorchTrainer(
        train_func,
        datasets={"train": train_dataset},
        scaling_config=ScalingConfig(num_workers=2)
    )
    trainer.fit()


有关更多详细信息，请参阅 :ref:`Ray 训练用户指南 <data-ingest-torch>`。

.. _transform_pytorch:

使用 Torch 张量进行变换
----------------------------------
使用 `map` 或 ``map_batches`` 变换，可以返回 Torch 张量。

.. caution::

    在底层，Ray Data 自动将 Torch 张量转换为 NumPy 数组。后续转换接受 NumPy 数组作为输入，而不是 Torch 张量。

.. tab-set::

    .. tab-item:: map

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            def convert_to_torch(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                return {"tensor": torch.as_tensor(row["image"])}

            # The tensor gets converted into a Numpy array under the hood
            transformed_ds = ds.map(convert_to_torch)
            print(transformed_ds.schema())

            # Subsequent transformations take in Numpy array as input.
            def check_numpy(row: Dict[str, np.ndarray]):
                assert isinstance(row["tensor"], np.ndarray)
                return row

            transformed_ds.map(check_numpy).take_all()

        .. testoutput::

            Column  Type
            ------  ----
            tensor  numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: map_batches

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            def convert_to_torch(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                return {"tensor": torch.as_tensor(batch["image"])}

            # The tensor gets converted into a Numpy array under the hood
            transformed_ds = ds.map_batches(convert_to_torch, batch_size=2)
            print(transformed_ds.schema())

            # Subsequent transformations take in Numpy array as input.
            def check_numpy(batch: Dict[str, np.ndarray]):
                assert isinstance(batch["tensor"], np.ndarray)
                return batch

            transformed_ds.map_batches(check_numpy, batch_size=2).take_all()

        .. testoutput::

            Column  Type
            ------  ----
            tensor  numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

有关转换数据的更多信息，请参阅 :ref:`转换数据 <transforming_data>`。

内置 PyTorch 转换
~~~~~~~~~~~~~~~~~~~~~~~~~~~

你可以使用 Ray Data 转换内置的 PyTorch 转换如 ``torchvision``， ``torchtext`` 和 ``torchaudio`` 。

.. tab-set::

    .. tab-item:: torchvision

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            from torchvision import transforms
            import ray

            # Create the Dataset.
            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            # Define the torchvision transform.
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(10)
                ]
            )

            # Define the map function
            def transform_image(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
                row["transformed_image"] = transform(row["image"])
                return row

            # Apply the transform over the dataset.
            transformed_ds = ds.map(transform_image)
            print(transformed_ds.schema())

        .. testoutput::

            Column             Type
            ------             ----
            image              numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            transformed_image  numpy.ndarray(shape=(3, 10, 10), dtype=float)

    .. tab-item:: torchtext

        .. testcode::

            from typing import Dict, List
            import numpy as np
            from torchtext import transforms
            import ray

            # Create the Dataset.
            ds = ray.data.read_text("s3://anonymous@ray-example-data/simple.txt")

            # Define the torchtext transform.
            VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
            transform = transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

            # Define the map_batches function.
            def tokenize_text(batch: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
                batch["tokenized_text"] = transform(list(batch["text"]))
                return batch

            # Apply the transform over the dataset.
            transformed_ds = ds.map_batches(tokenize_text, batch_size=2)
            print(transformed_ds.schema())

        .. testoutput::

            Column          Type
            ------          ----
            text            <class 'object'>
            tokenized_text  <class 'object'>

.. _batch_inference_pytorch:

使用 PyTorch 进行批量推理
----------------------------

借助 Ray 数据集，您可以通过将预先训练的模型映射到数据上，使用 Torch 模型进行可扩展的离线批量推理。

.. testcode::

    from typing import Dict
    import numpy as np
    import torch
    import torch.nn as nn

    import ray

    # Step 1: Create a Ray Dataset from in-memory Numpy arrays.
    # You can also create a Ray Dataset from many other sources and file
    # formats.
    ds = ray.data.from_numpy(np.ones((1, 100)))

    # Step 2: Define a Predictor class for inference.
    # Use a class to initialize the model just once in `__init__`
    # and re-use it for inference across multiple batches.
    class TorchPredictor:
        def __init__(self):
            # Load a dummy neural network.
            # Set `self.model` to your pre-trained PyTorch model.
            self.model = nn.Sequential(
                nn.Linear(in_features=100, out_features=1),
                nn.Sigmoid(),
            )
            self.model.eval()

        # Logic for inference on 1 batch of data.
        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            tensor = torch.as_tensor(batch["data"], dtype=torch.float32)
            with torch.inference_mode():
                # Get the predictions from the input batch.
                return {"output": self.model(tensor).numpy()}

    # Use 2 parallel actors for inference. Each actor predicts on a
    # different partition of data.
    scale = ray.data.ActorPoolStrategy(size=2)
    # Step 3: Map the Predictor over the Dataset to get predictions.
    predictions = ds.map_batches(TorchPredictor, compute=scale)
    # Step 4: Show one prediction output.
    predictions.show(limit=1)

.. testoutput::
    :options: +MOCK

    {'output': array([0.5590901], dtype=float32)}

更多信息，请参阅 :ref:`批量推理用户指南 <batch_inference_home>`。

.. _saving_pytorch:

保存包含 Torch 张量的数据集
----------------------------------------

包含 Torch 张量的数据集可以保存到文件中，例如 parquet 或 NumPy。

更多信息，请参阅 :ref:`保存数据 <saving-data>`。

.. caution::

    Torch 张量在 GPU 设备上无法序列化并写入磁盘。在保存数据之前将张量转换为 CPU（ ``tensor.to("cpu")``）。

.. tab-set::

    .. tab-item:: Parquet

        .. testcode::

            import torch
            import ray

            tensor = torch.Tensor(1)
            ds = ray.data.from_items([{"tensor": tensor}])

            ds.write_parquet("local:///tmp/tensor")

    .. tab-item:: Numpy

        .. testcode::

            import torch
            import ray

            tensor = torch.Tensor(1)
            ds = ray.data.from_items([{"tensor": tensor}])

            ds.write_numpy("local:///tmp/tensor", column="tensor")

.. _migrate_pytorch:

从 PyTorch 数据集和 DataLoaders 迁移
-----------------------------------------------

如果您正在使用 PyTorch 数据集和 DataLoader，您可以迁移到 Ray Data 以处理分布式数据集。

PyTorch 数据集被 :class:`Dataset <ray.data.Dataset>` 抽象替换，PyTorch DataLoader 被 :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` 替换。

内置 PyTorch 数据集
~~~~~~~~~~~~~~~~~~~~~~~~~

如果你使用的是内置的 PyTorch 数据集，例如来自 ``torchvision``，可以使用 :meth:`from_torch() <ray.data.from_torch>` API 将其转换为 Ray 数据集。

.. caution::

    :meth:`from_torch() <ray.data.from_torch>` 需要 PyTorch 数据集适合内存。仅将其用于小型内置数据集以进行原型设计或测试。

.. testcode::

    import torchvision
    import ray

    mnist = torchvision.datasets.MNIST(root="/tmp/", download=True)
    ds = ray.data.from_torch(mnist)

    # The data for each item of the Torch dataset is under the "item" key.
    print(ds.schema())

..
    The following `testoutput` is mocked to avoid illustrating download logs like
    "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz".

.. testoutput::
    :options: +MOCK

    Column  Type
    ------  ----
    item    <class 'object'>

自定义 PyTorch 数据集
~~~~~~~~~~~~~~~~~~~~~~~

如果您有自定义 PyTorch 数据集，则可以通过 ``__getitem__`` 替换 Ray 数据读取和准换操作逻辑来迁移到 Ray Data。

任何从云存储和磁盘读取数据的逻辑都可以通过 Ray Data ``read_*`` API 之一替换，任何转换逻辑都可以作为 Dataset 上的 :meth:`map <ray.data.Dataset.map>` 调用应用。

以下示例显示了一个自定义 PyTorch 数据集，以及使用 Ray Data 的类似情况。

.. note::

    与 PyTorch 地图样式数据集不同， Ray 数据集不可索引。

.. tab-set::

    .. tab-item:: PyTorch Dataset

        .. testcode::

            import tempfile
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            from torchvision import transforms
            from torch.utils.data import Dataset
            from PIL import Image

            class ImageDataset(Dataset):
                def __init__(self, bucket_name: str, dir_path: str):
                    self.s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
                    self.bucket = self.s3.Bucket(bucket_name)
                    self.files = [obj.key for obj in self.bucket.objects.filter(Prefix=dir_path)]

                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((128, 128)),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    img_name = self.files[idx]

                    # Infer the label from the file name.
                    last_slash_idx = img_name.rfind("/")
                    dot_idx = img_name.rfind(".")
                    label = int(img_name[last_slash_idx+1:dot_idx])

                    # Download the S3 file locally.
                    obj = self.bucket.Object(img_name)
                    tmp = tempfile.NamedTemporaryFile()
                    tmp_name = "{}.jpg".format(tmp.name)

                    with open(tmp_name, "wb") as f:
                        obj.download_fileobj(f)
                        f.flush()
                        f.close()
                        image = Image.open(tmp_name)

                    # Preprocess the image.
                    image = self.transform(image)

                    return image, label

            dataset = ImageDataset(bucket_name="ray-example-data", dir_path="batoidea/JPEGImages/")

    .. tab-item:: Ray Data

        .. testcode::

            import torchvision
            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages", include_paths=True)

            # Extract the label from the file path.
            def extract_label(row: dict):
                filepath = row["path"]
                last_slash_idx = filepath.rfind("/")
                dot_idx = filepath.rfind('.')
                label = int(filepath[last_slash_idx+1:dot_idx])
                row["label"] = label
                return row

            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((128, 128)),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

            # Preprocess the images.
            def transform_image(row: dict):
                row["transformed_image"] = transform(row["image"])
                return row

            # Map the transformations over the dataset.
            ds = ds.map(extract_label).map(transform_image)

PyTorch 数据加载器
~~~~~~~~~~~~~~~~~~

数据加载器可以通过调用 :meth:`Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` 来批次迭代数据集。

以下表格描述了 PyTorch DataLoader 的参数如何映射到 Ray Data。请注意，行为可能不一定相同。有关确切的语义和用法，请 :meth:`参见 API 参考 <ray.data.Dataset.iter_torch_batches>`。

.. list-table::
   :header-rows: 1

   * - PyTorch DataLoader arguments
     - Ray Data API
   * - ``batch_size``
     - ``batch_size`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``shuffle``
     - ``local_shuffle_buffer_size`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``collate_fn``
     - ``collate_fn`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``sampler``
     - Not supported. Can be manually implemented after iterating through the dataset with :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.
   * - ``batch_sampler``
     - Not supported. Can be manually implemented after iterating through the dataset with :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`.
   * - ``drop_last``
     - ``drop_last`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
   * - ``num_workers``
     - Use ``prefetch_batches`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to indicate how many batches to prefetch. The number of prefetching threads are automatically configured according to ``prefetch_batches``.
   * - ``prefetch_factor``
     - Use ``prefetch_batches`` argument to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to indicate how many batches to prefetch. The number of prefetching threads are automatically configured according to ``prefetch_batches``.
   * - ``pin_memory``
     - Pass in ``device`` to :meth:`ds.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>` to get tensors that have already been moved to the correct device.
