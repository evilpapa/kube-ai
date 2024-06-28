.. _working_with_images:

处理图像
===================

使用 Ray Data，您可以轻松读取和转换大型图像数据集。

本指南向您展示如何：

* :ref:`读取图像 <reading_images>`
* :ref:`变换图像 <transforming_images>`
* :ref:`对图像进行推理 <performing_inference_on_images>`
* :ref:`保存图像 <saving_images>`

.. _reading_images:

读取图像
--------------

Ray Data 可以读取多种格式的图像。

要查看支持的文件格式的完整列表，请参阅
:ref:`输入/输出参考 <input-output>`。

.. tab-set::

    .. tab-item:: Raw images

        要加载 JPEG 文件的原始图像，请调用 :func:`~ray.data.read_images` 。

        .. note::

            :func:`~ray.data.read_images` 使用
            `PIL <https://pillow.readthedocs.io/en/stable/index.html>`_。有关支持的文件格式的列表，请参阅
            `图像文件格式 <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_。

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: NumPy

        要加载存储为 NumPy 数组的图像，请调用 :func:`~ray.data.read_numpy`。

        .. testcode::

            import ray

            ds = ray.data.read_numpy("s3://anonymous@air-example-data/cifar-10/images.npy")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            data    numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: TFRecords

        图像数据集通常包含类似于以下内容的 ``tf.train.Example`` 消息：

        .. code-block::

            features {
                feature {
                    key: "image"
                    value {
                        bytes_list {
                            value: ...  # Raw image bytes
                        }
                    }
                }
                feature {
                    key: "label"
                    value {
                        int64_list {
                            value: 3
                        }
                    }
                }
            }

        要加载示例存储的这种格式，请调用 :func:`~ray.data.read_tfrecords`。
        然后，调用 :meth:`~ray.data.Dataset.map` 来解码原始图像字节。

        .. testcode::

            import io
            from typing import Any, Dict
            import numpy as np
            from PIL import Image
            import ray

            def decode_bytes(row: Dict[str, Any]) -> Dict[str, Any]:
                data = row["image"]
                image = Image.open(io.BytesIO(data))
                row["image"] = np.array(image)
                return row

            ds = (
                ray.data.read_tfrecords(
                    "s3://anonymous@air-example-data/cifar-10/tfrecords"
                )
                .map(decode_bytes)
            )

            print(ds.schema())

        ..
            The following `testoutput` is mocked because the order of column names can
            be non-deterministic. For an example, see
            https://buildkite.com/ray-project/oss-ci-build-branch/builds/4849#01892c8b-0cd0-4432-bc9f-9f86fcd38edd.

        .. testoutput::
            :options: +MOCK

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            label   int64

    .. tab-item:: Parquet

        To load image data stored in Parquet files, call :func:`ray.data.read_parquet`.

        .. testcode::

            import ray

            ds = ray.data.read_parquet("s3://anonymous@air-example-data/cifar-10/parquet")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)
            label   int64


更多创建数据集的信息，参考 :ref:`加载数据 <loading_data>`。

.. _transforming_images:

变换图像
-------------------

要变换图像，调用 :meth:`~ray.data.Dataset.map` 或
:meth:`~ray.data.Dataset.map_batches`。

.. testcode::

    from typing import Any, Dict
    import numpy as np
    import ray

    def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")
        .map_batches(increase_brightness)
    )

有关转换数据的更多信息，请参阅
:ref:`数据转换 <transforming_data>`。

.. _performing_inference_on_images:

对图像进行推理
------------------------------

要使用预先训练的模型执行推理，请首先加载并转换数据。

.. testcode::

    from typing import Any, Dict
    from torchvision import transforms
    import ray

    def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        row["image"] = transform(row["image"])
        return row

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/batoidea/JPEGImages")
        .map(transform_image)
    )

接下来，实现一个可调用类来设置和调用您的模型。

.. testcode::

    import torch
    from torchvision import models

    class ImageClassifier:
        def __init__(self):
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.model.eval()

        def __call__(self, batch):
            inputs = torch.from_numpy(batch["image"])
            with torch.inference_mode():
                outputs = self.model(inputs)
            return {"class": outputs.argmax(dim=1)}

最后，调用 :meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`。

.. testcode::

    predictions = ds.map_batches(
        ImageClassifier,
        compute=ray.data.ActorPoolStrategy(size=2),
        batch_size=4
    )
    predictions.show(3)

.. testoutput::

    {'class': 118}
    {'class': 153}
    {'class': 296}

有关执行推理的更多信息，请参阅
:ref:`端到端：离线批量推理 <batch_inference_home>`
和 :ref:`使用 actor 转换批次 <transforming_data_actors>`。

.. _saving_images:

保存图像
-------------

使用 PNG、Parquet 和 NumPy 等格式保存图像。要查看所有支持的格式，请参阅
参考 :ref:`输入/输出参考 <input-output>`。

.. tab-set::

    .. tab-item:: Images

        要将图像保存为图像文件，请调用 :meth:`~ray.data.Dataset.write_images`。

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_images("/tmp/simple", column="image", file_format="png")

    .. tab-item:: Parquet

        要将图像保存在 Parquet 文件中，请调用 :meth:`~ray.data.Dataset.write_parquet`。

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_parquet("/tmp/simple")


    .. tab-item:: NumPy

        要将图像保存在 NumPy 文件中，请调用 :meth:`~ray.data.Dataset.write_numpy`。

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_numpy("/tmp/simple", column="image")

有关保存数据的更多信息，请参阅 :ref:`保存数据 <loading_data>`。
