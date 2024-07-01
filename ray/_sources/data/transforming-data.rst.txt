.. _transforming_data:

=================
数据转换
=================

转换允许您处理和修改数据集。您可以组合转换来表达计算链。

.. note::
    默认情况下，转换是惰性的。在您通过 :ref:`迭代 Dataset <iterating-over-data>`、 :ref:`保存 Dataset <saving-data>`、或 :ref:`检查 Dataset 属性 <inspecting-data>` 来触发数据消耗之前，它们不会被执行。

本指南向您展示如何：

* `变换行 <#transforming-rows>`_
* `批量转换 <#transforming-batches>`_
* `Groupby 和变换组 <#groupby-and-transforming-groups>`_
* `随机排列行 <#shuffling-rows>`_
* `重新分区数据 <#repartitioning-data>`_

.. _transforming_rows:

转换行
=================

要转换行，请调用 :meth:`~ray.data.Dataset.map` 或
:meth:`~ray.data.Dataset.flat_map`。

使用 map 转换行
~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您的转换为每个输入行返回一行，请调用
:meth:`~ray.data.Dataset.map`。

.. testcode::

    import os
    from typing import Any, Dict
    import ray

    def parse_filename(row: Dict[str, Any]) -> Dict[str, Any]:
        row["filename"] = os.path.basename(row["path"])
        return row

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple", include_paths=True)
        .map(parse_filename)
    )

.. tip::

    如果您的转换是矢量化的，则需要 :meth:`~ray.data.Dataset.map_batches` 获取更好的性能。
    要了解更多信息，请参阅 `批量转换 <#transforming-batches>`_。

用 flat map 转换行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您的转换为每个输入行返回多行，请调用
:meth:`~ray.data.Dataset.flat_map`.

.. testcode::

    from typing import Any, Dict, List
    import ray

    def duplicate_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [row] * 2

    print(
        ray.data.range(3)
        .flat_map(duplicate_row)
        .take_all()
    )

.. testoutput::

    [{'id': 0}, {'id': 0}, {'id': 1}, {'id': 1}, {'id': 2}, {'id': 2}]

.. _transforming_batches:

批量转换
====================

如果您的转换像大多数 NumPy 或 pandas 操作一样进行矢量化，则批量转换的性能比转换行的性能更高。

在任务和 actor 之间进行选择
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 使用任务或 actor 进行批量转换。 Actor 只执行一次设置。相反，
任务需要每批进行设置。因此，如果您的转换涉及昂贵的设置（例如下载模型权重），
请使用 actor。否则，请使用任务。

要了解有关任务和 actor 的更多信息，请阅读
:ref:`Ray 核心关键概念 <core-key-concepts>`。

使用任务批量转换
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要使用任务批量转换，调用 :meth:`~ray.data.Dataset.map_batches`。 Ray Data
默认使用的是任务。

.. testcode::

    from typing import Dict
    import numpy as np
    import ray

    def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
        .map_batches(increase_brightness)
    )

.. _transforming_data_actors:

使用 actor 批量转换
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要使用 actor 批量转换，请完成以下步骤：

1. 实现一个类。 ``__init__`` 进行设置，在 ``__call__`` 转换数据。

2. 创建 :class:`~ray.data.ActorPoolStrategy` 并配置并发 worker 的数量。
   每个 worker 都会转换一个数据分区。

3. 调用 :meth:`~ray.data.Dataset.map_batches` 并传递 ``ActorPoolStrategy`` 到 ``compute``。

.. tab-set::

    .. tab-item:: CPU

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            class TorchPredictor:

                def __init__(self):
                    self.model = torch.nn.Identity()
                    self.model.eval()

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    inputs = torch.as_tensor(batch["data"], dtype=torch.float32)
                    with torch.inference_mode():
                        batch["output"] = self.model(inputs).detach().numpy()
                    return batch

            ds = (
                ray.data.from_numpy(np.ones((32, 100)))
                .map_batches(TorchPredictor, compute=ray.data.ActorPoolStrategy(size=2))
            )

        .. testcode::
            :hide:

            ds.materialize()

    .. tab-item:: GPU

        .. testcode::

            from typing import Dict
            import numpy as np
            import torch
            import ray

            class TorchPredictor:

                def __init__(self):
                    self.model = torch.nn.Identity().cuda()
                    self.model.eval()

                def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    inputs = torch.as_tensor(batch["data"], dtype=torch.float32).cuda()
                    with torch.inference_mode():
                        batch["output"] = self.model(inputs).detach().cpu().numpy()
                    return batch

            ds = (
                ray.data.from_numpy(np.ones((32, 100)))
                .map_batches(
                    TorchPredictor,
                    # Two workers with one GPU each
                    compute=ray.data.ActorPoolStrategy(size=2),
                    # Batch size is required if you're using GPUs.
                    batch_size=4,
                    num_gpus=1
                )
            )

        .. testcode::
            :hide:

            ds.materialize()

.. _configure_batch_format:

配置批处理格式
~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 将批次表示为 NumPy ndarray 或 pandas DataFrame 的字典。默认情况下，Ray Data 将批次表示为 NumPy ndarray 的字典。

要配置批处理类型，请在 ``batch_format`` 中指定
:meth:`~ray.data.Dataset.map_batches`。您可以从函数中返回任一格式。

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            from typing import Dict
            import numpy as np
            import ray

            def increase_brightness(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                batch["image"] = np.clip(batch["image"] + 4, 0, 255)
                return batch

            ds = (
                ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
                .map_batches(increase_brightness, batch_format="numpy")
            )

    .. tab-item:: pandas

        .. testcode::

            import pandas as pd
            import ray

            def drop_nas(batch: pd.DataFrame) -> pd.DataFrame:
                return batch.dropna()

            ds = (
                ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
                .map_batches(drop_nas, batch_format="pandas")
            )

配置批量大小
~~~~~~~~~~~~~~~~~~~~~~

增加 ``batch_size`` 可以提高 NumPy 函数和模型推理等矢量化转换的性能。
但是，如果您的批处理大小太大，您的程序可能会耗尽内存。
如果遇到内存不足错误，请减少
``batch_size``。

.. note::

    默认批量大小取决于您的资源类型。如果您使用的是 CPU，
    则默认批处理大小为 4096。
    如果您使用的是 GPU，则必须指定显式批处理大小。

.. _transforming_groupby:

Groupby 和转换组
===============================

要转换组，请调用 :meth:`~ray.data.Dataset.groupby` 进行分组 。然后，调用
:meth:`~ray.data.grouped_data.GroupedData.map_groups` 来转换组。

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            from typing import Dict
            import numpy as np
            import ray

            items = [
                {"image": np.zeros((32, 32, 3)), "label": label}
                for _ in range(10) for label in range(100)
            ]

            def normalize_images(group: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                group["image"] = (group["image"] - group["image"].mean()) / group["image"].std()
                return group

            ds = (
                ray.data.from_items(items)
                .groupby("label")
                .map_groups(normalize_images)
            )

    .. tab-item:: pandas

        .. testcode::

            import pandas as pd
            import ray

            def normalize_features(group: pd.DataFrame) -> pd.DataFrame:
                target = group.drop("target")
                group = (group - group.min()) / group.std()
                group["target"] = target
                return group

            ds = (
                ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
                .groupby("target")
                .map_groups(normalize_features)
            )

打乱行
==============

要随机打乱所有行，请调用 :meth:`~ray.data.Dataset.random_shuffle`。

.. testcode::

    import ray

    ds = (
        ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
        .random_shuffle()
    )

.. tip::

    :meth:`~ray.data.Dataset.random_shuffle` 很慢。为了获得更好的性能，请尝试
    `Iterating over batches with shuffling <iterating-over-data#iterating-over-batches-with-shuffling>`_。

重新分区数据
===================

:class:`~ray.data.dataset.Dataset` 对一系列分布式数据
:term:`blocks <block>` 进行操作。如果您想实现更细粒度的并行化，
请通过设置更高的 ``parallelism`` 读取时间来增加块数。

要更改现有数据集的块数，请调用
:meth:`Dataset.repartition() <ray.data.Dataset.repartition>`。

.. testcode::

    import ray

    ds = ray.data.range(10000, parallelism=1000)

    # Repartition the data into 100 blocks. Since shuffle=False, Ray Data will minimize
    # data movement during this operation by merging adjacent blocks.
    ds = ds.repartition(100, shuffle=False).materialize()

    # Repartition the data into 200 blocks, and force a full data shuffle.
    # This operation will be more expensive
    ds = ds.repartition(200, shuffle=True).materialize()
