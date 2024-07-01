.. _working_with_tensors:

处理张量
====================

N 维数组（即张量）在 ML 工作负载中无处不在。本指南
描述了处理此类数据的限制和最佳实践。

张量数据表示
--------------------------

Ray Data 将张量表示为
`NumPy ndarrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`__。

.. testcode::

    import ray

    ds = ray.data.read_images("s3://anonymous@air-example-data/digits")
    print(ds)

.. testoutput::

    Dataset(
       num_blocks=...,
       num_rows=100,
       schema={image: numpy.ndarray(shape=(28, 28), dtype=uint8)}
    )

批量固定形状张量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您的张量具有固定形状，则 Ray Data 将批次表示为常规 ndarray。

.. doctest::

    >>> import ray
    >>> ds = ray.data.read_images("s3://anonymous@air-example-data/digits")
    >>> batch = ds.take_batch(batch_size=32)
    >>> batch["image"].shape
    (32, 28, 28)
    >>> batch["image"].dtype
    dtype('uint8')

批量可变形状张量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果张量的形状不同， Ray Data 将批次表示为对象数据类型的数组。

.. doctest::

    >>> import ray
    >>> ds = ray.data.read_images("s3://anonymous@air-example-data/AnimalDetection")
    >>> batch = ds.take_batch(batch_size=32)
    >>> batch["image"].shape
    (32,)
    >>> batch["image"].dtype
    dtype('O')

这些对象数组的各个元素都是常规的 ndarray。

.. doctest::

    >>> batch["image"][0].dtype
    dtype('uint8')
    >>> batch["image"][0].shape  # doctest: +SKIP
    (375, 500, 3)
    >>> batch["image"][3].shape  # doctest: +SKIP
    (333, 465, 3)

.. _transforming_tensors:

转换张量数据
------------------------

调用 :meth:`~ray.data.Dataset.map` 或 :meth:`~ray.data.Dataset.map_batches` 来变换张量数据。

.. testcode::

    from typing import Any, Dict

    import ray
    import numpy as np

    ds = ray.data.read_images("s3://anonymous@air-example-data/AnimalDetection")

    def increase_brightness(row: Dict[str, Any]) -> Dict[str, Any]:
        row["image"] = np.clip(row["image"] + 4, 0, 255)
        return row

    # Increase the brightness, record at a time.
    ds.map(increase_brightness)

    def batch_increase_brightness(batch: Dict[str, np.ndarray]) -> Dict:
        batch["image"] = np.clip(batch["image"] + 4, 0, 255)
        return batch

    # Increase the brightness, batch at a time.
    ds.map_batches(batch_increase_brightness)

除了 NumPy ndarrays，Ray Data 还将返回的 NumPy ndarrays 列表和
实现 ``__array__`` 的对象（例如， ``torch.Tensor``）视为张量数据。

有关转换数据的更多信息，请阅读
:ref:`转换数据 <transforming_data>`。


保存张量数据
------------------

使用 Parquet、NumPy 和 JSON 等格式保存张量数据。要查看所有支持的格式，请参阅
:ref:`输入/输出 参考 <input-output>`。

.. tab-set::

    .. tab-item:: Parquet

        Call :meth:`~ray.data.Dataset.write_parquet` to save data in Parquet files.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_parquet("/tmp/simple")


    .. tab-item:: NumPy

        Call :meth:`~ray.data.Dataset.write_numpy` to save an ndarray column in NumPy
        files.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_numpy("/tmp/simple", column="image")

    .. tab-item:: JSON

        To save images in a JSON file, call :meth:`~ray.data.Dataset.write_json`.

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            ds.write_json("/tmp/simple")

有关保存数据的更多信息，请阅读 :ref:`保存数据 <loading_data>`。
