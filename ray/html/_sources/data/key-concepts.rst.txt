.. _data_key_concepts:

关键概念
============

了解 :class:`Dataset <ray.data.Dataset>` 及其提供的功能。

本指南提供了以下内容的简单介绍：

* :ref:`加载数据 <loading_key_concept>`
* :ref:`转换数据 <transforming_key_concept>`
* :ref:`消费数据 <consuming_key_concept>`
* :ref:`保存数据 <saving_key_concept>`

Datasets
--------

Ray Data 的主要抽象是 :class:`Dataset <ray.data.Dataset>`，它是一个分布式数据集合。
数据集是为机器学习而设计的，它们可以表示超出单台机器内存的数据集合。

.. _loading_key_concept:

加载数据
------------

从磁盘文件、Python 对象和 S3 等云存储服务创建数据集。 Ray Data 可以从 `Arrow 支持的任何文件系统
<http://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`__ 读取。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
    ds.show(limit=1)

.. testoutput::

    {'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}

要了解有关创建数据集的更多信息，请阅读 :ref:`加载数据 <loading_data>`。

.. _transforming_key_concept:

转换数据
-----------------

应用用户定义函数 (UDF) 来转换数据集。 Ray 并行执行转换以提高性能。

.. testcode::

    from typing import Dict
    import numpy as np

    # Compute a "petal area" attribute.
    def transform_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        vec_a = batch["petal length (cm)"]
        vec_b = batch["petal width (cm)"]
        batch["petal area (cm^2)"] = vec_a * vec_b
        return batch

    transformed_ds = ds.map_batches(transform_batch)
    print(transformed_ds.materialize())

.. testoutput::

    MaterializedDataset(
       num_blocks=...,
       num_rows=150,
       schema={
          sepal length (cm): double,
          sepal width (cm): double,
          petal length (cm): double,
          petal width (cm): double,
          target: int64,
          petal area (cm^2): double
       }
    )

要了解有关转换数据集的更多信息，请阅读
:ref:`转换数据 <transforming_data>`.

.. _consuming_key_concept:

消费数据
--------------

将数据集传递给 Ray Tasks 或 Actors，并使用
:meth:`~ray.data.Dataset.take_batch` 和 :meth:`~ray.data.Dataset.iter_batches` 等方法访问记录。

.. tab-set::

    .. tab-item:: Local

        .. testcode::

            print(transformed_ds.take_batch(batch_size=3))

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

            {'sepal length (cm)': array([5.1, 4.9, 4.7]),
             'sepal width (cm)': array([3.5, 3. , 3.2]),
             'petal length (cm)': array([1.4, 1.4, 1.3]),
             'petal width (cm)': array([0.2, 0.2, 0.2]),
             'target': array([0, 0, 0]),
             'petal area (cm^2)': array([0.28, 0.28, 0.26])}

    .. tab-item:: Tasks

       .. testcode::

            @ray.remote
            def consume(ds: ray.data.Dataset) -> int:
                num_batches = 0
                for batch in ds.iter_batches(batch_size=8):
                    num_batches += 1
                return num_batches

            ray.get(consume.remote(transformed_ds))

    .. tab-item:: Actors

        .. testcode::

            @ray.remote
            class Worker:

                def train(self, data_iterator):
                    for batch in data_iterator.iter_batches(batch_size=8):
                        pass

            workers = [Worker.remote() for _ in range(4)]
            shards = transformed_ds.streaming_split(n=4, equal=True)
            ray.get([w.train.remote(s) for w, s in zip(workers, shards)])


要了解有关使用数据集的更多信息，请参阅
:ref:`迭代数据 <iterating-over-data>` 和 :ref:`保存数据 <saving-data>`.

.. _saving_key_concept:

保存数据
-----------

调用 :meth:`~ray.data.Dataset.write_parquet` 等方法将数据集内容保存到本地或远程文件系统。

.. testcode::
    :hide:

    # The number of blocks can be non-determinstic. Repartition the dataset beforehand
    # so that the number of written files is consistent.
    transformed_ds = transformed_ds.repartition(2)

.. testcode::

    import os

    transformed_ds.write_parquet("/tmp/iris")

    print(os.listdir("/tmp/iris"))

.. testoutput::
    :options: +MOCK
    
    ['..._000000.parquet', '..._000001.parquet']


要了解有关保存数据集内容的更多信息，请参阅 :ref:`保存数据 <saving-data>`。
