.. _inspecting-data:

===============
检查数据
===============

检查 :class:`Datasets <ray.data.Dataset>` 来更好的了解你的数据。

本指南将向您展示如何：

* `描述 datasets <#describing-datasets>`_
* `检查行 <#inspecting-rows>`_
* `检验批次 <#inspecting-batches>`_
* `检查执行统计 <#inspecting-stats>`_

.. _describing-datasets:

描述 Dataset
===================

:class:`Datasets <ray.data.Dataset>` 是表格形式。要查看数据集的列
名称和类型，请调用 :meth:`Dataset.schema() <ray.data.Dataset.schema>`。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    print(ds.schema())

.. testoutput::

    Column             Type
    ------             ----
    sepal length (cm)  double
    sepal width (cm)   double
    petal length (cm)  double
    petal width (cm)   double
    target             int64

如需更多信息（例如行数），请打印数据集。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    print(ds)

.. testoutput::

    Dataset(
       num_blocks=...,
       num_rows=150,
       schema={
          sepal length (cm): double,
          sepal width (cm): double,
          petal length (cm): double,
          petal width (cm): double,
          target: int64
       }
    )

.. _inspecting-rows:

检查行
===============

要获取行列表，请调用 :meth:`Dataset.take() <ray.data.Dataset.take>` 或
:meth:`Dataset.take_all() <ray.data.Dataset.take_all>`。Ray Data 将每一行表示为一个字典。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    rows = ds.take(1)
    print(rows)

.. testoutput::

    [{'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}]


有关处理行的更多信息，请参阅
:ref:`转换行 <transforming_rows>` 和
:ref:`迭代行 <iterating-over-rows>`。

.. _inspecting-batches:

检查批次
==================

一个批次包含来自多行的数据。要检查批次，请调用
`Dataset.take_batch() <ray.data.Dataset.take_batch>`。

默认情况下，Ray Data 将批次表示为 NumPy ndarrays 的字典。要更改
返回批次的类型，请设置 ``batch_format``。

.. tab-set::

    .. tab-item:: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            batch = ds.take_batch(batch_size=2, batch_format="numpy")
            print("Batch:", batch)
            print("Image shape", batch["image"].shape)

        .. testoutput::
            :options: +MOCK

            Batch: {'image': array([[[[...]]]], dtype=uint8)}
            Image shape: (2, 32, 32, 3)

    .. tab-item:: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            batch = ds.take_batch(batch_size=2, batch_format="pandas")
            print(batch)

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

               sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
            0                5.1               3.5  ...               0.2       0
            1                4.9               3.0  ...               0.2       0
            <BLANKLINE>
            [2 rows x 5 columns]

有关使用批次的更多信息，请参阅
:ref:`转换批次 <transforming_batches>` 和
:ref:`迭代批次 <iterating-over-batches>`。


检查执行统计信息
===============================

Ray Data 在执行期间计算统计数据，例如不同阶段的挂钟时间和内存使用情况。

要查看有关 :class:`Datasets <ray.data.Dataset>` 的统计数据，请在已执行的数据集上调用 :meth:`Dataset.stats() <ray.data.Dataset.stats>` 。统计数据也会保留在 `/tmp/ray/session_*/logs/ray-data.log` 下。

.. testcode::
    import ray
    import time

    def pause(x):
        time.sleep(.0001)
        return x

    ds = (
        ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
        .map(lambda x: x)
        .map(pause)
    )

    for batch in ds.iter_batches():
        pass

    print(ds.stats())

.. testoutput::
    :options: +MOCK

    Stage 1 ReadCSV->Map(<lambda>)->Map(pause): 1/1 blocks executed in 0.23s
    * Remote wall time: 222.1ms min, 222.1ms max, 222.1ms mean, 222.1ms total
    * Remote cpu time: 15.6ms min, 15.6ms max, 15.6ms mean, 15.6ms total
    * Peak heap memory usage (MiB): 157953.12 min, 157953.12 max, 157953 mean
    * Output num rows: 150 min, 150 max, 150 mean, 150 total
    * Output size bytes: 6000 min, 6000 max, 6000 mean, 6000 total
    * Tasks per node: 1 min, 1 max, 1 mean; 1 nodes used
    * Extra metrics: {'obj_store_mem_alloc': 6000, 'obj_store_mem_freed': 5761, 'obj_store_mem_peak': 6000}

    Dataset iterator time breakdown:
    * Total time user code is blocked: 5.68ms
    * Total time in user code: 0.96us
    * Total time overall: 238.93ms
    * Num blocks local: 0
    * Num blocks remote: 0
    * Num blocks unknown location: 1
    * Batch iteration time breakdown (summed across prefetch threads):
        * In ray.get(): 2.16ms min, 2.16ms max, 2.16ms avg, 2.16ms total
        * In batch creation: 897.67us min, 897.67us max, 897.67us avg, 897.67us total
        * In batch formatting: 836.87us min, 836.87us max, 836.87us avg, 836.87us total
