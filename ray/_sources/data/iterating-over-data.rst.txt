.. _iterating-over-data:

===================
数据迭代
===================

Ray Data 允许您迭代数据行或批量迭代数据。

本指南向您展示如何：

* `迭代行 <#iterating-over-rows>`_
* `批量迭代 <#iterating-over-batches>`_
* `通过 shuffling 批量迭代 <#iterating-over-batches-with-shuffling>`_
* `分割数据集以进行分布式并行训练 <#splitting-datasets-for-distributed-parallel-training>`_

.. _iterating-over-rows:

迭代行
===================

要迭代数据集的行，请调用
:meth:`Dataset.iter_rows() <ray.data.Dataset.iter_rows>`。Ray Data 将每一行表示为一个字典。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

    for row in ds.iter_rows():
        print(row)

.. testoutput::

    {'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}
    {'sepal length (cm)': 4.9, 'sepal width (cm)': 3.0, 'petal length (cm)': 1.4, 'petal width (cm)': 0.2, 'target': 0}
    ...
    {'sepal length (cm)': 5.9, 'sepal width (cm)': 3.0, 'petal length (cm)': 5.1, 'petal width (cm)': 1.8, 'target': 2}


有关使用行的更多信息，请参阅
:ref:`转换行 <transforming_rows>` 和
:ref:`检查行 <inspecting-rows>`。

.. _iterating-over-batches:

批量迭代
======================

一个批次包含来自多行的数据。通过调用以下方法之一迭代不同格式的批量数据集：

* `Dataset.iter_batches() <ray.data.Dataset.iter_batches>`
* `Dataset.iter_torch_batches() <ray.data.Dataset.iter_torch_batches>`
* `Dataset.to_tf() <ray.data.Dataset.to_tf>`

.. tab-set::

    .. tab-item:: NumPy
        :sync: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_batches(batch_size=2, batch_format="numpy"):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': array([[[[...]]]], dtype=uint8)}
            ...
            {'image': array([[[[...]]]], dtype=uint8)}

    .. tab-item:: pandas
        :sync: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            for batch in ds.iter_batches(batch_size=2, batch_format="pandas"):
                print(batch)

        .. testoutput::
            :options: +MOCK

               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                5.1               3.5                1.4               0.2       0
            1                4.9               3.0                1.4               0.2       0
            ...
               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                6.2               3.4                5.4               2.3       2
            1                5.9               3.0                5.1               1.8       2

    .. tab-item:: Torch
        :sync: Torch

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_torch_batches(batch_size=2):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': tensor([[[[...]]]], dtype=torch.uint8)}
            ...
            {'image': tensor([[[[...]]]], dtype=torch.uint8)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            tf_dataset = ds.to_tf(
                feature_columns="sepal length (cm)",
                label_columns="target",
                batch_size=2
            )
            for features, labels in tf_dataset:
                print(features, labels)

        .. testoutput::

            tf.Tensor([5.1 4.9], shape=(2,), dtype=float64) tf.Tensor([0 0], shape=(2,), dtype=int64)
            ...
            tf.Tensor([6.2 5.9], shape=(2,), dtype=float64) tf.Tensor([2 2], shape=(2,), dtype=int64)

有关使用批量的更多信息，请参阅
:ref:`批量转换 <transforming_batches>` 和
:ref:`批量检查 <inspecting-batches>`。

.. _iterating-over-batches-with-shuffling:

通过 shuffling 批量迭代
=====================================

:class:`Dataset.random_shuffle <ray.data.Dataset.random_shuffle>` 速度很慢，因为它会打乱所有行。
如果不需要完整的全局混洗，您可以通过指定在迭代期间将行子集混洗到提供的缓冲区大小
``local_shuffle_buffer_size``。虽然这不是真正的全局随机播放
``random_shuffle``，但它的性能更高，因为它不需要过多的数据移动。

.. tip::

    要配置 ``local_shuffle_buffer_size``，请选择实现足够随机性的最小值。
    较高的值会导致更多的随机性，但代价是迭代速度较慢。

.. tab-set::

    .. tab-item:: NumPy
        :sync: NumPy

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")

            for batch in ds.iter_batches(
                batch_size=2,
                batch_format="numpy",
                local_shuffle_buffer_size=250,
            ):
                print(batch)


        .. testoutput::
            :options: +MOCK

            {'image': array([[[[...]]]], dtype=uint8)}
            ...
            {'image': array([[[[...]]]], dtype=uint8)}

    .. tab-item:: pandas
        :sync: pandas

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            for batch in ds.iter_batches(
                batch_size=2,
                batch_format="pandas",
                local_shuffle_buffer_size=250,
            ):
                print(batch)

        .. testoutput::
            :options: +MOCK

               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                6.3               2.9                5.6               1.8       2
            1                5.7               4.4                1.5               0.4       0
            ...
               sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
            0                5.6               2.7                4.2               1.3       1
            1                4.8               3.0                1.4               0.1       0

    .. tab-item:: Torch
        :sync: Torch

        .. testcode::

            import ray

            ds = ray.data.read_images("s3://anonymous@ray-example-data/image-datasets/simple")
            for batch in ds.iter_torch_batches(
                batch_size=2,
                local_shuffle_buffer_size=250,
            ):
                print(batch)

        .. testoutput::
            :options: +MOCK

            {'image': tensor([[[[...]]]], dtype=torch.uint8)}
            ...
            {'image': tensor([[[[...]]]], dtype=torch.uint8)}

    .. tab-item:: TensorFlow
        :sync: TensorFlow

        .. testcode::

            import ray

            ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")

            tf_dataset = ds.to_tf(
                feature_columns="sepal length (cm)",
                label_columns="target",
                batch_size=2,
                local_shuffle_buffer_size=250,
            )
            for features, labels in tf_dataset:
                print(features, labels)

        .. testoutput::
            :options: +MOCK

            tf.Tensor([5.2 6.3], shape=(2,), dtype=float64) tf.Tensor([1 2], shape=(2,), dtype=int64)
            ...
            tf.Tensor([5.  5.8], shape=(2,), dtype=float64) tf.Tensor([0 0], shape=(2,), dtype=int64)

分割数据集以进行分布式并行训练
====================================================

如果您正在执行分布式数据并行训练，请调用
:meth:`Dataset.streaming_split <ray.data.Dataset.streaming_split>` 将数据集拆分为不相交的分片。

.. note::

  如果您使用 :ref:`Ray Train <train-docs>`，则无需拆分数据集。 
  Ray Train 会自动为您分割数据集。要了解更多信息，请参阅
  :ref:`ML 训练数据加载指南 <data-ingest-torch>`。

.. testcode::

    import ray

    @ray.remote
    class Worker:

        def train(self, data_iterator):
            for batch in data_iterator.iter_batches(batch_size=8):
                pass

    ds = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
    workers = [Worker.remote() for _ in range(4)]
    shards = ds.streaming_split(n=4, equal=True)
    ray.get([w.train.remote(s) for w, s in zip(workers, shards)])
