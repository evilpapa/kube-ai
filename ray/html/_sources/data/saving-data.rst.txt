.. _saving-data:

===========
数据保存
===========

Ray Data 允许您将数据保存在文件或其他 Python 对象中。

本指南向您展示如何：

* `将数据写入到文件 <#writing-data-to-files>`_
* `转换 Datasets 到其他 Python 类库 <#converting-datasets-to-other-python-libraries>`_

将数据写入文件
=====================

Ray Data写入本地磁盘和云存储。

将数据写入本地磁盘
~~~~~~~~~~~~~~~~~~~~~~~~~~

要将您的内容 :class:`~ray.data.dataset.Dataset` 保存到本地磁盘，请调用
:meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>` 方法并使用 `local://` 指定本地目录。

.. warning::

    如果您的集群包含多个节点并且您不使用 `local://`，Ray Data
    会将不同分区的数据写入不同的节点。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    ds.write_parquet("local:///tmp/iris/")

要将数据写入 Parquet 以外的格式，请阅读
:ref:`Input/Output 参考 <input-output>`。

将数据写入云存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要将您的数据 :class:`~ray.data.dataset.Dataset` 保存到云存储，请向您的云服务提供商验证所有节点。然后，调用类似方法
:meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>` 并使用适当的方案指定 URI。 URI 可以指向存储桶或文件夹。

.. tab-set::

    .. tab-item:: S3

        要将数据保存到 Amazon S3，请指定带有 ``s3://`` 方案的 URI。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("local:///tmp/iris.csv")

            ds.write_parquet("s3://my-bucket/my-folder")

    .. tab-item:: GCS

        To save data to Google Cloud Storage, install the
        `Filesystem interface to Google Cloud Storage <https://gcsfs.readthedocs.io/en/latest/>`_

        .. code-block:: console

            pip install gcsfs

        Then, create a ``GCSFileSystem`` and specify a URI with the ``gcs://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("local:///tmp/iris.csv")

            filesystem = gcsfs.GCSFileSystem(project="my-google-project")
            ds.write_parquet("gcs://my-bucket/my-folder", filesystem=filesystem)

    .. tab-item:: ABL

        To save data to Azure Blob Storage, install the
        `Filesystem interface to Azure-Datalake Gen1 and Gen2 Storage <https://pypi.org/project/adlfs/>`_

        .. code-block:: console

            pip install adlfs

        Then, create a ``AzureBlobFileSystem`` and specify a URI with the ``az://`` scheme.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("local:///tmp/iris.csv")

            filesystem = adlfs.AzureBlobFileSystem(account_name="azureopendatastorage")
            ds.write_parquet("az://my-bucket/my-folder", filesystem=filesystem)

要将数据写入 Parquet 以外的格式，请阅读
:ref:`Input/Output 参考 <input-output>`。

将数据写入 NFS
~~~~~~~~~~~~~~~~~~~

要将您的 :class:`~ray.data.dataset.Dataset` 保存到 NFS 文件系统，请调用类似方法
:meth:`Dataset.write_parquet <ray.data.Dataset.write_parquet>` 并指定挂载目录。

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    ds.write_parquet("/mnt/cluster_storage/iris")

要将数据写入 Parquet 以外的格式，请阅读
:ref:`Input/Output 参考 <input-output>`。

.. _changing-number-output-files:

更改输出文件的数量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当您调用写入方法时，Ray Data 会将您的数据写入每个 :term:`block` 的一个文件中。
要更改块数，请调用 :meth:`~ray.data.Dataset.repartition`。

.. testcode::

    import os
    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
    ds.repartition(2).write_csv("/tmp/two_files/")

    print(os.listdir("/tmp/two_files/"))

.. testoutput::
    :options: +MOCK

    ['26b07dba90824a03bb67f90a1360e104_000003.csv', '26b07dba90824a03bb67f90a1360e104_000002.csv']


将数据集转换为其他 Python 库
=============================================

将数据集转换为 pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要将 :class:`~ray.data.dataset.Dataset` 转换为 pandas DataFrame，请调用
:meth:`Dataset.to_pandas() <ray.data.Dataset.to_pandas>` 。
您的数据必须适合头节点上的内存。

.. testcode::

    import ray

    ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

    df = ds.to_pandas()
    print(df)

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target
    0                  5.1               3.5  ...               0.2       0
    1                  4.9               3.0  ...               0.2       0
    2                  4.7               3.2  ...               0.2       0
    3                  4.6               3.1  ...               0.2       0
    4                  5.0               3.6  ...               0.2       0
    ..                 ...               ...  ...               ...     ...
    145                6.7               3.0  ...               2.3       2
    146                6.3               2.5  ...               1.9       2
    147                6.5               3.0  ...               2.0       2
    148                6.2               3.4  ...               2.3       2
    149                5.9               3.0  ...               1.8       2
    <BLANKLINE>
    [150 rows x 5 columns]

将数据集转换为分布式 DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 与
:ref:`Dask <dask-on-ray>` 、 :ref:`Spark <spark-on-ray>` 、 :ref:`Modin <modin-on-ray>` 和
:ref:`Mars <mars-on-ray>` 等分布式数据处理框架进行互操作 。

.. tab-set::

    .. tab-item:: Dask

        要将 :class:`~ray.data.dataset.Dataset` 转换成
        `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__ ，调用
        :meth:`Dataset.to_dask() <ray.data.Dataset.to_dask>`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            df = ds.to_dask()

    .. tab-item:: Spark

        要将 :class:`~ray.data.dataset.Dataset` 转换成 `Spark DataFrame
        <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html>`__，
        调用 :meth:`Dataset.to_spark() <ray.data.Dataset.to_spark>`.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            df = ds.to_spark()

    .. tab-item:: Modin

        要将 :class:`~ray.data.dataset.Dataset` 转换成 Modin DataFrame，调用
        :meth:`Dataset.to_modin() <ray.data.Dataset.to_modin>`.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            mdf = ds.to_modin()

    .. tab-item:: Mars

       要将 :class:`~ray.data.dataset.Dataset` 转换成 Mars DataFrame，调用
        :meth:`Dataset.to_mars() <ray.data.Dataset.to_mars>`.

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

            mdf = ds.to_mars()