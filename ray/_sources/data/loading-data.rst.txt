.. _loading_data:

============
数据加载
============

Ray Data 从各种来源加载数据。本指南向您展示如何：

* `读取图像 <#reading-files>`_ 等文件
* `加载内存数据 <#loading-data-from-other-libraries>`_ 如 pandas DataFrames
* `读取数据 <#reading-databases>`_ 如 MySQL

.. _reading-files:

读取文件
=============

Ray Data 从本地磁盘或云存储中读取多种文件格式的文件。
要查看支持的文件格式的完整列表，请参阅
:ref:`Input/Output 参考 <input-output>`。

.. tab-set::

    .. tab-item:: Parquet

        要读取 Parquet 文件，请调用 :func:`~ray.data.read_parquet`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_parquet("local:///tmp/iris.parquet")

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

    .. tab-item:: Images

        读取原始图像，请调用 :func:`~ray.data.read_images`。Ray Data 将图像表示为 NumPy ndarray。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_images("local:///tmp/batoidea/JPEGImages/")

            print(ds.schema())

        .. testoutput::
            :skipif: True

            Column  Type
            ------  ----
            image   numpy.ndarray(shape=(32, 32, 3), dtype=uint8)

    .. tab-item:: Text

        要读取文本行，请调用 :func:`~ray.data.read_text`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_text("local:///tmp/this.txt")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            text    string

    .. tab-item:: CSV

        要读取 CSV 文件，请调用 :func:`~ray.data.read_csv`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_csv("local:///tmp/iris.csv")

            print(ds.schema())

        .. testoutput::

            Column             Type
            ------             ----
            sepal length (cm)  double
            sepal width (cm)   double
            petal length (cm)  double
            petal width (cm)   double
            target             int64

    .. tab-item:: Binary

        要读取原始二进制文件，请调用 :func:`~ray.data.read_binary_files`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_binary_files("local:///tmp/file.dat")

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            bytes   binary

    .. tab-item:: TFRecords

        要读取 TFRecords 文件，请调用 :func:`~ray.data.read_tfrecords`。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_tfrecords("local:///tmp/iris.tfrecords")

            print(ds.schema())

        .. testoutput::

            Column             Type
            ------             ----
            sepal length (cm)  double
            sepal width (cm)   double
            petal length (cm)  double
            petal width (cm)   double
            target             int64

从本地磁盘读取文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要从本地磁盘读取文件，请调用如 :func:`~ray.data.read_parquet` 函数，并使用
``local://`` 协议指定路径。路径可以指向文件或目录。

要读取 Parquet 以外的格式，请参阅 :ref:`Input/Output 参考 <input-output>`。

.. tip::

    如果您的文件可以在每个节点上访问，请排除 ``local://`` 以在集群中并行读取任务。

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_parquet("local:///tmp/iris.parquet")

    print(ds.schema())

.. testoutput::

    Column        Type
    ------        ----
    sepal.length  double
    sepal.width   double
    petal.length  double
    petal.width   double
    variety       string

从云存储读取文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要读取云存储中的文件，请向云服务提供商验证所有节点。然后，调用类似方法
:func:`~ray.data.read_parquet` 并指定具有适当架构的 URI。 
URI 可以指向存储桶、文件夹或对象。

要读取 Parquet 以外的格式，请参阅 :ref:`Input/Output 参考 <input-output>`。

.. tab-set::

    .. tab-item:: S3

        要从 Amazon S3 读取文件，请使用 ``s3://`` 协议。

        .. testcode::

            import ray

            ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

    .. tab-item:: GCS

        要从 Google Cloud Storage 读取文件，请安装
        `Google Cloud Storage 的文件系统接口 <https://gcsfs.readthedocs.io/en/latest/>`_

        .. code-block:: console

            pip install gcsfs

        然后，创建一个 ``GCSFileSystem`` 并使用 ``gcs://`` 指定 URI。

        .. testcode::
            :skipif: True

            import ray

            ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

    .. tab-item:: ABL

        要从 Azure Blob 存储读取文件，请将
        `文件系统接口安装到 Azure-Datalake Gen1 和 Gen2 存储 <https://pypi.org/project/adlfs/>`_

        .. code-block:: console

            pip install adlfs

        然后，创建一个 ``AzureBlobFileSystem`` 并使用 `az://` 协议的 URI。

        .. testcode::
            :skipif: True

            import adlfs
            import ray

            ds = ray.data.read_parquet(
                "az://ray-example-data/iris.parquet",
                adlfs.AzureBlobFileSystem(account_name="azureopendatastorage")
            )

            print(ds.schema())

        .. testoutput::

            Column        Type
            ------        ----
            sepal.length  double
            sepal.width   double
            petal.length  double
            petal.width   double
            variety       string

从 NFS 读取文件
~~~~~~~~~~~~~~~~~~~~~~

要从 NFS 文件系统读取文件，请调用类似函数 :func:`~ray.data.read_parquet`
并指定已挂载文件系统上的文件。路径可以指向文件或目录。

要读取 Parquet 以外的格式，请参阅 :ref:`Input/Output 参考 <input-output>`。

.. testcode::
    :skipif: True

    import ray

    ds = ray.data.read_parquet("/mnt/cluster_storage/iris.parquet")

    print(ds.schema())

.. testoutput::

    Column        Type
    ------        ----
    sepal.length  double
    sepal.width   double
    petal.length  double
    petal.width   double
    variety       string

处理压缩文件
~~~~~~~~~~~~~~~~~~~~~~~~~

要读取压缩文件，请再 ``compression`` 中指定 ``arrow_open_stream_args`` 。
您可以使用 `Arrow 支持的任何编解码器 <https://arrow.apache.org/docs/python/generated/pyarrow.CompressedInputStream.html>`__。

.. testcode::

    import ray

    ds = ray.data.read_csv(
        "s3://anonymous@ray-example-data/iris.csv.gz",
        arrow_open_stream_args={"compression": "gzip"},
    )

从其他库加载数据
=================================

从单节点数据库加载数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 与 pandas、NumPy 和 Arrow 等库进行互操作。

.. tab-set::

    .. tab-item:: Python objects

        要从Python 对象创建 :class:`~ray.data.dataset.Dataset` ，调用
        :func:`~ray.data.from_items` 并传入 ``Dict``. Ray Data 将每个据 ``Dict`` 数据视为一行。

        .. testcode::

            import ray

            ds = ray.data.from_items([
                {"food": "spam", "price": 9.34},
                {"food": "ham", "price": 5.37},
                {"food": "eggs", "price": 0.94}
            ])

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=3,
               num_rows=3,
               schema={food: string, price: double}
            )

        您还可以从常规 Python 对象列表中创建一个 :class:`~ray.data.dataset.Dataset` 。

        .. testcode::

            import ray

            ds = ray.data.from_items([1, 2, 3, 4, 5])

            print(ds)

        .. testoutput::

            MaterializedDataset(num_blocks=5, num_rows=5, schema={item: int64})

    .. tab-item:: NumPy

        To create a :class:`~ray.data.dataset.Dataset` from a NumPy array, call
        :func:`~ray.data.from_numpy`. Ray Data treats the outer axis as the row
        dimension.

        .. testcode::

            import numpy as np
            import ray

            array = np.ones((3, 2, 2))
            ds = ray.data.from_numpy(array)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={data: numpy.ndarray(shape=(2, 2), dtype=double)}
            )

    .. tab-item:: pandas

        To create a :class:`~ray.data.dataset.Dataset` from a pandas DataFrame, call
        :func:`~ray.data.from_pandas`.

        .. testcode::

            import pandas as pd
            import ray

            df = pd.DataFrame({
                "food": ["spam", "ham", "eggs"],
                "price": [9.34, 5.37, 0.94]
            })
            ds = ray.data.from_pandas(df)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={food: object, price: float64}
            )

    .. tab-item:: PyArrow

        To create a :class:`~ray.data.dataset.Dataset` from an Arrow table, call
        :func:`~ray.data.from_arrow`.

        .. testcode::

            import pyarrow as pa

            table = pa.table({
                "food": ["spam", "ham", "eggs"],
                "price": [9.34, 5.37, 0.94]
            })
            ds = ray.data.from_arrow(table)

            print(ds)

        .. testoutput::

            MaterializedDataset(
               num_blocks=1,
               num_rows=3,
               schema={food: string, price: double}
            )

.. _loading_datasets_from_distributed_df:

从分布式 DataFrame 库加载数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 与
:ref:`Dask <dask-on-ray>`、 :ref:`Spark <spark-on-ray>`、 :ref:`Modin <modin-on-ray>` 和
:ref:`Mars <mars-on-ray>` 等分布式数据处理框架进行互操作 。

.. tab-set::

    .. tab-item:: Dask

        要从
        `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__ 创建 :class:`~ray.data.dataset.Dataset`，调用
        :func:`~ray.data.from_dask`。
        该函数构造一个由 Dask DataFrame 的分布式 Pandas DataFrame 分区支持的 ``Dataset``。

        .. testcode::
            :skipif: True

            import dask.dataframe as dd
            import pandas as pd
            import ray

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            ddf = dd.from_pandas(df, npartitions=4)
            # Create a Dataset from a Dask DataFrame.
            ds = ray.data.from_dask(ddf)

            ds.show(3)

        .. testoutput::

            {'string': 'spam', 'number': 0}
            {'string': 'ham', 'number': 1}
            {'string': 'eggs', 'number': 2}

    .. tab-item:: Spark

        从 `Spark DataFrame
        <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html>`__ 创建 :class:`~ray.data.dataset.Dataset`，调用
        :func:`~ray.data.from_spark`. 
        该函数构造一个由 Spark DataFrame 的分布式 Pandas DataFrame 分区支持的 ``Dataset``。

        .. testcode::
            :skipif: True

            import ray
            import raydp

            spark = raydp.init_spark(app_name="Spark -> Datasets Example",
                                    num_executors=2,
                                    executor_cores=2,
                                    executor_memory="500MB")
            df = spark.createDataFrame([(i, str(i)) for i in range(10000)], ["col1", "col2"])
            ds = ray.data.from_spark(df)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

    .. tab-item:: Modin

        要从 Modin DataFrame 创建 :class:`~ray.data.dataset.Dataset`，调用
        :func:`~ray.data.from_modin`。
        该函数构造一个由 Modin DataFrame 的分布式 Pandas DataFrame 分区支持的 ``Dataset``。

        .. testcode::
            :skipif: True

            import modin.pandas as md
            import pandas as pd
            import ray

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            mdf = md.DataFrame(df)
            # Create a Dataset from a Modin DataFrame.
            ds = ray.data.from_modin(mdf)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

    .. tab-item:: Mars

        从  Mars DataFrame 创建 :class:`~ray.data.dataset.Dataset` ，调用
        :func:`~ray.data.from_mars`. 
        该函数构造一个由 Mars DataFrame 的分布式 Pandas DataFrame 分区支持的 ``Dataset``。

        .. testcode::
            :skipif: True

            import mars
            import mars.dataframe as md
            import pandas as pd
            import ray

            cluster = mars.new_cluster_in_ray(worker_num=2, worker_cpu=1)

            df = pd.DataFrame({"col1": list(range(10000)), "col2": list(map(str, range(10000)))})
            mdf = md.DataFrame(df, num_partitions=8)
            # Create a tabular Dataset from a Mars DataFrame.
            ds = ray.data.from_mars(mdf)

            ds.show(3)

        .. testoutput::

            {'col1': 0, 'col2': '0'}
            {'col1': 1, 'col2': '1'}
            {'col1': 2, 'col2': '2'}

.. _loading_datasets_from_ml_libraries:

从 ML 库加载数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data 与 HuggingFace 和 TensorFlow 数据集互操作。

.. tab-set::

    .. tab-item:: HuggingFace

        要将 🤗 数据集转换为 Ray 数据集，请调用
        :func:`~ray.data.from_huggingface`。
        该函数构造一个由 🤗 数据集的分布式 Pandas DataFrame 分区支持的 ``Dataset``。

        .. warning::
            :class:`~ray.data.from_huggingface` 不支持并行读取。对于内存中 🤗 数据集来说这不是问题，但对于大型内存映射 🤗 数据集可能会失败。此外， 🤗 ``IterableDataset`` 对象不支持。

        .. testcode::

            import ray.data
            from datasets import load_dataset

            hf_ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            ray_ds = ray.data.from_huggingface(hf_ds["train"])
            ray_ds.take(2)

        .. testoutput::
            :options: +MOCK

            [{'text': ''}, {'text': ' = Valkyria Chronicles III = \n'}]

    .. tab-item:: TensorFlow

        要转换 TensorFlow dataset 为 Ray Dataset，调用 :func:`~ray.data.from_tf`。

        .. warning::
            :class:`~ray.data.from_tf` 不支持并行读取。仅将此函数用于 MNIST 或 CIFAR 等小型数据集。

        .. testcode::

            import ray
            import tensorflow_datasets as tfds

            tf_ds, _ = tfds.load("cifar10", split=["train", "test"])
            ds = ray.data.from_tf(tf_ds)

            print(ds)

        ..
            The following `testoutput` is mocked to avoid illustrating download logs like
            "Downloading and preparing dataset 162.17 MiB".

        .. testoutput::
            :options: +MOCK

            MaterializedDataset(
               num_blocks=...,
               num_rows=50000,
               schema={
                  id: binary,
                  image: numpy.ndarray(shape=(32, 32, 3), dtype=uint8),
                  label: int64
               }
            )

读取数据库
=================

Ray Data 从 MySQL、PostgreSQL 和 MongoDB 等数据库读取。

.. _reading_sql:

读取 SQL 数据库
~~~~~~~~~~~~~~~~~~~~~

调用 `Python DB API2 标准 <https://peps.python.org/pep-0249/>`_ 连接器的 :func:`~ray.data.read_sql` 从数据库中读取数据 。

.. tab-set::

    .. tab-item:: MySQL

        要从 MySQL 读取数据，请安装
        `MySQL Connector/Python <https://dev.mysql.com/doc/connector-python/en/>`_。它是第一方 MySQL 数据库连接器。

        .. code-block:: console

            pip install mysql-connector-python

        然后，定义连接逻辑并查询数据库。

        .. testcode::
            :skipif: True

            import mysql.connector

            import ray

            def create_connection():
                return mysql.connector.connect(
                    user="admin",
                    password=...,
                    host="example-mysql-database.c2c2k1yfll7o.us-west-2.rds.amazonaws.com",
                    connection_timeout=30,
                    database="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )


    .. tab-item:: PostgreSQL

        To read from PostgreSQL, install `Psycopg 2 <https://www.psycopg.org/docs>`_. It's
        the most popular PostgreSQL database connector.

        .. code-block:: console

            pip install psycopg2-binary

        Then, define your connection logic and query the database.

        .. testcode::
            :skipif: True

            import psycopg2

            import ray

            def create_connection():
                return psycopg2.connect(
                    user="postgres",
                    password=...,
                    host="example-postgres-database.c2c2k1yfll7o.us-west-2.rds.amazonaws.com",
                    dbname="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )

    .. tab-item:: Snowflake

        To read from Snowflake, install the
        `Snowflake Connector for Python <https://docs.snowflake.com/en/user-guide/python-connector>`_.

        .. code-block:: console

            pip install snowflake-connector-python

        Then, define your connection logic and query the database.

        .. testcode::
            :skipif: True

            import snowflake.connector

            import ray

            def create_connection():
                return snowflake.connector.connect(
                    user=...,
                    password=...
                    account="ZZKXUVH-IPB52023",
                    database="example",
                )

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )


    .. tab-item:: Databricks

        To read from Databricks, install the
        `Databricks SQL Connector for Python <https://docs.databricks.com/dev-tools/python-sql-connector.html>`_.

        .. code-block:: console

            pip install databricks-sql-connector


        Then, define your connection logic and read from the Databricks SQL warehouse.

        .. testcode::
            :skipif: True

            from databricks import sql

            import ray

            def create_connection():
                return sql.connect(
                    server_hostname="dbc-1016e3a4-d292.cloud.databricks.com",
                    http_path="/sql/1.0/warehouses/a918da1fc0b7fed0",
                    access_token=...,


            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )

    .. tab-item:: BigQuery

        To read from BigQuery, install the
        `Python Client for Google BigQuery <https://cloud.google.com/python/docs/reference/bigquery/latest>`_.
        This package includes a DB API2-compliant database connector.

        .. code-block:: console

            pip install google-cloud-bigquery

        Then, define your connection logic and query the dataset.

        .. testcode::
            :skipif: True

            from google.cloud import bigquery
            from google.cloud.bigquery import dbapi

            import ray

            def create_connection():
                client = bigquery.Client(...)
                return dbapi.Connection(client)

            # Get all movies
            dataset = ray.data.read_sql("SELECT * FROM movie", create_connection)
            # Get movies after the year 1980
            dataset = ray.data.read_sql(
                "SELECT title, score FROM movie WHERE year >= 1980", create_connection
            )
            # Get the number of movies per year
            dataset = ray.data.read_sql(
                "SELECT year, COUNT(*) FROM movie GROUP BY year", create_connection
            )

.. _reading_mongodb:

读取 MongoDB
~~~~~~~~~~~~~~~

要从 MongoDB 读取数据，请调用 :func:`~ray.data.read_mongo` 并指定源 URI、数据库和集合。
您还需要指定针对集合运行的管道。

.. testcode::
    :skipif: True

    import ray

    # Read a local MongoDB.
    ds = ray.data.read_mongo(
        uri="mongodb://localhost:27017",
        database="my_db",
        collection="my_collection",
        pipeline=[{"$match": {"col": {"$gte": 0, "$lt": 10}}}, {"$sort": "sort_col"}],
    )

    # Reading a remote MongoDB is the same.
    ds = ray.data.read_mongo(
        uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin",
        database="my_db",
        collection="my_collection",
        pipeline=[{"$match": {"col": {"$gte": 0, "$lt": 10}}}, {"$sort": "sort_col"}],
    )

    # Write back to MongoDB.
    ds.write_mongo(
        MongoDatasource(),
        uri="mongodb://username:password@mongodb0.example.com:27017/?authSource=admin",
        database="my_db",
        collection="my_collection",
    )

创建合成数据
=======================

综合数据集可用于测试和基准测试。

.. tab-set::

    .. tab-item:: Int Range

        要从一系列整数创建合成 :class:`~ray.data.Dataset` ，调用
        :func:`~ray.data.range`。 Ray Data 将整数范围存储在单列中。

        .. testcode::

            import ray

            ds = ray.data.range(10000)

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            id      int64

    .. tab-item:: Tensor Range

        To create a synthetic :class:`~ray.data.Dataset` containing arrays, call
        :func:`~ray.data.range_tensor`. Ray Data packs an integer range into ndarrays of
        the provided shape.

        .. testcode::

            import ray

            ds = ray.data.range_tensor(10, shape=(64, 64))

            print(ds.schema())

        .. testoutput::

            Column  Type
            ------  ----
            data    numpy.ndarray(shape=(64, 64), dtype=int64)

加载其他数据源
==========================

如果 Ray Data 无法加载您的数据，请使用
:class:`~ray.data.datasource.Datasource`。然后，构建自定义数据源的实例并将其传递给
给 :func:`~ray.data.read_datasource`。

.. testcode::
    :skipif: True

    # Read from a custom datasource.
    ds = ray.data.read_datasource(YourCustomDatasource(), **read_args)

    # Write to a custom datasource.
    ds.write_datasource(YourCustomDatasource(), **write_args)

有关示例，请参阅 :ref:`实现自定义数据源 <custom_datasources>`。

性能考虑
==========================

``parallelism`` 数据集决定了基础数据被分割成并行读取的块数。
 Ray Data 在内部决定同时运行多少个读取任务，以充分利用集群，范围从 ``1...parallelism`` 个读取任务。
换句话说，并行度越高，Dataset 中的数据块越小，因此并行执行的机会就越多。

.. image:: images/dataset-read.svg
   :width: 650px
   :align: center

可以通过 ``parallelism`` 数覆盖此默认并行性； 有关如何调整读取并行性的更多信息，请参阅
:ref:`性能指南 <data_performance_tips>` 。
