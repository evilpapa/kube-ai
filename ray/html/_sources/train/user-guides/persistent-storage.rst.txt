.. _persistent-storage-guide:

.. _train-log-dir:

配置持久存储
==============================

Ray Train 运行会产生一系列的 :ref:`reported metrics <train-monitoring-and-logging>`、
:ref:`checkpoints <train-checkpointing>` 和 :ref:`other artifacts <train-artifacts>`。
你可以配置这些内容保存到持久存储位置。

.. figure:: ../images/persistent_storage_checkpoint.png
    :align: center
    :width: 600px

    多个工作程序分布在多个节点上，将检查点上传到持久存储的示例。

**Ray Train 期望所有 worker 都能够将文件写入同一个持久存储位置。**
因此，Ray Train 需要某种形式的外部持久存储，
例如云存储（例如 S3、GCS）或
共享文件系统（例如 AWS EFS、Google Filestore、HDFS）来进行多节点训练。

以下是持久存储支持的一些功能：

- **检查点和容错**: 将检查点保存到持久存储位置，
  您可以在发生节点故障时从最后一个检查点恢复训练。
  有关如何设置检查点的详细指南，请参阅 :ref:`train-checkpointing`。
- **实验后分析**: 存储所有试验数据的合并位置对于实验后分析很有用，
  例如在集群终止后访问最佳检查点和超参数配置。
- **通过下游服务和批量推理任务桥接训练/微调**: ：您可以轻松访问模型和工件，
  与他人共享或在下游任务中使用它们。


云存储（AWS S3、Google Cloud 存储 ）
--------------------------------------------

.. tip::

    云存储是推荐的持久存储选项。

通过使用 :class:`RunConfig(storage_path) <ray.train.RunConfig>` 指定 桶 URI 存储位置来使用云存储：

.. code-block:: python

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path/",
            name="experiment_name",
        )
    )


确保 Ray 集群中的所有节点都可以访问云存储，这样工作器的输出就可以上传到共享云存储桶。
此例中，所有文件都上传到共享存储位置 ``s3://bucket-name/sub-path/experiment_name`` 以供进一步处理。


共享文件系统 (NFS、HDFS) 
-----------------------------

使用 :class:`RunConfig(storage_path) <ray.train.RunConfig>` 指定共享存储位置：

.. code-block:: python

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="/mnt/cluster_storage",
            # HDFS example:
            # storage_path=f"hdfs://{hostname}:{port}/subpath",
            name="experiment_name",
        )
    )

确保 Ray 集群中的所有节点都可以访问共享文件系统，例如 AWS EFS、Google Cloud Filestore 或 HDFS，
以便将输出保存到那里。
在此示例中，所有文件都保存到 ``/mnt/cluster_storage/experiment_name`` 以便进一步处理。


本地存储
-------------

对单节点集群使用本地存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您只是在单个节点（例如笔记本电脑）上运行实验，
Ray Train 将使用本地文件系统作为检查点和其他工件的存储位置。
默认情况下，结果将保存到 ``~/ray_results`` 具有唯一自动生成名称的子目录中，
除非您使用在 :class:`~ray.train.RunConfig` 自定义 ``storage_path`` 和 ``name``。


.. code-block:: python

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="/tmp/custom/storage/path",
            name="experiment_name",
        )
    )


在此示例中，所有实验结果均可在本地找到 ``/tmp/custom/storage/path/experiment_name`` 以进行进一步处理。


对多节点集群使用本地存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    在多个节点上运行时，不再支持使用头节点的本地文件系统作为持久存储位置。

    如果使用 :meth:`ray.train.report(..., checkpoint=...) <ray.train.report>` 保存检查点
    并在该集群上运行，则如果未设置 NFS 或云存储，Ray Train 将引发错误。
    这是因为 Ray Train 期望所有 worker 都能够将检查点写入相同的持久存储位置。

    如果您的训练循环没有保存检查点，
    则报告的指标仍将聚合到头节点上的本地存储路径。

    参考 `此问题 <https://github.com/ray-project/ray/issues/37177>`_ 以了解更多信息。


.. _custom-storage-filesystem:

自定义存储
--------------

如果上述情况不符合您的需求，Ray Train 可以支持自定义文件系统并执行自定义逻辑。
Ray Train 标准化了 ``pyarrow.fs.FileSystem`` 接口来集成存储
(`参阅这里的 API 参考 <https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html>`_)。

默认情况下，传递 ``storage_path=s3://bucket-name/sub-path/`` 会使用 pyarrow 的
`默认 S3 文件系统实现 <https://arrow.apache.org/docs/python/generated/pyarrow.fs.S3FileSystem.html>`_
来上传文件。 (`参考其他默认实现。 <https://arrow.apache.org/docs/python/api/filesystems.html#filesystem-implementations>`_)

通过 :class:`RunConfig(storage_filesystem) <ray.train.RunConfig>` 提供 ``pyarrow.fs.FileSystem`` 的自定义实现
来存储上传和下载逻辑。

.. warning::

    提供自定义文件系统时，关联 ``storage_path`` 应该是一个
    *没有协议前缀的* 合格文件系统路径。

    例如，如果您为 ``storage_path`` 提供了 ``s3://bucket-name/sub-path/``，
    ``storage_path`` 应是剥离 ``s3://`` 的 ``bucket-name/sub-path/``.
    参考以下使用示例。

.. code-block:: python

    import pyarrow.fs

    from ray import train
    from ray.train.torch import TorchTrainer

    fs = pyarrow.fs.S3FileSystem(
        endpoint_override="http://localhost:9000",
        access_key=...,
        secret_key=...
    )

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_filesystem=fs,
            storage_path="bucket-name/sub-path",
            name="experiment_name",
        )
    )


``fsspec`` 文件系统
~~~~~~~~~~~~~~~~~~~~~~~

`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ 提供许多文件系统实现，
例如 ``s3fs``、 ``gcsfs`` 等。

您可以通过用实用程序 ``pyarrow.fs`` 包装 ``fsspec`` 文件系统来使用这些实现。

.. code-block:: python

    # Make sure to install: `pip install -U s3fs`
    import s3fs
    import pyarrow.fs

    s3_fs = s3fs.S3FileSystem(
        key='miniokey...',
        secret='asecretkey...',
        endpoint_url='https://...'
    )
    custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

    run_config = RunConfig(storage_path="minio_bucket", storage_filesystem=custom_fs)

.. seealso::

    请参阅包装器实用程序 ``pyarrow.fs`` 的 API 参考：

    * https://arrow.apache.org/docs/python/generated/pyarrow.fs.PyFileSystem.html
    * https://arrow.apache.org/docs/python/generated/pyarrow.fs.FSSpecHandler.html



MinIO 和其他与 S3 兼容的存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

您可以按照 :ref:`如上示例 <custom-storage-filesystem>` 配置
自定义 S3 文件系统以与 MinIO 一起使用。

请注意，直接将这些作为查询参数包含在 ``storage_path`` URI 中是另一种选择：

.. code-block:: python

    from ray import train
    from ray.train.torch import TorchTrainer

    trainer = TorchTrainer(
        ...,
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path?endpoint_override=http://localhost:9000",
            name="experiment_name",
        )
    )


Ray Train 输出概述
-----------------------------

到目前为止，我们介绍了如何配置 Ray Train 输出的存储位置。
让我们通过一个具体的示例来了解这些输出到底是什么，
以及它们在存储中的结构。

.. seealso::

    本示例包含检查点，详细信息请参阅 :ref:`train-checkpointing`。

.. code-block:: python

    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import TorchTrainer

    def train_fn(config):
        for i in range(10):
            # Training logic here

            metrics = {"loss": ...}

            # Save arbitrary artifacts to the working directory
            rank = train.get_context().get_world_rank()
            with open(f"artifact-rank={rank}-iter={i}.txt", "w") as f:
                f.write("data")

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(..., os.path.join(temp_checkpoint_dir, "checkpoint.pt"))
                train.report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir)
                )

    trainer = TorchTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            storage_path="s3://bucket-name/sub-path/",
            name="experiment_name",
            sync_config=train.SyncConfig(sync_artifacts=True),
        )
    )
    result: train.Result = trainer.fit()
    last_checkpoint: Checkpoint = result.checkpoint

以下是将保存到存储中的所有文件的概要：

.. code-block:: text

    s3://bucket-name/sub-path (RunConfig.storage_path)
    └── experiment_name (RunConfig.name)          <- The "experiment directory"
        ├── experiment_state-*.json
        ├── basic-variant-state-*.json
        ├── trainer.pkl
        ├── tuner.pkl
        └── TorchTrainer_46367_00000_0_...        <- The "trial directory"
            ├── events.out.tfevents...            <- Tensorboard logs of reported metrics
            ├── result.json                       <- JSON log file of reported metrics
            ├── checkpoint_000000/                <- Checkpoints
            ├── checkpoint_000001/
            ├── ...
            ├── artifact-rank=0-iter=0.txt        <- Worker artifacts (see the next section)
            ├── artifact-rank=1-iter=0.txt
            └── ...

``trainer.fit`` 返回的 :class:`~ray.train.Result` 及 :class:`~ray.train.Checkpoint` 对象是访问这些文件中的数据的最简单方法：

.. code-block:: python

    result.filesystem, result.path
    # S3FileSystem, "bucket-name/sub-path/experiment_name/TorchTrainer_46367_00000_0_..."

    result.checkpoint.filesystem, result.checkpoint.path
    # S3FileSystem, "bucket-name/sub-path/experiment_name/TorchTrainer_46367_00000_0_.../checkpoint_000009"


参考 :ref:`train-inspect-results` 以获取有关与训练 :class:`Results <ray.train.Result>` 交互的完整指南。


.. _train-artifacts:

保存训练成果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如上示例中，我们将训练循环中的一些成功保存到 worker 的 *当前工作目录* 。
如果你正在训练一个稳定的扩散模型，你可以保存一些生成的样本图像作为训练工件。

默认，worker 的当前工作目录设置为“试验目录”的本地版本。
比如，上面示例中的 ``~/ray_results/experiment_name/TorchTrainer_46367_00000_0_...``。

如果 :class:`RunConfig(SyncConfig(sync_artifacts=True)) <ray.train.SyncConfig>`，
那么保存在此目录中的所有工件都将被持久保存到存储中。

可以通过 :class:`SyncConfig <ray.train.SyncConfig>` 配置工件同步的频率。
请注意，此行为默认处于关闭状态。

.. figure:: ../images/persistent_storage_artifacts.png
    :align: center
    :width: 600px

    分布在多个节点的多个 worker 将工件保存到其本地工作目录中，
    然后将其持久保存到存储中。

.. warning::

    *每个 worker* 保存的工件都将同步到存储。 如果您有多个 worker
    位于同一个节点上，请
    确保工作器不会删除其共享工作目录中的文件。

    最佳做法是只从单个 worker 那里写入工件，除非您确实需要来自多个 worker 的工件。

    .. code-block:: python

        from ray import train

        if train.get_context().get_world_rank() == 0:
            # Only the global rank 0 worker saves artifacts.
            ...

        if train.get_context().get_local_rank() == 0:
            # Every local rank 0 worker saves artifacts.
            ...


高级配置
----------------------

设置中间本地路径
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当指定 ``storage_path`` ，训练输出将保存到
*中间本地目录*，然后持久化（复制/上传）到 ``storage_path``。
认情况下，此中间本地目录是 ``~/ray_results`` 的子目录。

使用环境变量 ``RAY_AIR_LOCAL_CACHE_DIR`` 定制这个中间本地目录：

.. code-block:: python

    import os
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = "/tmp/custom/"

    ...

.. _train-ray-storage:

自动设置持久存储
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

您可以使用环境变量 ``RAY_STORAGE`` 控制存储训练结果的位置。

例如，如果您设置 ``RAY_STORAGE="s3://my_bucket/train_results"``，您的
结果将自动保存在那里。

如果您手动设置 :attr:`RunConfig.storage_path <ray.train.RunConfig.storage_path>`，它
将优先于此环境变量。
