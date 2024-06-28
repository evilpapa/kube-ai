.. _train-tensorflow-overview:

TensorFlow 和 Keras 入门
=====================================

Ray Train 的 `TensorFlow <https://www.tensorflow.org/>`__ 集成使您
能够将 TensorFlow 和 Keras 训练功能扩展到许多机器和 GPU。

从技术层面上讲，Ray Train 会为您安排训练工作程序并进行配置
``TF_CONFIG`` ， 让您可以运行
``MultiWorkerMirroredStrategy`` 训练脚本。有关更多信息，请参阅 `使用 TensorFlow 进行分布式训练 <https://www.tensorflow.org/guide/distributed_training>`_ 
获取更多信息。

本指南中的大多数示例均使用带有 Keras 的 TensorFlow，
但 Ray Train 也可使用原始 TensorFlow。


快速开始
-----------
.. literalinclude:: ./doc_code/tf_starter.py
  :language: python
  :start-after: __tf_train_start__
  :end-before: __tf_train_end__


更新你的训练函数
-----------------------------

首先，更新您的 :ref:`训练函数 <train-overview-training-function>` 以支持分布式训练。


.. note::
   当前 TensorFlow 实现支持
   ``MultiWorkerMirroredStrategy`` (和 ``MirroredStrategy``)。 如果您希望 Ray Train 支持其他策略，
   请 `在 GitHub 上提交功能请求 <https://github.com/ray-project/ray/issues>`_。

本说明与 Tensorflow `Keras 多 worker 
训练 <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras>`_ 教程紧密相关。
一个关键区别是 Ray Train 为您
处理了环境变量设置。

**步骤 1:** 在 ``MultiWorkerMirroredStrategy`` 中包装模型。

`MultiWorkerMirroredStrategy <https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy>`_
可实现同步分布式训练。 你 *必须* 在策略范围内构建和编译 ``Model``。

.. code-block:: python

    with tf.distribute.MultiWorkerMirroredStrategy().scope():
        model = ... # build model
        model.compile()

**步骤 2:** 更新 ``Dataset`` 批大小为 *全局* 批次大小。
size.

适当设置 ``batch_size`` 是因为 `batch <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch>`_ 在
工作进程之间均匀分割。

.. code-block:: diff

    -batch_size = worker_batch_size
    +batch_size = worker_batch_size * train.get_context().get_world_size()


.. warning::
    :ref:`除了 "OMP_NUM_THREADS" <omp-num-thread-note>`，
    Ray 不会自动设置任何与本地并行或线程相关的环境变量或配置。
    如果您想要更好地控制 TensorFlow 线程，请在 ``train_loop_per_worker`` 函数开头使用
    ``tf.config.threading`` 模块 （例如：
    ``tf.config.threading.set_inter_op_parallelism_threads(num_cpus)``）。

创建 TensorflowTrainer
--------------------------

``Trainer``\s 是 Ray Train 的主要类，
用于管理状态和执行训练。针对分布式 Tensorflow，
使用 :class:`~ray.train.tensorflow.TensorflowTrainer` 
可以这样设置：

.. code-block:: python

    from ray.train import ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer
    # For GPU Training, set `use_gpu` to True.
    use_gpu = False
    trainer = TensorflowTrainer(
        train_func,
        scaling_config=ScalingConfig(use_gpu=use_gpu, num_workers=2)
    )

要自定义后端设置，您可以传递
:class:`~ray.train.tensorflow.TensorflowConfig`：

.. code-block:: python

    from ray.train import ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer, TensorflowConfig

    trainer = TensorflowTrainer(
        train_func,
        tensorflow_backend=TensorflowConfig(...),
        scaling_config=ScalingConfig(num_workers=2),
    )


更多配置信息，请参考 :py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` API。


运行训练函数
-----------------------

借助分布式训练函数和 Ray Train ``Trainer``，
您现在可以开始训练了。

.. code-block:: python

    trainer.fit()

加载和预处理数据
------------------------

Tensorflow 默认使用其内部数据集分片策略，如 `指南中所述 <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#dataset_sharding>`__。
如果你的 TensorFlow 数据集与分布式加载兼容，你不需要改变任何东西。

如果你需要更多高级预处理，你可能想考虑使用 Ray Data 进行
分布式数据提取。请参阅 :ref:`Ray Data with Ray Train <data-ingest-torch>`。

主要不同是您可能希望在训练函数中将 Ray Data 数据集分片
转换为 TensorFlow 数据集，以便您
可以使用 Keras API 进行模型训练。

`参阅示例 <https://github.com/ray-project/ray/blob/master/python/ray/train/examples/tf/tune_tensorflow_autoencoder_example.py>`__
获取分布式数据加载。相关部分如下：

.. code-block:: python

    import tensorflow as tf
    from ray import train
    from ray.train.tensorflow import prepare_dataset_shard

    def train_func(config: dict):
        # ...

        # Get dataset shard from Ray Train
        dataset_shard = train.get_context().get_dataset_shard("train")

        # Define a helper function to build a TensorFlow dataset
        def to_tf_dataset(dataset, batch_size):
            def to_tensor_iterator():
                for batch in dataset.iter_tf_batches(
                    batch_size=batch_size, dtypes=tf.float32
                ):
                    yield batch["image"], batch["label"]

            output_signature = (
                tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
            )
            tf_dataset = tf.data.Dataset.from_generator(
                to_tensor_iterator, output_signature=output_signature
            )
            # Call prepare_dataset_shard to disable automatic sharding
            # (since the dataset is already sharded)
            return prepare_dataset_shard(tf_dataset)

        for epoch in range(epochs):
            # Call our helper function to build the dataset
            tf_dataset = to_tf_dataset(
                dataset=dataset_shard,
                batch_size=64,
            )
            history = multi_worker_model.fit(tf_dataset)



报告结果
--------------
在训练期间，训练环应向 Ray Train 报告中间结果和检查点。 
这些报告日志将结果记录到控制台输出并将其附加到本地日志文件。
日志还会触发 :ref:`检查点记录 <train-dl-configure-checkpoints>`。

最简单的使用 Keras 报告结果的方法是使用
:class:`~ray.train.tensorflow.keras.ReportCheckpointCallback`: 

.. code-block:: python

    from ray.train.tensorflow.keras import ReportCheckpointCallback

    def train_func(config: dict):
        # ...
        for epoch in range(epochs):
            model.fit(dataset, callbacks=[ReportCheckpointCallback()])


此回调会自动将所有结果和检查点从 Keras 训练函数
转发到 Ray Train。


结果汇总
~~~~~~~~~~~~~~~~~

TensorFlow Keras 会自动汇总所有 worker 的指标。 如果您希望对此有更多控制权，
考虑实现一个 `自定义训练循环 <https://www.tensorflow.org/tutorials/distribute/custom_training>`__。


保存并加载检查点
-------------------------

你可以通过调用在训练函数中的 ``train.report(metrics, checkpoint=Checkpoint(...))`` 来
保存 :class:`Checkpoints <ray.train.Checkpoint>`。
当执行你的 python 脚本，这些调用会将来自分布式 worker 的检查点状态保存在 ``Trainer`` 上。

你可以通过 :py:class:`~ray.train.Result` 的 ``checkpoint`` 属性访问最新保存的检查点，
并通过 ``best_checkpoints`` 属性访问最佳保存的检查点。

这些具体的例子展示了 Ray Train 如何在分布式训练中适当地保存检查点、模型权重但不保存模型。


.. code-block:: python

    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer

    import numpy as np

    def train_func(config):
        import tensorflow as tf
        n = 100
        # create a toy dataset
        # data   : X - dim = (n, 4)
        # target : Y - dim = (n, 1)
        X = np.random.normal(0, 1, size=(n, 4))
        Y = np.random.uniform(0, 1, size=(n, 1))

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            # toy neural network : 1-layer
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="linear", input_shape=(4,))])
            model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mse"])

        for epoch in range(config["num_epochs"]):
            history = model.fit(X, Y, batch_size=20)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                model.save(os.path.join(temp_checkpoint_dir, "model.keras"))
                checkpoint_dict = os.path.join(temp_checkpoint_dir, "checkpoint.json")
                with open(checkpoint_dict, "w") as f:
                    json.dump({"epoch": epoch}, f)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"loss": history.history["loss"][0]}, checkpoint=checkpoint)

    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
    )
    result = trainer.fit()
    print(result.checkpoint)

默认情况下，检查点持久化到本地磁盘中的
每个运行的 :ref:`日志目录 <train-log-dir>`。

加载检查点
~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import tempfile

    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.tensorflow import TensorflowTrainer

    import numpy as np

    def train_func(config):
        import tensorflow as tf
        n = 100
        # create a toy dataset
        # data   : X - dim = (n, 4)
        # target : Y - dim = (n, 1)
        X = np.random.normal(0, 1, size=(n, 4))
        Y = np.random.uniform(0, 1, size=(n, 1))

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            # toy neural network : 1-layer
            checkpoint = train.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    model = tf.keras.models.load_model(
                        os.path.join(checkpoint_dir, "model.keras")
                    )
            else:
                model = tf.keras.Sequential(
                    [tf.keras.layers.Dense(1, activation="linear", input_shape=(4,))]
                )
            model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mse"])

        for epoch in range(config["num_epochs"]):
            history = model.fit(X, Y, batch_size=20)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                model.save(os.path.join(temp_checkpoint_dir, "model.keras"))
                extra_json = os.path.join(temp_checkpoint_dir, "checkpoint.json")
                with open(extra_json, "w") as f:
                    json.dump({"epoch": epoch}, f)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"loss": history.history["loss"][0]}, checkpoint=checkpoint)

    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
    )
    result = trainer.fit()
    print(result.checkpoint)

    # Start a new run from a loaded checkpoint
    trainer = TensorflowTrainer(
        train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=2),
        resume_from_checkpoint=result.checkpoint,
    )
    result = trainer.fit()


进一步阅读
---------------
请参阅 :ref:`用户指南 <train-user-guides>` 以探索更多主题：

- :ref:`实验追踪 <train-experiment-tracking-native>`
- :ref:`容错和现场实例训练 <train-fault-tolerance>`
- :ref:`超参数优化 <train-tune>`
