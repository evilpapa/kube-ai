.. _spark-on-ray:

**************************
Ray 上运行 Spark (RayDP)
**************************

RayDP 合并了您的 Spark 和 Ray 集群，使得使用 PySpark API 进行大规模数据处理变得容易，
并且可以无缝地使用这些数据来训练您的 TensorFlow 和 PyTorch 模型。

更多信息和示例，请参阅 RayDP Github 页面：
https://github.com/oap-project/raydp


================
安装 RayDP
================

RayDP 可以从 PyPI 安装，支持 PySpark 3.0 和 3.1。

.. code-block bash

  pip install raydp

.. note::
  RayDP requires ray >= 1.2.0

.. note::
  要运行 Spark，head 和 worker 节点需要安装 Java。

========================
创建 Spark 会话
========================

要创建一个 Spark 会话，调用 ``raydp.init_spark``

例如，

.. code-block:: python

  import ray
  import raydp

  ray.init()
  spark = raydp.init_spark(
    app_name = "example",
    num_executors = 10,
    executor_cores = 64,
    executor_memory = "256GB"
  )

====================================
使用 Spark DataFrame 进行深度学习
====================================

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
使用 Spark DataFrame 进行 TensorFlow 训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``raydp.tf.TFEstimator`` 使用 Spark DataFrame 进行训练，支持分布式训练。

.. code-block:: python

  from pyspark.sql.functions import col
  df = spark.range(1, 1000)
  # calculate z = x + 2y + 1000
  df = df.withColumn("x", col("id")*2)\
    .withColumn("y", col("id") + 200)\
    .withColumn("z", col("x") + 2*col("y") + 1000)
  
  from raydp.utils import random_split
  train_df, test_df = random_split(df, [0.7, 0.3])

  # TensorFlow code
  from tensorflow import keras
  input_1 = keras.Input(shape=(1,))
  input_2 = keras.Input(shape=(1,))

  concatenated = keras.layers.concatenate([input_1, input_2])
  output = keras.layers.Dense(1, activation='sigmoid')(concatenated)
  model = keras.Model(inputs=[input_1, input_2],
                      outputs=output)

  optimizer = keras.optimizers.Adam(0.01)
  loss = keras.losses.MeanSquaredError()

  from raydp.tf import TFEstimator
  estimator = TFEstimator(
    num_workers=2,
    model=model,
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy", "mse"],
    feature_columns=["x", "y"],
    label_column="z",
    batch_size=1000,
    num_epochs=2,
    use_gpu=False,
    config={"fit_config": {"steps_per_epoch": 2}})

  estimator.fit_on_spark(train_df, test_df)

  tensorflow_model = estimator.get_model()

  estimator.shutdown()


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
使用 Spark DataFrame 进行 PyTorch 训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

相似的，``raydp.torch.TorchEstimator`` 提供了一个 PyTorch 训练的 API。

.. code-block:: python

  from pyspark.sql.functions import col
  df = spark.range(1, 1000)
  # calculate z = x + 2y + 1000
  df = df.withColumn("x", col("id")*2)\
    .withColumn("y", col("id") + 200)\
    .withColumn("z", col("x") + 2*col("y") + 1000)
  
  from raydp.utils import random_split
  train_df, test_df = random_split(df, [0.7, 0.3])

  # PyTorch Code 
  import torch
  class LinearModel(torch.nn.Module):
      def __init__(self):
          super(LinearModel, self).__init__()
          self.linear = torch.nn.Linear(2, 1)

      def forward(self, x, y):
          x = torch.cat([x, y], dim=1)
          return self.linear(x)

  model = LinearModel()
  optimizer = torch.optim.Adam(model.parameters())
  loss_fn = torch.nn.MSELoss()

  def lr_scheduler_creator(optimizer, config):
      return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250, 350], gamma=0.1)

  # You can use the RayDP Estimator API or libraries like Ray Train for distributed training.
  from raydp.torch import TorchEstimator
  estimator = TorchEstimator(
    num_workers = 2,
    model = model,
    optimizer = optimizer,
    loss = loss_fn,
    lr_scheduler_creator=lr_scheduler_creator,
    feature_columns = ["x", "y"],
    label_column = ["z"],
    batch_size = 1000,
    num_epochs = 2
  )

  estimator.fit_on_spark(train_df, test_df)

  pytorch_model = estimator.get_model()

  estimator.shutdown()
  
