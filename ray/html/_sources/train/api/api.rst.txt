.. _train-api:
.. _air-trainer-ref:

Ray 训练 API
=============

.. _train-integration-api:
.. _train-framework-specific-ckpts:

.. currentmodule:: ray

PyTorch 生态
-----------------

.. autosummary::
    :toctree: doc/

    ~train.torch.TorchTrainer
    ~train.torch.TorchConfig

.. _train-pytorch-integration:

PyTorch
~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.torch.get_device
    ~train.torch.prepare_model
    ~train.torch.prepare_data_loader
    ~train.torch.enable_reproducibility

.. _train-lightning-integration:

PyTorch Lightning
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.lightning.prepare_trainer
    ~train.lightning.RayLightningEnvironment
    ~train.lightning.RayDDPStrategy
    ~train.lightning.RayFSDPStrategy
    ~train.lightning.RayDeepSpeedStrategy
    ~train.lightning.RayTrainReportCallback

.. _train-transformers-integration:

Hugging Face Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.huggingface.transformers.prepare_trainer
    ~train.huggingface.transformers.RayTrainReportCallback


更多框架
---------------

Tensorflow/Keras
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.tensorflow.TensorflowTrainer
    ~train.tensorflow.TensorflowConfig
    ~train.tensorflow.prepare_dataset_shard
    ~train.tensorflow.keras.ReportCheckpointCallback

Horovod
~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.horovod.HorovodTrainer
    ~train.horovod.HorovodConfig


XGBoost
~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.xgboost.XGBoostTrainer


LightGBM
~~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.lightgbm.LightGBMTrainer


.. _ray-train-configs-api:

Ray Train 配置
-----------------------

.. autosummary::
    :toctree: doc/

    ~train.CheckpointConfig
    ~train.DataConfig
    ~train.FailureConfig
    ~train.RunConfig
    ~train.ScalingConfig
    ~train.SyncConfig

.. _train-loop-api:

Ray Train 实用工具
-------------------

**类**

.. autosummary::
    :toctree: doc/

    ~train.Checkpoint
    ~train.context.TrainContext

**函数**

.. autosummary::
    :toctree: doc/
    
    ~train.get_checkpoint
    ~train.get_context
    ~train.get_dataset_shard
    ~train.report


Ray Train 输出
----------------

.. autosummary::
    :template: autosummary/class_without_autosummary.rst
    :toctree: doc/

    ~train.Result


Ray Train 开发者 API
------------------------

.. _train-base-trainer:

Trainer 基础类
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: doc/

    ~train.trainer.BaseTrainer
    ~train.data_parallel_trainer.DataParallelTrainer
    ~train.gbdt_trainer.GBDTTrainer


Train 后端基础类
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _train-backend:
.. _train-backend-config:

.. autosummary::
    :toctree: doc/
    :template: autosummary/class_without_autosummary.rst

    ~train.backend.Backend
    ~train.backend.BackendConfig
