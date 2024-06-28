.. _train-experiment-tracking-native:

===================
实现追踪
===================

.. note::
    本指南适用于您定义自定义训练循环的所有训练器。
    这包括 :class:`TorchTrainer <ray.train.torch.TorchTrainer>` 和 
    :class:`TensorflowTrainer <ray.train.tensorflow.TensorflowTrainer>`。

多数实验追踪库可以直接与 Ray Train 配合使用。
本指南提供了如何设置代码的说明，以便将您喜欢的实验追踪库可以与 Ray Train 的分布式训练配合使用。
指南最后列出了常见错误，以帮助调试设置。

以下伪代码演示了如何在 Ray Train 中使用原生实验追踪库调用：

.. code-block:: python

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    def train_func(config):
        # Training code and native experiment tracking library calls go here.

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()

Ray Train 让您可以使用原生实验追踪类库在 :ref:`train_func<train-overview-training-function>` 函数中自定义追踪逻辑。
这样，你只需进行最少的更改就可以将实验追踪逻辑移植到 Ray Train 中。

入门
===============

让我们先看一些代码片段。

以下示例使用权重和偏差（W＆B）和 MLflow，但它可以适用于其他框架。

.. tabs::

    .. tab:: W&B

        .. code-block:: python
            
            import ray
            from ray import train
            import wandb

            # Step 1
            # This ensures that all ray worker processes have `WANDB_API_KEY` set.
            ray.init(runtime_env={"env_vars": {"WANDB_API_KEY": "your_api_key"}})

            def train_func(config):
                # Step 1 and 2
                if train.get_context().get_world_rank() == 0:
                    wandb.init(
                        name=...,
                        project=...,
                        # ...
                    )

                # ...
                loss = optimize()
                metrics = {"loss": loss}

                # Step 3
                if train.get_context().get_world_rank() == 0:
                    wandb.log(metrics)

                # ...

                # Step 4
                # Make sure that all loggings are uploaded to the W&B backend.
                if train.get_context().get_world_rank() == 0:
                    wandb.finish()

    .. tab:: MLflow

        .. code-block:: python
            
            from ray import train
            import mlflow

            # Run the following on the head node:
            # $ databricks configure --token
            # mv ~/.databrickscfg YOUR_SHARED_STORAGE_PATH
            # This function assumes `databricks_config_file` in config
            def train_func(config):
                # Step 1 and 2
                os.environ["DATABRICKS_CONFIG_FILE"] = config["databricks_config_file"]
                mlflow.set_tracking_uri("databricks")
                mlflow.set_experiment_id(...)
                mlflow.start_run()

                # ...

                loss = optimize()

                metrics = {"loss": loss}
                # Only report the results from the first worker to MLflow 
                to avoid duplication

                # Step 3
                if train.get_context().get_world_rank() == 0:
                    mlflow.log_metrics(metrics)

.. tip::

    分布式和非分布式训练之间的主要区别在于，在分布式训练中，多个进程并行运行，并且在某些设置下它们具有相同的结果。
    如果所有进程都将结果报告给跟踪后端，您可能会得到重复的结果。为了解决这个问题，
    Ray Train 允许您使用以下方法将日志记录逻辑仅应用于等级 0 的 worker ：
    :meth:`ray.train.get_context().get_world_rank() <ray.train.context.TrainContext.get_world_rank>`.

    .. code-block:: python

        from ray import train
        def train_func(config):
            ...
            if train.get_context().get_world_rank() == 0:
                # Add your logging logic only for rank0 worker.
            ...

与 :ref:`train_func<train-overview-training-function>` 中的实验跟踪后端的交互
有 4 个逻辑步骤：

#. 设置与跟踪后端的连接
#. 配置并启动运行
#. 记录指标
#. 结束运行

以下是关于每个步骤的更多详细信息。

步骤 1：连接到你的跟踪后端
----------------------------------------

首先，决定使用哪个跟踪后端：W&B、MLflow、TensorBoard、Comet 等。
如果适用，请确保在每个训练 worker 上正确设置凭据。

.. tabs::

    .. tab:: W&B
        
        W&B 提供 *在线* 和 *离线* 模式。

        **在线**

        针对 *在线* 模式，因为您要记录到 W&B 的跟踪服务，
        确保您在 :ref:`train_func<train-overview-training-function>` 中设置凭据。
        参考 :ref:`设置凭据<set-up-credentials>` 获取更多信息。

        .. code-block:: python
            
            # This is equivalent to `os.environ["WANDB_API_KEY"] = "your_api_key"`
            wandb.login(key="your_api_key")

        **离线**

        针对 *离线* 模式，因为您要记录到本地文件系统，
        指定一个所有节点都可以写入的共享存储路径。
        参考 :ref:`设置共享文件存储<set-up-credentials>` 获取更多信息。
        
        .. code-block:: python

            os.environ["WANDB_MODE"] = "offline"
            wandb.init(dir="some_shared_storage_path/wandb") 

    .. tab:: MLflow
        
        MLflow 提供了 *本地* 和 *远程*（例如，到 Databrick 的 MLflow 服务）模式。

        **本地**

        针对 *本地* 模式，因为您要记录到本地文件系统，将离线目录指向共享存储路径。
        这样所有节点都可以写入。
        参考 :ref:`设置共享文件系统<set-up-shared-file-system>` 获取更多信息。
        
        .. code-block:: python

            mlflow.start_run(tracking_uri="file:some_shared_storage_path/mlruns")

        **远程，由 Databricks 托管**
            
        确保所有节点都可以访问 Databricks 配置文件。
        参考 :ref:`设置凭据<set-up-credentials>` 获取更多信息。
        
        .. code-block:: python

            # The MLflow client looks for a Databricks config file 
            # at the location specified by `os.environ["DATABRICKS_CONFIG_FILE"]`.
            os.environ["DATABRICKS_CONFIG_FILE"] = config["databricks_config_file"]
            mlflow.set_tracking_uri("databricks")
            mlflow.start_run()

.. _set-up-credentials:

设置凭据
~~~~~~~~~~~~~~~~~~

请参阅每个跟踪库的 API 文档以了解如何设置凭据。
此步骤通常涉及设置环境变量或访问配置文件。

最简单的方式是将环境变量凭据通过 :ref:`runtime environments <runtime-environments>` 传递给训练 workers，
您可以使用以下代码初始化：

.. code-block:: python

    import ray
    # This makes sure that training workers have the same env var set
    ray.init(runtime_env={"env_vars": {"SOME_API_KEY": "your_api_key"}})

要访问配置文件，请确保配置文件对所有节点都是可访问的。
一种方法是设置共享存储。另一种方法是在每个节点中保存一份副本。

.. _set-up-shared-file-system:

设置共享文件系统
~~~~~~~~~~~~~~~~~~~~~~~~~~~

设置集群中所有节点均可访问的网络文件系统。
例如 AWS EFS 或 Google Cloud Filestore。

步骤 2: 配置并开始运行
-----------------------------------

此步骤通常涉及为运行选择一个标识符并将其与项目关联。
请参阅跟踪库的文档以了解语义。

.. To conveniently link back to Ray Train run, you may want to log the persistent storage path 
.. of the run as a config.

.. .. code-block:: python

..     def train_func(config):
..       if ray.train.get_context().get_world_rank() == 0:
..                 wandb.init(..., config={"ray_train_persistent_storage_path": "TODO: fill in when API stablizes"})

.. tip::
    
    执行具有自动恢复功能的 **容错训练** 时，
    请使用一致的 ID 来配置逻辑上属于同一训练运行的所有跟踪运行。
    获取唯一 ID 的一种方法是使用以下方法：
    :meth:`ray.train.get_context().get_trial_id() <ray.train.context.TrainContext.get_trial_id>`.

    .. code-block:: python

        import ray
        from ray.train import ScalingConfig, RunConfig, FailureConfig
        from ray.train.torch import TorchTrainer

        def train_func(config):
            if ray.train.get_context().get_world_rank() == 0:
                wandb.init(id=ray.train.get_context().get_trial_id())
            ...

        trainer = TorchTrainer(
            train_func, 
            run_config=RunConfig(failure_config=FailureConfig(max_failures=3))
        )

        trainer.fit()
            

步骤 3: 记录指标
-------------------

你可以在 :ref:`train_func<train-overview-training-function>` 中自定义如何记录参数、指标、模型或媒体内容，就像在非分布式训练脚本中一样。
使用特定跟踪框架与特定训练框架的本机集成。比如，``mlflow.pytorch.autolog()``, 
``lightning.pytorch.loggers.MLFlowLogger`` 等。

步骤 4: 结束运行
----------------------

此步骤可确保所有日志都同步到跟踪服务。根据各种跟踪库的实现，有时日志会先在本地缓存，然后以异步方式同步到跟踪服务。
完成运行可确保在训练 worker 退出时所有日志都已同步。

.. tabs::

    .. tab:: W&B
        
        .. code-block:: python

            # https://docs.wandb.ai/ref/python/finish
            wandb.finish()

    .. tab:: MLflow

        .. code-block:: python

            # https://mlflow.org/docs/1.2.0/python_api/mlflow.html
            mlflow.end_run()

    .. tab:: Comet

        .. code-block:: python

            # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#experimentend
            Experiment.end()    

例子
========

以下是 PyTorch 和 PyTorch Lightning 的可运行示例。

PyTorch
-------

.. dropdown:: Log to W&B

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking//torch_exp_tracking_wandb.py
            :emphasize-lines: 15, 16, 17, 21, 22, 51, 52, 54, 55
            :language: python
            :start-after: __start__

.. dropdown:: Log to file-based MLflow

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/torch_exp_tracking_mlflow.py
        :emphasize-lines: 22, 23, 24, 25, 54, 55, 57, 58, 64
        :language: python
        :start-after: __start__
        :end-before: __end__

PyTorch Lightning
-----------------

您可以将 PyTorch Lightning 中的本机 Logger 集成与 W&B、CometML、MLFlow 和 Tensorboard 一起使用，
同时使用 Ray Train 的 TorchTrainer。

以下示例将引导您完成该过程。此处的代码可运行。

.. dropdown:: W&B

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_wandb.py
        :language: python
        :start-after: __lightning_experiment_tracking_wandb_start__

.. dropdown:: MLflow

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_mlflow.py
        :language: python
        :start-after: __lightning_experiment_tracking_mlflow_start__
        :end-before: __lightning_experiment_tracking_mlflow_end__

.. dropdown:: Comet

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_comet.py
        :language: python
        :start-after: __lightning_experiment_tracking_comet_start__

.. dropdown:: TensorBoard
  
    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_model_dl.py
        :language: python
        :start-after: __model_dl_start__

    .. literalinclude:: ../../../../python/ray/train/examples/experiment_tracking/lightning_exp_tracking_tensorboard.py
        :language: python
        :start-after: __lightning_experiment_tracking_tensorboard_start__
        :end-before: __lightning_experiment_tracking_tensorboard_end__

常见错误
=============

缺少凭证
-------------------

**我已经调用了 `wandb login` cli，但仍然得到** 

.. code-block:: none

    wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key]).

这可能是由于未在工作节点上正确设置 wandb 凭据引起的。
确保您运行 ``wandb.login`` 或将 ``WANDB_API_KEY`` 传递给每个训练函数。
参考 :ref:`设置凭据<set-up-credentials>` 获取更多信息。

缺少配置
----------------------

**我已经运行了 `databricks configure` ，但仍然**

.. code-block:: none

    databricks_cli.utils.InvalidConfigurationError: You haven't configured the CLI yet!

这通常由于运行 ``databricks configure`` 而生成的 ``~/.databrickscfg`` 仅在 head 节点上。 
移动此文件到共享位置或将其复制到每个节点。
参考 :ref:`设置凭据<set-up-credentials>` 获取更多信息。
