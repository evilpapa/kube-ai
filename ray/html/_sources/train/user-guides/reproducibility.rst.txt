.. _train-reproducibility:

可重复性
---------------

.. tab-set::

    .. tab-item:: PyTorch

        为了限制不确定性行为的来源，请将
        :func:`ray.train.torch.enable_reproducibility` 添加到训练功能的顶部。

        .. code-block:: diff

             def train_func():
            +    train.torch.enable_reproducibility()

                 model = NeuralNetwork()
                 model = train.torch.prepare_model(model)

                 ...

        .. warning:: :func:`ray.train.torch.enable_reproducibility` 无法保证
            在执行过程中完全可重现结果。 要了解更多信息，请阅读
            `PyTorch 关于随机性的说明 <https://pytorch.org/docs/stable/notes/randomness.html>`_。

..
    import ray
    from ray import tune

    def training_func(config):
        dataloader = ray.train.get_dataset()\
            .get_shard(torch.rank())\
            .iter_torch_batches(batch_size=config["batch_size"])

        for i in config["epochs"]:
            ray.train.report(...)  # use same intermediate reporting API

    # Declare the specification for training.
    trainer = Trainer(backend="torch", num_workers=12, use_gpu=True)
    dataset = ray.dataset.window()

    # Convert this to a trainable.
    trainable = trainer.to_tune_trainable(training_func, dataset=dataset)

    tuner = tune.Tuner(trainable,
        param_space={"lr": tune.uniform(), "batch_size": tune.randint(1, 2, 3)},
        tune_config=tune.TuneConfig(num_samples=12))
    results = tuner.fit()
