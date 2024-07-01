模式：使用 supervisor actor 来管理 actor 树
============================================================

Actor 监督是一种模式，其中监督 Actor 管理一组工作 Actor。
监督者将任务委托给下属并处理他们的失败。
这种模式简化了驱动程序，因为它只管理几个监督者，而不直接处理工作 Actor 的失败。
此外，多个监督者可以并行执行以并行处理更多工作。

.. figure:: ../images/tree-of-actors.svg

    actor 树

.. note::

    - 如果 supervisor 死亡（或 driver 死亡），那么根据 actor 引用计数，工作者 actor 将自动终止。
    - Actor 可以嵌套多层，形成一棵树。

用例
----------------

您想要进行数据并行训练，并并行训练具有不同超参数的同一模型。
对于每个超参数，您可以启动一个 supervisor  actor 来进行编排，它将创建工作者 actor 来对每个数据分片进行实际训练。

.. note::
    对于数据并行训练和超参数调整，建议使用:ref:`Ray Train <train-key-concepts>` (:py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` and :ref:`Ray Tune's Tuner <tune-main>`)
    ，它在底层应用此模式。

代码
------------

.. literalinclude:: ../doc_code/pattern_tree_of_actors.py
    :language: python
