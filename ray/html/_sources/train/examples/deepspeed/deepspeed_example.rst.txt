:orphan:

.. _deepspeed_example:

使用 DeepSpeed ZeRO-3 和 Ray Train 进行训练
=========================================

这是一个中间示例，展示了如何使用 DeepSpeed ZeRO-3 和 Ray Train 进行分布式训练。
它演示了如何将 :ref:`Ray Data <data>` 与 DeepSpeed ZeRO-3 和 Ray Train 结合使用。
如果您只是想快速将现有的 TorchTrainer 脚本转换为 Ray Train，可以参考 :ref:`使用 DeepSpeed 进行训练 <train-deepspeed>`。


代码示例
------------

.. literalinclude:: /../../python/ray/train/examples/deepspeed/deepspeed_torch_trainer.py


同样看看
--------

* 更多 :ref:`Ray Train 示例 <train-examples>`

* :ref:`开始使用 DeepSpeed <train-deepspeed>` 的教程。
