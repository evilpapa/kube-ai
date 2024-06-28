:orphan:

.. _accelerate_example:

使用 Hugging Face Accelerate 进行分布式训练
=================================================

此示例使用 Hugging Face Accelerate、Ray Train 和 Ray Data 进行分布式数据并行训练。
它微调了 BERT 模型，改编自
https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py


代码示例
------------

.. literalinclude:: /../../python/ray/train/examples/accelerate/accelerate_torch_trainer.py

同样看看
--------

* :ref:`开始使用 Hugging Face Accelerate <train-hf-accelerate>`，查看有关使用 Ray Train 和 HF Accelerate 的教程

* 更多 :ref:`Ray Train 示例 <train-examples>`
