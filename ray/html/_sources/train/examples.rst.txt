.. _train-examples:

Ray 训练示例
==================

.. Organize example .rst files in the same manner as the
   .py files in ray/python/ray/train/examples.

以下是使用 Ray Train 与各种框架和用例的示例。

初学者
--------

.. list-table::
  :widths: 1 5
  :header-rows: 1

  * - 框架
    - 示例
  * - PyTorch
    - :ref:`使用 PyTorch 训练时尚 MNIST 图像分类器 <torch_fashion_mnist_ex>`
  * - Lightning
    - :ref:`使用 Lightning 训练 MNIST 图像分类器 <lightning_mnist_example>`
  * - Transformers
    - :ref:`使用 Hugging Face Transformers 在 Yelp 评论数据集上微调文本分类器 <transformers_torch_trainer_basic_example>`
  * - Accelerate
    - :ref:`利用 Hugging Face Accelerate 进行分布式数据并行训练 <accelerate_example>`
  * - DeepSpeed
    - :ref:`使用 DeepSpeed ZeRO-3 进行训练 <deepspeed_example>`
  * - TensorFlow
    - :ref:`使用 TensorFlow 训练 MNIST 图像分类器 <tensorflow_mnist_example>`
  * - Horovod
    - :ref:`使用 Horovod 和 PyTorch 进行训练 <horovod_example>`

熟练者
------------

.. list-table::
  :widths: 1 5
  :header-rows: 1

  * - 框架
    - 示例
  * - PyTorch
    - :ref:`使用 DreamBooth 和 Ray Train 对稳定扩散进行微调 <torch_finetune_dreambooth_ex>`
  * - Lightning
    - :ref:`使用 PyTorch Lightning 和 Ray Data 进行训练 <lightning_advanced_example>`
  * - Transformers
    - :ref:`使用 Hugging Face Accelerate 在 GLUE 基准上微调文本分类器 <train_transformers_glue_example>`


高阶者
--------

.. list-table::
  :widths: 1 5
  :header-rows: 1

  * - 框架
    - 示例
  * - Accelerate, DeepSpeed
    - `使用 Deepspeed、Accelerate 和 Ray Train TorchTrainer 微调 Llama-2 系列模型 <https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed>`_
  * - Transformers, DeepSpeed
    - :ref:`使用 Ray Train 和 DeepSpeed 对 GPT-J-6B 进行微调 <gptj_deepspeed_finetune>`
  * - Lightning, DeepSpeed
    - :ref:`使用 PyTorch Lightning 和 DeepSpeed 对 vicuna-13b 进行微调 <vicuna_lightning_deepspeed_finetuning>`
  * - Lightning
    - :ref:`使用 PyTorch Lightning 和 FSDP 对 dolly-v2-7b 进行微调 <dolly_lightning_fsdp_finetuning>`
