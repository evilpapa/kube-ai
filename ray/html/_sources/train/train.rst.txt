.. _train-docs:

Ray Train：可扩展模型训练
==================================

|

.. figure:: images/logo.png
   :align: center
   :width: 50%

|

Ray Train 是一个可扩展的机器学习库，用于分布式训练和微调。

Ray Train 可让您将模型训练代码从单台机器扩展到云端的机器集群，并消除分布式计算的复杂性。
无论您拥有大型模型还是大型数据集，Ray Train 都是最简单的分布式训练解决方案。

Ray Train 为许多框架提供支持：

.. list-table::
   :widths: 1 1
   :header-rows: 1

   * - PyTorch 生态
     - 更多框架
   * - PyTorch
     - TensorFlow
   * - PyTorch Lightning
     - Keras
   * - Hugging Face Transformers
     - Horovod
   * - Hugging Face Accelerate
     - XGBoost
   * - DeepSpeed
     - LightGBM

安装 Ray Train
-----------------

要安装 Ray Train，请运行：

.. code-block:: console

    $ pip install -U "ray[train]"

学习更多关于安装 Ray 和其库的信息，请参见 
:ref:`安装 Ray <installation>`。

开始
-----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **概述**
        ^^^

        了解使用 Ray Train 进行分布式训练的关键概念。

        +++
        .. button-ref:: train-overview
            :color: primary
            :outline:
            :expand:

            了解基础知识

    .. grid-item-card::

        **PyTorch**
        ^^^

        使用 Ray Train 和 PyTorch 开始进行分布式模型训练。

        +++
        .. button-ref:: train-pytorch
            :color: primary
            :outline:
            :expand:

            通过 PyTorch 尝试 Ray Train

    .. grid-item-card::

        **PyTorch Lightning**
        ^^^

        使用 Ray Train 和 Lightning 开始进行分布式模型训练。

        +++
        .. button-ref:: train-pytorch-lightning
            :color: primary
            :outline:
            :expand:

            通过 Lightning 尝试 Ray Train

    .. grid-item-card::

        **Hugging Face Transformers**
        ^^^

        使用 Ray Train 和 Transformers 开始进行分布式模型训练。

        +++
        .. button-ref:: train-pytorch-transformers
            :color: primary
            :outline:
            :expand:

            通过 Transformers 尝试 Ray Train

了解更多
----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **更多框架**
        ^^^

        没有找到你的框架？请参阅这些指南。

        +++
        .. button-ref:: train-more-frameworks
            :color: primary
            :outline:
            :expand:

            尝试将 Ray Train 与其他框架一起使用

    .. grid-item-card::

        **用户指南**
        ^^^

        获取使用 Ray Train 进行常见训练任务的操作说明。

        +++
        .. button-ref:: train-user-guides
            :color: primary
            :outline:
            :expand:

            阅读操作指南

    .. grid-item-card::

        **示例**
        ^^^

        浏览不同用例的端到端代码示例。

        +++
        .. button-ref:: train-examples
            :color: primary
            :outline:
            :expand:

            通过例子学习

    .. grid-item-card::

        **API**
        ^^^

        请参阅 API 参考，了解 Ray Train API 的完整描述。

        +++
        .. button-ref:: air-trainer-ref
            :color: primary
            :outline:
            :expand:

            阅读 API 参考
