.. _data:

==================================
Ray Data: 可扩展的机器学习数据集
==================================

Ray Data 是一个可扩展的机器学习数据处理库。它提供了灵活且高性能的 API，用于扩展 :ref:`离线批量推断 <batch_inference_overview>` 和 :ref:`用于 ML 训练的数据预处理和摄取 <ml_ingest_overview>`。Ray Data 使用 `流式执行 <https://www.anyscale.com/blog/streaming-distributed-execution-across-cpus-and-gpus>`__ 来高效处理大型数据集。

.. image:: images/dataset.svg

..
  https://docs.google.com/drawings/d/16AwJeBNR46_TsrkOmMbGaBK7u-OPsf_V8fHjU-d2PPQ/edit

安装 Ray Data
----------------

要安装 Ray Data，请运行以下命令：

.. code-block:: console

    $ pip install -U 'ray[data]'

要了解有关 Ray 机器类库的更多信息，请参阅
 :ref:`安装 Ray <installation>`。

学习更多内容
----------

.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-6

    .. grid-item-card::

        **Ray Data Overview**
        ^^^

        获取 Ray Data 的概述，支持的工作负载以及与其他替代方案的比较。

        +++
        .. button-ref:: data_overview
            :color: primary
            :outline:
            :expand:

            Ray Data Overview

    .. grid-item-card::

        **Key Concepts**
        ^^^

        
        了解 Ray Data 背后的关键概念。了解什么是 
        :ref:`Datasets <dataset_concept>` ，以及它们是如何使用的。

        +++
        .. button-ref:: data_key_concepts
            :color: primary
            :outline:
            :expand:

            学习关键概念

    .. grid-item-card::

        **User Guides**
        ^^^

        学习如何使用 Ray Data，从基本用法到端到端指南。

        +++
        .. button-ref:: data_user_guide
            :color: primary
            :outline:
            :expand:

            学习如何使用 Ray Data

    .. grid-item-card::

        **Examples**
        ^^^

        查找使用 Ray Data 的简单示例和扩展示例。

        +++
        .. button-ref:: data-recipes
            :color: primary
            :outline:
            :expand:

            Ray Data Examples

    .. grid-item-card::

        **API**
        ^^^

        获取有关 Ray Data API 的更多深入信息。

        +++
        .. button-ref:: data-api
            :color: primary
            :outline:
            :expand:

            阅读 API 参考

    .. grid-item-card::

        **Ray blogs**
        ^^^

        获取 Ray 团队的工程更新以及公司如何使用 Ray Data 的最新信息。

        +++
        .. button-link:: https://www.anyscale.com/blog?tag=ray-datasets
            :color: primary
            :outline:
            :expand:

            阅读 Ray 博客
