.. _ray-for-ml-infra:

Ray 作为 ML 基础设施
=========================

.. tip::

    如果您正在使用 Ray 构建 ML 平台，我们很乐意听取您的意见！填些 `此简短表格 <https://forms.gle/wCCdbaQDtgErYycT6>`__ 以参与其中。

Ray 及其人工智能库为希望简化 ML 平台的团队提供了统一的计算运行时。
Ray 的库（如Ray Train、Ray Data 和 Ray Serve）可用于组成端到端 ML 工作流，为数据预处理提供功能和API，
作为训练的一部分，并从训练过渡到服务。

..
  https://docs.google.com/drawings/d/1PFA0uJTq7SDKxzd7RHzjb5Sz3o1WvP13abEJbD0HXTE/edit

.. image:: /images/ray-air.svg

为什么 Ray 可以作为 ML 基础设施？
------------------------------

Ray 的人工智能库通过为可扩展 ML 提供无缝、统一和开放的体验，简化了机器学习框架、平台和工具的生态系统：


.. image:: images/why-air-2.svg

..
  https://docs.google.com/drawings/d/1oi_JwNHXVgtR_9iTdbecquesUd4hOk0dWgHaTaFj6gk/edit

**1. 无缝开发到生产**: Ray 的人工智能库减少了从开发到生产的摩擦。使用 Ray 及其库，相同的 Python 代码可以从笔记本电脑无缝扩展到大型集群。

**2. 统一的 ML API 和运行时**: Ray 的 API 允许在 XGBoost、PyTorch 和 Hugging Face 等流行框架之间进行交换，只需最少的代码更改。从训练到服务的一切都在一个运行时（Ray+KubeRay）上运行。

**3. 开放且可扩展**: Ray 是完全开源的，可以在任何集群、云或 Kubernetes 上运行。在可扩展的开发人员 API 之上构建自定义组件和集成。

基于 Ray 的 ML 平台示例
---------------------------------

`Merlin <https://shopify.engineering/merlin-shopify-machine-learning-platform>`_ 是 Shopify 基于 Ray 的 ML 平台。它实现了 `分布式应用程序 <https://www.youtube.com/watch?v=kbvzvdKH7bc>`_ （如产品分类和推荐）的快速迭代和扩展。

.. figure:: /images/shopify-workload.png

  Shopify 基于 Ray 的 Merlin 架构。

Spotify `使用 Ray 进行高级应用 <https://www.anyscale.com/ray-summit-2022/agenda/sessions/180>`_ 其中包括对家庭播客的内容推荐进行个性化设置，以及对 Spotify Radio 曲目排序进行个性化设置。

.. figure:: /images/spotify.png

  Ray 生态系统如何为 Spotify 的 ML 科学家和工程师赋能。。


以下重点介绍了利用 Ray 的统一 API 构建更简单、更灵活的ML平台的功能公司。

- `[博客] 魔性 Merlin - Shopify 的新 ML 平台 <https://shopify.engineering/merlin-shopify-machine-learning-platform>`_
- `[演示] Large Scale Deep Learning Training and Tuning with Ray <https://drive.google.com/file/d/1BS5lfXfuG5bnI8UM6FdUrR7CiSuWqdLn/view>`_
- `[博客] Griffin: How Instacart’s ML Platform Tripled in a year <https://www.instacart.com/company/how-its-made/griffin-how-instacarts-ml-platform-tripled-ml-applications-in-a-year/>`_
- `[讨论] Predibase - 针对规模构建的低代码深度学习平台 <https://www.youtube.com/watch?v=B5v9B5VSI7Q>`_
- `[博客] 在 GKE 上用 Kubeflow 和 Ray 构建 ML 平台 <https://cloud.google.com/blog/products/ai-machine-learning/build-a-ml-platform-with-kubeflow-and-ray-on-gke>`_
- `[讨论] Ray Summit Panel - ML Platform on Ray <https://www.youtube.com/watch?v=_L0lsShbKaY>`_


.. 基于 Ray 的开发
.. include:: /ray-air/deployment.rst
