.. _ref-use-cases:

Ray 用例
=============

.. raw:: html

    <link rel="stylesheet" type="text/css" href="../_static/css/use_cases.css">

此页索引了用于 ML 扩缩 的常见 Ray 用例。
它包含对博客、示例和教程的突出显示参考，也位于
Ray 文档中的其他位置。

.. _ref-use-cases-llm:

LLMs 以及生成式 AI
---------------

大型语言模型（LLM）和生成人工智能正在迅速改变行业，需求计算速度惊人。Ray 提供了一个用于扩展这些模型的分布式计算框架，使开发人员能够更快、更高效地训练和部署模型。凭借用于数据流、训练、微调、超参数调整和服务的专用库，Ray 简化了开发和部署大规模人工智能模型的过程。

.. figure:: /images/llm-stack.png

.. query-param-ref:: ray-overview/examples
    :parameters: ?tags=llm
    :ref-type: doc
    :classes: example-gallery-link

    探索 LLMs 和生成式 AI 示例

.. _ref-use-cases-batch-infer:

批量预估
---------------

批量推理是对大量输入数据生成模型预测的过程。
Ray 针对于批量预估可工作于任何云厂商以及 ML 框架，
并且它对于现代深度学习应用程序来说既快速又便宜。
它从单机扩展到大型集群，只需最少的代码更改。
作为 Python 优先的框架，您可以在 Ray 中轻松地表达和交互式地开发推理工作负载。
了解有关使用 Ray 运行批处理推理的更多信息，参考 :ref:`批量预估指导<batch_inference_home>`.

.. figure:: ../data/images/batch_inference.png

.. query-param-ref:: ray-overview/examples
    :parameters: ?tags=inference
    :ref-type: doc
    :classes: example-gallery-link

    探索批量预估示例

.. _ref-use-cases-mmt:

多模型训练
-------------------

多模型训练在 ML 用例中很常见，如时间序列预测，它需要在与地点、产品等对应的多个数据批次上拟合模型。
重点是在数据集的子集上训练许多模型。这与在整个数据集上训练单个模型形成对比。

当要训练的任何给定模型都可以放在一个 GPU 上时，Ray 可以将每个训练运行分配给一个单独的 Ray 任务。通过这种方式，所有可用的 worker 都被用来运行独立的远程训练，而不是一个工人按顺序运行作业。

.. figure:: /images/training_small_models.png

  用于大型数据集上分布式训练的数据并行模式。

我如何在 Ray 上进行多模型训练？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要训练多个独立的模型，使用 Ray Tune (:ref:`Tutorial <mmt-tune>`) 类库。这是大多数情况下推荐的库。

如果数据源适合单个机器（节点）的内存，则可以将 Tune 与当前的数据预处理管道一起使用。
如果您需要扩展数据，或者希望规划未来的扩展，使用 :ref:`Ray Data <data>` 类库。
要使用 Ray Data，你的数据必须是 :ref:`支持的格式 <input-output>`。

对于不太常见的情况，存在替代解决方案：

#. 如果数据不是支持的格式，使用 Ray Core (:ref:`Tutorial <mmt-core>`) 来自定义应用。这是一种高级模式，并需要你理解 :ref:`设计模式和反模式 <core-patterns>`。
#. 如果您有一个大型预处理管道，你可以使用 Ray Data 类库来训练多个模型 (:ref:`Tutorial <mmt-datasets>`)。

.. query-param-ref:: ray-overview/examples
    :parameters: ?tags=training
    :ref-type: doc
    :classes: example-gallery-link

    探索模型训练示例

模型服务
-------------

:ref:`Ray Serve <rayserve>` 非常适合模型组合，使您能够构建由多个 ML 模型和业务逻辑组成的复杂推理服务，所有这些都使用 Python 代码。

它支持复杂的 `模型部署模式 <https://www.youtube.com/watch?v=mM4hJLelzSw>`_ 需要多个 Ray Actor 的编排，其中不同的 actor 为不同的模型提供推理。Serve 同时处理批处理和在线推理，并且可以扩展到生产中的数千个模型。

.. figure:: /images/multi_model_serve.png

  Ray Serve 部署模式。(点击图片放大)

使用以下资源了解有关模型服务的更多信息。

- `[讨论] 利用 Ray Serve 大规模生产 ML <https://www.youtube.com/watch?v=UtH-CMpmxvI>`_
- `[博客] 使用 Ray & Ray Serve 简化 MLOps <https://www.anyscale.com/blog/simplify-your-mlops-with-ray-and-ray-serve>`_
- :doc:`[指引] Ray Serve 入门 </serve/getting_started>`
- :doc:`[指引] Serve 中的模型组合 </serve/model_composition>`
- :doc:`[库] 服务示例库 </serve/tutorials/index>`
- `[库] 博客上的更多服务用例 <https://www.anyscale.com/blog?tag=ray_serve>`_

超参调优
---------------------

:ref:`Ray Tune <tune-main>` 类库使任何并行 Ray 工作负载能够在超参数调整算法下运行。

运行多个超参数调整实验是一种适用于分布式计算的模式，因为每个实验彼此独立。Ray Tune 处理了分布式超参数优化的难点，并提供了可用的关键功能，如最佳结果的检查点、优化调度和指定搜索模式。

.. figure:: /images/tuning_use_case.png

   分布式调优和每次试验的分布式训练。

通过以下讲座和用户指南了解有关 Tune 库的更多信息。

- :doc:`[指引] Ray Tune 入门 </tune/getting-started>`
- `[博客] 如何使用 Ray Tune 进行超参数调整 <https://www.anyscale.com/blog/how-to-distribute-hyperparameter-tuning-using-ray-tune>`_
- `[讨论] 简单的分布式超参数优化 <https://www.youtube.com/watch?v=KgYZtlbFYXE>`_
- `[博客] Hyperparameter Search with 🤗 Transformers <https://www.anyscale.com/blog/hyperparameter-search-hugging-face-transformers-ray-tune>`_
- :doc:`[库] Ray Tune 示例库 </tune/examples/index>`
- `博客上的更多 Tune 用例 <https://www.anyscale.com/blog?tag=ray-tune>`_

分布式训练
--------------------

:ref:`Ray Train <train-docs>` 该库在一个简单的 Trainer API 下集成了许多分布式培训框架，
提供了开箱即用的分布式编排和管理功能。

与训练许多模型不同，模型并行性将一个大模型划分为多台机器进行训练。Ray Train 内置了用于分发模型碎片和并行运行训练的抽象。

.. figure:: /images/model_parallelism.png

  用于分布式大模型训练的模型并行模式。

通过以下讲座和用户指南了解更多关于 Train 库的信息。

- `[讨论] Ray Train, PyTorch, TorchX 以及分布式机器学习 <https://www.youtube.com/watch?v=e-A93QftCfc>`_
- `[博客] XGBoost 在 Ray 上的弹性分布式训练 <https://www.uber.com/blog/elastic-xgboost-ray/>`_
- :doc:`[指引] Ray Train 入门 </train/train>`
- :doc:`[示例] Fine-tune a 🤗 Transformers model </train/examples/transformers/huggingface_text_classification>`
- :doc:`[库] Ray Train 示例库 </train/examples>`
- `[库] 博客上的更多 Train 用例 <https://www.anyscale.com/blog?tag=ray_train>`_

强化学习
----------------------

RLlib 是一个强化学习 (RL) 开源类库，为生产级、高度分布式的 RL 工作负载提供支持，同时为各种行业应用程序维护统一而简单的 API。 RLlib 被许多不同垂直领域的行业领导者使用，如气候控制、工业控制、制造和物流、金融、游戏、汽车、机器人、船舶设计等。

.. figure:: /images/rllib_use_case.png

   分布式近端优化（DD-PPO）架构。

使用以下资源了解有关强化学习的更多信息。

- `[课程] 使用 RLlib 应用于强化学习 <https://applied-rl-course.netlify.app/>`_
- `[博客] RLlib 介绍：环境示例 <https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70>`_
- :doc:`[指引] RLlib 入门 </rllib/rllib-training>`
- `[讨论] Riot Games 深度强化学习 <https://www.anyscale.com/events/2022/03/29/deep-reinforcement-learning-at-riot-games>`_
- :doc:`[库] RLlib 示例库 </rllib/rllib-examples>`
- `[库] 博客上的更多 RL 用例 <https://www.anyscale.com/blog?tag=rllib>`_

ML平台
-----------

Ray 及其人工智能库为希望简化ML平台的团队提供了统一的计算运行时。
Ray 及其库（如 Ray Train、Ray Data 和 Ray Serve）可用于组成端到端ML工作流，为数据预处理提供功能和 API，
作为训练的一部分，并从训练过渡到服务。

在 :ref:`本章节 <ray-for-ml-infra>` 中阅读有关使用Ray构建ML平台的更多信息。

..
  https://docs.google.com/drawings/d/1PFA0uJTq7SDKxzd7RHzjb5Sz3o1WvP13abEJbD0HXTE/edit

.. image:: /images/ray-air.svg

端到端 ML 工作流
-----------------------

以下重点介绍了利用 Ray AI 库实现端到端 ML 工作流的示例。

- :doc:`[示例] 基于 Ray 的文本分类 </train/examples/transformers/huggingface_text_classification>`
- :doc:`[示例] 基于 Ray 的对象检测 </train/examples/pytorch/torch_detection>`
- :doc:`[示例] 基于表格数据的机器学习 </train/examples/xgboost/xgboost_example>`
- :doc:`[示例] 基于 Ray 的时序 AutoML </ray-core/examples/automl_for_time_series>`

大规模工作负载编排
----------------------------------

以下重点介绍了利用 Ray Core 的分布式 API 简化大规模工作负载编排的功能项目。

- `[博客] 蚂蚁集团基于 Ray 的高可用性和可扩展的在线应用程序 <https://www.anyscale.com/blog/building-highly-available-and-scalable-online-applications-on-ray-at-ant>`_
- `[博客] Ray Forward 2022 Conference: Hyper-scale Ray Application Use Cases <https://www.anyscale.com/blog/ray-forward-2022>`_
- `[博客] 使用 Ray 在 CloudSort 基准测试上创下新的世界纪录 <https://www.anyscale.com/blog/ray-breaks-the-usd1-tb-barrier-as-the-worlds-most-cost-efficient-sorting>`_
- :doc:`[示例] 通过与 Ray 并行化来加快网络爬虫的速度 </ray-core/examples/web-crawler>`
