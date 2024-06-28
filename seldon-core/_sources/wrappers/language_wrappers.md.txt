# Seldon Core 语言封装

预封装推理服务器无法覆盖自定义的用例时，可以利用我们的语言封装来将机器学习模型和逻辑容器化。

Seldon 提供的所有预封装的模型服务器都是使用语言封装构建的，这意味着如果需要，您也可以构建自己的可重复使用的推论服务器。

本页提供了有关使用预封装模型服务器时的概念和最佳实践概述。

## 可用封装语言

支持的封装语言，包括它们当前的稳定性，如下所述

### 已毕业封装语言

以下是现在稳定版的封装语言。

#### Python [已毕业]

对于任何基于 Python 的机器学习模型，都有可能使用我们的 Python 语言封装将它们装箱，并通过简单的 Python 类暴露任何逻辑。

这是目前最流行的封装（其次是 Java 封装），目前用于大量的使用案例，可用于使用基于 [Keras](https://keras.io/)，[PyTorch](http://pytorch.org/)，[StatsModels](http://www.statsmodels.org/stable/index.html)，[XGBoost](https://github.com/dmlc/xgboost)，[scikit-learn](http://scikit-learn.org/stable/) 的训练模型，甚至是基于自定义操作系统的专用引擎。

请参考 [Python 模型封装](../python/index.xhtml) 章节查看更多使用信息。

### 孵化中的封装语言

以下为不稳定或者未毕业的语言，但是已经有了明确的路线图和毕业战略。

#### Java [孵化中]

目前，Java 封装正在大量关键环境中使用，但我们要求 Java 封装具有与 Python 封装相同的功能级别才能毕业。你可以通过 [GitHub 讨论 #1344](https://github.com/SeldonIO/seldon-core/issues/1344) 追踪进展。

基于Java的模型包括 [H2O](https://www.h2o.ai/)，[Deep Learning 4J](https://deeplearning4j.org/)，[Spark](https://spark.apache.org/)（导出的独立模型）。

请参考 [使用 s2i 封装 JAVA 模型](../java/README.xhtml) 获取有关如何进一步使用它的信息。

### Alpha封装语言

#### R [Alpha]

- [使用 s2i 封装 R 模型](../R/README.xhtml)

#### NodeJS [Alpha]

- [使用 s2i 封装 Javascript 模型](../nodejs/README.xhtml)


#### Go [Alpha]

- [Go 实现示例](https://github.com/SeldonIO/seldon-core/blob/master/incubating/wrappers/s2i/go/SeldonGoModel.ipynb)