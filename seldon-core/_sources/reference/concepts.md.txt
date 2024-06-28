# Seldon Core 核心概念

此页面是一个正在进行的工作，提供 Seldon Core 的核心理念。
这项工作正在进行中，我们欢迎反馈。

## 机器学习发布 / 推理图

### 机器学习部署/推理图的概念概述

机器学习部署（或推理图）是指 Seldon 系统下 Seldon（Seldon Deployments）相关的一组组件。它表示工作流程，将机器学习系统的组件分组到逻辑管道中。ML 部署包含组件的配置、系统输入和输出的定义以及每个组件的定义。

## 组件/推理服务器

### 组件/推理服务器的概念概述

组件（或推理服务器）是：
-   模型
    
-   路由
    
-   合并器
    
-   转换器
    
-   输出转换器

一个程序提供工作流中的一个步骤。

## 模型

### 模型的概念概述

机器学习部署中的组件，从训练数据中获取的训练模型。

## 语言封装

### 语言封装的概念概述

语言封装一种针对特定编程语言
A language wrapper is a model which enables cross language and/or runtime interoperability with a particular programming language.

## 预封装推理服务器

### 预封装推理服务器的概念概述

预封装推理服务器包含如下：

-   SKLearn Server
    
-   XGBoost Server
    
-   Tensorflow Serving
    
-   MLflow Server
    
可以发布训练模型的服务。

Seldon Core 中内置了预封装的推理服务器，允许用户轻松地从 artifact (即序列化模型) 到 ML 部署而无需考虑工具包。请参阅以下文档页面，供用户查找有关如何创建自己的「预封装」推理服务器的说明：
- [https://docs.seldon.io/projects/seldon-core/en/latest/servers/overview.html](https://docs.seldon.io/projects/seldon-core/en/latest/servers/overview.html)
- [https://docs.seldon.io/projects/seldon-core/en/latest/servers/custom.html](https://docs.seldon.io/projects/seldon-core/en/latest/servers/custom.html)

## 图

### 图的概念简述

图将机器学习组件表示为节点，其表示从一个组件到下一个组件的传递输入和输出的操作。

## 请求

### 请求的概念概述

请求代表对模型的单个调用以进行预测。该请求附加一个有效负载，其中包含通过特定协议传递的预测数据（通常以数组的形式）。它需遵循特定格式。

## Request 日志

### 请求日志的概念描述

请求日志记录是 Seldon 的一项功能，能够跟踪已提交给模型的请求。在默认设置中，请求被记录并存储在 Elasticsearch 中。

## 有用的链接

### Kubeflow pipelines 概念

[https://www.kubeflow.org/docs/pipelines/overview/concepts/](https://www.kubeflow.org/docs/pipelines/overview/concepts/)

### Google machine learning 词汇表

[https://developers.google.com/machine-learning/glossary](https://developers.google.com/machine-learning/glossary)

### Kubernetes standardized 词汇表

[https://kubernetes.io/docs/reference/glossary/?fundamental=true](https://kubernetes.io/docs/reference/glossary/?fundamental=true)

### Helm 词汇表

[https://helm.sh/docs/glossary/](https://helm.sh/docs/glossary/)

### Jaeger 术语

[https://www.jaegertracing.io/docs/1.18/architecture/#terminology](https://www.jaegertracing.io/docs/1.18/architecture/#terminology)

### Elastic search 术语

[https://www.elastic.co/guide/en/elastic-stack-glossary/current/terms.html](https://www.elastic.co/guide/en/elastic-stack-glossary/current/terms.html)
