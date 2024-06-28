
=================================
高级Helm Chart 配置
=================================

Seldon Core Operator Chart 配置
----------------------------------------

本页提供了使用 Helm 3.x 安装Seldon Core安装参数的详细概述。可以在高级别 `安装工作流 <../workflow/install.md>`_ 查看更多信息。

下面参数配置可在 `seldon-core-operator Helm chart <https://github.com/SeldonIO/seldon-core/tree/master/helm-charts/seldon-core-operator>`_ 的 ``values.yaml`` 文件找到，它包含了所有基础的配置和值，使用 ``--set value.path=YOUR_VALUE`` 在你的安装中进行配置的自定义设置。

该文件已被设置为自动记录所有核心参数及说明，文件中进一步的信息需要引用特定文档。(翻译版本更新不同步）

.. literalinclude:: ../../../helm-charts/seldon-core-operator/values.yaml
   :language: yaml
   :linenos:

