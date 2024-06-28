# Seldon Core Helm Charts

Helm charts 发布到我们的官方仓库。

## 核心 Charts

安装 Seldon Core 的核心 charts 如下

.. toctree::
   :maxdepth: 1

   seldon-core-operator <../charts/seldon-core-operator>

有关更多详细信息，请参见 [此处](../workflow/install.md).

## 推理图模板

提供了一组使用 Seldon Core 创建特定推理图的示例模板

.. toctree::
   :maxdepth: 1

   seldon-single-model <../charts/seldon-single-model>
   seldon-abtest <../charts/seldon-abtest>
   seldon-mab <../charts/seldon-mab>
   seldon-od-model <../charts/seldon-od-model>
   seldon-od-transformer <../charts/seldon-od-transformer>

提供了一个[其中包含使用上述图表的示例笔记](https://docs.seldon.io/projects/seldon-core/en/latest/examples/helm_examples.html)。

## 杂项

.. toctree::
   :maxdepth: 1

   seldon-core-loadtesting <../charts/seldon-core-loadtesting>
   seldon-core-analytics <../charts/seldon-core-analytics>
