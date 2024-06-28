==================================
Seldon Deployment 参考类型
==================================

SeldonDeployment 被定义为 Kubernetes 中的自定义资源。

如果您想了解更多使用 SeldonDeployment 类型的实际示例，可以查看 `Worfklow 文档部分的推理图部分 <../graph/inference-grpah.html>`_。

下面是我们的 `seldondeployment_types.go <https://github.com/SeldonIO/seldon-core/blob/master/operator/apis/machinelearning.seldon.io/v1/seldondeployment_types.go>`_ 文件的第二部分，其中包含 SeldonDeployment YAML 文件中使用的类型。

.. literalinclude:: ../../../operator/apis/machinelearning.seldon.io/v1/seldondeployment_types.go
   :language: go
   :lines: 203-


