.. _ref-autoscaler-sdk:

程序化集群扩展
============================

.. _ref-autoscaler-sdk-request-resources:

ray.autoscaler.sdk.request_resources
------------------------------------

在 Ray 程序中，你可以使用 ``request_resources()`` 调用命令自动缩放器将集群扩展到所需大小。集群将立即尝试扩展以容纳请求的资源，从而绕过正常的升级速度限制。

.. autofunction:: ray.autoscaler.sdk.request_resources
    :noindex:
