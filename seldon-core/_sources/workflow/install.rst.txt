安装 Seldon-Core
===================

准备：
---------------

-  Kubernetes 集群版本不小于 1.12

   -  Openshift 需要 4.2 或更高版本

-  安装方法

   -  Helm 不小于 3.0 版本
   -  Kustomize 不小于 0.1.0 版本

-  Ingress

   -  Istio（可在这里找到 Istio 1.5 版本的安装示例 
      https://github.com/SeldonIO/seldon-core/tree/master/examples/auth
      ）
   -  Ambassador v1 （v2 当前不支持）

正在运行旧版的 Seldon Core ？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

请确保您已阅读 `「Seldon Core 升级指引」 <../reference/upgrading.md>`__

-  **Seldon Core 已停止为 1.0 之前版本提供支持，请确保您已升级到最新版本。**
-  如果你在运行旧版本 Seldon Core，并想对它升级，请阅读 `「Seldon Core 升级指引」 <../reference/upgrading.md>`__ 来了解升级的改动和最佳实践。
-  请查看 `从 Helm v2 迁移到 Helm v3 <https://helm.sh/docs/topics/v2_v3_migration/>`__ 如若你已经使用 Helm v2 安装运行了 Seldon Core 并想进行升级。

以 Helm 方式安装Seldon Core
-----------------------------

首先 `安装 Helm 3.x <https://docs.helm.sh/docs/intro/install/>`__。
安装完 helm 之后，你就能通过发布 seldon 控制器来管理 Seldon Deployment 图。

如果想在安装时设置高级参数，请查看完整的 `Seldon Core Helm Chart 参考 <../reference/helm.html>`__。

首选命名空间 ``seldon-system``，如下创建：

.. code:: bash

    kubectl create namespace seldon-system

现在我们可以在命名空间 ``seldon-system`` 中安装 Seldon Core。

.. tab-set::

    .. tab-item:: Istio

        .. code:: bash

            helm install seldon-core seldon-core-operator \
                --repo https://storage.googleapis.com/seldon-charts \
                --set usageMetrics.enabled=true \
                --set istio.enabled=true \
                --namespace seldon-system

    .. tab-item:: Ambassador

        .. code:: bash

            helm install seldon-core seldon-core-operator \
                --repo https://storage.googleapis.com/seldon-charts \
                --set usageMetrics.enabled=true \
                --set ambassador.enabled=true \
                --namespace seldon-system


完整的 Istio 和 Ambassador 安装说明
请查看以下页面：

* `Ingress with Istio <../ingress/istio.md>`__ 
* `Ingress with Ambassador <../ingress/ambassador.md>`__

安装特殊版本
~~~~~~~~~~~~~~~~~~~~~~~~~~

按照上面的命令使用 ``--version`` 参数来安装你想运行的特定版本。

安装一个快照版本
~~~~~~~~~~~~~~~~~~~~~~~~~~

每当一个新的 PR 合并合并到主干，我们设置 CI 来构建一个「快照」版本，这将包含该特定开发/主分支代码的 Docker 镜像。
当镜像以快照被推送，会创建一个 ``"<next-version>-SNAPSHOT_<timestamp>"`` 标签并包含 helm charts 的镜像来指定特殊版本（如 ``version.txt`` 中描述）来指定安装。

这意味着，如果您想在开发版本发布之前尝试特定功能，则可以尝试主的开发版本。

为此，您将能够克隆存储库，然后签出相关的快照分支。

完成后，您可以使用以下
命令安装：

.. code:: bash

    helm install helm-charts/seldon-core-operator seldon-core-operator

下为包含 ``helm-charts/seldon-core-operator`` charts 的版本文件夹。

通过 cert-manager 安装
~~~~~~~~~~~~~~~~~~~~~~~~~

按照 `cert manager 文
档 <https://cert-manager.io/docs/installation/kubernetes/>`__ 来进行安装。

通过如下命令安装 seldon-core ：

.. code:: bash

    helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts \
        --set usageMetrics.enabled=true \
        --namespace seldon-system \
        --set certManager.enabled=true

通过 Kustomize 安装 Seldon Core
-------------------------------

`Kustomize <https://github.com/kubernetes-sigs/kustomize>`__ 安装在仓库 ``/operator/config`` 文件夹你可将模板拷贝到自己的 kustomize 路径进行编辑。

要直接使用模板，这有一个 Makefile，它包含一组有用的命令：

对于高于 1.15 的版本群集，确保
`注释掉 patch\_object\_selector
这块 <https://github.com/SeldonIO/seldon-core/blob/master/operator/config/webhook/kustomization.yaml#L8>`__。

安装 cert-manager

.. code:: bash

    make install-cert-manager

安装 Seldon 使用 cert-manager 来提供证书。

.. code:: bash

    make deploy

通过在 ``config/cert/`` 提供的证书安装 Seldon

.. code:: bash

    make deploy-cert

其他选项
-------------

生产集成安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seldon Core 安装好，你可以进行如下设置：

集成 Kubeflow 安装 
^^^^^^^^^^^^^^^^^^^^^

-  `将 Seldon 作为 Kubeflow 的一部分 <https://www.kubeflow.org/docs/guides/components/seldon/#seldon-serving>`__ 。

GCP 应用市场
^^^^^^^^^^^^^^^

如果有 Google Cloud Platform 账户，可通过 `GCP
Marketplace <https://console.cloud.google.com/marketplace/details/seldon-portal/seldon-core>`__ 安装。

OpenShift
^^^^^^^^^

可在 OpenShift console UI 通过 OperatorHub 安装 Seldon Core。

OperatorHub
^^^^^^^^^^^

你页可以通过 `Operator Hub <https://operatorhub.io/operator/seldon-operator>`__ 安装 Seldon Core。

从上一版本进行升级
--------------------------------

查看 `升级日志 <../reference/upgrading.md>`__

高级用法
--------------

在单独的命名空间中安装 Seldon Core (版本 >=1.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**你需要一个版本 >= 1.15 的 k8s 集群**

Helm
^^^^

可将 Seldon Core Operator 安装到指定的命名空间管理相关资源。
一个安装在 ``seldon-ns1`` 命名空间的示例：

.. code:: bash

    kubectl create namespace seldon-ns1
    kubectl label namespace seldon-ns1 seldon.io/controller-id=seldon-ns1

使用 `seldon.io/controller-id=<namespace>` 贴上标签，来确保全局全局性，Seldon Core Operator 将会忽略此命名空间。

安装 Operator 到命名空间：

.. code:: bash

    helm install seldon-namespaced seldon-core-operator  --repo https://storage.googleapis.com/seldon-charts  \
        --set singleNamespace=true \
        --set image.pullPolicy=IfNotPresent \
        --set usageMetrics.enabled=false \
        --set crd.create=true \
        --namespace seldon-ns1

可设置 ``crd.create=true`` 来创建 CRD。
同意集群中如果在前一版本之后安装 Seldon Core Operator 需要设置 ``crd.create=false``。

Kustomize
^^^^^^^^^

Operator 文件加下的 Makefile 提供了一个安装示例：

.. code:: bash

    make deploy-namespaced1

查看 `多服务器示例笔记 <../examples/multiple_operators.html>`__。

指定标签的 Seldon Core Operator (version >=1.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**需要 k8s 集群 >= 1.15**

您可以安装 Seldon Core Operator，来管理带有标签的 SeldonDeployments，其中标签 ``seldon.io/controller-id`` 的值与正在运行的 operator 的 controller-id 匹配。
示例 ``seldon-id1`` 命名空间如下所示：

Helm
^^^^

.. code:: bash

    kubectl create namespace seldon-id1

运行命令安装 Operator:

.. code:: bash

    helm install seldon-controllerid seldon-core-operator  --repo https://storage.googleapis.com/seldon-charts  \
        --set singleNamespace=false \
        --set image.pullPolicy=IfNotPresent \
        --set usageMetrics.enabled=false \
        --set crd.create=true \
        --set controllerId=seldon-id1 \
        --namespace seldon-id1

设置 ``crd.create=true`` 来创建 CRD。
如果您在同一集群上在以前的 Seldon Core Operator 之上安装 Seldon Core Operator，则需要设置 ``crd.create=false``。

针对 kustomize 你需要在此处 `去掉 patch\_object\_selector
<https://github.com/SeldonIO/seldon-core/blob/master/operator/config/webhook/kustomization.yaml>`__ 注释。

Kustomize
^^^^^^^^^

Operator 文件夹中的 Makefile 中提供了一个示例安装：

.. code:: bash

    make deploy-controllerid

查看 `多服务器示例笔记 <../examples/multiple_operators.html>`__。

通过代理安装
~~~~~~~~~~~~~~~~~~~~~~

当您的 kubernetes 集群位于代理后面时， ``kube-apiserver``通常会继承系统代理变量。
这可以阻止 ``kube-apiserver`` 访问创建 Seldon 所需的 webhook 资源。

你可能会看到如下错误：

.. code:: bash

    Internal error occurred: failed calling webhook "v1.vseldondeployment.kb.io": Post https://seldon-webhook-service.seldon-system.svc:443/validate-machinelearning-seldon-io-v1-seldondeployment?timeout=30s: Service Unavailable

要解决此问题，请确保 ``kube-apiserver`` 的环境变量 ``no_proxy``包含 ``.svc,.svc.cluster.local``。
查看 `这个 Github Issue 回复 <https://github.com/jetstack/cert-manager/issues/2640#issuecomment-601872165>`__来参考。
如那里所述，错误也可能发生在 ``cert-manager-webhook``。
