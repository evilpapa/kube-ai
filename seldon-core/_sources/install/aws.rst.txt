========================================
在 Amazon Web Services 上安装
========================================

本指南介绍了如何在 AWS 上运行的 Kubernetes 集群中设置和安装 Seldon Core。最后，您将启动并运行 Seldon Core，并准备开始部署机器学习模型。

先决条件
-----------------------------

AWS CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

您将需要 AWS CLI 来检索集群身份验证凭证。它还可用于为您创建集群和其他资源：

* `安装 AWS CLI <https://aws.amazon.com/cli/>`_

弹性 Kubernetes 服务 (EKS) 集群
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you haven't already created a Kubernetes cluster on EKS, you can follow this quickstart guide to get set up with your first cluster. We recommend using the `eksctl` path to create your cluster as it simplifies the process of creating IAM roles, VPCs and subnets.
如果您尚未在 EKS 上创建 Kubernetes 集群，您可以按照此快速入门指南设置您的第一个集群。我们建议使用 `eksctl` 路径来创建集群，因为它简化了创建 IAM 角色、VPC 和子网的过程。

* `安装 eksctl CLI <https://eksctl.io/introduction/#installation>`_
* `创建 EKS 集群 <https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html>`_

.. warning:: 

    如果你计划使用 Ambassador 作为入口，你的集群需要运行 Kubernetes

Kubectl
^^^^^^^^^^^^^
`kubectl <https://kubernetes.io/docs/reference/kubectl/overview/>`_ 是 Kubernetes 命令行工具。它允许您针对 Kubernetes 集群运行命令，这是我们在设置 Seldon Core 时需要执行的操作。

* `在 Linux 上安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux>`_
* `在 macOS 上安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-macos>`_
* `在 Windows 上安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-windows>`_

Helm
^^^^^^^^^^^^^
`Helm <https://helm.sh/>`_ 是一个软件包管理器，可让您轻松查找、共享和使用为 Kubernetes 构建的软件。如果您尚未在本地安装 Helm，可以在此处安装：

* `安装 Helm <https://helm.sh/docs/intro/install/>`_

连接到您的集群
------------------------------

您可以通过运行以下 `aws eks` 命令连接到您的集群：

.. code-block:: bash

    aws eks update-kubeconfig --region REGION_CODE --name CLUSTER_NAME

这将配置 ``kubectl`` 使用您的 aws kubernetes 集群。不要忘记用您创建集群时的名称替换 ``CLUSTER_NAME`` 。如果您忘记了集群名称，您可以运行 ``aws eks list-clusters``。

.. note:: 

    如果在运行上述命令时出现身份验证错误，请尝试运行 ``aws configure`` 以检查您是否正确登录。

安装 Cluster Ingress
------------------------------

``Ingress`` 是一个 Kubernetes 对象，为您的集群提供路由规则。它管理传入流量并将其路由到集群内运行的服务。

Seldon Core 支持使用 `Istio <https://istio.io/>`_ 或 `Ambassador <https://www.getambassador.io/>`_ 来管理传入流量。 Seldon Core 会自动创建将流量路由到已部署的机器学习模型所需的对象和规则。

.. tab-set::

    .. tab-item:: Istio

        Istio 是一个开源服务网格。如果您不熟悉 *服务网格* 这个术语，那么值得多读 `一些有关 Istio 的内容 <https://istio.io/latest/about/service-mesh/>`_。

        **下载 Istio**

        对于 Linux 和 macOS，下载 Istio 最简单的方法是使用以下命令：

        .. code-block:: bash 

            curl -L https://istio.io/downloadIstio | sh -

        移至 Istio 包目录。例如，如果包是 ``istio-1.11.4``：

        .. code-block:: bash

            cd istio-1.11.4

        将 istioctl 客户端添加到您的路径（Linux 或 macOS）：

        .. code-block:: bash

            export PATH=$PWD/bin:$PATH

        **安装 Istio**

        Istio 提供了一个命令行工具 ``istioctl`` ，使安装过程变得简单。``demo`` `配置文件 <https://istio.io/latest/docs/setup/additional-setup/config-profiles/>`_ 有一组很好的默认设置，可以在您的本地集群上运行。

        .. code-block:: bash

            istioctl install --set profile=demo -y

        命名空间标签 ``istio-injection=enabled`` 指示 Istio 自动注入代理以及我们在该命名空间中部署的任何内容。我们将为我们的  ``default`` 命名空间设置它：

        .. code-block:: bash 

            kubectl label namespace default istio-injection=enabled

        **创建 Istio 网关**

        为了让 Seldon Core 使用 Istio 的功能来管理集群流量，我们需要通过运行以下命令创建一个 `Istio 网关 <https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/>`_ ：

        .. warning:: 您需要从下面的代码块复制整个命令
        
        .. code-block:: yaml

            kubectl apply -f - << END
            apiVersion: networking.istio.io/v1alpha3
            kind: Gateway
            metadata:
            name: seldon-gateway
            namespace: istio-system
            spec:
            selector:
                istio: ingressgateway # use istio default controller
            servers:
            - port:
                number: 80
                name: http
                protocol: HTTP
                hosts:
                - "*"
            END
        
        有关自定义配置和使用 Istio 安装 seldon core 的更多详细信息，请参阅 `Istio Ingress <../ingress/istio.md>`_ 页面。

    .. tab-item:: Ambassador

        .. warning:: Ambassador 目前不支持 Kubernetes 1.22+，以下说明仅适用于 Kubernetes v1.21 或更早版本。

        `Ambassador <https://www.getambassador.io/>`_ 是 Kubernetes 入口控制器和 API 网关。它通过配置将传入流量路由到底层 Kubernetes 工作负载。

        **安装 Ambassador**

        .. note::
            目前仅支持 Ambassador V1 API。以下安装说明将安装最新的 v1 版本的 emissary ingress。

        首先添加 datawire helm 存储库：

        .. code-block:: bash

            helm repo add datawire https://www.getambassador.io
            helm repo update

        运行以下 `helm` 命令在您的 GKE 集群上安装 Ambassador：

        .. code-block:: bash

            helm install ambassador datawire/ambassador --set enableAES=false --namespace ambassador --create-namespace
            kubectl rollout status -n ambassador deployment/ambassador -w
            
        Ambassador 现已准备就绪。有关自定义配置以及使用 Ambassador 安装 seldon core 的更多详细信息，请参阅 `Ambassador Ingress <../ingress/ambassador.md>`_ 页面。

安装 Seldon Core
----------------------------

在安装 Seldon Core 之前，我们将为控制器创建一个名为 ``seldon-system`` 的新命名空间：

.. code:: bash

    kubectl create namespace seldon-system

现在，我们已准备好在集群中安装 Seldon Core。针对您选择的 Ingress 运行以下命令：

.. tab-set::

    .. tab-item:: Istio

        .. code:: bash

            helm install seldon-core seldon-core-operator \
                --repo https://storage.googleapis.com/seldon-charts \
                --set usageMetrics.enabled=true \
                --set istio.enabled=true \
                --namespace seldon-system

    .. tab-item:: Ambassador

        .. warning:: Ambassador 目前不支持 Kubernetes 1.22+，以下说明仅适用于 Kubernetes v1.21 或更早版本。

        .. code:: bash

            helm install seldon-core seldon-core-operator \
                --repo https://storage.googleapis.com/seldon-charts \
                --set usageMetrics.enabled=true \
                --set ambassador.enabled=true \
                --namespace seldon-system

您可以通过执行以下操作来检查 Seldon 控制器是否正在运行：

.. code-block:: bash

    kubectl get pods -n seldon-system

您应该会看到一个 ``seldon-controller-manager`` pod 状态为 ``STATUS=Running``。

访问你的模型
-------------------------

恭喜！Seldon Core 现已完全安装并运行。在继续部署模型之前，请记下您的集群 IP 和端口：

.. tab-set::

    .. tab-item:: Istio

        .. code-block:: bash 

            export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
            export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
            export INGRESS_URL=$INGRESS_HOST:$INGRESS_PORT
            echo $INGRESS_URL

        这是您用于访问集群中运行的模型的公共地址。

    .. tab-item:: Ambassador

        .. warning:: Ambassador 目前不支持 Kubernetes 1.22+，以下说明仅适用于 Kubernetes v1.21 或更早版本。

        .. code-block:: bash

            export INGRESS_HOST=$(kubectl -n ambassador get service ambassador -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
            export INGRESS_PORT=$(kubectl -n ambassador get service ambassador -o jsonpath='{.spec.ports[?(@.name=="http")].port}')
            export INGRESS_URL=$INGRESS_HOST:$INGRESS_PORT
            echo $INGRESS_URL

        这是您用于访问集群中运行的模型的公共地址。

您现在可以将 `模型部署到您的集群了 <../workflow/github-readme.md>`_。
