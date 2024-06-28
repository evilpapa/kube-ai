========================================
在 Microsoft Azure Cloud 安装
========================================

本指南介绍了如何在 Azure 云上运行的 Kubernetes 集群中设置和安装 Seldon Core。到最后，您将启动并运行 Seldon Core，并准备好开始部署机器学习模型。

要求
-----------------------------

Azure 云 CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

您将需要 Azure CLI 来检索集群身份验证凭据。它还可用于为您创建集群和其他资源：

* `安装 Azure CLI <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>`_

Azure Kubernetes Service (AKS) 集群
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您尚未在 AKS 上创建 Kubernetes 集群，则可以按照此快速入门指南设置您的第一个集群：

* `创建 AKS 集群 <https://docs.microsoft.com/en-us/azure/aks/tutorial-kubernetes-deploy-cluster?tabs=azure-cli>`_

.. note:: 

    如果您只是评估 Seldon Core 并且想要使用 http 而不是 https，请确保在网络配置中选择「启用 HTTP 应用程序路由」。

Kubectl
^^^^^^^^^^^^^
`kubectl <https://kubernetes.io/docs/reference/kubectl/overview/>`_ 是 Kubernetes 命令行工具。它允许您对 Kubernetes 集群运行命令，这是设置 Seldon Core 的一部分。

* `在 Linux 安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux>`_ 
* `在 macOS 安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-macos>`_ 
* `在 Windows 安装 kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-windows>`_ 

Helm
^^^^^^^^^^^^^
`Helm <https://helm.sh/>`_ 是一个包管理工具，可以轻松查找、共享和使用为 Kubernetes 构建的软件。如果你还没有在本地安装 Helm，你可以在这里安装它：

* `安装 Helm <https://helm.sh/docs/intro/install/>`_ 

连接到集群
------------------------------

通过执行 `az` 命令连接到集群：

.. code-block:: bash

    az aks get-credentials --resource-group myResourceGroup --name myAKSCluster

他将配置 ``kubectl`` 到你的 Azure kubernetes 集群。别忘记替换 ``myResourceGroup`` 及 ``myAKSCluster`` 为你创建的资源组及集群名称。如果忘记，运行 ``az aks list``。

.. note:: 

    如果运行如上命令出现认证错误，尝试运行 ``az login`` 来检查是否正确登录。

安装集群入口
------------------------------

``Ingress`` 是为您的集群提供路由规则的 Kubernetes 对象。它管理传入的流量并将其路由到集群内运行的服务。

Seldon Core 支持使用 `Istio <https://istio.io/>`_ 或 `Ambassador <https://www.getambassador.io/>`_ 来管理传入流量。Seldon Core 自动创建将流量路由到您部署的机器学习模型所需的对象和规则。

.. tab-set::

    .. tab-item:: Istio

        Istio 是一个开源服务网格。如果您对 *service mesh* 不熟悉，非常值得去阅读 `更多关于Istio 的内容 <https://istio.io/latest/about/service-mesh/>`_ 。

        **下载 Istio**

        对于 Linux 及 macOS，最简单的方式是使用以下命令下载 Istio：

        .. code-block:: bash 

            curl -L https://istio.io/downloadIstio | sh -

        M进入 Istio 包目录。比如，包 ``istio-1.11.4``：

        .. code-block:: bash

            cd istio-1.11.4

        添加 istioctl 客户端到 path (Linux or macOS):

        .. code-block:: bash

            export PATH=$PWD/bin:$PATH

        **安装 Istio**

        Istio 提供了一个命令工具 ``istioctl`` 来使安装更便捷。``示例`` `配置项 <https://istio.io/latest/docs/setup/additional-setup/config-profiles/>`_ 有一组很好的默认值来运行在你本地集群。

        .. code-block:: bash

            istioctl install --set profile=demo -y

        命名空间标签 ``istio-injection=enabled`` 指示 Istio 注入自动代理我们在该命名空间中部署的任何内容。我们将为我们的 ``default`` 命名空间设置它：

        .. code-block:: bash 

            kubectl label namespace default istio-injection=enabled

        **创建 Istio 网关**

        为了让 Seldon Core 使用 Istio 的特性来管理流量，我们使用如下命令来创建一个 `Istio Gateway <https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/>`_ ：

        .. warning:: 你需要拷贝下面全部的命令
        
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
        
        自定义配置及更多 seldon core 集成 Istio 安装的细节请查看 `Istio 入口 <../ingress/istio.md>`_ 页。

    .. tab-item:: Ambassador

        `Ambassador <https://www.getambassador.io/>`_ 是 Kubernetes 入口控制器及 API 网关。他通过配置路由请求流量到 kubernetes 负载。

        **安装 Ambassador**

        .. note::
            Seldon Core 现在只支持 Ambassador V1 APIs。以下安装说明将只安装 emissary ingress 最新的 v1 版本。


        首先添加 datawire helm 仓库：

        .. code-block:: bash

            helm repo add datawire https://www.getambassador.io
            helm repo update

        执行以下 `helm` 命令安装 Ambassador 到 GKE 集群：

        .. code-block:: bash

            helm install ambassador datawire/ambassador --set enableAES=false --namespace ambassador --create-namespace
            kubectl rollout status -n ambassador deployment/ambassador -w
            
        Ambassador 已就绪。自定义配置及更多集成 Ambassador 安装 seldon core 的细节请查看 `Ambassador 入口 <../ingress/ambassador.md>`_ 页。

安装 Seldon Core
----------------------------

在安装 Seldon Core 前，创建一个 operator 运行所在的命名空间 ``seldon-system`` ：

.. code:: bash

    kubectl create namespace seldon-system

现在我们已经为在集群安装 Seldon Core 准备就绪。根据选择的入口类型执行如下命令：

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

使用以下命令检查 Seldon Controller 运行状态：

.. code-block:: bash

    kubectl get pods -n seldon-system

你应当看到 ``seldon-controller-manager`` pod 的状态 ``STATUS=Running``。

访问您的模型
-------------------------

恭喜！Seldon Core 现在已完全安装并运行。在继续部署模型之前，请记下您的集群 IP 和端口：

.. tab-set::

    .. tab-item:: Istio

        .. code-block:: bash 

            export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
            export INGRESS_URL=$INGRESS_HOST:$INGRESS_PORT
            echo $INGRESS_URL

        这是您将用于访问集群中运行的模型的公共地址。

    .. tab-item:: Ambassador

        .. code-block:: bash

            export INGRESS_HOST=$(kubectl -n ambassador get service ambassador -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            export INGRESS_PORT=$(kubectl -n ambassador get service ambassador -o jsonpath='{.spec.ports[?(@.name=="http")].port}')
            export INGRESS_URL=$INGRESS_HOST:$INGRESS_PORT
            echo $INGRESS_URL

        这是您将用于访问集群中运行的模型的公共地址。

您现在已准备好 `将模型部署到您的集群 <../workflow/github-readme.md>`_。
