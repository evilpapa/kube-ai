====================
本地安装
====================

本指南介绍了如何在本地机器上运行的 Kubernetes 集群中设置和安装 Seldon Core。最终，您将启动并运行 Seldon Core，并准备好开始部署机器学习模型。

先决条件
-----------------

为了在本地安装 Seldon Core，您需要以下工具：

.. warning:: 根据您的本地计算机的权限和您正在使用的目录，某些工具可能需要 root 访问权限

Docker 或 Podman
^^^^^^^^^^^^^^^^^^^
`Docker <https://www.docker.com/>`_ 和 `Podman <https://podman.io/>`_ 是容器引擎。Kind 需要一个容器引擎（如 docker 或 podman）来实际运行集群内的容器。
您只需要 Docker 或 Podman 之一。请注意，Docker 不再免费为大公司供个人使用：

* 为 `Linux <https://docs.docker.com/engine/install/ubuntu/>`_ , `Mac <https://docs.docker.com/desktop/mac/install/>`_ , `Windows <https://docs.docker.com/desktop/windows/install/>`_ 安装 Docker

或

* `安装 Podman <https://podman.io/getting-started/installation>`_

.. note:: 如果使用 Podman 记得设置 ``alias docker=podman``

Kind
^^^^^^^^^^^^^
`Kind <https://kind.sigs.k8s.io/>`_ 是一个在本地运行 Kubernetes 集群的工具。我们将使用它在您的机器上创建一个集群，以便您可以在其中安装 Seldon Core。如果您的机器上还没有安装 `kind <https://kind.sigs.k8s.io/>`_，您需要按照他们的安装指南进行操作：

* `安装 Kind <https://kind.sigs.k8s.io/docs/user/quick-start/#installation>`_ 

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

设置 Kind
----------------



当在及其安装 kind 之后，通过运行以下命令创建一个先的 Kubernetes 集群

.. tab-set::

    .. tab-item:: Istio

        .. code-block:: bash

            kind create cluster --name seldon

    .. tab-item:: Ambassador

        .. code-block:: bash

            cat <<EOF | kind create cluster --name seldon --config=-
            kind: Cluster
            apiVersion: kind.x-k8s.io/v1alpha4
            nodes:
            - role: control-plane
            kubeadmConfigPatches:
            - |
                kind: InitConfiguration
                nodeRegistration:
                kubeletExtraArgs:
                    node-labels: "ingress-ready=true"
            extraPortMappings:
            - containerPort: 80
                hostPort: 80
                protocol: TCP
            - containerPort: 443
                hostPort: 443
                protocol: TCP
            EOF

在 ``kind`` 创建集群后，可通过设置上下文配置 ``kubectl`` 使用集群：

.. code-block:: bash

    kubectl cluster-info --context kind-seldon

从现在起，所有使用 ``kubectl`` 运行的命令都将直接作用于 ``kind`` 集群。 

.. note:: Kind 为您的集群名称添加 ``kind-`` 前缀，所以集群上下文是 ``kind-seldon`` 而非 ``seldon``

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

        进入 Istio 包目录。比如，包 ``istio-1.11.4``：

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

        首先通过命令安装 Custom Resource Definitions：

        .. code-block:: bash 

            kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml

        现在安装 kind-specific manifests in the ``ambassador`` namespace:

        .. code-block:: bash 

            kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
            kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

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

你应该能看到 ``seldon-controller-manager`` pod 的 ``STATUS=Running`` 状态。

本地端口转发
-------------------------------

因为 kubernetes 集群运行在本地，我们需要转发一个本地及其端口到集群，以便我们从外部访问。可通过命令尝试：

.. tab-set::

    .. tab-item:: Istio

        .. code-block:: bash

            kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80

    .. tab-item:: Ambassador

        .. code-block:: bash 

            kubectl port-forward -n ambassador svc/ambassador 8080:80

这将转发本地端口 8080 的任意流量到集群内 80 端口。

现在成功在本地安装 Seldon Core 并就绪 `开始部署模型 <../workflow/github-readme.md>`_ 作为生产微服务。