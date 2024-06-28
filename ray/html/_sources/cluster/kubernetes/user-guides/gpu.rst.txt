.. _kuberay-gpu:

使用 GPU
==========
本文档提供有关 KubeRay 的 GPU 使用技巧。

要在 Kubernetes 上使用 GPU，请配置 Kubernetes 设置并向 Ray 集群配置添加其他值。

要了解不同云上的 GPU 使用情况，请参阅 `GKE`_、 `EKS`_、 `AKS`_的说明。

快速入门：提供基于 GPU 的 StableDiffusion 模型
___________________________________________________

您可以在文档的 :ref:`示例 <kuberay-examples>` 部分找到几个 GPU 工作负载示例。
:ref:`StableDiffusion 示例 <kuberay-stable-diffusion-rayservice-example>` 是一个很好的起点。

基于 GPU 的机器学习的依赖项
___________________________________________

`Ray Docker Hub <https://hub.docker.com/r/rayproject/>`_ 托管与 Ray 和某些机器学习库打包的基于 CUDA 的容器映像。
例如，镜像 ``rayproject/ray-ml:2.6.3-gpu`` 非常适合使用 Ray 2.6.3 运行基于 GPU 的 ML 工作负载。
Ray ML 镜像与这些文档中使用的 Ray 库所需的依赖项（例如 TensorFlow 和 PyTorch）一起打包。
要添加自定义依赖项，请使用以下一种或两种方法：

* 使用官方 :ref:`Ray docker 镜像 <docker-images>` 作为基础构建 docker 镜像。
* 使用 :ref:`Ray 运行时环境 <runtime-environments>`。


配置 Ray pod 以使用 GPU
__________________________________

要使用 Nvidia GPU 需要在 `RayCluster` 的 `headGroupSpec` 及 `workerGroupSpecs` 的容器 `limits` 和 `requests` 字段指定 `nvidia.com/gpu` 资源。

以下是最多包含 5 个 GPU worker 线程的 RayCluster 工作组的配置片段。

.. code-block:: yaml

   groupName: gpu-group
   replicas: 0
   minReplicas: 0
   maxReplicas: 5
   ...
   template:
       spec:
        ...
        containers:
         - name: ray-node
           image: rayproject/ray-ml:2.6.3-gpu
           ...
           resources:
            nvidia.com/gpu: 1 # Optional, included just for documentation.
            cpu: 3
            memory: 50Gi
           limits:
            nvidia.com/gpu: 1 # Required to use GPU.
            cpu: 3
            memory: 50Gi
            ...

组中的每个 Ray Pod 都可以在 AWS `p2.xlarge` 示例 (1 GPU、4vCPU、61Gi RAM) 上进行调度。

.. tip::

    GPU 实例非常昂贵 - 请考虑为 GPU Ray 工作线程设置自动缩放，
    如上面的 `minReplicas:0` 和 `maxReplicas:5` 。
    要启用自动缩放，还请记住需要为 RayCluster 的 `spec` 设置 `enableInTreeAutoscaling:True` 。
    最后，确保将 GPU Kubernetes 节点组或池配置为自动缩放。
    有关自动缩放节点池的详细信息，请参阅 :ref:`云提供商的文档 <kuberay-k8s-setup>` 。

GPU 多租户
_________________

如果 Pod 在其资源配置中未包含 `nvidia.com/gpu` ，用户通常会期望 Pod 不知道任何 GPU 设备，即使它调度在 GPU 节点上。
但是，如果 `nvidia.com/gpu` 未指定，则默认值 `NVIDIA_VISIBLE_DEVICES` 为 `all`，使 Pod 能够感知节点上的所有 GPU 设备。
这种行为并非 KubeRay 所独有，而是 NVIDIA 的一个已知问题。
解决方法是在不需要 GPU 设备的 Pod 中设置 `NVIDIA_VISIBLE_DEVICES` 环境变量为 `void` 。

一些有用的链接：

- `NVIDIA/k8s-device-plugin#61`_
- `NVIDIA/k8s-device-plugin#87`_
- `[NVIDIA] 防止对 Kubernetes 中的 GPU 进行非特权访问`_
- `ray-project/ray#29753`_

GPU 和 Ray
____________

节讨论在 Kubernetes 上运行的 Ray 应用程序的 GPU 使用情况。
有关 Ray 的 GPU 使用的一般指南，另请参阅 :ref:`gpu-support`。


KubeRay operator 将容器 GPU 资源限制通告给 Ray 调度器和 Ray 自动缩放器。
通常，Ray 容器的入口点 `ray start` 使用 `--num-gpus` 选项会自动配置。

GPU 工作负载调度
~~~~~~~~~~~~~~~~~~~~~~~
部署可访问 GPU 的 Ray Pod 后，它将能够执行使用 GPU 请求注释的任务和 Actor。
例如，装饰器 `@ray.remote(num_gpus=1)` 注释了需要 1 个 GPU 的任务或 Actor。


GPU 自动缩放
~~~~~~~~~~~~~~~
Ray 自动缩放器知道每个 Ray 工作组的 GPU 容量。
假设我们配置了一个 RayCluster，如上面的配置片段所示：

- 有一个由 Ray pod 组成的工作组，每个工作组有 1 个单位的 GPU 容量。
- Ray 集群当前没有来自该组的任何 worker。
- 改组的 `maxReplicas` 至少为 2 。

然后下面的 Ray 程序将触发 2 个 GPU 工作线程的升级。

.. code-block:: python

    import ray

    ray.init()

    @ray.remote(num_gpus=1)
    class GPUActor:
        def say_hello(self):
            print("I live in a pod with GPU access.")

    # Request actor placement.
    gpu_actors = [GPUActor.remote() for _ in range(2)]
    # The following command will block until two Ray pods with GPU access are scaled
    # up and the actors are placed.
    ray.get([actor.say_hello.remote() for actor in gpu_actors])

程序退出后，actor 将会被垃圾回收。
GPU Worker Pod 将在空闲超时（默认为 60 秒）后缩容。
如果 GPU Worker Pod 在 Kubernetes 节点的自动缩放池上运行，则 Kubernetes 节点也将缩小。

请求 GPU
~~~~~~~~~~~~~~~
您还可以 :ref:`直接向自动缩放器发出请求 <ref-autoscaler-sdk-request-resources>` 以扩展 GPU 资源。

.. code-block:: python

    import ray

    ray.init()
    ray.autoscaler.sdk.request_resources(bundles=[{"GPU": 1}] * 2)

节点扩展后，它们将持续存在，直到请求被显式覆盖。
以下程序将删除资源请求。

.. code-block:: python

    import ray

    ray.init()
    ray.autoscaler.sdk.request_resources(bundles=[])

然后 GPU 工作线程就可以缩小规模。

.. _kuberay-gpu-override:

覆盖 Ray GPU 容量（高级）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
对于专门的用例，可以覆盖向 Ray 公布的 Ray pod GPU 容量。
为此，请在头或者工作组的 `rayStartParams` 中设置 `num-gpus` 键的值。
如，

.. code-block:: yaml

    rayStartParams:
        # Note that all rayStartParam values must be supplied as strings.
        num-gpus: "2"

Ray 调度器和自动缩放器将为组中的每个 Ray pod 分配 2 个 GPU 容量单位，即使容器限制不指示 GPU 的存在。

GPU Pod 调度（高级）
_____________________________

GPU 污染和容忍
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::

  托管 Kubernetes 服务通常会为您处理与 GPU 相关的污染和容忍问题。
  如果您使用的是托管 Kubernetes 服务，则可能不需要担心此部分。

Kubernetes 的 `Nvidia gpu plugin`_ 将 `taints`_ 应用于 GPU 节点；这些污点会阻止非 GPU pod 在 GPU 节点上进行调度。 
GKE、EKS 和 AKS 等托管 Kubernetes 服务会自动将匹配的 `tolerations`_ 应用于请求 GPU 资源的 Pod。
容忍是通过 Kubernetes 的 `ExtendedResourceToleration`_ `admission controller`_ 准入控制器来应用的。
如果您的 Kubernetes 集群未启用此准入控制器，您可能需要手动向每个 GPU Pod 配置添加 GPU 容忍。例如，

.. code-block:: yaml

  apiVersion: v1
  kind: Pod
  metadata:
   generateName: example-cluster-ray-worker
   spec:
   ...
   tolerations:
   - effect: NoSchedule
     key: nvidia.com/gpu
     operator: Exists
   ...
   containers:
   - name: ray-node
     image: rayproject/ray:nightly-gpu
     ...

节点选择器和节点标签
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
为了确保 Ray Pod 绑定到满足特定条件（例如存在 GPU 硬件）的 Kubernetes 节点，您可能希望使用`workerGroup` 的 `pod`` 模板 `nodeSelector` 字段。
有关 Pod 到节点分配的更多信息，请参阅 `Kubernetes docs`_。


进一步参考和讨论
--------------------------------
`这里 <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>`__ 阅读有关 Kubernetes 设备插件的信息，
`这里 <https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus>`__ 阅读有关 Kubernetes GPU 插件的信息，
`这里 <https://github.com/NVIDIA/k8s-device-plugin>`__ 阅读有关 Nvidia 的 Kubernetes GPU 插件的信息。

.. _`GKE`: https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
.. _`EKS`: https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
.. _`AKS`: https://docs.microsoft.com/en-us/azure/aks/gpu-cluster

.. _`NVIDIA/k8s-device-plugin#61`: https://github.com/NVIDIA/k8s-device-plugin/issues/61
.. _`NVIDIA/k8s-device-plugin#87`: https://github.com/NVIDIA/k8s-device-plugin/issues/87
.. _`[NVIDIA] Preventing unprivileged access to GPUs in Kubernetes`: https://docs.google.com/document/d/1zy0key-EL6JH50MZgwg96RPYxxXXnVUdxLZwGiyqLd8/edit?usp=sharing
.. _`ray-project/ray#29753`: https://github.com/ray-project/ray/issues/29753

.. _`tolerations`: https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/
.. _`taints`: https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/
.. _`Nvidia gpu plugin`: https://github.com/NVIDIA/k8s-device-plugin
.. _`admission controller`: https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/
.. _`ExtendedResourceToleration`: https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/#extendedresourcetoleration
.. _`Kubernetes docs`: https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/
.. _`bug`: https://github.com/ray-project/kuberay/pull/497/
