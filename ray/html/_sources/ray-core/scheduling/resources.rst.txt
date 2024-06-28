.. _core-resources:

资源
=========

Ray 允许你在不改变代码的情况下，将应用程序从笔记本电脑扩展到集群。
**Ray 资源** 是实现这一功能的关键。
它们抽象出物理机器，让你以资源的形式表达计算，而系统则根据资源请求进行调度和自动扩展。

Ray 的一个资源是一个键值对，其中键表示资源名称，值是一个浮点数数量。
为方便起见，Ray 本身支持 CPU、GPU 和内存资源类型；CPU、GPU 和内存被称为 **预定义资源**。
除了这些，Ray 还支持 :ref:`自定义资源 <custom-resources>`。

.. _logical-resources:

物理资源和逻辑资源
----------------------------------------

物理资源是机器上实际存在的资源，例如物理 CPU 和 GPU，而逻辑资源是系统定义的虚拟资源。

Ray 资源是 **逻辑** 的，不需要与物理资源一一对应。
比如，你可以通过 ``ray start --head --num-cpus=0`` 来启动一个没有逻辑 CPU 的 Ray head 节点。

甚至在物理上有八个 CPU 的情况下（这样可以告诉 Ray 调度器不要在 head 节点上调度任何需要逻辑 CPU 资源的任务或 actor，主要是为了保留 head 节点用于运行 Ray 系统进程。）。
他们主要用于调度期间的准入控制。

实际上资源是逻辑的，这有几个影响：

- Resource requirements of tasks or actors do NOT impose limits on actual physical resource usage. 
- 任务和 actor 的资源需求并不限制实际的物理资源使用。
  比如，Ray 不会阻止一个 ``num_cpus=1`` 的任务启动多个线程并使用多个物理 CPU。
  你需要确保任务或 actor 使用的资源不超过通过资源需求指定的资源。
- Ray 不提供任务或 actor 的 CPU 隔离。
  比如，Ray 不会专门保留一个物理 CPU 并将 ``num_cpus=1`` 的任务固定在上面。
  Ray 会让操作系统调度和运行任务。
  如果需要，你可以使用操作系统 API（如 ``sched_setaffinity``）将任务固定到一个物理 CPU 上。
- Ray 提供 *visible devices* 形式的 :ref:`GPU <gpu-support>` 隔离，通过自动设置 ``CUDA_VISIBLE_DEVICES`` 环境变量，大多数 ML 框架将遵守这个环境变量来分配 GPU。

.. figure:: ../images/physical_resources_vs_logical_resources.svg

  物理资源对比逻辑资源

.. _custom-resources:

自定义资源
----------------

除了预定义资源，你还可以指定 Ray 节点的自定义资源，并在任务或 actor 中请求它们。
一些自定义资源的用例：

- 你的节点有特殊的硬件，你可以将其表示为一个自定义资源。
  然后，你的任务或 actor 可以通过 ``@ray.remote(resources={"special_hardware": 1})`` 请求自定义资源，
  Ray 将任务或 actor 调度到具有自定义资源的节点。
- 你可以使用自定义资源作为标签来标记节点，从而实现基于标签的亲和调度。
  比如，你可以使用 ``ray.remote(resources={"custom_label": 0.001})`` 将任务或 actor 调度到具有 ``custom_label`` 自定义资源的节点。
  对于这个用例，实际数量并不重要，约定是指定一个很小的数字，以便标签资源不会成为并行性的限制因素。

.. _specify-node-resources:

指定节点资源
-------------------------

默认，Ray 节点启动时会使用预定义的 CPU、GPU 和内存资源。这些逻辑资源的数量是 Ray 自动检测到的物理数量。
默认，逻辑资源按照以下规则配置。

.. warning::

    Ray **不会在启动后动态更新资源容量**。

- **逻辑 CPU 数量（``num_cpus``）**：设置为机器/容器的 CPU 数量。
- **逻辑 GPU 数量（``num_gpus``）**：设置为机器/容器的 GPU 数量。
- **内存（``memory``）**：设置为 ray 运行时启动时的 "可用内存" 的 70%。
- **对象存储内存（``object_store_memory``）**：设置为 ray 运行时启动时的 "可用内存" 的 30%。请注意，对象存储内存不是逻辑资源，用户不能用它来调度。

然而，你可以通过手动指定预定义资源的数量和添加自定义资源来覆盖这些默认值。
这里有几种方法，取决于你如何启动 Ray 集群：

.. tab-set::

    .. tab-item:: ray.init()

        如果你使用 :func:`ray.init() <ray.init>` 启动单节点 Ray 集群，你可以通过以下方式手动指定节点资源：

        .. literalinclude:: ../doc_code/resources.py
            :language: python
            :start-after: __specifying_node_resources_start__
            :end-before: __specifying_node_resources_end__

    .. tab-item:: ray start

        如果你使用 :ref:`ray start <ray-start-doc>` 启动 Ray 节点，你可以运行：

        .. code-block:: shell

            ray start --head --num-cpus=3 --num-gpus=4 --resources='{"special_hardware": 1, "custom_label": 1}'

    .. tab-item:: ray up

        如果你使用 :ref:`ray up <ray-up-doc>` 启动 Ray 集群，你可以在 yaml 文件中设置 :ref:`resources 字段 <cluster-configuration-resources-type>`：

        .. code-block:: yaml

            available_node_types:
              head:
                ...
                resources:
                  CPU: 3
                  GPU: 4
                  special_hardware: 1
                  custom_label: 1

    .. tab-item:: KubeRay

        如果你使用 :ref:`KubeRay <kuberay-index>` 启动 Ray 集群，你可以在 yaml 文件中设置 :ref:`rayStartParams 字段 <rayStartParams>`：

        .. code-block:: yaml

            headGroupSpec:
              rayStartParams:
                num-cpus: "3"
                num-gpus: "4"
                resources: '"{\"special_hardware\": 1, \"custom_label\": 1}"'


.. _resource-requirements:

指定任务或 actor 的资源需求
----------------------------------------------

Ray 允许指定任务或 actor 的逻辑资源需求（例如 CPU、GPU 和自定义资源）。
任务和 actor 会在调度时使用资源需求来决定在哪个节点上运行。

By default, Ray tasks use 1 logical CPU resource and Ray actors use 1 logical CPU for scheduling, and 0 logical CPU for running.
默认的，Ray 任务使用 1 个逻辑 CPU 资源，Ray actor 使用 1 个逻辑 CPU 资源进行调度，0 个逻辑 CPU 资源进行运行。
（这意味着，默认情况下，actor 不能在零 CPU 节点上调度，但是无限数量的 actor 可以在任何非零 CPU 节点上运行。
默认的 actor 资源需求是为了历史原因而选择的。
推荐的做法是始终为 actor 明确设置 ``num_cpus``，以避免任何意外。
如果资源是显式指定的，它们对调度和运行都是必需的。）

你可以通过 :func:`ray.remote() <ray.remote>` 和 :meth:`task.options() <ray.remote_function.RemoteFunction.options>`/:meth:`actor.options() <ray.actor.ActorClass.options>` 明确指定任务或 actor 的逻辑资源需求
（例如，一个任务可能需要一个 GPU）。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/resources.py
            :language: python
            :start-after: __specifying_resource_requirements_start__
            :end-before: __specifying_resource_requirements_end__

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources.
            Ray.task(MyRayApp::myFunction).setResource("CPU", 1.0).setResource("GPU", 1.0).setResource("special_hardware", 1.0).remote();

            Ray.actor(Counter::new).setResource("CPU", 2.0).setResource("GPU", 1.0).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources.
            ray::Task(MyFunction).SetResource("CPU", 1.0).SetResource("GPU", 1.0).SetResource("special_hardware", 1.0).Remote();

            ray::Actor(CreateCounter).SetResource("CPU", 2.0).SetResource("GPU", 1.0).Remote();

任务和 actor 的资源需求对 Ray 的调度并发性有影响。
具体而言，同一节点上所有并发执行的任务和 actor 的逻辑资源需求之和不能超过节点的总逻辑资源。
这个属性可以用来 :ref:`限制并发运行的任务或 actor 的数量，以避免像 OOM 这样的问题 <core-patterns-limit-running-tasks>`。

.. _fractional-resource-requirements:

分数资源需求
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray 支持分数资源需求。
比如，如果你的任务或 actor 是 IO 绑定的，并且 CPU 使用率很低，你可以指定分数 CPU ``num_cpus=0.5``，甚至是零 CPU ``num_cpus=0``。
分数资源需求的精度是 0.0001，所以你应该避免指定超出这个精度的双精度。

.. literalinclude:: ../doc_code/resources.py
    :language: python
    :start-after: __specifying_fractional_resource_requirements_start__
    :end-before: __specifying_fractional_resource_requirements_end__

.. tip::

  除了资源需求，你还可以为任务或 actor 指定一个运行环境，其中可以包括 Python 包、本地文件、环境变量等，详情请参见 :ref:`运行时环境 <runtime-environments>`。
