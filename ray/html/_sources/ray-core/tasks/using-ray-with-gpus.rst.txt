.. _gpu-support:

GPU 支持
===========

GPU 对于许多机器学习应用程序至关重要。
Ray 远程支持 GPU 作为预定义的 :ref:`resource <core-resources>` 类型，并允许任务和 actor 指定它们的 GPU :ref:`资源需求 <resource-requirements>`。

启动 GPU 的 Ray 节点
----------------------------

默认的，Ray 会将节点的 GPU 资源数量设置为 Ray 自动检测到的物理 GPU 数量。
如果需要，你可以 :ref:`覆盖 <specify-node-resources>` 这个设置。

.. note::

  没什么可以阻止你在机器上指定一个比真实 GPU 数量更大的 ``num_gpus`` 值，因为 Ray 资源是 :ref:`逻辑的 <logical-resources>`。
  在此情况下，Ray 会表现的好像机器上有你指定的 GPU 数量一样，用于调度需要 GPU 的任务和 actor。
  只有当这些任务和 actor 尝试使用不存在的 GPU 时才会出现问题。

.. tip::

  你可以在启动 Ray 节点之前设置 ``CUDA_VISIBLE_DEVICES`` 环境变量来限制 Ray 可见的 GPU。
  比如，``CUDA_VISIBLE_DEVICES=1,3 ray start --head --num-gpus=2`` 将只让 Ray 看到设备 1 和 3。

在任务和 actor 中使用 GPU
------------------------------

如果一个任务或 actor 需要 GPU，你可以指定相应的 :ref:`资源需求 <resource-requirements>`（比如 ``@ray.remote(num_gpus=1)``）。
Ray 然后会将任务或 actor 调度到一个有足够空闲 GPU 资源的节点上，
并在运行任务或 actor 代码之前通过设置 ``CUDA_VISIBLE_DEVICES`` 环境变量来分配 GPU。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __get_gpu_ids_start__
    :end-before: __get_gpu_ids_end__

在任务或 actor 中，:func:`ray.get_gpu_ids() <ray.get_gpu_ids>` 会返回一个可用于任务或 actor 的 GPU ID 列表。
通常，不需要调用 ``ray.get_gpu_ids()``，因为 Ray 会自动设置 ``CUDA_VISIBLE_DEVICES`` 环境变量，
这在大多数机器学习框架中都会被用于 GPU 分配。

**注意：** 上面定义的 ``use_gpu``  函数实际上并没有使用任何 GPU。Ray 会将其调度到至少有一个 GPU 的节点上，
并在其执行时为其保留一个 GPU，但实际上使用 GPU 是由函数自己决定的。
这通常通过外部库（比如 TensorFlow）来完成。下面是一个实际使用 GPU 的例子。
为了使这个例子工作，你需要安装 TensorFlow 的 GPU 版本。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __gpus_tf_start__
    :end-before: __gpus_tf_end__

**注意：** 使用了 ``use_gpu`` 的人员可以忽略 ``ray.get_gpu_ids()`` 并使用机器上的所有 GPU。
Ray 不会阻止这种情况发生，这可能导致太多任务或 actor 同时使用同一个 GPU。然而，Ray 会自动设置 ``CUDA_VISIBLE_DEVICES`` 环境变量，
它会限制大多数深度学习框架使用的 GPU，假设用户没有覆盖它。

分数 GPU
---------------

Ray 支持 :ref:`分数资源需求 <fractional-resource-requirements>`，这样多个任务和 actor 可以共享同一个 GPU。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __fractional_gpus_start__
    :end-before: __fractional_gpus_end__

**注意：** 用户需要确保单个任务不会使用超过其 GPU 内存份额。
TensorFlow 可以配置为限制其内存使用量。

当 Ray 将节点的 GPU 分配给具有分数资源需求的任务或 actor 时，它会先分配一个 GPU，然后再分配下一个 GPU，以避免碎片化。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __fractional_gpus_packing_start__
    :end-before: __fractional_gpus_packing_end__

.. _gpu-leak:

Worker 不释放 GPU 资源
-----------------------------------

目前，当 worker 执行使用 GPU 的任务（比如，通过 TensorFlow）时，任务可能会在 GPU 上分配内存，并且在任务执行完毕后可能不会释放它。
这可能会导致下次任务尝试使用相同的 GPU 时出现问题。为了解决这个问题，Ray 默认情况下禁用了 GPU 任务之间的 worker 进程重用，其中 GPU 资源在任务进程退出后被释放。
由于这会增加 GPU 任务调度的开销，你可以通过在 :func:`ray.remote <ray.remote>` 装饰器中设置 ``max_calls=0`` 来重新启用 worker 重用。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __leak_gpus_start__
    :end-before: __leak_gpus_end__

.. _accelerator-types:

加速器类型
-----------------

Ray 支持特定资源的加速器类型。``accelerator_type`` 选项可以用于强制任务或 actor 在具有特定类型加速器的节点上运行。
在底层，加速器类型选项被实现为 ``"accelerator_type:<type>": 0.001`` 的 :ref:`自定义资源需求 <custom-resources>`。
这会强制任务或 actor 被放置在具有该特定加速器类型的节点上。
这还让多节点类型自动缩放器知道有需求的资源类型，可能会触发提供该加速器的新节点的启动。

.. literalinclude:: ../doc_code/gpus.py
    :language: python
    :start-after: __accelerator_type_start__
    :end-before: __accelerator_type_end__

参考 ``ray.util.accelerators`` 获取可用的加速器类型。当前自动检测到的加速器类型包括：

 - Nvidia GPU
 - AWS Neuron Cores

AWS Neuron Core 加速器（实验性）
------------------------------------------

类似于 Nvidia GPU，Ray 默认情况下会自动检测 `AWS Neuron Cores`_。
用户可以在任务或 actor 的资源需求中指定 `resources={"neuron_cores": some_number}` 来分配 Neuron Core(s)。

.. _`AWS Neuron Cores` : https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/model-architecture-fit.html

.. note::

  Ray 支持异构的 GPU 和 Neuron Core 集群，但不允许为任务或 actor 指定 ``num_gpus`` 和 ``neuron_cores`` 的资源需求。

.. literalinclude:: ../doc_code/neuron_core_accelerator.py
    :language: python
    :start-after: __neuron_core_accelerator_start__
    :end-before: __neuron_core_accelerator_end__
