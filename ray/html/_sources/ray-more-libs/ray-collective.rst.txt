..
  This part of the docs is generated from the ray.util.collective readme using m2r
  To update:
  - run `m2r RAY_ROOT/python/ray/util/collective/README.md`
  - copy the contents of README.rst here
  - Be sure not to delete the API reference section in the bottom of this file.

.. _ray-collective:

Ray 集体通信库
================================

Ray 集体通信库(\ ``ray.util.collective``\ ) 提供了一组原生的集体原语，用于在分布式 CPU 或 GPU 之间进行通信。

Ray 集体通信库


* 使 Ray actor 和 task 进程之间的集体通信效率提高了 10 倍，
* 可在分布式 CPU 和 GPU 上运行，
* 使用 NCCL 和 GLOO 作为可选的高性能通信后端，
* 适用于 Ray 上的分布式 ML 程序。

集体原语支持矩阵
------------------------------------

查看下面的支持矩阵，了解不同后端的所有集体调用的当前支持情况。

.. list-table::
   :header-rows: 1

   * - Backend
     - `gloo <https://github.com/ray-project/pygloo>`_
     -
     - `nccl <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`_
     -
   * - Device
     - CPU
     - GPU
     - CPU
     - GPU
   * - send
     - ✔
     - ✘
     - ✘
     - ✔
   * - recv
     - ✔
     - ✘
     - ✘
     - ✔
   * - broadcast
     - ✔
     - ✘
     - ✘
     - ✔
   * - allreduce
     - ✔
     - ✘
     - ✘
     - ✔
   * - reduce
     - ✔
     - ✘
     - ✘
     - ✔
   * - allgather
     - ✔
     - ✘
     - ✘
     - ✔
   * - gather
     - ✘
     - ✘
     - ✘
     - ✘
   * - scatter
     - ✘
     - ✘
     - ✘
     - ✘
   * - reduce_scatter
     - ✔
     - ✘
     - ✘
     - ✔
   * - all-to-all
     - ✘
     - ✘
     - ✘
     - ✘
   * - barrier
     - ✔
     - ✘
     - ✘
     - ✔


支持的张量类型
----------------------


* ``torch.Tensor``
* ``numpy.ndarray``
* ``cupy.ndarray``

用法
-----

安装及引用
^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray 集体库与发布的 Ray 轮包捆绑在一起。除了 Ray 之外，用户还需要安装 `pygloo <https://github.com/ray-project/pygloo>`_
或 `cupy <https://docs.cupy.dev/en/stable/install.html>`_ 以便使用 GLOO 和 NCCL 后端的集体通信。

.. code-block:: python

   pip install pygloo
   pip install cupy-cudaxxx # replace xxx with the right cuda version in your environment

要使用这些 APIs，请通过以下方式在 actor/task 或 driver 代码中导入 collective 包：

.. code-block:: python

   import ray.util.collective as col

初始化
^^^^^^^^^^^^^^

集合函数在集合组上运行。
集合组包含一组进程（在 Ray 中，它们通常是 Ray 管理的 actor 或 task），这些进程将一起进入集合函数调用。
在进行集体调用之前，用户需要将一组 actor/task 静态地声明为一个集体组。

以下是一个示例代码片段，使用两个 API ``init_collective_group()`` 和 ``declare_collective_group()`` 在几个远程 actor 之间初始化集体组。
参考 `APIs <#api-reference>`_ 以获取这两个 API 的详细描述。

.. code-block:: python

   import ray
   import ray.util.collective as collective

   import cupy as cp


   @ray.remote(num_gpus=1)
   class Worker:
      def __init__(self):
          self.send = cp.ones((4, ), dtype=cp.float32)
          self.recv = cp.zeros((4, ), dtype=cp.float32)

      def setup(self, world_size, rank):
          collective.init_collective_group(world_size, rank, "nccl", "default")
          return True

      def compute(self):
          collective.allreduce(self.send, "default")
          return self.send

      def destroy(self):
          collective.destroy_group()

   # imperative
   num_workers = 2
   workers = []
   init_rets = []
   for i in range(num_workers):
      w = Worker.remote()
      workers.append(w)
      init_rets.append(w.setup.remote(num_workers, i))
   _ = ray.get(init_rets)
   results = ray.get([w.compute.remote() for w in workers])


   # declarative
   for i in range(num_workers):
      w = Worker.remote()
      workers.append(w)
   _options = {
      "group_name": "177",
      "world_size": 2,
      "ranks": [0, 1],
      "backend": "nccl"
   }
   collective.declare_collective_group(workers, **_options)
   results = ray.get([w.compute.remote() for w in workers])

注意，对于相同的 actor/task 进程集合，可以构建多个集合组，其中 ``group_name`` 是它们的唯一标识符。
这使得可以在不同（子）进程集之间指定复杂的通信模式。

集合通信
^^^^^^^^^^^^^^^^^^^^^^^^

检查 `支持矩阵 <#collective-primitives-support-matrix>`_ 以了解支持的集体调用和后端的当前状态。

注意，当前的集体通信 API 是命令式的，并表现出以下行为：


* 所有的集体 API 都是同步阻塞调用
* 由于每个 API 仅指定集体通信的一部分，因此预计该 API 将由（预先声明的）集体组的每个参与进程调用。当所有进程都进行了调用并相互会合后，集体通信就会发生并继续进行。
* API 是命令式的，并且通信发生在带外 —— 它们需要在集体流程（ actor /任务）代码内使用。

一个使用 ``ray.util.collective.allreduce`` 的示例如下：

.. code-block:: python

   import ray
   import cupy
   import ray.util.collective as col


   @ray.remote(num_gpus=1)
   class Worker:
       def __init__(self):
           self.buffer = cupy.ones((10,), dtype=cupy.float32)

       def compute(self):
           col.allreduce(self.buffer, "default")
           return self.buffer

   # Create two actors A and B and create a collective group following the previous example...
   A = Worker.remote()
   B = Worker.remote()
   # Invoke allreduce remotely
   ray.get([A.compute.remote(), B.compute.remote()])

点对点通信
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ray.util.collective`` 也提供了进程之间的 P2P 发送/接收通信。

send/recv 与集体函数表现出相同的行为：
它们是同步阻塞调用 - 必须在成对进程上一起调用一对 send 和 receive，以便指定整个通信，
并且必须成功地彼此会合才能继续。请参阅下面的代码示例：

.. code-block:: python

   import ray
   import cupy
   import ray.util.collective as col


   @ray.remote(num_gpus=1)
   class Worker:
       def __init__(self):
           self.buffer = cupy.ones((10,), dtype=cupy.float32)

       def get_buffer(self):
           return self.buffer

       def do_send(self, target_rank=0):
           # this call is blocking
           col.send(target_rank)

       def do_recv(self, src_rank=0):
           # this call is blocking
           col.recv(src_rank)

       def do_allreduce(self):
           # this call is blocking as well
           col.allreduce(self.buffer)
           return self.buffer

   # Create two actors
   A = Worker.remote()
   B = Worker.remote()

   # Put A and B in a collective group
   col.declare_collective_group([A, B], options={rank=[0, 1], ...})

   # let A to send a message to B; a send/recv has to be specified once at each worker
   ray.get([A.do_send.remote(target_rank=1), B.do_recv.remote(src_rank=0)])

   # An anti-pattern: the following code will hang, because it doesn't instantiate the recv side call
   ray.get([A.do_send.remote(target_rank=1)])

单 GPU 和多 GPU 集体基元
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在许多集群设置中，一台机器通常具有多个 GPU；
有效利用GPU-GPU带宽，例如 `NVLINK <https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/>`_\ ，
可以显着提高通信性能。

``ray.util.collective`` 支持多GPU集体调用，在这种情况下，一个进程 (actor/tasks) 管理超过1个GPU（例如，通过 ``ray.remote(num_gpus=4)``\ ）。
使用这些多 GPU 集体函数通常比使用单 GPU 集体 API 更具性能优势，
并且生成的进程数量等于 GPU 数量。
请参阅 API 参考，了解多 GPU 集体 API 的签名。

另请注意，所有多 GPU API 均具有以下限制：


* 仅支持 NCCL 后端。
* 进行多GPU集体或P2P调用的集体进程需要拥有相同数量的GPU设备。
* 多 GPU 集体函数的输入通常是张量列表，每个张量位于调用者进程拥有的不同 GPU 设备上。

下面提供了利用多 GPU 集体 API 的示例代码：

.. code-block:: python

   import ray
   import ray.util.collective as collective

   import cupy as cp
   from cupy.cuda import Device


   @ray.remote(num_gpus=2)
   class Worker:
      def __init__(self):
          with Device(0):
              self.send1 = cp.ones((4, ), dtype=cp.float32)
          with Device(1):
              self.send2 = cp.ones((4, ), dtype=cp.float32) * 2
          with Device(0):
              self.recv1 = cp.ones((4, ), dtype=cp.float32)
          with Device(1):
              self.recv2 = cp.ones((4, ), dtype=cp.float32) * 2

      def setup(self, world_size, rank):
          self.rank = rank
          collective.init_collective_group(world_size, rank, "nccl", "177")
          return True

      def allreduce_call(self):
          collective.allreduce_multigpu([self.send1, self.send2], "177")
          return [self.send1, self.send2]

      def p2p_call(self):
          if self.rank == 0:
             collective.send_multigpu(self.send1 * 2, 1, 1, "8")
          else:
             collective.recv_multigpu(self.recv2, 0, 0, "8")
          return self.recv2

   # Note that the world size is 2 but there are 4 GPUs.
   num_workers = 2
   workers = []
   init_rets = []
   for i in range(num_workers):
      w = Worker.remote()
      workers.append(w)
      init_rets.append(w.setup.remote(num_workers, i))
   a = ray.get(init_rets)
   results = ray.get([w.allreduce_call.remote() for w in workers])
   results = ray.get([w.p2p_call.remote() for w in workers])

更多资源
--------------

以下链接提供了有关如何有效利用 ``ray.util.collective`` 库的有用资源。


* 更多运行在 ``ray.util.collective.examples`` 的 `示例 <https://github.com/ray-project/ray/tree/master/python/ray/util/collective/examples>`_ 。
* 使用 Ray 集体库 `扩展 Spacy 名称实体识别 (NER) 管道 <https://github.com/explosion/spacy-ray>`_ 。
* `实现了 AllReduce 策略 <https://github.com/ray-project/distml/blob/master/distml/strategy/allreduce_strategy.py>`_ 的数据并行分布式机器学习训练。

API 参考
--------------

.. automodule:: ray.util.collective.collective
    :members:
