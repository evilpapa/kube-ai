.. _ray-multiprocessing:

分布式 multiprocessing.Pool
================================

.. _`GitHub 上的讨论`: https://github.com/ray-project/ray/issues

Ray 支持使用 `Ray Actors <actors.html>`__ 的 `multiprocessing.Pool API`_ 运行分布式 Python 程序而不是本地进程来运行。这使得将现有 ``multiprocessing.Pool`` 应用程序从单节点扩展到集群变得容易。

.. _`multiprocessing.Pool API`: https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool

快速入门
----------

要开始，首先 `安装 Ray <installation.html>`__，
然后使用 ``ray.util.multiprocessing.Pool`` 代替 ``multiprocessing.Pool``。
这将在第一次创建 ``Pool`` 时启动一个本地 Ray 集群，并在集群中分发您的任务。
请参见下面的 `在集群上运行`_ 部分，了解如何在多节点 Ray 集群上运行。

.. code-block:: python

  from ray.util.multiprocessing import Pool

  def f(index):
      return index

  pool = Pool()
  for result in pool.map(f, range(100)):
      print(result)

完整的 ``multiprocessing.Pool`` API 目前受支持。有关详细信息，请参见 `multiprocessing 文档`_。

.. warning::
  在 ``Pool`` 构造器中的 ``context`` 参数在使用 Ray 时会被忽略。

.. _`multiprocessing 文档`: https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool

在集群上运行
----------------

本章节假设您已经有一个运行中的 Ray 集群。要启动一个 Ray 集群，请参考 `集群设置 <cluster/index.html>`__ 说明。

要在 ``Pool`` 中连接到运行中的 Ray 集群，可以通过以下两种方式之一指定 head 节点的地址：

- 通过设置 ``RAY_ADDRESS`` 环境变量。
- 通过设置 ``ray_address`` 关键字参数到 ``Pool`` 构造器。

.. code-block:: python

  from ray.util.multiprocessing import Pool

  # Starts a new local Ray cluster.
  pool = Pool()

  # Connects to a running Ray cluster, with the current node as the head node.
  # Alternatively, set the environment variable RAY_ADDRESS="auto".
  pool = Pool(ray_address="auto")

  # Connects to a running Ray cluster, with a remote node as the head node.
  # Alternatively, set the environment variable RAY_ADDRESS="<ip_address>:<port>".
  pool = Pool(ray_address="<ip_address>:<port>")

你也可以在创建 ``Pool`` 之前手动调用 ``ray.init()`` (使用任何支持的配置选项) 来启动 Ray。
