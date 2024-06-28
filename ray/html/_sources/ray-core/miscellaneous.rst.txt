杂项主题
====================

本页将介绍 Ray 中的一些杂项主题。

.. contents::
  :local:

动态远程参数
-------------------------

你可以在执行过程中使用 ``.options`` 动态调整 ``ray.remote`` 的资源需求或返回值。

例如，我们实例化了多个具有不同资源需求的相同 actor。请注意，要成功创建这些 actor，Ray 需要足够的 CPU 资源和相关的自定义资源：

.. testcode::

  import ray

  @ray.remote(num_cpus=4)
  class Counter(object):
      def __init__(self):
          self.value = 0

      def increment(self):
          self.value += 1
          return self.value

  a1 = Counter.options(num_cpus=1, resources={"Custom1": 1}).remote()
  a2 = Counter.options(num_cpus=2, resources={"Custom2": 1}).remote()
  a3 = Counter.options(num_cpus=3, resources={"Custom3": 1}).remote()

您可以为任务指定不同的资源需求（但不能为 actor 方法指定）：

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    ray.init(num_cpus=1, num_gpus=1)

    @ray.remote
    def g():
        return ray.get_gpu_ids()

    object_gpu_ids = g.remote()
    assert ray.get(object_gpu_ids) == []

    dynamic_object_gpu_ids = g.options(num_cpus=1, num_gpus=1).remote()
    assert ray.get(dynamic_object_gpu_ids) == [0]

并改变任务（以及 actor 方法）的返回值的数量：

.. testcode::

    @ray.remote
    def f(n):
        return list(range(n))

    id1, id2 = f.options(num_returns=2).remote(2)
    assert ray.get(id1) == 0
    assert ray.get(id2) == 1

并在提交任务时指定任务（以及 actor 方法）的名称：

.. testcode::

   import setproctitle

   @ray.remote
   def f(x):
      assert setproctitle.getproctitle() == "ray::special_f"
      return x + 1

   obj = f.options(name="special_f").remote(3)
   assert ray.get(obj) == 4

该名称将出现在仪表板的机器视图中的任务名称中，
将在执行此任务时作为工作进程名称出现（如果是 Python 任务），
并将出现在日志中的任务名称中。

.. image:: images/task_name_dashboard.png


重载函数
--------------------
Ray Java API 支持远程调用重载的 Java 函数。但是，由于 Java 编译器类型推断的限制，必须将方法引用显式转换为正确的函数类型。例如，考虑以下内容。

重载正常任务调用：

.. code:: java

    public static class MyRayApp {

      public static int overloadFunction() {
        return 1;
      }

      public static int overloadFunction(int x) {
        return x;
      }
    }

    // Invoke overloaded functions.
    Assert.assertEquals((int) Ray.task((RayFunc0<Integer>) MyRayApp::overloadFunction).remote().get(), 1);
    Assert.assertEquals((int) Ray.task((RayFunc1<Integer, Integer>) MyRayApp::overloadFunction, 2).remote().get(), 2);

重载 Actor 任务调用：

.. code:: java

    public static class Counter {
      protected int value = 0;

      public int increment() {
        this.value += 1;
        return this.value;
      }
    }

    public static class CounterOverloaded extends Counter {
      public int increment(int diff) {
        super.value += diff;
        return super.value;
      }

      public int increment(int diff1, int diff2) {
        super.value += diff1 + diff2;
        return super.value;
      }
    }

.. code:: java

    ActorHandle<CounterOverloaded> a = Ray.actor(CounterOverloaded::new).remote();
    // Call an overloaded actor method by super class method reference.
    Assert.assertEquals((int) a.task(Counter::increment).remote().get(), 1);
    // Call an overloaded actor method, cast method reference first.
    a.task((RayFunc1<CounterOverloaded, Integer>) CounterOverloaded::increment).remote();
    a.task((RayFunc2<CounterOverloaded, Integer, Integer>) CounterOverloaded::increment, 10).remote();
    a.task((RayFunc3<CounterOverloaded, Integer, Integer, Integer>) CounterOverloaded::increment, 10, 10).remote();
    Assert.assertEquals((int) a.task(Counter::increment).remote().get(), 33);

检查集群状态
------------------------

在 Ray 上编写的应用程序通常需要获取有关集群的一些信息或诊断信息。
一些常见问题包括：

    1. 我的自动扩展集群中有多少个节点？
    2. 我的集群中当前可用的资源有哪些（已使用资源和总计资源）？
    3. 我的集群中当前有哪些对象？

为此，您可以使用全局状态 API。

节点信息
~~~~~~~~~~~~~~~~

要获取有关集群中当前节点的信息，您可以使用 ``ray.nodes()``：

.. autofunction:: ray.nodes
    :noindex:

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import ray

    ray.init()
    print(ray.nodes())

.. testoutput::
  :options: +MOCK

    [{'NodeID': '2691a0c1aed6f45e262b2372baf58871734332d7',
      'Alive': True,
      'NodeManagerAddress': '192.168.1.82',
      'NodeManagerHostname': 'host-MBP.attlocal.net',
      'NodeManagerPort': 58472,
      'ObjectManagerPort': 52383,
      'ObjectStoreSocketName': '/tmp/ray/session_2020-08-04_11-00-17_114725_17883/sockets/plasma_store',
      'RayletSocketName': '/tmp/ray/session_2020-08-04_11-00-17_114725_17883/sockets/raylet',
      'MetricsExportPort': 64860,
      'alive': True,
      'Resources': {'CPU': 16.0, 'memory': 100.0, 'object_store_memory': 34.0, 'node:192.168.1.82': 1.0}}]

上述信息包括：

  - `NodeID`: raylet 的唯一标识符。
  - `alive`: 节点是否还存活。
  - `NodeManagerAddress`: raylet 所在节点的私有 IP。
  - `Resources`: 节点上资源总容量。
  - `MetricsExportPort`: 通过 `Prometheus 端点 <ray-metrics.html>`_ 公开指标的端口号。

资源信息
~~~~~~~~~~~~~~~~~~~~

要获取有关集群当前总资源容量的信息，您可以使用 ``ray.cluster_resources()``。

.. autofunction:: ray.cluster_resources
    :noindex:


要获取有关集群当前可用资源容量的信息，您可以使用 ``ray.available_resources()``。

.. autofunction:: ray.available_resources
    :noindex:

运行大型 Ray 集群
--------------------------

以下是运行超过 1000 个节点的 Ray 的一些技巧。
当运行具有如此大量节点的 Ray 时，可能需要调整几个系统设置，
以实现如此大量机器之间的通信。

调整操作系统设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由于所有节点和工作程序都连接到 GCS，因此将创建许多网络连接，并且操作系统必须支持该数量的连接。

最大打开文件数
******************

由于每个 worker 和 raylet 都连接到 GCS，因此必须配置 OS 以支持打开多个 TCP 连接。
在 POSIX 系统中，可以通过 ``ulimit -n`` 检查当前限制，
如果限制较小，则应根据 OS 手册增加限制。

ARP 缓存
*********

需要配置的另一件事是 ARP 缓存。
在大型集群中，所有 worker 节点都连接到头节点，这会向 ARP 表添加大量条目。
确保 ARP 缓存大小足够大以处理这么多节点。不这样做将导致头节点挂起。
发生这种情况时，
``dmesg`` 将显示类似 ``neighbor table overflow message`` 的错误。

在 Ubuntu 中，可以通过增加 ``/etc/sysctl.conf`` 中 ``net.ipv4.neigh.default.gc_thresh1`` - ``net.ipv4.neigh.default.gc_thresh3`` 的值来调整 ARP 缓存大小。
有关更多详细信息，请参阅操作系统手册。

调整 Ray 设置
~~~~~~~~~~~~~~~~~~~

.. note::
  目前正在进行的 `project <https://github.com/ray-project/ray/projects/15>`_ 致力于
  提高 Ray 的可扩展性和稳定性。欢迎分享您的想法和用例。

要运行大型集群，需要在 Ray 中调整几个参数。

Resource 广播
*********************

在 Ray 2.3+ 版本中，轻量级资源广播作为一项实验性功能被支持。
开启该功能可以显著降低 GCS 负载，从而提高其整体稳定性和可扩展性。
要开启该功能，需要设置以下 OS 环境： ``RAY_use_ray_syncer=true``。
此功能将在 2.4+ 版本中默认开启。

基准
~~~~~~~~~

机器设置：

- 1 个头节点：m5.4xlarge（16 个 vCPU/64GB 内存）
- 2000 个 worker 节点：m5.large（2 个 vCPU/8GB 内存）

操作系统设置：

- 设置最大打开文件数为1048576
- 增加 ARP 缓存大小：
    - ``net.ipv4.neigh.default.gc_thresh1=2048``
    - ``net.ipv4.neigh.default.gc_thresh2=4096``
    - ``net.ipv4.neigh.default.gc_thresh3=8192``


Ray 设置：

- ``RAY_use_ray_syncer=true``
- ``RAY_event_stats=false``

测试工作量：

- Test 脚本: `代码 <https://github.com/ray-project/ray/blob/master/release/benchmarks/distributed/many_nodes_tests/actor_test.py>`_



.. list-table:: Benchmark result
   :header-rows: 1

   * - actor 数量
     - Actor 上线时间
     - Actor 准备时间
     - 总时间
   * - 20k (10 actors / node)
     - 14.5s
     - 136.1s
     - 150.7s
