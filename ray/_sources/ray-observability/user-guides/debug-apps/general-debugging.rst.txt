.. _observability-general-debugging:

一般调试
=======================

分布式应用程序比非分布式应用程序更强大但更复杂。 Ray 的某些行为可能会让用户措手不及，而这些设计选择可能有合理的论据。

此页面列出了用户可能遇到的一些常见问题。特别是，用户认为 Ray 在本地计算机上运行，​​虽然有时确实如此，但这会导致很多问题。

环境变量不会从Driver进程传递到Worker进程
---------------------------------------------------------------------------------

**问题**: 如果您在命令行（运行驱动程序的位置）设置环境变量，并且集群之前已启动，则该环境变量不会传递给集群中运行的所有工作线程。

**示例**: 如果运行 Ray 的目录中有一个文件 ``baz.py`` ，则运行以下命令：

.. literalinclude:: /ray-observability/doc_code/gotchas.py
  :language: python
  :start-after: __env_var_start__
  :end-before: __env_var_end__

**预期行为**: 大多数人会期望（就好像它是一台机器上的单个进程一样）所有 Worker 中的环境变量都是相同的。不会是这样的。

**修复**: 使用运行时环境显式传递环境变量。
如果您调用 ``ray.init(runtime_env=...)``，
那么 Workers 将设置环境变量。


.. literalinclude:: /ray-observability/doc_code/gotchas.py
  :language: python
  :start-after: __env_var_fix_start__
  :end-before: __env_var_fix_end__


文件名有时有效，有时无效
-----------------------------------------------

**问题**: 如果您在任务或参与者中按名称引用文件，它有时会起作用，有时会失败。
这是因为如果任务或 actor 运行在集群的头节点上，它就会工作，
但如果任务或 actor 运行在另一台机器上，它就不会工作。

**示例**: 假设我们执行以下命令：

.. code-block:: bash

	% touch /tmp/foo.txt

我有这个代码：

.. testcode::

  import os
  import ray

  @ray.remote
  def check_file():
    foo_exists = os.path.exists("/tmp/foo.txt")
    return foo_exists

  futures = []
  for _ in range(1000):
    futures.append(check_file.remote())

  print(ray.get(futures))


那么你会得到 True 和 False 的混合结果。 如果
``check_file()`` 在头节点上运行，或者我们在本地运行，它就可以工作。
但如果它在工作节点上运行，则会返回 ``False``。

**预期行为**: M大多数人会期望这要么失败要么成功。毕竟是相同的代码。

**修复**

- 仅对此类应用程序使用共享路径 - 例如，如果您使用网络文件系统，则可以使用该系统，或者文件可以位于 S3 上。
- 不要依赖文件路径一致性。



占位组不可组合
-----------------------------------

**问题**: 若有一个从占位组进行调用的任务，那么资源永远不会被分配并会挂起。

**示例**: 你正在使用 Ray Tune 创建占位组，并希望将其应用于目标函数，但是该目标函数使用了 Ray Task 本身，例如：

.. testcode::

  import ray
  from ray import tune

  def create_task_that_uses_resources():
    @ray.remote(num_cpus=10)
    def sample_task():
      print("Hello")
      return

    return ray.get([sample_task.remote() for i in range(10)])

  def objective(config):
    create_task_that_uses_resources()

  tuner = tune.Tuner(objective, param_space={"a": 1})
  tuner.fit()

这将出错并显示信息：

.. testoutput::
  :options: +MOCK

    ValueError: Cannot schedule create_task_that_uses_resources.<locals>.sample_task with the placement group
    because the resource request {'CPU': 10} cannot fit into any bundles for the placement group, [{'CPU': 1.0}].

**预期行为**: 以上会进行执行。

**修复**: 在 ``create_task_that_uses_resources()`` 调用的 ``@ray.remote`` 声明中，包含
``scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)``。

.. code-block:: diff

  def create_task_that_uses_resources():
  +     @ray.remote(num_cpus=10, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None))
  -     @ray.remote(num_cpus=10)

过期的函数定义
-----------------------------

由于 Python 的微妙，如果你重新定义远程函数，你可能并不总能获得预期的行为。在这种情况下， Ray 可能没有运行该函数的最新版本。

假设你定义了一个远程函数 ``f`` ，然后重新定义它。Ray 应该运行最新的版本。

.. testcode::

  import ray

  @ray.remote
  def f():
      return 1

  @ray.remote
  def f():
      return 2

  print(ray.get(f.remote()))  # This should be 2.

.. testoutput::

  2

但是，以下情况 Ray 不会将远程函数修改为最新版本（至少在不停止并重新启动 Ray 的情况下）。

- **该函数作为外部文件导入:** 本例中，
  ``f`` 在外部文件 ``file.py`` 定义。如果你 ``import file``，
  在 ``file.py`` 修改 ``f`` 定义，然后重新 ``import file``，
  函数 ``f`` 不会更新。

  因为第二个导入被是为无操作并被忽略，``f`` 仍由第一个文件导入定义。

  解决方案是使用 ``reload(file)`` 来代替 ``import file``。重新加载会重新执行新的定义，并将其
  导出到其他机器。请注意，在 Python 3 中，你需要执行 ``from importlib import reload``。

- **该函数依赖于外部文件中的辅助函数：**
  在这种情况下， ``f`` 可以在 Ray 应用程序中定义，但依赖某些外部文件 ``file.py`` 定义的辅助
  函数 ``h`` 。如果 ``file.py`` 中的 ``h`` 定义发生了变化，重新定义的 ``f`` 不会更新 Ray 使用最新版本的 ``h``。

  因为 ``f`` 最初定义之后，它的定义会被发送到所有工作进程，并不会被 unpickled。在 unpick 期间，
  ``file.py`` 被导入到 worker 。然后当 ``f`` 被重新定义，定义重新被所有 worker 并 unpickled。
  但是 ``file.py`` 已经被导入，会作为第二次导入并忽略来对待。

  不幸的是，重新加载 driver 不会更新 ``h``，重新加载需要在 worker 上进行。

  A solution to this problem is to redefine ``f`` to reload ``file.py`` before
  it calls ``h``. For example, if inside ``file.py`` you have
  解决这个问题的方式是在 ``f`` 调用 ``h`` 之前重新加载 ``file.py``。例如，在 ``file.py`` 中

  .. testcode::

    def h():
        return 1

  远程方法 ``f`` 定义

  .. testcode::

    @ray.remote
    def f():
        return file.h()

  按照如下重新定义 ``f`` 。

  .. testcode::

    @ray.remote
    def f():
        reload(file)
        return file.h()

  这会强制 worker 按照需要机型重新加载。注意，在 Python 3 在，你需要 ``from importlib import reload`` 。

本文档讨论了人们在使用 Ray 时遇到的一些常见问题以及一些已知问题。如果您遇到其他问题， `请告知我们`_ 。

.. _`let us know`: https://github.com/ray-project/ray/issues
