启动 Ray
============

本页介绍如何在单台机器或机器集群上启动 Ray。

.. tip:: 在按照本页上的说明进行操作之前，请确保已 :ref:`安装 Ray <installation>` 。


Ray 运行时是什么？
------------------------

Ray 程序通过利用底层的 *Ray 运行时* 来并行化和分发。
Ray 运行时由在后台启动的多个服务/进程组成，用于通信、数据传输、调度等。Ray 运行时可以在笔记本电脑、单台服务器或多台服务器上启动。

有三种方式启动 Ray 运行时：

* 隐式通过 ``ray.init()`` (:ref:`start-ray-init`)
* 通过 CLI 明确启动 (:ref:`start-ray-cli`)
* 通过集群启动器明确启动 (:ref:`start-ray-up`)

在所有情况下， ``ray.init()`` 都会尝试自动找到要连接的 Ray 实例。
它会按顺序检查：
1. ``RAY_ADDRESS`` 操作系统环境变量。
2. 传递给 ``ray.init(address=<address>)`` 的具体地址。
3. 如果没有提供地址，则使用 ``ray start`` 在同一台机器上启动的最新 Ray 实例。

.. _start-ray-init:

在单台机器上启动 Ray
--------------------------------

调用 ``ray.init()`` 将在您的笔记本电脑/机器上启动本地 Ray 实例。此笔记本电脑/机器将成为“头节点”。

.. note::

  在 Ray 的最新版本（>=1.5）中，第一次使用 Ray 远程 API 时 ``ray.init()`` 会自动调用。

.. tab-set::

    .. tab-item:: Python

        .. testcode::
          :hide:

          import ray
          ray.shutdown()

        .. testcode::

          import ray
          # Other Ray APIs will not work until `ray.init()` is called.
          ray.init()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

              public static void main(String[] args) {
                // Other Ray APIs will not work until `Ray.init()` is called.
                Ray.init();
                ...
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>
            // Other Ray APIs will not work until `ray::Init()` is called.
            ray::Init()

当 ``ray.init()`` 进程调用终止时，Ray 运行时也将终止。要显式停止或重新启动 Ray，请使用 shutdown API。

.. tab-set::

    .. tab-item:: Python

        .. testcode::
          :hide:

          ray.shutdown()

        .. testcode::

            import ray
            ray.init()
            ... # ray program
            ray.shutdown()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

              public static void main(String[] args) {
                Ray.init();
                ... // ray program
                Ray.shutdown();
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>
            ray::Init()
            ... // ray program
            ray::Shutdown()

要检查 Ray 是否已初始化，请使用 ``is_initialized`` API。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray
            ray.init()
            assert ray.is_initialized()

            ray.shutdown()
            assert not ray.is_initialized()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

            public static void main(String[] args) {
                    Ray.init();
                    Assert.assertTrue(Ray.isInitialized());
                    Ray.shutdown();
                    Assert.assertFalse(Ray.isInitialized());
                }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>

            int main(int argc, char **argv) {
                ray::Init();
                assert(ray::IsInitialized());

                ray::Shutdown();
                assert(!ray::IsInitialized());
            }

参考 `配置 <configure.html>`__ 文档了解配置 Ray 的各种方式。

.. _start-ray-cli:

通过 CLI 启动 Ray (``ray start``)
----------------------------------------

在 CLI 中使用 ``ray start`` 启动一个 Ray 运行时节点。这个节点将成为“头节点”。

.. code-block:: bash

  $ ray start --head --port=6379

  Local node IP: 192.123.1.123
  2020-09-20 10:38:54,193 INFO services.py:1166 -- View the Ray dashboard at http://localhost:8265

  --------------------
  Ray runtime started.
  --------------------

  ...


你可以通过在与 ``ray start`` 同一节点上启动驱动程序进程来连接到此 Ray 实例。
``ray.init()`` 会自动连接到最新的 Ray 实例。

.. tab-set::

    .. tab-item:: Python

      .. testcode::

        import ray
        ray.init()

    .. tab-item:: java

        .. code-block:: java

          import io.ray.api.Ray;

          public class MyRayApp {

            public static void main(String[] args) {
              Ray.init();
              ...
            }
          }

        .. code-block:: bash

          java -classpath <classpath> \
            -Dray.address=<address> \
            <classname> <args>

    .. tab-item:: C++

        .. code-block:: c++

          #include <ray/api.h>

          int main(int argc, char **argv) {
            ray::Init();
            ...
          }

        .. code-block:: bash

          RAY_ADDRESS=<address> ./<binary> <args>


你可以通过在其他节点上调用 ``ray start`` 来连接到头节点，从而创建一个 Ray 集群。在 :ref:`on-prem` 中查看更多细节。在集群中的任何一台机器上调用 ``ray.init()`` 将连接到同一个 Ray 集群。

.. _start-ray-up:

启动 Ray 集群 (``ray up``)
------------------------------------

Ray 集群可以通过 :ref:`集群启动器 <cluster-index>` 启动。
``ray up`` 命令使用 Ray 集群启动器在云上启动集群，创建指定的“头节点”和 worker 节点。在底层，它会自动调用 ``ray start`` 来创建 Ray 集群。

你的代码 **只需要** 在集群中的一台机器上执行（通常是头节点）。了解更多关于 :ref:`在 Ray 集群上运行程序 <cluster-index>` 的信息。

要连接到 Ray 集群，请在集群中的一台机器上调用 ``ray.init``。这将连接到最新的 Ray 集群：

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  ray.init()

请注意，调用 ``ray up`` 的机器不会被视为 Ray 集群的一部分，因此在同一台机器上调用 ``ray.init`` 将不会连接到集群。

接下来是什么？
------------

查看我们的 `部署部分 <cluster/index.html>`_ 了解有关在不同环境中部署 Ray 的更多信息，包括 Kubernetes、YARN 和 SLURM。
