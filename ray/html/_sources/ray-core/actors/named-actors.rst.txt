具名 Actor
============

一个 Actor 可以在它的 :ref:`namespace <namespaces-guide>` 赋予一个唯一的名字。
这允许你从 Ray 集群中的任何作业中检索 Actor。
这在无法直接传递 actor 句柄到任务时必须，或者尝试在其他驱动时访问 actor。
注意，如果没有句柄指向它，actor 仍然会被垃圾回收。
更多细节请参考 :ref:`actor-lifetimes`。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Counter:
                pass

            # Create an actor with a name
            counter = Counter.options(name="some_name").remote()

            # Retrieve the actor later somewhere
            counter = ray.get_actor("some_name")

    .. tab-item:: Java

        .. code-block:: java

            // Create an actor with a name.
            ActorHandle<Counter> counter = Ray.actor(Counter::new).setName("some_name").remote();

            ...

            // Retrieve the actor later somewhere
            Optional<ActorHandle<Counter>> counter = Ray.getActor("some_name");
            Assert.assertTrue(counter.isPresent());

    .. tab-item:: C++

        .. code-block:: c++

            // Create an actor with a globally unique name
            ActorHandle<Counter> counter = ray::Actor(CreateCounter).SetGlobalName("some_name").Remote();

            ...

            // Retrieve the actor later somewhere
            boost::optional<ray::ActorHandle<Counter>> counter = ray::GetGlobalActor("some_name");

        We also support non-global named actors in C++, which means that the actor name is only valid within the job and the actor cannot be accessed from another job

        .. code-block:: c++

            // Create an actor with a job-scope-unique name
            ActorHandle<Counter> counter = ray::Actor(CreateCounter).SetName("some_name").Remote();

            ...

            // Retrieve the actor later somewhere in the same job
            boost::optional<ray::ActorHandle<Counter>> counter = ray::GetActor("some_name");

.. note::

     命名 actor 的作用域是由命名空间决定的。
     如果没有分配命名空间，它们将默认放在一个匿名命名空间中。

.. tab-set::

    .. tab-item:: Python

        .. testcode::
            :skipif: True

            import ray

            @ray.remote
            class Actor:
              pass

            # driver_1.py
            # Job 1 creates an actor, "orange" in the "colors" namespace.
            ray.init(address="auto", namespace="colors")
            Actor.options(name="orange", lifetime="detached").remote()

            # driver_2.py
            # Job 2 is now connecting to a different namespace.
            ray.init(address="auto", namespace="fruit")
            # This fails because "orange" was defined in the "colors" namespace.
            ray.get_actor("orange")
            # You can also specify the namespace explicitly.
            ray.get_actor("orange", namespace="colors")

            # driver_3.py
            # Job 3 connects to the original "colors" namespace
            ray.init(address="auto", namespace="colors")
            # This returns the "orange" actor we created in the first job.
            ray.get_actor("orange")

    .. tab-item:: Java

        .. code-block:: java

            import ray

            class Actor {
            }

            // Driver1.java
            // Job 1 creates an actor, "orange" in the "colors" namespace.
            System.setProperty("ray.job.namespace", "colors");
            Ray.init();
            Ray.actor(Actor::new).setName("orange").remote();

            // Driver2.java
            // Job 2 is now connecting to a different namespace.
            System.setProperty("ray.job.namespace", "fruits");
            Ray.init();
            // This fails because "orange" was defined in the "colors" namespace.
            Optional<ActorHandle<Actor>> actor = Ray.getActor("orange");
            Assert.assertFalse(actor.isPresent());  // actor.isPresent() is false.

            // Driver3.java
            System.setProperty("ray.job.namespace", "colors");
            Ray.init();
            // This returns the "orange" actor we created in the first job.
            Optional<ActorHandle<Actor>> actor = Ray.getActor("orange");
            Assert.assertTrue(actor.isPresent());  // actor.isPresent() is true.

获取或创建一个命名 actor
---------------------------

常用的方法是创建一个 actor，如果它不存在的话。
Ray 提供了一个 ``get_if_exists`` 选项，用于创建 actor。
该方法在你通过 ``.options()`` 为 actor 设置名称后可用。

如果 actor 已经存在，将返回 actor 的句柄并且参数将被忽略。
否则，将使用指定的参数创建一个新的 actor。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/get_or_create.py

    .. tab-item:: Java

        .. code-block:: java

            // This feature is not yet available in Java.

    .. tab-item:: C++

        .. code-block:: c++

            // This feature is not yet available in C++.


.. _actor-lifetimes:

Actor 生命周期
---------------

特别的，actor 的生命周期可以与作业分离，允许 actor 在作业的驱动程序进程退出后继续存在。我们称这些 actor 为 *detached*。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            counter = Counter.options(name="CounterActor", lifetime="detached").remote()

        这个 ``CounterActor`` 将会在上面的脚本退出后继续存在。
        因此可以在另一个驱动程序中运行以下脚本：

        .. testcode::

            counter = ray.get_actor("CounterActor")

        注意，一个 actor 可以被命名但不是游离的。
        如果我们只指定了名称而没有指定 ``lifetime="detached"``，
        那么 CounterActor 只能在原始驱动程序仍在运行时检索。

    .. tab-item:: Java

        .. code-block:: java

            System.setProperty("ray.job.namespace", "lifetime");
            Ray.init();
            ActorHandle<Counter> counter = Ray.actor(Counter::new).setName("some_name").setLifetime(ActorLifetime.DETACHED).remote();

        CounterActor 会一直保持活动状态，即使上面的进程退出。
        因此，可以在另一个驱动程序中运行以下代码：

        .. code-block:: java

            System.setProperty("ray.job.namespace", "lifetime");
            Ray.init();
            Optional<ActorHandle<Counter>> counter = Ray.getActor("some_name");
            Assert.assertTrue(counter.isPresent());

    .. tab-item:: C++

        自定义 actor 的生命周期在 C++ 中尚未实现。


和通常的 actor 不同，Ray 不会自动回收游离的 actor。
游离的 actor 必须在确定不再需要它们时手动销毁。
要完成此操作，请使用 ``ray.kill`` 来 :ref:`手动终止 <ray-kill-actors>` actor。
在此调用之后，actor 的名称可能会被重用。
