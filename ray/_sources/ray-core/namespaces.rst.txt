.. _namespaces-guide:

使用命名空间
================

命名空间是作业和命名 actor 的逻辑分组。当 actor 被命名时，其名称必须在命名空间内唯一。

为了设置应用程序的命名空间，应该在首次连接到集群时指定。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __init_namespace_start__
          :end-before: __init_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

          System.setProperty("ray.job.namespace", "hello"); // set it before Ray.init()
          Ray.init();

    .. tab-item:: C++

        .. code-block:: c++

          ray::RayConfig config;
          config.ray_namespace = "hello";
          ray::Init(config);

请参阅 `Driver Options <configure.html#driver-options>`__ 了解配置 Java 应用程序的方法。

命名 actor 仅在其命名空间内可访问。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __actor_namespace_start__
          :end-before: __actor_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 creates two actors, "orange" and "purple" in the "colors" namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                Ray.actor(Actor::new).setName("orange").remote();
                Ray.actor(Actor::new).setName("purple").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 2 is now connecting to a different namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "fruits");
            try {
                Ray.init();
                // This fails because "orange" was defined in the "colors" namespace.
                Ray.getActor("orange").isPresent(); // return false
                // This succceeds because the name "orange" is unused in this namespace.
                Ray.actor(Actor::new).setName("orange").remote();
                Ray.actor(Actor::new).setName("watermelon").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 3 connects to the original "colors" namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                // This fails because "watermelon" was in the fruits namespace.
                Ray.getActor("watermelon").isPresent(); // return false
                // This returns the "orange" actor we created in the first job, not the second.
                Ray.getActor("orange").isPresent(); // return true
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 creates two actors, "orange" and "purple" in the "colors" namespace.
            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("orange").Remote();
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("purple").Remote();
            ray::Shutdown();

            // Job 2 is now connecting to a different namespace.
            ray::RayConfig config;
            config.ray_namespace = "fruits";
            ray::Init(config);
            // This fails because "orange" was defined in the "colors" namespace.
            ray::GetActor<Counter>("orange"); // return nullptr;
            // This succeeds because the name "orange" is unused in this namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("orange").Remote();
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("watermelon").Remote();
            ray::Shutdown();

            // Job 3 connects to the original "colors" namespace.
            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            // This fails because "watermelon" was in the fruits namespace.
            ray::GetActor<Counter>("watermelon"); // return nullptr;
            // This returns the "orange" actor we created in the first job, not the second.
            ray::GetActor<Counter>("orange");
            ray::Shutdown();

为命名 actor 指定命名空间
-------------------------------------

你可以在创建命名 actor 时为其指定命名空间。创建的 actor 属于指定的命名空间，而不管当前作业的命名空间是什么。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __specify_actor_namespace_start__
          :end-before: __specify_actor_namespace_end__


    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                // Create an actor with specified namespace.
                Ray.actor(Actor::new).setName("my_actor", "actor_namespace").remote();
                // It is accessible in its namespace.
                Ray.getActor("my_actor", "actor_namespace").isPresent(); // return true

            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block::

            // `ray start --head` has been run to launch a local cluster.
            ray::RayConfig config;
            ray::Init(config);
            // Create an actor with specified namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor", "actor_namespace").Remote();
            // It is accessible in its namespace.
            ray::GetActor<Counter>("orange");
            ray::Shutdown();`


匿名命名空间
--------------------

当命名空间未指定时，Ray 会将作业放在匿名命名空间中。
在匿名命名空间中，作业将拥有自己的命名空间，并且无法访问其他命名空间中的 actor。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __anonymous_namespace_start__
          :end-before: __anonymous_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 connects to an anonymous namespace by default.
            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                Ray.actor(Actor::new).setName("my_actor").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 2 connects to a _different_ anonymous namespace by default
            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                // This succeeds because the second job is in its own namespace.
                Ray.actor(Actor::new).setName("my_actor").remote();
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 connects to an anonymous namespace by default.
            ray::RayConfig config;
            ray::Init(config);
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor").Remote();
            ray::Shutdown();

            // Job 2 connects to a _different_ anonymous namespace by default
            ray::RayConfig config;
            ray::Init(config);
            // This succeeds because the second job is in its own namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor").Remote();
            ray::Shutdown();

.. note::

     Anonymous namespaces are implemented as UUID's. This makes it possible for
     a future job to manually connect to an existing anonymous namespace, but
     it is not recommended.


获取当前命名空间
-----------------------------
你可以使用 :ref:`runtime_context APIs <runtime-context-apis>` 访问当前命名空间。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __get_namespace_start__
          :end-before: __get_namespace_end__


    .. tab-item:: Java

        .. code-block:: java

            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                // Will print namespace name "colors".
                System.out.println(Ray.getRuntimeContext().getNamespace());
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            // Will print namespace name "colors".
            std::cout << ray::GetNamespace() << std::endl;
            ray::Shutdown();
