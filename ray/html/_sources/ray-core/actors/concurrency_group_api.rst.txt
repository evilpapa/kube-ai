使用并发组来限制每个方法的并发数
=======================================================

除了为异步IO Actor 设置整体最大并发数之外，Ray 还允许将方法分成 *并发组* ，每个组都有自己的线程。这允许您限制每个方法的并发数，例如，允许为健康检查方法分配独立于请求服务方法的并发配额。

.. warning:: 并发组只支持异步 actor，而不是线程 actor。

.. _defining-concurrency-groups:

定义并发组
---------------------------

这里定义了两个并发组，"io" 的最大并发数为 2 ，
"compute" 的最大并发数为 4。
方法 ``f1`` 和 ``f2`` 被放置在“io” 组中，方法 ``f3`` 和 ``f4 ``被放置在 “compute” 组中。
请注意，对于 actor ，始终有一个默认并发组，其默认并发数 Python 为 1000， Java 为 1。

.. tab-set::

    .. tab-item:: Python

        您可以使用装饰器参数 ``concurrency_group`` 为异步IO actor 定义并发组：

        .. testcode::

            import ray

            @ray.remote(concurrency_groups={"io": 2, "compute": 4})
            class AsyncIOActor:
                def __init__(self):
                    pass

                @ray.method(concurrency_group="io")
                async def f1(self):
                    pass

                @ray.method(concurrency_group="io")
                async def f2(self):
                    pass

                @ray.method(concurrency_group="compute")
                async def f3(self):
                    pass

                @ray.method(concurrency_group="compute")
                async def f4(self):
                    pass

                async def f5(self):
                    pass

            a = AsyncIOActor.remote()
            a.f1.remote()  # executed in the "io" group.
            a.f2.remote()  # executed in the "io" group.
            a.f3.remote()  # executed in the "compute" group.
            a.f4.remote()  # executed in the "compute" group.
            a.f5.remote()  # executed in the default group.

    .. tab-item:: Java

        你可以使用 ``setConcurrencyGroups()`` 参数为并发 actor 定义并发组：

        .. code-block:: java

            class ConcurrentActor {
                public long f1() {
                    return Thread.currentThread().getId();
                }

                public long f2() {
                    return Thread.currentThread().getId();
                }

                public long f3(int a, int b) {
                    return Thread.currentThread().getId();
                }

                public long f4() {
                    return Thread.currentThread().getId();
                }

                public long f5() {
                    return Thread.currentThread().getId();
                }
            }

            ConcurrencyGroup group1 =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("io")
                    .setMaxConcurrency(1)
                    .addMethod(ConcurrentActor::f1)
                    .addMethod(ConcurrentActor::f2)
                    .build();
            ConcurrencyGroup group2 =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("compute")
                    .setMaxConcurrency(1)
                    .addMethod(ConcurrentActor::f3)
                    .addMethod(ConcurrentActor::f4)
                    .build();

            ActorHandle<ConcurrentActor> myActor = Ray.actor(ConcurrentActor::new)
                .setConcurrencyGroups(group1, group2)
                .remote();

            myActor.task(ConcurrentActor::f1).remote();  // executed in the "io" group.
            myActor.task(ConcurrentActor::f2).remote();  // executed in the "io" group.
            myActor.task(ConcurrentActor::f3, 3, 5).remote();  // executed in the "compute" group.
            myActor.task(ConcurrentActor::f4).remote();  // executed in the "compute" group.
            myActor.task(ConcurrentActor::f5).remote();  // executed in the "default" group.


.. _default-concurrency-group:

默认并发组
-------------------------

默认情况下，方法被放置在一个默认并发组中，该组的并发限制为 1000（Python）或 1（Java）。
可以通过设置 actor 选项 ``max_concurrency`` 来更改默认组的并发数。

.. tab-set::

    .. tab-item:: Python

        以下 AsyncIOActor 有 2 个并发组："io" 和 "default"。
        “io”最大并发为2，“default”最大并发为10。

        .. testcode::

            @ray.remote(concurrency_groups={"io": 2})
            class AsyncIOActor:
                async def f1(self):
                    pass

            actor = AsyncIOActor.options(max_concurrency=10).remote()

    .. tab-item:: Java

        以下 AsyncIOActor 有 2 个并发组："io" 和 "default"。
        “io”最大并发为2，“default”最大并发为10。

        .. code-block:: java

            class ConcurrentActor:
                public long f1() {
                    return Thread.currentThread().getId();
                }

            ConcurrencyGroup group =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("io")
                    .setMaxConcurrency(2)
                    .addMethod(ConcurrentActor::f1)
                    .build();

            ActorHandle<ConcurrentActor> myActor = Ray.actor(ConcurrentActor::new)
                  .setConcurrencyGroups(group1)
                  .setMaxConcurrency(10)
                  .remote();


.. _setting-the-concurrency-group-at-runtime:

为运行时设置设置并发组
----------------------------------------

您还可以在运行时将 actor 方法分派到特定的并发组中。

以下代码片段演示了 ``f2`` 在运行时动态设置方法的并发组。

.. tab-set::

    .. tab-item:: Python

        您可以使用 ``.options`` 方法。

        .. testcode::

            # Executed in the "io" group (as defined in the actor class).
            a.f2.options().remote()

            # Executed in the "compute" group.
            a.f2.options(concurrency_group="compute").remote()

    .. tab-item:: Java

        您可以使用 ``setConcurrencyGroup`` 方法。

        .. code-block:: java

            // Executed in the "io" group (as defined in the actor creation).
            myActor.task(ConcurrentActor::f2).remote();

            // Executed in the "compute" group.
            myActor.task(ConcurrentActor::f2).setConcurrencyGroup("compute").remote();
