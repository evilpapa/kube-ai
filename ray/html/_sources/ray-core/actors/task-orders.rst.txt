.. _actor-task-order:

Actor 任务执行顺序
==========================

同步、单线程 Actor
----------------------------------
在 Ray 中，一个 actor 从多个提交者（包括 driver 和 worker）接收任务。
对于来自同一提交者的任务，同步、单线程 actor 按照提交顺序执行它们。
换句话说，给定任务在之前提交的任务执行完之前不会执行。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Counter:
                def __init__(self):
                    self.value = 0

                def add(self, addition):
                    self.value += addition
                    return self.value

            counter = Counter.remote()

            # For tasks from the same submitter,
            # they are executed according to submission order.
            value0 = counter.add.remote(1)
            value1 = counter.add.remote(2)

            # Output: 1. The first submitted task is executed first.
            print(ray.get(value0))
            # Output: 3. The later submitted task is executed later.
            print(ray.get(value1))

        .. testoutput::

            1
            3


然而，actor 不保证来自不同提交者的任务的执行顺序。
例如，假设一个未满足的参数阻塞了之前提交的任务。
在这种情况下，actor 仍然可以执行由不同 worker 提交的任务。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time
            import ray

            @ray.remote
            class Counter:
                def __init__(self):
                    self.value = 0

                def add(self, addition):
                    self.value += addition
                    return self.value

            counter = Counter.remote()

            # Submit task from a worker
            @ray.remote
            def submitter(value):
                return ray.get(counter.add.remote(value))

            # Simulate delayed result resolution.
            @ray.remote
            def delayed_resolution(value):
                time.sleep(5)
                return value

            # Submit tasks from different workers, with
            # the first submitted task waiting for
            # dependency resolution.
            value0 = submitter.remote(delayed_resolution.remote(1))
            value1 = submitter.remote(2)

            # Output: 3. The first submitted task is executed later.
            print(ray.get(value0))
            # Output: 2. The later submitted task is executed first.
            print(ray.get(value1))

        .. testoutput::

            3
            2


异步或线程 Actor
------------------------------
:ref:`异步或线程 Actor <async-actors>` 不保证
任务执行顺序。这意味着系统可能会执行一个任务，
即使之前提交的任务正在等待执行。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time
            import ray

            @ray.remote
            class AsyncCounter:
                def __init__(self):
                    self.value = 0

                async def add(self, addition):
                    self.value += addition
                    return self.value

            counter = AsyncCounter.remote()

            # Simulate delayed result resolution.
            @ray.remote
            def delayed_resolution(value):
                time.sleep(5)
                return value

            # Submit tasks from the driver, with
            # the first submitted task waiting for
            # dependency resolution.
            value0 = counter.add.remote(delayed_resolution.remote(1))
            value1 = counter.add.remote(2)

            # Output: 3. The first submitted task is executed later.
            print(ray.get(value0))
            # Output: 2. The later submitted task is executed first.
            print(ray.get(value1))

        .. testoutput::

            3
            2
