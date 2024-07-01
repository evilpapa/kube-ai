首次用户提示
=========================

Ray 提供了高度灵活、简约且易于使用的 API。
在本页中，我们描述了一些技巧，可帮助首次使用 Ray 的用户避免一些常见错误，这些错误可能会严重损害其程序的性能。
如需深入了解高级设计模式，请阅读 :ref:`核心设计模式 <core-patterns>`.

.. list-table:: 我们在本文档中使用的核心 Ray API。
   :header-rows: 1

   * - API
     - 描述
   * - ``ray.init()``
     - 初始化 Ray 上下文。
   * - ``@ray.remote``
     - | 函数或类装饰器指定函数将作为任务执行，
       | 或者作为不同进程中的 actor 执行。
   * - ``.remote()``
     - | 每个远程函数、远程类声明的后缀，或
       | 调用远程类方法。
       | 远程操作是异步的。
   * - ``ray.put()``
     - | 将对象存储在对象存储中，并返回其 ID。
       | 此 ID 可用于将对象作为参数传递
       | 任何远程函数或方法调用。
       | 这是一个同步操作。
   * - ``ray.get()``
     - | 根据对象 ID 返回对象或对象列表
       | 或对象 ID 列表。
       | 这是一个同步（即阻塞）操作。
   * - ``ray.wait()``
     - | 从对象 ID 列表中返回
       | （1）已就绪对象的 ID 列表，以及
       | （2）尚未准备好的对象的ID列表。
       | 默认情况下，它每次返回一个就绪对象 ID。


本页报告的所有结果均在配备 2.7 GHz Core i7 CPU 和 16GB RAM 的 13 英寸 MacBook Pro 上获得。
虽然 ``ray.init()`` 在单台机器上运行时会自动检测内核数量，
但为了减少在运行以下代码时在您的机器上观察到的结果的差异，
我们在此处指定 num_cpus = 4，即具有 4 个 CPU 的机器。

由于每个任务默认请求一个 CPU，因此此设置允许我们并行执行最多四个任务。
因此，我们的 Ray 系统由一个执行程序的驱动程序和最多四个运行远程任务或 actor 的工作程序组成。

.. _tip-delay-get:

Tip 1: 延迟 ray.get()
----------------------

使用 Ray，每个远程操作（例如任务、 actor 方法）的调用都是异步的。这意味着操作会立即返回一个 promise/future，这本质上是操作结果的标识符 (ID)。这是实现并行性的关键，因为它允许驱动程序并行启动多个操作。要获得实际结果，程序员需要使用 ``ray.get()`` 调用 ID 的结果。此调用会阻塞，直到结果可用。作为副作用，此操作还会阻止驱动程序调用其他操作，这可能会损害并行性。

不幸的是，对于新的 Ray 用户来说，无意中使用 ``ray.get()`` 是很自然的。为了说明这一点，请考虑以下简单的 Python 代码，它四次调用该函数 ``do_some_work()`` ，每次调用大约需要 1 秒：

.. testcode::

    import ray
    import time

    def do_some_work(x):
        time.sleep(1) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [do_some_work(x) for x in range(4)]
    print("duration =", time.time() - start)
    print("results =", results)


程序执行的输出如下。正如预期的那样，该程序大约需要 4 秒：

.. testoutput::
    :options: +MOCK

    duration = 4.0149290561676025
    results = [0, 1, 2, 3]

现在，让我们用 Ray 将上述程序并行化。一些初次使用的用户只需将函数设为远程即可，即

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import time
    import ray

    ray.init(num_cpus=4) # Specify this system has 4 CPUs.

    @ray.remote
    def do_some_work(x):
        time.sleep(1) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [do_some_work.remote(x) for x in range(4)]
    print("duration =", time.time() - start)
    print("results =", results)

然而，执行上述程序时会得到：

.. testoutput::
    :options: +MOCK

    duration = 0.0003619194030761719
    results = [ObjectRef(df5a1a828c9685d3ffffffff0100000001000000), ObjectRef(cb230a572350ff44ffffffff0100000001000000), ObjectRef(7bbd90284b71e599ffffffff0100000001000000), ObjectRef(bd37d2621480fc7dffffffff0100000001000000)]

查看此输出时，有两点值得注意。首先，程序立即完成，即在不到 1 毫秒的时间内完成。其次，我们得到的不是预期的结果（即 [0, 1, 2, 3]），而是一堆标识符。回想一下，远程操作是异步的，它们返回的是 Future（即对象 ID），而不是结果本身。这正是我们在这里看到的。我们只测量调用任务所需的时间，而不是它们的运行时间，并且我们得到了与四个任务相对应的结果的 ID。

为了获得实际结果，我们需要使用 ray.get()，这里的第一直觉就是调用 ``ray.get()`` 远程操作调用，即将第 12 行替换为：

.. testcode::

    results = [ray.get(do_some_work.remote(x)) for x in range(4)]

更改后重新运行程序，我们得到：

.. testoutput::
    :options: +MOCK

    duration = 4.018050909042358
    results =  [0, 1, 2, 3]

现在结果是正确的，但仍然需要 4 秒，因此没有加速！发生了什么？细心的读者肯定已经知道答案了：``ray.get()`` 是阻塞的，因此在每次远程操作后调用它意味着我们等待该操作完成，这实际上意味着我们一次执行一个操作，因此没有并行性！

为了实现并行性，我们需要在调用所有任务后调用 ``ray.get()`` 。在我们的示例中，我们可以轻松地做到这一点，只需将第 12 行替换为：

.. testcode::

    results = ray.get([do_some_work.remote(x) for x in range(4)])

通过在更改后重新运行程序，我们现在得到：

.. testoutput::
    :options: +MOCK

    duration = 1.0064549446105957
    results =  [0, 1, 2, 3]

终于成功了！我们的 Ray 程序现在仅需 1 秒即可运行，这意味着所有 ``do_some_work()`` 调用在并行运行。

总之，请始终记住， ``ray.get()`` 是一个阻塞操作，因此如果急切调用，可能会损害并行性。相反，您应该尝试编写程序， ``ray.get()`` 可能晚地被调用。

Tip 2: 避免执行微小的任务
-----------------------

当初次开发的人想要使用 Ray 并行化他们的代码时，自然的本能是让每个函数或类都远程运行。不幸的是，这可能会导致不良后果；如果任务非常小，Ray 程序可能比同等的 Python 程序花费的时间更长。

让我们再次考虑上述示例，但这次我们使任务变得更短（即每个任务仅需要 0.1 毫秒），并将任务调用次数大幅增加到 100,000 次。

.. testcode::

    import time

    def tiny_work(x):
        time.sleep(0.0001) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [tiny_work(x) for x in range(100000)]
    print("duration =", time.time() - start)

通过运行该程序我们得到：

.. testoutput::
    :options: +MOCK

    duration = 13.36544418334961

这个结果应该是可以预料到的，因为执行 100,000 个每个耗时 0.1 毫秒的任务的下限是 10 秒，我们还需要添加其他开销，例如函数调用等。

现在让我们使用 Ray 并行化此代码，通过每次调用远程 ``tiny_work()`` :

.. testcode::

    import time
    import ray

    @ray.remote
    def tiny_work(x):
        time.sleep(0.0001) # Replace this with work you need to do.
        return x

    start = time.time()
    result_ids = [tiny_work.remote(x) for x in range(100000)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

运行该代码的结果是：

.. testoutput::
    :options: +MOCK

    duration = 27.46447515487671

令人惊讶的是，Ray 不仅没有改善执行时间，而且 Ray 程序实际上比顺序程序更慢！发生了什么？嗯，这里的问题是每个任务调用都有不小的开销（例如，调度、进程间通信、更新系统状态），并且这些开销决定了执行任务所需的实际时间。

加快此程序速度的一种方法是增大远程任务，以分摊调用开销。以下是一种可能的解决方案，我们将 1000 个 ``tiny_work()`` 远程调用聚合到一个更大的远程函数中：

.. testcode::

    import time
    import ray

    def tiny_work(x):
        time.sleep(0.0001) # replace this is with work you need to do
        return x

    @ray.remote
    def mega_work(start, end):
        return [tiny_work(x) for x in range(start, end)]

    start = time.time()
    result_ids = []
    [result_ids.append(mega_work.remote(x*1000, (x+1)*1000)) for x in range(100)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

现在，如果我们运行上述程序，我们会得到：

.. testoutput::
    :options: +MOCK

    duration = 3.2539820671081543

这大约是顺序执行的四分之一，符合我们的预期（回想一下，我们可以并行运行四个任务）。当然，自然而然的问题是，多大的任务才能足以分摊远程调用开销。找到这一点的一种方法是运行以下简单程序来估算每个任务的调用开销：

.. testcode::

    @ray.remote
    def no_work(x):
        return x

    start = time.time()
    num_calls = 1000
    [ray.get(no_work.remote(x)) for x in range(num_calls)]
    print("per task overhead (ms) =", (time.time() - start)*1000/num_calls)

在 2018 款 MacBook Pro 笔记本上运行上述程序显示：

.. testoutput::
    :options: +MOCK

    per task overhead (ms) = 0.4739549160003662

换句话说，执行一个空任务需要将近半毫秒的时间。这意味着我们需要确保任务至少需要几毫秒才能分摊调用开销。需要注意的是，每个任务的开销会因机器而异，并且在同一台机器上运行的任务与远程运行的任务之间也会有所不同。话虽如此，确保任务至少需要几毫秒是开发 Ray 程序时的一个好经验法则。

Tip 3: 避免将同一对象重复传递给远程任务
-----------------------------------------------------------

当我们将大型对象作为参数传递给远程函数时，Ray 会在后台调用该函数 ``ray.put()`` 该对象存储在本地对象存储中。 当远程任务在本地执行时，这可以显著提高远程任务调用的性能，因为所有本地任务都共享对象存储。

但是，有时任务自动调用 ``ray.put()`` 会导致性能问题。。一个例子是重复传递同一个大对象作为参数，如下面的程序所示：

.. testcode::

    import time
    import numpy as np
    import ray

    @ray.remote
    def no_work(a):
        return

    start = time.time()
    a = np.zeros((5000, 5000))
    result_ids = [no_work.remote(a) for x in range(10)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

该程序输出：

.. testoutput::
    :options: +MOCK

    duration = 1.0837509632110596


对于一个只调用 10 个不执行任何操作的远程任务的程序来说，这个运行时间相当长。运行时间出乎意料地长的原因是，每次我们调用 ``no_work(a)`` 时，Ray 都会调用 ``ray.put(a)`` ，这会导致将数组复制 ``a`` 到对象存储中。由于数组 ``a`` 有 250 万个条目，因此复制它需要花费不少时间。

为了避免每次 ``no_work()`` 调用时复制 ``a`` 数组，一个简单的解决方案是显式调用 ``ray.put(a)``，然后将 ``a`` 的 ID 传递给 ``no_work()``，如下所示：

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import time
    import numpy as np
    import ray

    ray.init(num_cpus=4)

    @ray.remote
    def no_work(a):
        return

    start = time.time()
    a_id = ray.put(np.zeros((5000, 5000)))
    result_ids = [no_work.remote(a_id) for x in range(10)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

运行该程序仅需：

.. testoutput::
    :options: +MOCK

    duration = 0.132796049118042

这比原始程序快 7 倍，这是可以预料的，因为调用 ``no_work(a)`` 的主要开销是将数组 ``a`` 复制到对象存储中，现在只需要一次。

可以说，避免将同一对象多次复制到对象存储的一个更重要的优点是，它可以防止对象存储过早填满并产生对象驱逐的成本。


Tip 4: Pipeline 数据处理
-------------------------------

如果我们使用 ``ray.get()`` 在多个任务的结果上，我们将不得不等到这些任务中的最后一个完成。如果任务花费的时间差异很大，这可能会成为一个问题。

为了说明这个问题，请考虑以下示例，我们并行运行四个 ``do_some_work()`` 任务，每个任务花费的时间在 0 到 4 秒之间均匀分布。接下来，假设这些任务的结果由 ``process_results()`` 处理，每个结果需要 1 秒。预期的运行时间是（1）执行 ``do_some_work()`` 任务中最慢的任务所需的时间，加上（2）执行 ``process_results()`` 所需的 4 秒。

.. testcode::

    import time
    import random
    import ray

    @ray.remote
    def do_some_work(x):
        time.sleep(random.uniform(0, 4)) # Replace this with work you need to do.
        return x

    def process_results(results):
        sum = 0
        for x in results:
            time.sleep(1) # Replace this with some processing code.
            sum += x
        return sum

    start = time.time()
    data_list = ray.get([do_some_work.remote(x) for x in range(4)])
    sum = process_results(data_list)
    print("duration =", time.time() - start, "\nresult = ", sum)

程序的输出显示运行需要接近 8 秒：

.. testoutput::
    :options: +MOCK

    duration = 7.82636022567749
    result =  6

当其他任务可能早已完成时，等待最后一个任务完成可能会增加程序运行时间。更好的解决方案是一旦数据可用就立即处理数据。
幸运的是，Ray 允许您通过在对象 ID 列表上调用 ``ray.wait()`` 来实现这一点。在不指定任何其他参数的情况下，此函数会在其参数列表中的对象准备就绪时立即返回。此调用有两个返回值：（1）准备就绪的对象的 ID，以及（2）包含尚未准备好的对象的 ID 列表。修改后的程序如下。请注意，我们需要做的一个更改是将 ``process_results()`` 替换为 ``process_incremental()``，后者一次处理一个结果。

.. testcode::

    import time
    import random
    import ray

    @ray.remote
    def do_some_work(x):
        time.sleep(random.uniform(0, 4)) # Replace this with work you need to do.
        return x

    def process_incremental(sum, result):
        time.sleep(1) # Replace this with some processing code.
        return sum + result

    start = time.time()
    result_ids = [do_some_work.remote(x) for x in range(4)]
    sum = 0
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        sum = process_incremental(sum, ray.get(done_id[0]))
    print("duration =", time.time() - start, "\nresult = ", sum)

该程序现在仅需 4.8 秒多一点，这是一个显著的进步：

.. testoutput::
    :options: +MOCK

    duration = 4.852453231811523
    result =  6

为了帮助直观理解，图 1 显示了两种情况下的执行时间线：在使用 ``ray.get()`` 等待所有结果可用之前处理它们，以及在使用 ``ray.wait()`` 在结果可用时立即开始处理结果。

.. figure:: /images/pipeline.png

    图 1：（a）使用 ray.get() 等待所有结果从 ``do_some_work()`` 任务中调用 ``process_results()`` 之前的执行时间线。 （b）使用 ``ray.wait()`` 处理结果的执行时间线，一旦结果可用就立即处理。
