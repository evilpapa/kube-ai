.. _ray-debugger:

使用 Ray 调试器
======================

Ray 有一个内置的调试器，允许您调试分布式应用程序。
它允许在 Ray 任务和 actor 中设置断点，当遇到断点时，
您可以放入 PDB 会话，然后可以使用该会话：

- 检查该上下文中的变量 
- 步入该任务或 actor
- 在堆栈中向上或向下移动

.. warning::

    Ray 调试器是一项实验性功能，目前不稳定。接口可能会发生变化。

入门
---------------

.. note::

    在 Python 3.6 上， ``breakpoint()`` 不支持该函数，您需要使用
    ``ray.util.pdb.set_trace()`` 替代它。

举个例子：

.. testcode::
    :skipif: True

    import ray

    @ray.remote
    def f(x):
        breakpoint()
        return x * x

    futures = [f.remote(i) for i in range(2)]
    print(ray.get(futures))

将程序放入名为 ``debugging.py`` 的文件中，并使用以下命令执行它：

.. code-block:: bash

    python debugging.py


当执行该行 ``breakpoint()`` 时，2 个执行的任务中的每一个都会落入断点。
您可以通过在集群的头节点上运行以下命令来附加到调试器：

.. code-block:: bash

    ray debug

``ray debug`` 命令将打印如下输出：

.. code-block:: text

    2021-07-13 16:30:40,112	INFO scripts.py:216 -- Connecting to Ray instance at 192.168.2.61:6379.
    2021-07-13 16:30:40,112	INFO worker.py:740 -- Connecting to existing Ray cluster at address: 192.168.2.61:6379
    Active breakpoints:
    index | timestamp           | Ray task | filename:lineno
    0     | 2021-07-13 23:30:37 | ray::f() | debugging.py:6
    1     | 2021-07-13 23:30:37 | ray::f() | debugging.py:6
    Enter breakpoint index or press enter to refresh:


您现在可按 ``0`` 并按 Enter 键跳转到第一个断点。您将在断点处进入 PDB，并可以使用
``help`` 查看可用的操作。运行 ``bt`` 以查看执行的回溯：

.. code-block:: text

    (Pdb) bt
      /home/ubuntu/ray/python/ray/workers/default_worker.py(170)<module>()
    -> ray.worker.global_worker.main_loop()
      /home/ubuntu/ray/python/ray/worker.py(385)main_loop()
    -> self.core_worker.run_task_loop()
    > /home/ubuntu/tmp/debugging.py(7)f()
    -> return x * x

您可以通过 ``print(x)`` 检查 ``x`` 的值。 您可以使用 ``ll`` 查看上下文，
并使用 ``up`` 和 ``down`` 更改堆栈帧。 现在让我们继续执行，使用 ``c``。

继续执行后，使用 ``Control + D`` 返回到断点列表。选择另一个断点，然后再次按 ``c`` 继续执行。

Ray 程序 ``debugging.py`` 现在已经完成，应该已经打印了 ``[0, 1]``。
恭喜，您已经完成了第一个 Ray 调试会话！

在集群上运行
--------------------

Ray 调试器支持在运行在 Ray 集群上的任务和 actor 中设置断点。
要使用 ``ray debug`` 从集群的头节点附加到这些任务和 actor，
您需要确保在启动集群时使用 ``ray start`` 传递 ``--ray-debugger-external`` 标志（可能在您的 ``cluster.yaml`` 文件或 k8s Ray 集群规范中）。

注意，这个标志会导致 worker 在外部 IP 地址上监听 PDB 命令，
所以 *只有* 在您的集群在防火墙后面时才应该使用。

调试命令
-----------------

Ray 调试器支持
`与 PDB 相同的命令
<https://docs.python.org/3/library/pdb.html#debugger-commands>`_。

在 Ray 任务之间步进
--------------------------

您可以使用调试器在 Ray 任务之间进行单步执行。
我们以下面的递归函数为例：

.. testcode::
    :skipif: True

    import ray

    @ray.remote
    def fact(n):
        if n == 1:
            return n
        else:
            n_ref = fact.remote(n - 1)
            return n * ray.get(n_ref)

    @ray.remote
    def compute():
        breakpoint()
        result_ref = fact.remote(5)
        result = ray.get(result_ref)

    ray.get(compute.remote())


执行Python文件并调用 
``ray debug``运行程序后，可以通过 ``0`` 和 回车 选择断点。
这将产生以下输出：

.. code-block:: shell

    Enter breakpoint index or press enter to refresh: 0
    > /home/ubuntu/tmp/stepping.py(16)<module>()
    -> result_ref = fact.remote(5)
    (Pdb)

您可以使用Ray 调试器中的命令 ``remote`` 跳入调用。
在函数内部，打印 ``p(n)`` 的 `n` 值，产生以下输出：

.. code-block:: shell

    -> result_ref = fact.remote(5)
    (Pdb) remote
    *** Connection closed by remote host ***
    Continuing pdb session in different process...
    --Call--
    > /home/ubuntu/tmp/stepping.py(5)fact()
    -> @ray.remote
    (Pdb) ll
      5  ->	@ray.remote
      6  	def fact(n):
      7  	    if n == 1:
      8  	        return n
      9  	    else:
     10  	        n_ref = fact.remote(n - 1)
     11  	        return n * ray.get(n_ref)
    (Pdb) p(n)
    5
    (Pdb)

现在使用 ``remote`` 步入下一个远程调用并打印 `n` 。
你现在可以通多多次调用 ``remote`` 继续递归进入函数，或者通过使用 ``get`` 调试器命令跳转到调用 ``ray.get`` 的位置。
使用 ``get`` 再次跳转到原始调用位置，并使用 ``p(result)`` 打印结果：

.. code-block:: shell

    Enter breakpoint index or press enter to refresh: 0
    > /home/ubuntu/tmp/stepping.py(14)<module>()
    -> result_ref = fact.remote(5)
    (Pdb) remote
    *** Connection closed by remote host ***
    Continuing pdb session in different process...
    --Call--
    > /home/ubuntu/tmp/stepping.py(5)fact()
    -> @ray.remote
    (Pdb) p(n)
    5
    (Pdb) remote
    *** Connection closed by remote host ***
    Continuing pdb session in different process...
    --Call--
    > /home/ubuntu/tmp/stepping.py(5)fact()
    -> @ray.remote
    (Pdb) p(n)
    4
    (Pdb) get
    *** Connection closed by remote host ***
    Continuing pdb session in different process...
    --Return--
    > /home/ubuntu/tmp/stepping.py(5)fact()->120
    -> @ray.remote
    (Pdb) get
    *** Connection closed by remote host ***
    Continuing pdb session in different process...
    --Return--
    > /home/ubuntu/tmp/stepping.py(14)<module>()->None
    -> result_ref = fact.remote(5)
    (Pdb) p(result)
    120
    (Pdb)


事后调试
---------------------

通常我们事先并不知道错误发生在哪里，因此无法设置断点。
在这些情况下，当发生错误或抛出异常时，我们可以自动进入调试器。这称为 *事后调试*。

我们将使用 Ray 服务应用程序展示其工作原理。首先，安装所需的依赖项：

.. code-block:: bash

    pip install "ray[serve]" scikit-learn

接下来，将以下代码复制到名为 ``serve_debugging.py`` 的文件中:

.. testcode::
    :skipif: True

    import time

    from sklearn.datasets import load_iris
    from sklearn.ensemble import GradientBoostingClassifier

    import ray
    from ray import serve

    serve.start()

    # Train model
    iris_dataset = load_iris()
    model = GradientBoostingClassifier()
    model.fit(iris_dataset["data"], iris_dataset["target"])

    # Define Ray Serve model,
    @serve.deployment(route_prefix="/iris")
    class BoostingModel:
        def __init__(self):
            self.model = model
            self.label_list = iris_dataset["target_names"].tolist()

        async def __call__(self, starlette_request):
            payload = (await starlette_request.json())["vector"]
            print(f"Worker: received request with data: {payload}")

            prediction = self.model.predict([payload])[0]
            human_name = self.label_list[prediction]
            return {"result": human_name}

    # Deploy model
    serve.start()
    BoostingModel.deploy()

    time.sleep(3600.0)

让我们在激活事后调试  (``RAY_PDB=1``) 的情况下启动程序:

.. code-block:: bash

    RAY_PDB=1 python serve_debugging.py

``RAY_PDB=1`` 标志的作用是，如果发生异常，Ray 将进入调试器而不是进一步传播它。
让我们看看这是如何工作的！
首先使用无效请求查询模型

.. code-block:: bash

    python -c 'import requests; response = requests.get("http://localhost:8000/iris", json={"vector": [1.2, 1.0, 1.1, "a"]})'

当 ``serve_debugging.py`` 驱动到达断点时，它会告诉你运行 ``ray debug``。
执行此操作后，我们会看到如下输出：

.. code-block:: text

    Active breakpoints:
    index | timestamp           | Ray task                                     | filename:lineno
    0     | 2021-07-13 23:49:14 | ray::RayServeWrappedReplica.handle_request() | /home/ubuntu/ray/python/ray/serve/backend_worker.py:249
    Traceback (most recent call last):

      File "/home/ubuntu/ray/python/ray/serve/backend_worker.py", line 242, in invoke_single
        result = await method_to_call(*args, **kwargs)

      File "serve_debugging.py", line 24, in __call__
        prediction = self.model.predict([payload])[0]

      File "/home/ubuntu/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/_gb.py", line 1188, in predict
        raw_predictions = self.decision_function(X)

      File "/home/ubuntu/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/_gb.py", line 1143, in decision_function
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')

      File "/home/ubuntu/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py", line 63, in inner_f
        return f(*args, **kwargs)

      File "/home/ubuntu/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py", line 673, in check_array
        array = np.asarray(array, order=order, dtype=dtype)

      File "/home/ubuntu/anaconda3/lib/python3.7/site-packages/numpy/core/_asarray.py", line 83, in asarray
        return array(a, dtype, copy=False, order=order)

    ValueError: could not convert string to float: 'a'

    Enter breakpoint index or press enter to refresh:

现在按 ``0`` 然后按 Enter 来进入调试器。使用 ``ll`` 可以看到上下文，
``print(a)`` 打印导致问题的数组。 正如我们所看到的，它包含一个字符串 ( ``'a'`` ) 而不是数字作为最后一个元素。

与上面类似的方式，您也可以调试 Ray actor。调试愉快！

调试 API
--------------

参阅 :ref:`package-ref-debugging-apis`。
