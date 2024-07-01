.. _ray-dag-guide:

使用 Ray DAG API 的惰性计算图
============================================

使用 ``ray.remote``，您可以在运行时远程执行计算。
对于 ``ray.remote`` 装饰的类或函数，
您还可以在函数体上使用 ``.bind`` 来构建静态计算图。

.. note::

     Ray DAG 旨在成为面向开发人员的 API，其中推荐的用例是

     1) 在本地迭代并测试由更高级别的库编写的应用程序。

     2) 在 Ray DAG API 之上构建库。


当 ``.bind()`` 被调用在一个 ``ray.remote`` 装饰的类或函数上时，它将生成一个中间表示（IR）节点，
该节点作为 DAG 的骨干和构建块，静态地保持计算图在一起，其中每个 IR 节点在执行时按照拓扑顺序解析为值。

IR 节点也可以赋值给一个变量，并作为参数传递给其他节点。

带有函数的 Ray DAG
----------------------

IR 节点生成的 ``.bind()`` 在 ``ray.remote`` 装饰的函数上，将在执行时作为 Ray 任务执行，该任务将被解析为任务输出。

此示例展示了如何构建一个函数链，其中每个节点都可以作为根节点执行，
同时迭代，或者作为其他函数的输入参数或关键字参数，以形成更复杂的 DAG。

任何 IR 节点都可以直接执行 ``dag_node.execute()``，作为 DAG 的根节点，
其中所有其他从根节点不可达的节点将被忽略。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_tasks_begin__
          :end-before: __dag_tasks_end__


具有类和类方法的 Ray DAG
--------------------------------------

IR 节点生成的 ``.bind()`` 在 ``ray.remote`` 装饰的类上，将在执行时作为 Ray Actor 执行。
Actor 将在每次执行节点时实例化，而 classmethod 调用可以形成特定于父 Actor 实例的函数调用链。

从函数、类或类方法生成的 DAG IR 节点可以组合在一起形成 DAG。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_actors_begin__
          :end-before: __dag_actors_end__



带有自定义 InputNode 的 Ray DAG
-----------------------------

``InputNode`` 是 DAG 的单例节点，表示运行时的用户输入值。
它应该在没有参数的上下文管理器中使用，并作为 ``dag_node.execute()`` 的参数调用。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_input_node_begin__
          :end-before: __dag_input_node_end__

更多资源
--------------

您可以在以下资源中找到更多基于 Ray DAG API 并使用相同机制构建的其他 Ray 库的应用模式和示例。

| `DAG 可视化 <https://docs.ray.io/en/master/serve/model_composition.html#visualizing-the-graph>`_
