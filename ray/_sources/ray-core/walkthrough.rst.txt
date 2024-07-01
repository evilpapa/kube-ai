.. _core-walkthrough:

什么是 Ray Core？
=================

Ray Core 为构建和扩展分布式应用程序提供了少量的核心基元（即 task、actor、对象）。下面我们将介绍一些简单的示例，这些示例向您展示如何将函数和类轻松地转换为 Ray task 和 actor，以及如何使用 Ray 对象。

入门
---------------

要开始，请通过 ``pip-install-U Ray`` 安装 Ray。有关更多安装选项，请参阅：:ref:`安装 Ray<installation>`。以下几节将介绍使用 Ray Core 的基本知识。

第一步导入并实例化 Ray：

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __starting_ray_start__
    :end-before: __starting_ray_end__

.. note::

  Ray 在最近的版本 (>=1.5), ``ray.init()`` 在第一个 Ray 远程 API 调用时会自动执行。

运行一个任务
--------------

Ray 将你的函数作为集群远程任务。为此，你需要使用 ``@ray.remote`` 装饰你的函数，以声明你想要远程运行这个函数。
然后，你使用 ``.remote()`` 而不是通常调用方式。
这个远程调用返回一个 future，一个所谓的 Ray *对象引用*，你可以使用 ``ray.get`` 获取它：

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __running_task_start__
    :end-before: __running_task_end__

Actor 调用
----------------

Ray 提供了 actor 来允许您在多个 actor 实例之间并行计算。当您实例化一个 Ray actor 类时，Ray 将在集群中启动该类的远程实例。这个 actor 之后可以执行远程方法调用并维护自己的内部状态：

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __calling_actor_start__
    :end-before: __calling_actor_end__

上面的内容涵盖了非常基本的 actor 使用。要了解更深入的示例，包括如何同时使用任务和 actor，请查看 :ref:`monte-carlo-pi`。

对象传递
-----------------

如上所述，Ray 将任务和 actor 调用结果存储在其 :ref:`分布式对象存储 <objects-in-ray>` 中，返回 *对象引用*，稍后可以检索。对象引用也可以通过 ``ray.put`` 明确创建，对象引用可以作为参数值的替代传递给任务：

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __passing_object_start__
    :end-before: __passing_object_end__

下一步
----------

.. tip:: 要检查你的程序当前的运行情况，可以使用 :ref:`Ray 控制面板 <observability-getting-started>`。

Ray 的核心基础很简单，但可以组合在一起表达几乎任何类型的分布式计算。
通过以下用户指南学习更多关于 Ray 的 :ref:`核心概念 <core-key-concepts>` ：

.. grid:: 1 2 3 3
    :gutter: 1
    :class-container: container pb-3


    .. grid-item-card::
        :img-top: /images/tasks.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ray-remote-functions

            Using remote functions (Tasks)

    .. grid-item-card::
        :img-top: /images/actors.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ray-remote-classes

            Using remote classes (Actors)

    .. grid-item-card::
        :img-top: /images/objects.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: objects-in-ray

            Working with Ray Objects
