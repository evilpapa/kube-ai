.. _core-patterns:

设计模型 及 反模式
===============================

本节是关于编写 Ray 应用程序的常见设计模式和反模式的集合。

.. toctree::
    :maxdepth: 1

    nested-tasks
    generators
    limit-pending-tasks
    limit-running-tasks
    concurrent-operations-async-actor
    actor-sync
    tree-of-actors
    pipelining
    return-ray-put
    ray-get-loop
    unnecessary-ray-get
    ray-get-submission-order
    ray-get-too-many-objects
    too-fine-grained-tasks
    redefine-task-actor-loop
    pass-large-arg-by-value
    closure-capture-large-objects
    global-variables
