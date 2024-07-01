.. _fault-tolerance-actors:
.. _actor-fault-tolerance:

Actor 容错
=====================

如果 Actor 进程失败，或者 Actor 的 owner 进程失败，Actor 可能会失败。
Actor 的所有者是通过调用 ``ActorClass.remote()`` 创建 Actor 的 worker。
:ref:`Detached actors <actor-lifetimes>` 没有 owner 进程，当 Ray 集群被销毁时，它们会被清理。


Actor 进程失败
---------------------

Ray 可能在 actor 进程失败后自动重启 actor。
这个行为由 ``max_restarts`` 控制，它设置了 actor 可以重启的最大次数。
``max_restarts`` 的默认值是 0，意味着 actor 不会被重启。如果设置为 -1，actor 将无限次重启。
当 actor 重启时，它的状态将通过重新运行其构造函数来重新创建。
在指定的重启次数之后，后续的 actor 方法将引发 ``RayActorError``。

默认的，actor 任务以最多一次的语义执行（``@ray.remote`` 中 :func:``decorator <ray.remote>`` 的 ``max_task_retries=0``）。
这意味着如果一个 actor 任务被提交到一个不可达的 actor，Ray 将会报告错误，抛出 ``RayActorError``，这是一个 Python 级别的异常，当调用 ``ray.get`` 时，会抛出这个异常。
请注意，即使任务确实成功执行，也可能抛出这个异常。例如，如果 actor 在执行任务后立即死亡，就会发生这种情况。

Ray 还为 actor 任务提供了至少一次的执行语义（ ``max_task_retries=-1`` 或 ``max_task_retries > 0``）。
这意味着如果一个 actor 任务被提交到一个不可达的 actor，系统将自动重试任务。
使用此选项，系统只会在发生以下情况之一时向应用程序抛出 ``RayActorError``：(1) actor 的 ``max_restarts`` 限制已经超过，actor 不能再重启，
或者 (2) 此特定任务的 ``max_task_retries`` 限制已经超过。
请注意，如果 actor 在提交任务时正在重启，这将计为一次重试。
重试限制可以通过 ``max_task_retries = -1`` 设置为无限。

你可以通过运行以下代码来尝试这个行为。

.. literalinclude:: ../doc_code/actor_restart.py
  :language: python
  :start-after: __actor_restart_begin__
  :end-before: __actor_restart_end__

对于至少一次执行的 actor，系统仍然会根据初始提交顺序保证执行顺序。
例如，任何在失败的 actor 任务之后提交的任务都不会在 actor 上执行，直到失败的 actor 任务成功重试。
系统不会尝试重新执行任何在失败之前成功执行的任务（除非 ``max_task_retries`` 不为零且任务对于 :ref:`对象重建 <fault-tolerance-objects-reconstruction>` 是必要的）。

.. note::

  对于 :ref:`异步或线程 actor <async-actors>`，:ref:`任务可能会无序执行 <actor-task-order>`。
  Actor 重启后，系统只会重试 *未完成* 的任务。之前已完成的任务不会被重新执行。
  之前已完成的任务不会被重新执行。


至少一次执行最适合只读 actor 或具有不需要在失败后重建的临时状态的 actor。
对于具有关键状态的 actor，应用程序负责恢复状态，例如，通过定期检查点并在 actor 重启时从检查点恢复。


Actor 检查点
~~~~~~~~~~~~~~~~~~~

``max_restarts`` 会自动重启崩溃的 actor，但不会自动恢复 actor 的应用程序级状态。
相反，您应该手动检查点 actor 的状态，并在 actor 重启时恢复。

对于手动重启的 actor，actor 的创建者应该管理检查点，并在失败时手动重启和恢复 actor。如果你希望创建者决定何时重启 actor，或者如果创建者正在协调 actor 检查点与其他执行，这是推荐的做法：

.. literalinclude:: ../doc_code/actor_checkpointing.py
  :language: python
  :start-after: __actor_checkpointing_manual_restart_begin__
  :end-before: __actor_checkpointing_manual_restart_end__

或者，如果你使用 Ray 的自动 actor 重启，actor 可以手动检查点自己，并在构造函数中从检查点恢复：

.. literalinclude:: ../doc_code/actor_checkpointing.py
  :language: python
  :start-after: __actor_checkpointing_auto_restart_begin__
  :end-before: __actor_checkpointing_auto_restart_end__

.. note::

  如果检查点保存在外部存储中，请确保整个集群都可以访问它，因为 actor 可能会在不同的节点上重启。
  例如，将检查点保存到云存储（例如 S3）或共享目录（例如通过 NFS）。


Actor 创建者失败
---------------------

对于 :ref:`non-detached actors <actor-lifetimes>`，actor 的所有者是创建它的 worker，即调用 ``ActorClass.remote()`` 的 worker。
类似于 :ref:`objects <fault-tolerance-objects>`，如果 actor 的所有者死亡，actor 也会与所有者共享命运。
Ray 不会自动恢复其所有者已死亡的 actor，即使它有非零的 ``max_restarts``。

由于 :ref:`detached actors <actor-lifetimes>` 没有所有者，即使它们的原始创建者死亡，Ray 仍会重启它们。
直到达到最大重启次数、actor 被销毁，或者 Ray 集群被销毁，detached actors 仍会被自动重启。

你可以在以下代码中尝试这个行为。

.. literalinclude:: ../doc_code/actor_creator_failure.py
  :language: python
  :start-after: __actor_creator_failure_begin__
  :end-before: __actor_creator_failure_end__

强制杀死行为不端的 actor

---------------------------------

有时，应用程序级代码可能会导致 actor 挂起或泄漏资源。
这种情况下，Ray 允许您通过 :ref:`手动终止 <ray-kill-actors>` actor 来从失败中恢复。
你可以通过在 actor 的任何句柄上调用 ``ray.kill`` 来做到这一点。请注意，它不需要是 actor 的原始句柄。
请注意，它不需要是 actor 的原始句柄。

如果 ``max_restarts`` 被设置，你也可以通过将 ``no_restart=False`` 传递给 ``ray.kill`` 来允许 Ray 自动重启 actor。
