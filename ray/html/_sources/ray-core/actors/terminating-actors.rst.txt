终止 Actor
==================

当所有 actor handle 的副本都超出作用域时，actor 进程将在 Python 中自动终止，
或者如果原始创建者进程死亡。

注意，Java 和 C++ 中的 actor 的自动终止尚未支持。

.. _ray-kill-actors:

通过 actor 句柄手动终止
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在大多数情况下，Ray 会自动终止已经超出作用域的 actor，但是有时您可能需要强制终止 actor。
这应该保留给 actor 出现意外挂起或泄漏资源的情况，
以及必须手动销毁的 :ref:`游离 actors <actor-lifetimes>`。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Actor:
                pass

            actor_handle = Actor.remote()

            ray.kill(actor_handle)
            # This will not go through the normal Python sys.exit
            # teardown logic, so any exit handlers installed in
            # the actor using ``atexit`` will not be called.


    .. tab-item:: Java

        .. code-block:: java

            actorHandle.kill();
            // This will not go through the normal Java System.exit teardown logic, so any
            // shutdown hooks installed in the actor using ``Runtime.addShutdownHook(...)`` will
            // not be called.

    .. tab-item:: C++

        .. code-block:: c++

            actor_handle.Kill();
            // This will not go through the normal C++ std::exit
            // teardown logic, so any exit handlers installed in
            // the actor using ``std::atexit`` will not be called.


这会引起 actor 立即退出其进程，
导致任何当前、挂起和未来的任务失败并抛出 ``RayActorError``。
如果你想让 Ray :ref:`自动重启 <fault-tolerance-actors>` actor，
确保在 actor 的 ``@ray.remote`` 选项中设置一个非零的 ``max_restarts``，
然后将标志 ``no_restart=False`` 传递给 ``ray.kill``。

对于 :ref:`命名的游离 actor <actor-lifetimes>`, 在 actor 句柄调用 ``ray.kill``
会销毁 actor 并可重复使用命名。

从 :ref:`State API <state-api-overview-ref>` 使用 `ray list actors --detail` 查看死亡 actor 的死亡原因:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list actors --detail

.. code-block:: bash

  ---
  -   actor_id: e8702085880657b355bf7ef001000000
      class_name: Actor
      state: DEAD
      job_id: '01000000'
      name: ''
      node_id: null
      pid: 0
      ray_namespace: dbab546b-7ce5-4cbb-96f1-d0f64588ae60
      serialized_runtime_env: '{}'
      required_resources: {}
      death_cause:
          actor_died_error_context: # <---- You could see the error message w.r.t why the actor exits. 
              error_message: The actor is dead because `ray.kill` killed it.
              owner_id: 01000000ffffffffffffffffffffffffffffffffffffffffffffffff
              owner_ip_address: 127.0.0.1
              ray_namespace: dbab546b-7ce5-4cbb-96f1-d0f64588ae60
              class_name: Actor
              actor_id: e8702085880657b355bf7ef001000000
              never_started: true
              node_ip_address: ''
              pid: 0
              name: ''
      is_detached: false
      placement_group_id: null
      repr_name: ''


在 actor 内部手动终止
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果需要，您可以从 actor 方法内部手动终止 actor。
这会杀死 actor 进程并释放与 actor 相关的资源。

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            @ray.remote
            class Actor:
                def exit(self):
                    ray.actor.exit_actor()

            actor = Actor.remote()
            actor.exit.remote()

        此方法通常不需要，因为 actor 会自动进行垃圾回收。
        ``ObjectRef`` 的结果可以等待 actor 退出时
        获得（在其上调用 ``ray.get()`` 会引发 ``RayActorError``）。

    .. tab-item:: Java

        .. code-block:: java

            Ray.exitActor();

        垃圾回收的实现尚未完成，因此这是目前唯一优雅终止 actor 的方法。
        任务的结果是一个 ``ObjectRef``，
        可以等待 actor 退出（在其上调用 ``ObjectRef::get`` 会引发 ``RayActorException``）。

    .. tab-item:: C++

        .. code-block:: c++

            ray::ExitActor();

        垃圾回收的实现尚未完成，因此这是目前唯一优雅终止 actor 的方法。
        任务的结果是一个 ``ObjectRef``，
        可以等待 actor 退出（在其上调用 ``ObjectRef::get`` 会引发 ``RayActorException``）。

注意，这种终止方法会等待任何先前提交的任务执行完毕，然后使用 sys.exit 优雅地退出进程。


    
你可以看到 actor 死于用户的 `exit_actor()` 调用:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list actors --detail

.. code-block:: bash

  ---
  -   actor_id: 070eb5f0c9194b851bb1cf1602000000
      class_name: Actor
      state: DEAD
      job_id: '02000000'
      name: ''
      node_id: 47ccba54e3ea71bac244c015d680e202f187fbbd2f60066174a11ced
      pid: 47978
      ray_namespace: 18898403-dda0-485a-9c11-e9f94dffcbed
      serialized_runtime_env: '{}'
      required_resources: {}
      death_cause:
          actor_died_error_context:
              error_message: 'The actor is dead because its worker process has died.
                  Worker exit type: INTENDED_USER_EXIT Worker exit detail: Worker exits
                  by an user request. exit_actor() is called.'
              owner_id: 02000000ffffffffffffffffffffffffffffffffffffffffffffffff
              owner_ip_address: 127.0.0.1
              node_ip_address: 127.0.0.1
              pid: 47978
              ray_namespace: 18898403-dda0-485a-9c11-e9f94dffcbed
              class_name: Actor
              actor_id: 070eb5f0c9194b851bb1cf1602000000
              name: ''
              never_started: false
      is_detached: false
      placement_group_id: null
      repr_name: ''