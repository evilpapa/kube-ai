Core API
========

.. autosummary::
    :toctree: doc/

    ray.init
    ray.shutdown
    ray.is_initialized
    ray.job_config.JobConfig

任务
-----

.. autosummary::
    :toctree: doc/

    ray.remote
    ray.remote_function.RemoteFunction.options
    ray.cancel

Actors
------

.. autosummary::
    :toctree: doc/

    ray.remote
    ray.actor.ActorClass.options
    ray.method
    ray.get_actor
    ray.kill

对象
-------

.. autosummary::
    :toctree: doc/

    ray.get
    ray.wait
    ray.put

.. _runtime-context-apis:

运行时上下文
---------------
.. autosummary::
    :toctree: doc/

    ray.runtime_context.get_runtime_context
    ray.runtime_context.RuntimeContext
    ray.get_gpu_ids

跨语言
--------------
.. autosummary::
    :toctree: doc/

    ray.cross_language.java_function
    ray.cross_language.java_actor_class
