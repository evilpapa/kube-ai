.. _fault-tolerance-tasks:
.. _task-fault-tolerance:

任务容错
====================

任务可能因为不同应用级别的错误而失败，例如 Python 级别的异常，
或者系统级别的故障，例如机器故障。
在这里，我们将描述应用程序开发人员可以使用的机制来从这些错误中恢复。

捕捉应用级别的失败
-----------------------------------

Ray 将应用程序级别的失败作为 Python 级别的异常呈现出来。当一个任务在远程工作器
或者 actor 上因为 Python 级别的异常而失败时，
Ray 会将原始异常包装在 ``RayTaskError`` 中，并将其作为任务的返回值存储。
这个包装的异常将被抛出给任何尝试获取结果的工作器，
无论是通过调用 ``ray.get`` 还是如果工作器正在执行依赖于该对象的另一个任务。

.. literalinclude:: ../doc_code/task_exceptions.py
  :language: python
  :start-after: __task_exceptions_begin__
  :end-before: __task_exceptions_end__

使用 :ref:`状态 API CLI <state-api-overview-ref>` 的 `ray list tasks` 来查询任务退出详情：

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list tasks 

.. code-block:: bash

  ======== List: 2023-05-26 10:32:00.962610 ========
  Stats:
  ------------------------------
  Total: 3

  Table:
  ------------------------------
      TASK_ID                                             ATTEMPT_NUMBER  NAME    STATE      JOB_ID  ACTOR_ID    TYPE         FUNC_OR_CLASS_NAME    PARENT_TASK_ID                                    NODE_ID                                                   WORKER_ID                                                 ERROR_TYPE
   0  16310a0f0a45af5cffffffffffffffffffffffff01000000                 0  f       FAILED   01000000              NORMAL_TASK  f                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION
   1  c2668a65bda616c1ffffffffffffffffffffffff01000000                 0  g       FAILED   01000000              NORMAL_TASK  g                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION
   2  c8ef45ccd0112571ffffffffffffffffffffffff01000000                 0  f       FAILED   01000000              NORMAL_TASK  f                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION

.. _task-retries:

重试失败的任务
---------------------

当一个 worker 执行一个任务时，如果 worker 意外死亡，可能是因为进程崩溃或者机器故障，
Ray 会返回任务直到任务成功或者重试次数超过最大次数。默认的重试次数是 3，并可再 ``@ray.remote`` 构造器中
指定 ``max_retries`` 来覆盖。指定 -1 允许无限重试，0 禁用重试。要覆盖所有提交任务的默认重试次数，
请设置操作系统环境变量 ``RAY_TASK_MAX_RETRIES``。
例如，通过将其传递给驱动程序脚本或使用 :ref:`运行时环境<runtime-environments>`。

你可以通过运行以下代码来尝试这种行为。

.. literalinclude:: ../doc_code/tasks_fault_tolerance.py
  :language: python
  :start-after: __tasks_fault_tolerance_retries_begin__
  :end-before: __tasks_fault_tolerance_retries_end__

当一个任务返回一个结果到 Ray 对象存储时，可能会丢失结果对象， **在** 原始任务已经完成之后。
这种情况下，Ray 也会尝试通过重新执行创建对象的任务来自动恢复对象。
这可以通过相同的 ``max_retries`` 选项进行配置。
参考 :ref:`对象容错 <fault-tolerance-objects>` 获取更多信息。

默认的，Ray **不会** 重试应用程序代码抛出的异常。
但是，你可以通过 ``retry_exceptions`` 参数控制应用程序级别的错误是否重试，
甚至 **哪些** 应用程序级别的错误重试。默认情况下，这是 ``False``。
要启用应用程序级别错误的重试，设置 ``retry_exceptions=True`` 来重试任何异常，或者传递一个可重试异常的列表。
示例如下。

.. literalinclude:: ../doc_code/tasks_fault_tolerance.py
  :language: python
  :start-after: __tasks_fault_tolerance_retries_exception_begin__
  :end-before: __tasks_fault_tolerance_retries_exception_end__


使用 :ref:`状态 API CLI <state-api-overview-ref>` 的 `ray list tasks -f task_id=\<task_id\>` 来查询任务退出详情：

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list tasks -f task_id=16310a0f0a45af5cffffffffffffffffffffffff01000000

.. code-block:: bash

  ======== List: 2023-05-26 10:38:08.809127 ========
  Stats:
  ------------------------------
  Total: 2

  Table:
  ------------------------------
      TASK_ID                                             ATTEMPT_NUMBER  NAME              STATE       JOB_ID  ACTOR_ID    TYPE         FUNC_OR_CLASS_NAME    PARENT_TASK_ID                                    NODE_ID                                                   WORKER_ID                                                 ERROR_TYPE
   0  16310a0f0a45af5cffffffffffffffffffffffff01000000                 0  potentially_fail  FAILED    01000000              NORMAL_TASK  potentially_fail      ffffffffffffffffffffffffffffffffffffffff01000000  94909e0958e38d10d668aa84ed4143d0bf2c23139ae1a8b8d6ef8d9d  b36d22dbf47235872ad460526deaf35c178c7df06cee5aa9299a9255  WORKER_DIED
   1  16310a0f0a45af5cffffffffffffffffffffffff01000000                 1  potentially_fail  FINISHED  01000000              NORMAL_TASK  potentially_fail      ffffffffffffffffffffffffffffffffffffffff01000000  94909e0958e38d10d668aa84ed4143d0bf2c23139ae1a8b8d6ef8d9d  22df7f2a9c68f3db27498f2f435cc18582de991fbcaf49ce0094ddb0


取消不当行为的任务
----------------------------

如果任何任务挂起，你可能希望取消任务以继续进行。你可以通过调用 ``ray.cancel`` 来取消任务的 ``ObjectRef``。
默认的，如果任务的 worker 正在执行中，这将发送一个 KeyboardInterrupt 给任务的 worker。
传递 ``force=True`` 给 ``ray.cancel`` 将强制退出 worker。查看 :func:`API 参考 <ray.cancel>` 的 ``ray.cancel`` 获取更多细节。

请注意，目前 Ray 不会自动重试已取消的任务。

有时，应用程序级别的代码可能会导致 worker 在重复执行任务后发生内存泄漏，例如由于第三方库中的错误。
为了在这些情况下取得进展，你可以在任务的 ``@ray.remote`` 装饰器中设置 ``max_calls`` 选项。
一旦 worker 执行了给定远程函数的这么多调用，它将自动退出。默认情况下，``max_calls`` 设置为无限大。