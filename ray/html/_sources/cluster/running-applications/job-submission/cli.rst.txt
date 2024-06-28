.. _ray-job-submission-cli-ref:

Ray Jobs CLI API 参考
==========================

本节包含了 :ref:`Ray Job Submission <jobs-quickstart>` 相关命令。

.. _ray-job-submit-doc:

.. click:: ray.dashboard.modules.job.cli:submit
   :prog: ray job submit

.. warning::

    使用 CLI 时，不要将 entrypoint 命令括在引号中。比如，使用
    ``ray job submit --working_dir="." -- python script.py`` 代替 ``ray job submit --working_dir="." -- "python script.py"``.
    否则你可能会遇到错误 ``/bin/sh: 1: python script.py: not found``.

.. warning::

   你必须提供 entrypoint 命令， ``python script.py``, 最后（在 ``--`` 之后）以及 `ray job submit` 的任何其他参数（例如 ``--working_dir="."``）必须在两个连字符（``--``）之前提供。
   比如使用 ``ray job submit --working_dir="." -- python script.py`` 而不是 ``ray job submit -- python script.py --working_dir="."``.
   此语法支持使用 “--” 将 “ray job submit” 的参数与 entrypoint 命令的参数分开。

.. _ray-job-status-doc:

.. click:: ray.dashboard.modules.job.cli:status
   :prog: ray job status
   :show-nested:

.. _ray-job-stop-doc:

.. click:: ray.dashboard.modules.job.cli:stop
   :prog: ray job stop
   :show-nested:

.. _ray-job-logs-doc:

.. click:: ray.dashboard.modules.job.cli:logs
   :prog: ray job logs
   :show-nested:

.. _ray-job-list-doc:

.. click:: ray.dashboard.modules.job.cli:list
   :prog: ray job list
   :show-nested:
