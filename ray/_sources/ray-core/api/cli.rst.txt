Ray Core CLI
============

.. _ray-cli:

应用调试
----------------------
本节包含用于检查和调试当前集群的命令。

.. _ray-stack-doc:

.. click:: ray.scripts.scripts:stack
   :prog: ray stack
   :show-nested:

.. _ray-memory-doc:

.. click:: ray.scripts.scripts:memory
   :prog: ray memory
   :show-nested:

.. _ray-timeline-doc:

.. click:: ray.scripts.scripts:timeline
   :prog: ray timeline
   :show-nested:

.. _ray-status-doc:

.. click:: ray.scripts.scripts:status
   :prog: ray status
   :show-nested:

.. click:: ray.scripts.scripts:debug
   :prog: ray debug
   :show-nested:


使用统计
-----------
本节包含用于启用/禁用 :ref:`Ray 使用统计 <ref-usage-stats>` 的命令。

.. _ray-disable-usage-stats-doc:

.. click:: ray.scripts.scripts:disable_usage_stats
   :prog: ray disable-usage-stats
   :show-nested:

.. _ray-enable-usage-stats-doc:

.. click:: ray.scripts.scripts:enable_usage_stats
   :prog: ray enable-usage-stats
   :show-nested: