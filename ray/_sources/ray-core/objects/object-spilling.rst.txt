对象溢出
===============
.. _object-spilling:

一旦对象存储已满，Ray 1.3+ 就会将对象溢出到外部存储。默认情况下，对象会溢出到本地文件系统中 Ray 的临时目录中。

单节点
-----------

Ray 默认使用对象溢出。如果不进行任何设置，对象将溢出到 `[temp_folder]/spill`。在 Linux 和 MacOS 上，默认情况下 `temp_folder` 是 `/tmp` 。

要配置对象溢出到的目录，请使用：

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
            )
        },
    )

您还可以指定多个目录进行溢出，以便在需要时将 IO 负载和磁盘空间使用情况分散到多个物理设备（例如 SSD 设备）上：

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for parallelism.
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    # Multiple directories can be specified to distribute
                    # IO across multiple mounted physical devices.
                    "directory_path": [
                      "/tmp/spill",
                      "/tmp/spill_1",
                      "/tmp/spill_2",
                    ]
                  },
                }
            )
        },
    )


.. note::

    为了优化性能，建议在使用对象溢出处理内存密集型工作负载时使用 SSD 而不是 HDD。

如果您使用的是 HDD，建议您指定较大的缓冲区大小 (> 1MB) 以减少溢出期间的 IO 请求。

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    "directory_path": "/tmp/spill",
                    "buffer_size": 1_000_000,
                  }
                },
            )
        },
    )

为防止磁盘空间耗尽，如果磁盘利用率超过预定义阈值，将抛出本地对象溢出 ``OutOfDiskError`` 。
如果使用多个物理设备，任何物理设备的过度使用都将触发 ``OutOfDiskError``。
默认阈值为 0.95（95%）。您可以通过设置 ``local_fs_capacity_threshold``，或将其设置为 1 以禁用保护。

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            # Allow spilling until the local disk is 99% utilized.
            # This only affects spilling to the local file system.
            "local_fs_capacity_threshold": 0.99,
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    "directory_path": "/tmp/spill",
                  }
                },
            )
        },
    )


使对象溢出到远程存储（任何支持 `smart_open <https://pypi.org/project/smart-open/>`__ 的URI）：

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::
  :skipif: True

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for remote storage.
            "min_spilling_size": 100 * 1024 * 1024,  # Spill at least 100MB at a time.
            "object_spilling_config": json.dumps(
                {
                  "type": "smart_open",
                  "params": {
                    "uri": "s3://bucket/path"
                  },
                  "buffer_size": 100 * 1024 * 1024,  # Use a 100MB buffer for writes
                },
            )
        },
    )

建议您指定一个较大的缓冲区大小 (> 1MB) 以减少溢出期间的 IO 请求。

还支持溢出到多个远程存储。

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::
  :skipif: True

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for remote storage.
            "min_spilling_size": 100 * 1024 * 1024,  # Spill at least 100MB at a time.
            "object_spilling_config": json.dumps(
                {
                  "type": "smart_open",
                  "params": {
                    "uri": ["s3://bucket/path1", "s3://bucket/path2", "s3://bucket/path3"],
                  },
                  "buffer_size": 100 * 1024 * 1024, # Use a 100MB buffer for writes
                },
            )
        },
    )

远程存储支持仍处于实验阶段。

集群模式
------------
要在多节点群集中启用对象溢出：

.. code-block:: bash

  # Note that `object_spilling_config`'s value should be json format.
  # You only need to specify the config when starting the head node, all the worker nodes will get the same config from the head node.
  ray start --head --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/tmp/spill\"}}"}'

统计
-----

当发生溢出时，以下 INFO 级别的消息将被打印到 raylet 日志中（例如 ``/tmp/ray/session_latest/logs/raylet.out``）::

  local_object_manager.cc:166: Spilled 50 MiB, 1 objects, write throughput 230 MiB/s
  local_object_manager.cc:334: Restored 50 MiB, 1 objects, read throughput 505 MiB/s

您还可以使用以下命令  ``ray memory`` 查看集群范围的溢出统计信息 ::

  --- Aggregate object store stats across all nodes ---
  Plasma memory usage 50 MiB, 1 objects, 50.0% full
  Spilled 200 MiB, 4 objects, avg write throughput 570 MiB/s
  Restored 150 MiB, 3 objects, avg read throughput 1361 MiB/s

如果您只想显示集群范围的溢出统计数据，请使用。 ``ray memory --stats-only``。
