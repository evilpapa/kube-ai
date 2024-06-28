:orphan:

配置概述
======================

.. _train-run-config:

在 Train 中运行配置 （``RunConfig``）
------------------------------------------

``RunConfig`` 是 Ray Train 中用来定义与调用 ``trainer.fit()`` 相对应的实验规范的配置对象。

它包含了设置诸如实验名称、结果存储路径、停
止条件、自定义回调、检查点配置、详细程度和日志选项等的设置。

某些设置是通过其他配置对象进行配置并通过 ``RunConfig`` 传递的。
以下各小节包含这些配置的描述。

运行配置的属性是 :ref:`不可调整的 <tune-search-space-tutorial>`。

.. literalinclude:: ../doc_code/key_concepts.py
    :language: python
    :start-after: __run_config_start__
    :end-before: __run_config_end__

.. seealso::

    查看 :class:`~ray.train.RunConfig` API 参考。

    查看 :ref:`persistent-storage-guide` 以获取存储配置示例 （关联 ``storage_path``）。

