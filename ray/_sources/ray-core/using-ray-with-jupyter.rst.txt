使用 Jupyter Notebook 和 JupyterLab
===========================================

本文档介绍了将 Ray 与 Jupyter Notebook/JupyterLab 结合使用的最佳实践。
我们使用 AWS 进行说明，但这些论点也适用于其他云提供商。
如果您认为本文档缺少任何内容，请随时做出贡献。

设置 Notebook
-------------------

1. 如果您计划在 EC2 实例上运行 Notebook，请确保它具有足够的 EBS 卷。
默认情况下，深度学习 AMI、预安装的库和环境设置在 Ray 开始工作之前将占用约 76% 的磁盘空间。
在运行其他应用程序时，Notebook 可能会因磁盘已满而频繁失败。
内核重启会丢失正在进行的单元输出，尤其是当我们依赖它们来跟踪实验进度时。
相关问题： `Autoscaler 应允许配置磁盘空间，并使用更大的默认值 <https://github.com/ray-project/ray/issues/1376>`_.

2. 避免不必要的内存使用。
IPython 将每个单元的输出无限期地存储在本地 Python 变量中。
这会导致 Ray 固定对象，即使您的应用程序实际上可能没有使用它们。
因此，显式调用 ``print`` 或者 ``repr`` 比让 Notebook 自动生成输出更好。
另一个选择是使用以下命令完全禁用 IPython 缓存（从 bash/zsh 运行）：

.. code-block:: console

    echo 'c = get_config()
    c.InteractiveShell.cache_size = 0 # disable cache
    ' >>  ~/.ipython/profile_default/ipython_config.py

这仍然允许打印，但是会完全停止 IPython 缓存。

.. tip::
  虽然上述设置有助于减少内存占用，但删除应用程序不再需要的引用以释放对象存储中的空间始终是一个好习惯。

3. 了解节点的职责。
假设 Notebook 在 EC2 实例上运行，您计划在此实例上本地启动 ray 运行时，还是计划将此实例用作集群启动器？
Jupyter Notebook 更适合第一种情况。
诸如 ``ray exec`` 和 ``ray submit`` 之类的 CLI 更适合第二种情况。

4. 转发端口。
假设 Notebook 在 EC2 实例上运行，则应转发 Notebook 端口和 Ray Dashboard 端口。
默认端口分别为 8888 和 8265。如果默认端口不可用，则端口数量会增加。
您可以使用以下命令转发它们（从 bash/zsh 运行）：

.. code-block:: console

    ssh -i /path/my-key-pair.pem -N -f -L localhost:8888:localhost:8888 my-instance-user-name@my-instance-IPv6-address
    ssh -i /path/my-key-pair.pem -N -f -L localhost:8265:localhost:8265 my-instance-user-name@my-instance-IPv6-address
