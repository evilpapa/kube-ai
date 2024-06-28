.. _installation:

安装 Ray
==============

Ray 当前官方支持 x86_64, aarch64 (ARM) Linux，以及 Apple silicon (M1) 硬件。
Ray 在 Windows 中当前处于 beta 状态。

官方发布
-----------------

通过 Wheels
~~~~~~~~~~~
您可以通过选择最适合您用例的选项，在 Linux、Windows 和 macOS 上安装
来自 PyPI 的 Ray 的最新官方版本。

.. tab-set::

    .. tab-item:: 推荐

        **对于机器学习应用程序**

        .. code-block:: shell

          pip install -U "ray[data,train,tune,serve]"

          # For reinforcement learning support, install RLlib instead.
          # pip install -U "ray[rllib]"

        **对于一般的 Python 应用程序**

        .. code-block:: shell

          pip install -U "ray[default]"

          # If you don't want Ray Dashboard or Cluster Launcher, install Ray with minimal dependencies instead.
          # pip install -U "ray"

    .. tab-item:: 高级选项

        .. list-table::
          :widths: 2 3
          :header-rows: 1

          * - 命令
            - 安装的组件
          * - `pip install -U "ray"`
            - Core
          * - `pip install -U "ray[default]"`
            - Core, Dashboard, Cluster Launcher
          * - `pip install -U "ray[data]"`
            - Core, Data
          * - `pip install -U "ray[train]"`
            - Core, Train
          * - `pip install -U "ray[tune]"`
            - Core, Tune
          * - `pip install -U "ray[serve]"`
            - Core, Dashboard, Cluster Launcher, Serve
          * - `pip install -U "ray[serve-grpc]"`
            - Core, Dashboard, Cluster Launcher, Serve with gRPC support
          * - `pip install -U "ray[rllib]"`
            - Core, Tune, RLlib
          * - `pip install -U "ray[air]"`
            - Core, Dashboard, Cluster Launcher, Data, Train, Tune, Serve
          * - `pip install -U "ray[all]"`
            - Core, Dashboard, Cluster Launcher, Data, Train, Tune, Serve, RLlib

        .. tip::

          您可以组合安装附加功能。
          例如，要安装带有 Dashboard、Cluster Launcher 和 Train 支持的 Ray，您可以运行：

          .. code-block:: shell

            pip install -U "ray[default,train]"

.. _install-nightlies:

每日版本 (Nightlies)
--------------------------

您可以通过以下链接安装 nightly Ray wheels。这些每日发布是通过自动化测试进行测试的，但不会经历完整的发布过程。要安装这些轮子，请使用以下 ``pip`` 命令和 wheels：

.. code-block:: bash

  # Clean removal of previous install
  pip uninstall -y ray
  # Install Ray with support for the dashboard + cluster launcher
  pip install -U "ray[default] @ LINK_TO_WHEEL.whl"

  # Install Ray with minimal dependencies
  # pip install -U LINK_TO_WHEEL.whl

.. tab-set::

    .. tab-item:: Linux

        =============================================== ================================================
               Linux (x86_64)                                   Linux (arm64/aarch64)
        =============================================== ================================================
        `Linux Python 3.10 (x86_64)`_                    `Linux Python 3.10 (aarch64)`_
        `Linux Python 3.9 (x86_64)`_                     `Linux Python 3.9 (aarch64)`_
        `Linux Python 3.8 (x86_64)`_                     `Linux Python 3.8 (aarch64)`_
        `Linux Python 3.7 (x86_64)`_                     `Linux Python 3.7 (aarch64)`_
        `Linux Python 3.11 (x86_64) (EXPERIMENTAL)`_     `Linux Python 3.11 (aarch64) (EXPERIMENTAL)`_
        =============================================== ================================================

    .. tab-item:: MacOS

        ============================================  ==============================================
         MacOS (x86_64)                                MacOS (arm64)
        ============================================  ==============================================
        `MacOS Python 3.10 (x86_64)`_                  `MacOS Python 3.10 (arm64)`_
        `MacOS Python 3.9 (x86_64)`_                   `MacOS Python 3.9 (arm64)`_
        `MacOS Python 3.8 (x86_64)`_                   `MacOS Python 3.8 (arm64)`_
        `MacOS Python 3.7 (x86_64)`_                   `MacOS Python 3.11 (arm64) (EXPERIMENTAL)`_
        `MacOS Python 3.11 (x86_64) (EXPERIMENTAL)`_
        ============================================  ==============================================

    .. tab-item:: Windows (beta)

        .. list-table::
           :header-rows: 1

           * - Windows (beta)
           * - `Windows Python 3.10`_
           * - `Windows Python 3.9`_
           * - `Windows Python 3.8`_
           * - `Windows Python 3.7`_
           * - `Windows Python 3.11 (EXPERIMENTAL)`_

.. note::

  在 Windows 上，对多节点 Ray 集群的支持目前处于实验阶段，未经测试。
  如果您遇到问题，请在以下地址提交报告：https://github.com/ray-project/ray/issues.

.. note::

  :ref:`使用统计 <ref-usage-stats>` 收集选项 (可 :ref:`禁用 <usage-disable>`) 针对每日版本包括本地集群通过 ``ray.init()`` 以及远程集群运行的 cli 默认开启。

.. note::

  Python 3.11 支持是实验性的。

.. _`Linux Python 3.11 (x86_64) (EXPERIMENTAL)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl
.. _`Linux Python 3.10 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl
.. _`Linux Python 3.9 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl
.. _`Linux Python 3.8 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
.. _`Linux Python 3.7 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl

.. _`Linux Python 3.11 (aarch64) (EXPERIMENTAL)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_aarch64.whl
.. _`Linux Python 3.10 (aarch64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_aarch64.whl
.. _`Linux Python 3.9 (aarch64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_aarch64.whl
.. _`Linux Python 3.8 (aarch64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-manylinux2014_aarch64.whl
.. _`Linux Python 3.7 (aarch64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-manylinux2014_aarch64.whl


.. _`MacOS Python 3.11 (x86_64) (EXPERIMENTAL)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-macosx_10_15_x86_64.whl
.. _`MacOS Python 3.10 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-macosx_10_15_x86_64.whl
.. _`MacOS Python 3.9 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl
.. _`MacOS Python 3.8 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-macosx_10_15_x86_64.whl
.. _`MacOS Python 3.7 (x86_64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-macosx_10_15_x86_64.whl


.. _`MacOS Python 3.11 (arm64) (EXPERIMENTAL)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-macosx_11_0_arm64.whl
.. _`MacOS Python 3.10 (arm64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-macosx_11_0_arm64.whl
.. _`MacOS Python 3.9 (arm64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-macosx_11_0_arm64.whl
.. _`MacOS Python 3.8 (arm64)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-macosx_11_0_arm64.whl


.. _`Windows Python 3.11 (EXPERIMENTAL)`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-win_amd64.whl
.. _`Windows Python 3.10`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-win_amd64.whl
.. _`Windows Python 3.9`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-win_amd64.whl
.. _`Windows Python 3.8`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-win_amd64.whl
.. _`Windows Python 3.7`: https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp37-cp37m-win_amd64.whl

从指定提交处安装
---------------------------------

您可以使用以下模板在 ``master`` 上安装任何特定提交的 Ray wheels。您需要指定提交哈希、Ray 版本、操作系统和 Python 版本：

.. code-block:: bash

    pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/{COMMIT_HASH}/ray-{RAY_VERSION}-{PYTHON_VERSION}-{PYTHON_VERSION}-{OS_VERSION}.whl

例如，这里针对 Ray 3.0.0.dev0 wheels 的 Python 3.9，MacOS 的 ``4f2ec46c3adb6ba9f412f09a9732f436c4a5d0c9`` commit：

.. code-block:: bash

    pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/4f2ec46c3adb6ba9f412f09a9732f436c4a5d0c9/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl

针对 wheel 文件名格式有少量的变量组成；最好和 :ref:`每日构建 章节 <install-nightlies>` 列出的 URL 格式相匹配。
以下是变化的总结：

* 针对 MacOS，早于 2021 年 8 月 7 日的文件名会由 ``macosx_10_13`` 替代 ``macosx_10_15``。

.. _ray-install-java:

通过 Maven 安装 Ray Java
---------------------------
在通过 Maven 安装 Ray Java 之前，你需要通过 `pip install -U ray` 安装 Ray Python。注意 Ray Java 和 Ray Python 版本必须匹配。
注意当安装 Ray Java snapshot 版本时，也需要安装 nightly Ray Python wheels。

最新的 Ray Java 版本可在 `仓库中心 <https://mvnrepository.com/artifact/io.ray>`__ 找到。要在您的应用程序中使用最新的 Ray Java 版本，请在 ``pom.xml`` 中添加以下条目：

.. code-block:: xml

    <dependency>
      <groupId>io.ray</groupId>
      <artifactId>ray-api</artifactId>
      <version>${ray.version}</version>
    </dependency>
    <dependency>
      <groupId>io.ray</groupId>
      <artifactId>ray-runtime</artifactId>
      <version>${ray.version}</version>
    </dependency>

最新的 Ray Java snapshot 可在 `sonatype 仓库 <https://oss.sonatype.org/#nexus-search;quick~io.ray>`__ 中找到。要在您的应用程序中使用最新的 Ray Java snapshot，请在 ``pom.xml`` 中添加以下条目：

.. code-block:: xml

  <!-- only needed for snapshot version of ray -->
  <repositories>
    <repository>
      <id>sonatype</id>
      <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
      <releases>
        <enabled>false</enabled>
      </releases>
      <snapshots>
        <enabled>true</enabled>
      </snapshots>
    </repository>
  </repositories>

  <dependencies>
    <dependency>
      <groupId>io.ray</groupId>
      <artifactId>ray-api</artifactId>
      <version>${ray.version}</version>
    </dependency>
    <dependency>
      <groupId>io.ray</groupId>
      <artifactId>ray-runtime</artifactId>
      <version>${ray.version}</version>
    </dependency>
  </dependencies>

.. note::

  当你运行 ``pip install`` 来安装 Ray，Java jars 也会被安装。上述依赖项仅用于构建您的 Java 代码以及在本地模式下运行您的代码。

  如果你想在多节点 Ray 集群运行你的 Java 代码，如果（通过 ``pip install`` 安装的 Ray 和 maven 依赖项）的版本不匹配，最好在打包代码时排除 Ray jars，以避免 jar 冲突。

.. _ray-install-cpp:

安装 Ray C++
---------------

你可以通过以下方式安装和使用 Ray C++ API。

.. code-block:: bash

  pip install -U ray[cpp]

  # Create a Ray C++ project template to start with.
  ray cpp --generate-bazel-project-template-to ray-template

.. note::

  如果通过源码构建 Ray，请在运行应用程序之前从文件 ``cpp/example/.bazelrc`` 中删除构建选项 ``build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"``。相关问题是 `这个 <https://github.com/ray-project/ray/issues/26031>`_。

.. _apple-silcon-supprt:

M1 Mac (Apple Silicon) 支持
------------------------------

Ray 支持运行 Apple Silicon（例如 M1 Mac）的机器。
多节点集群未经测试。要开始本地 Ray 开发：

#. 安装 `miniforge <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>`_。

   * ``wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh``

   * ``bash Miniforge3-MacOSX-arm64.sh``

   * ``rm Miniforge3-MacOSX-arm64.sh # Cleanup.``

#. 确保你在使用 miniforge 环境 (你将在终端看到 (base) )。

   * ``source ~/.bash_profile``

   * ``conda activate``

#. 像往常一样安装 Ray。

   * ``pip install ray``

.. _windows-support:

Windows 支持
---------------

Windows 支持现在处理 Beta。Ray 支持在Windows上运行，但需要注意以下事项 (只有第一个是
Ray 指定，其余都适用于任何使用Windows的地方):

* 多节点 Ray 集群未经测试。

* 文件名在 Windows 上很棘手，Ray 可能仍有一些地方
  假定使用 UNIX 文件名而不是 Windows 文件名。
  在下游的包也是如此。

* Windows 上的性能已知较慢，因为在 Windows 上打开文件比
  其他操作系统慢得多。这可能会影响日志记录。

* Windows 没有写时复制 fork 模型，因此启动新的
  进程可能需要更多的内存。


将您遇到的任何问题提交给
`GitHub <https://github.com/ray-project/ray/issues/>`_ 。

在 Arch Linux 上安装 Ray
----------------------------

Note: 在 Arch Linux 上安装 Ray 未经 Project Ray 开发者测试。

Ray 在 Arch Linux 上通过 Arch 用户存储库 (`AUR`_) 作为 ``python-ray``。

你可以手动安装包，按照 `Arch Wiki`_ 上的说明，
或者使用 `AUR helper`_ 如 `yay`_ (推荐的简便安装)：

.. code-block:: bash

  yay -S python-ray

任何相关问题的讨论请参考 ``python-ray`` 的 AUR 页面上的 `评论部分`_ 。

.. _`AUR`: https://wiki.archlinux.org/index.php/Arch_User_Repository
.. _`Arch Wiki`: https://wiki.archlinux.org/index.php/Arch_User_Repository#Installing_packages
.. _`AUR helper`: https://wiki.archlinux.org/index.php/Arch_User_Repository#Installing_packages
.. _`yay`: https://aur.archlinux.org/packages/yay
.. _`here`: https://aur.archlinux.org/packages/python-ray

.. _ray_anaconda:

通过 conda-forge 安装
---------------------------
Ray 可作为 conda 包在 Linux 和 Windows 上安装。

.. code-block:: bash

  # 也适用于 mamba
  conda create -c conda-forge python=3.9 -n ray
  conda activate ray

  # 安装支持仪表板 + 集群启动器
  conda install -c conda-forge "ray-default"

  # 安装最小依赖的 Ray
  # conda install -c conda-forge ray

要安装 Ray 库包，使用如上 ``pip`` 或 ``conda``/``mamba``。

.. code-block:: bash

  conda install -c conda-forge "ray-air"    # installs Ray + dependencies for Ray AI Runtime
  conda install -c conda-forge "ray-tune"   # installs Ray + dependencies for Ray Tune
  conda install -c conda-forge "ray-rllib"  # installs Ray + dependencies for Ray RLlib
  conda install -c conda-forge "ray-serve"  # installs Ray + dependencies for Ray Serve

要查看在 Conda-forge 上的完成可用 ``ray`` 库包，
参考 https://anaconda.org/conda-forge/ray-default

.. note::

  Ray conda 库包由社区维护，而非 Ray 团队。即使
  使用 conda 环境，也建议使用
  `pip-install-ray` 在新创建的环境中从 PyPi 安装 Ray。

从源码构建 Ray
------------------------

对于大多数 Ray 用户来说，从 ``pip`` 进行安装就足够了。

然而，如果需要从源码构建，参考 :ref:`这篇 Ray 构建说明 <building-ray>`。


.. _docker-images:

Docker 源镜像
--------------------

多数用户应该从 `Ray Docker Hub <https://hub.docker.com/r/rayproject/>`__ 拉取 Ray 镜像。

- ``rayproject/ray`` `镜像 <https://hub.docker.com/r/rayproject/ray>`__ 包含 Ray 和 所有必须的依赖。它附带了 anaconda 和各种版本的 Python。
- ``rayproject/ray-ml`` `镜像 <https://hub.docker.com/r/rayproject/ray-ml>`__ 包含以上以及众多 ML 类库。
- ``rayproject/base-deps`` 和 ``rayproject/ray-deps`` 镜像分别为 Linux 和 Python 依赖。

镜像 `tagged` 格式为 ``{Ray version}[-{Python version}][-{Platform}]``。``Ray version`` 标签可能为以下某种：

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Ray 版本 tag
     - 描述
   * - latest
     - 最新的 Ray 版本。
   * - x.y.z
     - 特定 Ray 版本，如：1.12.1
   * - nightly
     - 最近的 Ray 开发构建 (最新的 Github ``master`` 提交)
   * - 6 字符 Git SHA 前缀
     - 特定开发构建 (使用来自 Github ``master`` 的 SHA，例如 ``8960af``)。

可选的 ``Python version`` 标签指定了镜像的 Python 版本。Ray 支持所有可用的 Python 版本。如 ``py37``, ``py38``, ``py39`` and ``py310``。如果未指定，镜像使用 ``Python 3.7``。

可选的 ``Platform`` 标签指定用于镜像平台：

.. list-table::
   :widths: 16 40
   :header-rows: 1

   * - 平台标签
     - 描述
   * - -cpu
     - 以 Ubuntu 为基础镜像。
   * - -cuXX
     - 这里以 NVIDIA CUDA 进行集成特定的 CUDA 版本为基础。需要 Nvidia Docker Runtime。
   * - -gpu
     - 为 ``-cuXX`` 镜像标签的别名。
   * - <no tag>
     - 为 ``-cpu`` 镜像标签的别名。针对 ``ray-ml`` 镜像，为 ``-gpu`` 镜像标签别名。

例如：对于基于 ``Python 3.8`` 的 nightly 镜像且不支持 GPU，标签为 ``nightly-py38-cpu``。

如果您想调整这些镜像的某些方面并在本地构建它们，请参考以下脚本：

.. code-block:: bash

  cd ray
  ./build-docker.sh

除了创建上面的 Docker 镜像，这个脚本还可以生成以下两个镜像。


- ``rayproject/development`` 镜像包含了为开发而设置的 ray 源码。
- ``rayproject/examples`` 镜像添加了额外的运行示例的类库。

通过命令查看镜像：


.. code-block:: bash

  docker images

输出应该类似于以下内容：

.. code-block:: bash

  REPOSITORY                          TAG                 IMAGE ID            CREATED             SIZE
  rayproject/ray                      latest              7243a11ac068        2 days ago          1.11 GB
  rayproject/ray-deps                 latest              b6b39d979d73        8 days ago          996  MB
  rayproject/base-deps                latest              5606591eeab9        8 days ago          512  MB
  ubuntu                              focal               1e4467b07108        3 weeks ago         73.9 MB


在 Docker 运行 Ray
~~~~~~~~~~~~~~~~~~~~

从运行本地容器开始。

.. code-block:: bash

  docker run --shm-size=<shm-size> -t -i rayproject/ray

针对你的系统设置适当的 ``<shm-size>``，例如
``512M`` 或 ``2G``。合理的预估大致为可用内存的 30% (这是因为
Ray 内部会使用它自己的对象存储)。``-t`` 和 ``-i`` 选项用于
容器的交互。

如果您使用 GPU 版本的 Docker 镜像，请记得添加 ``--gpus all`` 选项。在以下命令中替换 ``<ray-version>`` 为您的目标 ray 版本：

.. code-block:: bash

  docker run --shm-size=<shm-size> -t -i --gpus all rayproject/ray:<ray-version>-gpu

**Note:** Ray 需要 **巨大** 数量的共享内存，以为每个对象存储
的共享内存都保存了所有的对象，所有共享内存大小将
限制对象存储的大小。

现在，您应该会看到一个提示，内容如下：


.. code-block:: bash

  root@ebc78f68d100:/ray#

测试安装是否成功
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要测试安装是否成功，请尝试运行一些测试。这假定
您已经克隆了 git 仓库。

.. code-block:: bash

  python -m pytest -v python/ray/tests/test_mini.py


安装 Python 依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们的 Docker 镜像已经预装了 Ray 和
所需的 Python 依赖。

我们发布了 ``ray`` 和 ``ray-ml`` Docker 镜像中安装的依赖项，
适用于 Python 3.9。

.. tabs::

    .. group-tab:: ray (Python 3.9)

        Ray version: 2.7.1 (`9f07c12 <https://github.com/ray-project/ray/commit/9f07c12615958c3af3760604f6dcacc4b3758a47>`_)

        .. literalinclude:: ./pip_freeze_ray-py39-cpu.txt

    .. group-tab:: ray-ml (Python 3.9)

        Ray version: 2.7.1 (`9f07c12 <https://github.com/ray-project/ray/commit/9f07c12615958c3af3760604f6dcacc4b3758a47>`_)

        .. literalinclude:: ./pip_freeze_ray-ml-py39-cpu.txt
