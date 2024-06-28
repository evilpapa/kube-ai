.. _handling_dependencies:

环境依赖
========================

你的 Ray 应用程序可能具有 Ray 脚本之外的依赖项。例如：

* 您的 Ray 脚本可能会导入/依赖一些 Python 包。
* 您的 Ray 脚本可能正在寻找一些可用的特定环境变量。
* 你的 Ray 脚本可能会导入一些脚本之外的文件。

在集群上运行时，一个常见问题是 Ray 期望这些“依赖项”存在于每个 Ray 节点上。如果不存在这些，您可能会遇到诸如 ``ModuleNotFoundError`` 或 ``FileNotFoundError`` 等问题。

为了解决这个问题，您可以 (1) 使用 Ray :ref:`Cluster Launcher <vm-cluster-quick-start>` 提前准备集群上的依赖项（例如使用容器映像），或者 (2) 使用 Ray 的 :ref:`运行时环境 <runtime-environments>` 动态安装它们。

对于生产用途或非变更环境，我们建议将依赖项安装到容器映像中，并使用 Cluster Launcher 指定映像。
对于动态环境（例如用于开发和实验），我们建议使用运行时环境。


概念
--------

- **Ray 应用程序**。  包含 Ray 脚本的程序，用于调用 ``ray.init()`` 和使用 Ray task 或 actor。

- **依赖项** 或 **环境**。  应用程序需要运行的 Ray 脚本之外的任何内容，包括文件、包和环境变量。

- **文件**。 你的 Ray 应用程序运行所需的代码文件、数据文件或其他文件。

- **包**。 Ray 应用程序所需的外部库或可执行文件，通常通过 ``pip`` 或 ``conda`` 安装。

- **本地机器** 和 **集群**。  通常，您可能希望将 Ray 集群计算机器/pod 与处理和提交应用程序的机器/pod 分开。您可以通过 :ref:`Ray 作业提交机制 <jobs-overview>` 提交 Ray 作业，或使用 `ray attach` 以交互方式连接到集群。 我们将提交作业的机器称为 *本地机器*.

- **作业**.  :ref:`Ray 作业 <cluster-clients-and-jobs>` 是单个应用程序：它是来自同一脚本的 Ray 任务、对象和 actor 的集合。

.. _using-the-cluster-launcher:

使用 Ray Cluster launcher 准备环境
-------------------------------------------------------

设置依赖项的第一种方法是在启动 Ray 运行时之前准备跨集群的单一环境。

- 您可以将所有文件和依赖项构建到容器映像中，并在 :ref:`Cluster YAML 配置项 <cluster-config>` 指定。

- 您还可以在Ray Cluster 配置文件（:ref:`参考 <cluster-configuration-setup-commands>`）使用 ``setup_commands`` 安装依赖项；这些命令将在每个节点加入集群时运行。
  请注意，对于生产设置，建议将任何必要的软件包构建到容器映像中。

- 您可以使用使用 ``ray rsync_up`` (:ref:`参考<ray-rsync>`) 将本地文件推送到集群。

.. _runtime-environments:

运行环境
--------------------

.. note::

    此特性需要使用 ``pip install "ray[default]"`` 完整安装 Ray 。此功能从 Ray 1.4.0 开始可用，目前支持 macOS 和 Linux，在 Windows 上提供测试版支持。

设置依赖项的第二种方法是在 Ray 运行时动态安装它们。

**运行时环境** 描述了 Ray 应用程序运行时所需的依赖项，包括 :ref:`文件、包、环境变量等 <runtime-environments-api-ref>`。
它在运行时动态安装在集群上并缓存以供将来使用 （有关生命周期的详细信息，请参阅 :ref:`缓存和垃圾收集 <runtime-environments-caching>`）。

如果使用了 :ref:`Ray 集群启动器 <using-the-cluster-launcher>` ，则可以在准备好的环境上使用运行时环境。
例如，您可以使用 Cluster 启动器安装一组基本软件包，然后使用运行时环境安装其他软件包。
与基本集群环境相比，运行时环境仅对 Ray 进程有效。（例如，如果使用运行时环境 指定的 ``pip`` 包 ``my_pkg``，则在 Ray 任务、 actor 或作业之外调用 ``import my_pkg`` 语句将失败。）

运行时环境还允许您在长期运行的 Ray 集群上设置每个任务、actor 和每个作业的依赖项。

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

    import ray

    runtime_env = {"pip": ["emoji"]}

    ray.init(runtime_env=runtime_env)

    @ray.remote
    def f():
      import emoji
      return emoji.emojize('Python is :thumbs_up:')

    print(ray.get(f.remote()))

.. testoutput::

    Python is 👍

运行时环境可以用 Python  `dict` 来描述：

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __runtime_env_pip_def_start__
   :end-before: __runtime_env_pip_def_end__

或者，你可以使用 :class:`ray.runtime_env.RuntimeEnv <ray.runtime_env.RuntimeEnv>`：

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __strong_typed_api_runtime_env_pip_def_start__
   :end-before: __strong_typed_api_runtime_env_pip_def_end__

欲了解更多示例，请跳转至 :ref:`API 参考 <runtime-environments-api-ref>`。


您可以为两个主要范围指定运行时环境：

* :ref:`每个 Job <rte-per-job>`，以及
* :ref:`在一个 job 内的每个 Task/Actor <rte-per-task-actor>`。

.. _rte-per-job:

每个作业指定运行时环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

您可以为整个作业指定运行环境，无论是直接在集群上运行脚本，还是使用 :ref:`Ray 作业 API <jobs-overview>`:

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __ray_init_start__
   :end-before: __ray_init_end__

.. testcode::
    :skipif: True

    # Option 2: Using Ray Jobs API (Python SDK)
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient("http://<head-node-ip>:8265")
    job_id = client.submit_job(
        entrypoint="python my_ray_script.py",
        runtime_env=runtime_env,
    )

.. code-block:: bash

    # Option 3: Using Ray Jobs API (CLI). (Note: can use --runtime-env to pass a YAML file instead of an inline JSON string.)
    $ ray job submit --address="http://<head-node-ip>:8265" --runtime-env-json='{"working_dir": "/data/my_files", "pip": ["emoji"]}' -- python my_ray_script.py

.. warning::

    如果使用 Ray Jobs API（Python SDK 或 CLI），请在调用 ``submit_job`` 或者 ``ray job submit`` 指定 ``runtime_env`` 参数，而不是在入口点脚本中的 ``ray.init()`` 调用中指定（在这个示例中为 ``my_ray_script.py``）。

    这可确保在运行入口点脚本之前在集群上安装运行时环境。

.. note::

  何时安装运行环境有两种选择：

  1. 一旦作业开始（例如， ``ray.init()`` 被调用），依赖项将被急切地下载和安装。
  2. 仅当调用任务或创建 actor 时才会安装依赖项。

  默认是选项 1。要将行为更改为选项 2，请添加 ``"eager_install": False``到 ``config`` 的 ``runtime_env``。

.. _rte-per-task-actor:

为每个任务或每个 Actor 指定运行时
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

您可以使用 ``@ray.remote`` 装饰器或者 ``.options()`` 为每个 actor 或每个任务指定不同的运行时环境：

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __per_task_per_actor_start__
   :end-before: __per_task_per_actor_end__

这样，actor 和任务就可以在自己的环境中运行，不受周围环境的影响。（周围环境可能是作业的运行时环境，也可能是集群的系统环境。）

.. warning::

  Ray 不保证运行时环境冲突的任务和 Actor 之间的兼容性。
  例如，如果运行时环境包含某个 ``pip`` 包的 Actor 尝试与使用该包的不同版本的 Actor 进行通信，则可能会导致意外行为，例如解包错误。

常见工作流程
^^^^^^^^^^^^^^^^

本节介绍运行时环境的一些常见用例。这些用例并不相互排斥；下面描述的所有选项都可以组合在一个运行时环境中。

.. _workflow-local-files:

使用本地文件
"""""""""""""""""

您的 Ray 应用程序可能依赖源文件或数据文件。
对于开发工作流程，这些文件可能位于您的本地机器上，但当需要大规模运行时，您需要将它们转移到远程集群。

以下简单示例说明如何在集群上获取本地文件。

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

  import os
  import ray

  os.makedirs("/tmp/runtime_env_working_dir", exist_ok=True)
  with open("/tmp/runtime_env_working_dir/hello.txt", "w") as hello_file:
    hello_file.write("Hello World!")

  # Specify a runtime environment for the entire Ray job
  ray.init(runtime_env={"working_dir": "/tmp/runtime_env_working_dir"})

  # Create a Ray task, which inherits the above runtime env.
  @ray.remote
  def f():
      # The function will have its working directory changed to its node's
      # local copy of /tmp/runtime_env_working_dir.
      return open("hello.txt").read()

  print(ray.get(f.remote()))

.. testoutput::

  Hello World!

.. note::
  上面的例子是为了在本地机器上运行而编写的，但是对于所有这些例子，在指定要连接的 Ray 集群时它也同样有效
  （比如：使用 ``ray.init("ray://123.456.7.89:10001", runtime_env=...)`` 或 ``ray.init(address="auto", runtime_env=...)``）。

指定的本地目录将在 ``ray.init()`` 调用时自动推送到集群节点。

您还可以通过远程云存储 URI 指定文件；有关详细信息，请参阅 :ref:`remote-uris`。

使用 ``conda`` 或 ``pip`` 包
"""""""""""""""""""""""""""""""""""

您的 Ray 应用程序可能通过 ``import`` 语句依赖了 Python 包（比如，``pendulum`` 或 ``requests``）。

Ray 通常希望所有导入的包都预先安装在集群的每个节点上；特别是，这些包不会自动从本地机器运送到集群或从任何存储库下载。

但是，使用运行时环境，您可以动态指定要在虚拟环境中自动下载并安装的包，以供您的 Ray 作业或特定的 Ray 任务或 actor 使用。

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

  import ray
  import requests

  # This example runs on a local machine, but you can also do
  # ray.init(address=..., runtime_env=...) to connect to a cluster.
  ray.init(runtime_env={"pip": ["requests"]})

  @ray.remote
  def reqs():
      return requests.get("https://www.ray.io/").status_code

  print(ray.get(reqs.remote()))0

.. testoutput::

  200


你也可以通过 Python 列表或本地 ``requirements.txt`` 文件指定 ``pip`` 依赖项。
或者，你可以指定 ``conda`` 环境，可是通过 Python 词典或者一个本地的 ``environment.yml`` 文件。
这个 conda 环境可以包含 ``pip`` 包。
有关详细信息，请查看 :ref:`API 参考 <runtime-environments-api-ref>`。

.. warning::

  由于 ``runtime_env`` 中的包是在运行时安装，因此在指定需要从源代码构建的 ``conda`` 或 ``pip`` 包时要小心，因为这可能会很慢。

.. note::

  当使用 ``"pip"`` 字段时，指定的包将使用 ``virtualenv`` 安装在基础环境之上，因此集群上的现有包仍然可以导入。 相比之下，当使用 ``conda`` 字段时，您的 Ray 任务和 actor 将在隔离的环境中运行。 ``conda`` 和 ``pip`` 字段不能同时在单个 ``runtime_env`` 中使用。

.. note::

  ``ray[default]`` 软件包，本身会自动安装到环境中。仅对于 ``conda`` 字段，如果使用了任何其他 Ray 类库（如：Ray Serve），则需要在运行时环境指定类库（如： ``runtime_env = {"conda": {"dependencies": ["pytorch", "pip", {"pip": ["requests", "ray[serve]"]}]}}``。）

.. note::

  ``conda`` 环境必须与 Ray 集群是统一 Python 版本。不要在 ``conda`` 依赖列出 ``ray``，因为它会自动安装。

类库开发
"""""""""""""""""""

假设您正在开发一个 Ray 上的库 ``my_module``。

典型的迭代周期包括：

1. ``my_module`` 对 ``my_module`` 源代码进行一些更改。
2. 运行一个 Ray 脚本来测试更改，可能在分布式集群上。

要确保你的本地更改在所有 Ray workers 上显示并且可以正确导入，请使用 ``py_modules`` 字段。

.. testcode::
  :skipif: True

  import ray
  import my_module

  ray.init("ray://123.456.7.89:10001", runtime_env={"py_modules": [my_module]})

  @ray.remote
  def test_my_module():
      # No need to import my_module inside this function.
      my_module.test()

  ray.get(f.remote())

注意：此功能目前仅限于包含 ``__init__.py`` 文件的单个目录的包。对于单文件模块，您可以使用 ``working_dir``。

.. _runtime-environments-api-ref:

API 参考
^^^^^^^^^^^^^

``runtime_env`` 是个 Python 字典或者一个 Python 类 :class:`ray.runtime_env.RuntimeEnv <ray.runtime_env.RuntimeEnv>` 包含以下一个或多个字段：

- ``working_dir`` (str): 指定 Ray worker 的工作目录。必须是 (1) 一个大小最多在 100 MiB 的本地存在的目录，(2) 一个解压后大小最多 100 MiB 的压缩文件（注意：``excludes`` 会不生效），或者 (3) 包含作业工作目录的远程存储压缩文件的 URI。有关详细信息，请参阅 :ref:`remote-uris` 。
  指定的目录将下载到集群上的每个节点，并且 Ray 工作程序将在其节点的此目录副本中启动。

  - 示例

    - ``"."  # cwd``

    - ``"/src/my_project"``

    - ``"/src/my_project.zip"``

    - ``"s3://path/to/my_dir.zip"``

  注意：目前不支持按任务或按 actor 设置本地目录；只能按作业进行设置（即，在 ``ray.init()`` ）。

  注意：如果本地目录包含 ``.gitignore`` 文件，则其中指定的文件和路径不会上传到集群。您可以通过在执行上传的机器上设置环境变量 `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` 来禁用此功能。

- ``py_modules`` (List[str|module]): 指定可在 Ray worker 中导入的 Python 模块。  （有关指定包的更多方法，另请参阅下面的 ``pip`` 和 ``conda`` 字段。）
  每个条目必须是 (1) 本地目录的路径、(2) 远程 zip 文件的 URI、(3) Python 模块对象、或者 (4) 本地 `.whl` 文件的路径。

  - 列表中条目的示例：

    - ``"."``

    - ``"/local_dependency/my_module"``

    - ``"s3://bucket/my_module.zip"``

    - ``my_module # Assumes my_module has already been imported, e.g. via 'import my_module'``

    - ``my_module.whl``

  这些模块将被下载到集群上的每个节点。

  注意：目前不支持按任务或按 actor 设置选项 (1)、(3) 和 (4)，只能按作业设置（即，在 ``ray.init()`` 中）。

  注意：对于选项 (1)，如果本地目录包含 ``.gitignore`` 文件，则不会将其中指定的文件和路径上传到集群。您可以通过在 `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` 执行上传的机器上设置环境变量来禁用此功能。

  注意：此功能目前仅限于包含 ``__init__.py`` 文件的单个目录的包。对于单文件模块，您可以使用 ``working_dir``。

- ``excludes`` (List[str]): 当使用了 ``working_dir`` 或 ``py_modules``，指定要排除在上传到集群之外的文件或路径列表。
  此字段使用 ``.gitignore`` 文件使用的模式匹配语法：有关详细信息，请参阅 `<https://git-scm.com/docs/gitignore>`_。
  注意：根据 ``.gitignore`` 语法，如果模式的开头或中间（或两者）有分隔符（``/``），则该模式相对于 ``working_dir`` 的级别进行解释。
  特别是，你不应该在 `excludes` 使用绝对路径（例如 `/Users/my_working_dir/subdir/`）；相反，你应该使用相对路径 `/subdir/`（这里写成 `/` 开头，以匹配只有顶级 `subdir` 目录，而不是所有级别的所有名为 `subdir` 的目录。）

  - 示例: ``{"working_dir": "/Users/my_working_dir/", "excludes": ["my_file.txt", "/subdir/, "path/to/dir", "*.log"]}``

- ``pip`` (dict | List[str] | str): (1) pip `requirements specifiers <https://pip.pypa.io/en/stable/cli/pip_install/#requirement-specifiers>`_ 列表， (2) 包含本地 pip 
  `“requirements.txt” <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_ 文件路径的字符串，或者 (3) 具有三个字段的 python 字典： (a) ``packages`` (必须， List[str]): pip 包列表，
  (b) ``pip_check`` (可选, bool): 是否在 pip install 结束时启用 `pip check <https://pip.pypa.io/en/stable/cli/pip_check/>`_ ，默认为 ``False``。
  (c) ``pip_version`` (可选, str): pip 的版本； Ray 将在  ``pip_version`` 前面拼写 “pip” 包名称以形成最终的需求字符串。
  requirement 定义语法在 `PEP 508 <https://www.python.org/dev/peps/pep-0508/>`_ 中有详细说明。
  这将在运行时安装在 Ray worker 上。 集群预安装环境中的包仍然可用。
  要使用像 Ray Serve 或 Ray Tune 这样的库，你需要在这里包含 ``"ray[serve]"`` 或 ``"ray[tune]"``。
  Ray 版本必须与集群的版本匹配，因此您可能不应该手动指定它们。

  - 示例: ``["requests==1.0.0", "aiohttp", "ray[serve]"]``

  - 示例: ``"./requirements.txt"``

  - 示例: ``{"packages":["tensorflow", "requests"], "pip_check": False, "pip_version": "==22.0.2;python_version=='3.8.11'"}``

  指定 ``requirements.txt`` 文件路径时，文件必须存在于本地计算机上，并且必须是相对于本地当前工作目录的有效绝对路径或相对文件路径，而 *不是* 相对于 `runtime_env` 指定的 `working_dir`。
  此外，不支持在 `requirements.txt` 文件中引用本地文件。（如：``-r ./my-laptop/more-requirements.txt``，``./my-pkg.whl``）。

- ``conda`` (dict | str): (1) 示 conda 环境 YAML 的字典， (2) 包含本地
  `conda “environment.yml” <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually>`_ 文件的路径字符串，
  或 (3) 已安装在集群中每个节点上的本地 conda 环境的名称（例如， ``"pytorch_p36"``）。
  前两种情况下，Ray 和 Python 依赖项将自动注入到环境中以确保兼容性，因此无需手动包含它们。
  Python 和 Ray 版本必须与集群的版本匹配，因此您可能不应该手动指定它们。
  请注意，``conda`` 和 ``pip`` 的 ``runtime_env`` 键不能同时指定---要一起使用它们，请使用 ``conda`` 并在 conda 的 ``environment.yaml`` 中的 ``"pip"`` 字段中添加您的 pip 依赖项。

  - 示例: ``{"dependencies": ["pytorch", "torchvision", "pip", {"pip": ["pendulum"]}]}``

  - 示例: ``"./environment.yml"``

  - 示例: ``"pytorch_p36"``

  指定 ``environment.yml`` 文件路径时，文件必须存在于本地计算机上，并且必须是相对于本地当前工作目录的有效绝对路径或相对文件路径，而 *不是* 相对于 `runtime_env` 指定的 `working_dir`。
  此外，不支持在 `environment.yml` 文件中引用本地文件。

- ``env_vars`` (Dict[str, str]): 要设置的环境变量。  集群上已经设置的环境变量仍旧对于 Ray worker 可见；所以不需要在 ``env_vars`` 字段中包含 ``os.environ`` 或类似的内容。
  默认，这些环境变量会覆盖集群上相同名称的环境变量。
  你可以使用 ${ENV_VAR} 引用现有的环境变量来实现追加行为。
  如果环境变量不存在，它会变成一个空字符串 `""`。

  - 示例: ``{"OMP_NUM_THREADS": "32", "TF_WARNINGS": "none"}``

  - 示例: ``{"LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}:/home/admin/my_lib"}``

  - 不存在的变量示例: ``{"ENV_VAR_NOT_EXIST": "${ENV_VAR_NOT_EXIST}:/home/admin/my_lib"}`` -> ``ENV_VAR_NOT_EXIST=":/home/admin/my_lib"``.

- ``container`` (dict): 需要给定的 (Docker) 映像，, 工作进程将在具有此映像的容器中运行。
  `worker_paht` 是默认的 `default_worker.py` 路径。只有当容器中的 ray 安装目录与 raylet 主机不同时才需要。
  `run_options` 列表规范在 `这里 <https://docs.docker.com/engine/reference/run/>`__。

  - 示例: ``{"image": "anyscale/ray-ml:nightly-py38-cpu", "worker_path": "/root/python/ray/workers/default_worker.py", "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]}``

  注意：``container`` 目前是实验性的。如果您有一些要求或遇到任何问题，请在 `github <https://github.com/ray-project/ray/issues>`__ 提交讨论。

- ``config`` (dict | :class:`ray.runtime_env.RuntimeEnvConfig <ray.runtime_env.RuntimeEnvConfig>`): 运行时配置。可以是 dict 或 RuntimeEnvConfig。
  字段：
  (1) setup_timeout_seconds ，运行时环境创建的超时时间，单位为秒。

  - 示例: ``{"setup_timeout_seconds": 10}``

  - 示例: ``RuntimeEnvConfig(setup_timeout_seconds=10)``

  (2) ``eager_install`` (bool): 表示是否在 ``ray.init()`` 时在集群上安装运行时环境，而不是在工作程序被租用之前。默认情况下，此标志设置为 ``True``。
  如果设置为 ``False``，则只有在调用第一个任务或创建第一个 actor 时才会安装运行时环境。
  当前，不支持为每个 actor 或每个任务指定此选项。

  - 示例: ``{"eager_install": False}``

  - 示例: ``RuntimeEnvConfig(eager_install=False)``

.. _runtime-environments-caching:

缓存和垃圾回收
""""""""""""""""""""""""""""""
在每个节点上的运行时资源（如 conda 环境、pip 包或下载的 ``working_dir`` 或 ``py_modules`` 目录）将被缓存在集群上，以便在作业内的不同运行时环境之间快速重用。
每个字段（``working_dir``、``py_modules`` 等）都有自己的缓存，默认大小为 10 GB。要更改此默认值，您可以在启动 Ray 之前在集群中的每个节点上设置环境变量 ``RAY_RUNTIME_ENV_<field>_CACHE_SIZE_GB``，例如 ``export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB=1.5``。

当超过缓存大小限制时，不被任何 actor、任务或作业当前使用的资源将被删除。

继承
"""""""""""

运行时环境是可继承的，因此它将应用于作业中的所有任务 / actor 以及task 或 actor的所有子任务 / actor，除非它被覆盖。

如果一个 actor 或任务指定了一个新的 ``runtime_env``，它将覆盖父级的 ``runtime_env``（即，父级 actor/任务的 ``runtime_env``，或者如果没有父级 actor 或任务，则是作业的 ``runtime_env``）如下：

* ``runtime_env["env_vars"]`` 字段会与父级的 ``runtime_env["env_vars"]`` 字段合并。
  这允许在父级的运行时环境中设置的环境变量自动传播到子级，即使在子级的运行时环境中设置了新的环境变量。
* ``runtime_env`` 字段中的其他每个字段都将被子级 *覆盖*，而不是合并。 例如，如果指定了 ``runtime_env["py_modules"]``，它将替换父级的 ``runtime_env["py_modules"]`` 字段。

示例:

.. testcode::

  # Parent's `runtime_env`
  {"pip": ["requests", "chess"],
  "env_vars": {"A": "a", "B": "b"}}

  # Child's specified `runtime_env`
  {"pip": ["torch", "ray[serve]"],
  "env_vars": {"B": "new", "C": "c"}}

  # Child's actual `runtime_env` (merged with parent's)
  {"pip": ["torch", "ray[serve]"],
  "env_vars": {"A": "a", "B": "new", "C": "c"}}

.. _runtime-env-faq:

常见问题
^^^^^^^^^^^^^^^^^^^^^^^^^^

环境会被安装到每个节点吗？
"""""""""""""""""""""""""""""""""""""""""

如果在 ``ray.init(runtime_env=...)`` 中指定了运行时环境，则该环境将安装在每个节点上。有关更多详细信息，请参阅 :ref:`Per-Job <rte-per-job>`。
（注意，默认情况下，运行时环境将急切地安装在集群中的每个节点上。如果您想按需懒惰地安装运行时环境，请将 ``eager_install`` 选项设置为 false：``ray.init(runtime_env={..., "config": {"eager_install": False}}``。）

环境什么时候安装？
""""""""""""""""""""""""""""""""""

当指定为每个作业时，环境在调用 ``ray.init()`` 安装（除非设置了 ``"eager_install": False``）。
当指定为每个任务或每个 actor 时，环境在调用任务或创建 actor 时安装（即，当你调用 ``my_task.remote()`` 或 ``my_actor.remote()`` 时）。
参考 :ref:`Per-Job <rte-per-job>` 和 :ref:`Per-Task/Actor, within a job <rte-per-task-actor>` 以获取更多详细信息。

环境缓存在哪里？
""""""""""""""""""""""""""""""""""

环境下载的任何本地文件都缓存在 ``/tmp/ray/session_latest/runtime_resources``。

安装或从缓存加载需要多长时间？
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

安装时间通常取决于您使用的 ``runtime_env`` 选项，主要是运行 ``pip install`` 或 ``conda create`` / ``conda activate``，或者上传/下载 ``working_dir``，
这可能需要几秒钟或几分钟。

另一方面，从缓存加载运行时环境应该几乎与普通 Ray worker 启动时间一样快，大约几秒钟。对于每个需要新运行时环境的 Ray actor 或任务，都会启动一个新的 Ray worker。

（请注意，加载缓存的 ``conda`` 环境可能仍然很慢，因为 ``conda activate`` 命令有时需要几秒钟。）

你可以设置 ``setup_timeout_seconds`` 配置来避免安装时间过长。如果安装在此时间内没有完成，您的task 或 actor将无法启动。

运行环境和 Docker 有什么关系？
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

它们可以单独使用，也可以一起使用。
可以在 :ref:`Cluster Launcher <vm-cluster-quick-start>` 中指定容器映像，用于大型或静态依赖项，并且可以为每个作业或每个任务/ actor 指定运行时环境，以用于更动态的用例。
运行时环境将从容器映像继承包、文件和环境变量。

我的 ``runtime_env`` 已经安装了，但是当我登录到节点时，我无法导入这些包。
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

运行时环境仅对 Ray 工作进程有效；它不会在节点上“全局”安装任何包。

.. _remote-uris:

远程 URI
-----------

``worker_dir`` 和 ``py_modules`` 参数在 ``runtime_env`` 字典中可以指定本地路径或远程 URI。

本地路径必须是一个目录路径。目录的内容将直接作为 ``working_dir`` 或 ``py_module`` 访问。
远程 URI 必须是一个直接指向 zip 文件的链接。 **zip 文件必须只包含一个顶级目录。**
这些目录的内容将直接作为 ``working_dir`` 或 ``py_module`` 访问。

比如，你想把 ``/some_path/example_dir`` 目录的内容作为你的 ``working_dir``。
如果你想指定这个目录作为一个本地路径，你的 ``runtime_env`` 字典应该包含：

.. testcode::
  :skipif: True

  runtime_env = {..., "working_dir": "/some_path/example_dir", ...}

假设你想把你的文件放在 ``/some_path/example_dir`` 目录中，并提供一个远程 URI。
你首先需要将 ``example_dir`` 目录压缩成一个 zip 文件。

这里应该没有其他文件或目录在 zip 文件的顶层，除了 ``example_dir``。
你可以使用以下命令在终端中执行：

.. code-block:: bash

    cd /some_path
    zip -r zip_file_name.zip example_dir

注意，这个命令必须从所需 ``working_dir`` 的 *父目录* 运行，以确保生成的 zip 文件包含一个顶级目录。
通常，zip 文件的名称和顶级目录的名称可以是任何东西。
顶级目录的内容将被用作 ``working_dir``（或 ``py_module``）。

你可以通过在终端中运行以下命令来检查 zip 文件是否包含一个顶级目录：

.. code-block:: bash

  zipinfo -1 zip_file_name.zip
  # example_dir/
  # example_dir/my_file_1.txt
  # example_dir/subdir/my_file_2.txt

假设你将压缩的 ``example_dir`` 目录上传到 AWS S3 的 S3 URI ``s3://example_bucket/example.zip``。
你的 ``runtime_env`` 字典应该包含：

.. testcode::
  :skipif: True

  runtime_env = {..., "working_dir": "s3://example_bucket/example.zip", ...}

.. warning::

  检查在压缩文件中是否有隐藏文件或元数据目录。
  你可以通过在终端中运行 ``zipinfo -1 zip_file_name.zip`` 命令来检查 zip 文件的内容。
  一些压缩方法可能会导致隐藏文件或元数据目录出现在 zip 文件的顶层。
  要避免这种情况，请直接在要压缩的目录的父目录上使用 ``zip -r`` 命令。例如，如果你有一个目录结构，如：``a/b``，你想压缩 ``b``，请在目录 ``a`` 上发出 ``zip -r b`` 命令。
  如果 Ray 检测到顶层有多个目录，它将使用整个 zip 文件而不是顶层目录，这可能会导致意外行为。

当前支持三种远程 URI 类型来托管 ``working_dir`` 和 ``py_modules`` 包：

- ``HTTPS``: ``HTTPS`` 引用时以 ``https`` 打头的 URL。
  这些 URL 特别有用，因为远程 Git 提供商（如：GitHub、Bitbucket、GitLab 等）使用 ``https`` URL 作为存储库存档的下载链接。
  这允许你将依赖项托管在远程 Git 提供商上，推送更新，并指定作业应使用哪些依赖项版本（即提交版本）。
  要通过 ``HTTPS`` URI 使用包，你必须安装 ``smart_open`` 库（你可以使用 ``pip install smart_open`` 安装它）。

  - 示例:

    - ``runtime_env = {"working_dir": "https://github.com/example_username/example_respository/archive/HEAD.zip"}``

- ``S3``: ``S3`` 引用是以 ``s3://`` 打头的 URI，它指向了存储在 `AWS S3 <https://aws.amazon.com/s3/>`_ 压缩包。
  要使用 ``S3`` URI，你必须安装 ``smart_open`` 和 ``boto3`` 库（你可以使用 ``pip install smart_open`` 和 ``pip install boto3`` 安装它们）。
  Ray 不会为认证传递任何凭据给 ``boto3``。
  ``boto3`` 将使用你的环境变量、共享凭据文件和/或 AWS 配置文件来验证访问。
  参考 `AWS S3 文档 <https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html>`_ 来设置你的远程包。

  - 示例:

    - ``runtime_env = {"working_dir": "s3://example_bucket/example_file.zip"}``

- ``GS``: ``GS`` 引用是以 ``gs://`` 打头的 URI，它指向了存储在 `Google Cloud Storage <https://cloud.google.com/storage>`_ 压缩包。
  要通过 ``GS`` URI 使用包，你必须安装 ``smart_open`` 和 ``google-cloud-storage`` 库（你可以使用 ``pip install smart_open`` 和 ``pip install google-cloud-storage`` 安装它们）。
  Ray 不会为 ``google-cloud-storage`` 的 ``Client`` 对象传递任何凭据。
  ``google-cloud-storage`` 默认情况下将使用你的本地服务帐户密钥和环境变量。
  参考 `Google Cloud Storage 文档 <https://cloud.google.com/storage/docs>`_ 来设置认证，它允许你接入远程包。

  - 示例:

    - ``runtime_env = {"working_dir": "gs://example_bucket/example_file.zip"}``

注意，``smart_open``、``boto3`` 和 ``google-cloud-storage`` 包默认情况下不会被安装，仅在 ``runtime_env`` 的 ``pip`` 部分中指定它们是不够的。
在 Ray 启动时，集群的所有节点上必须已经安装了相关包。

在远程 Git 提供商上托管依赖项：手把手指南
-----------------------------------------------------------------

您可以将依赖项存储在远程 Git 提供商（例如 GitHub、Bitbucket、GitLab 等）的存储库中，并且可以定期推送更改以使其保持更新。
在本节中，您将学习如何将依赖项存储在 GitHub 上并在运行时环境中使用它。

.. note::
  如果您使用其他大型远程 Git 提供商（例如 BitBucket、GitLab 等），这些步骤也将很有用。
  为简单起见，本节仅指 GitHub，但您可以在提供商上继续操作。

首先，在 GitHub 上创建一个存储 ``working_dir`` 内容或 ``py_module`` 依赖项的存储库。
默认的，当你下载存储库的 zip 文件时，zip 文件将已经包含一个顶级目录，其中包含存储库的内容，
所以你可以直接上传你的 ``working_dir`` 内容或 ``py_module`` 依赖项到 GitHub 存储库。

一旦你上传了 ``working_dir`` 内容或 ``py_module`` 依赖项，你需要存储库 zip 文件的 HTTPS URL，这样你就可以在 ``runtime_env`` 字典中指定它。

你有两种方法获取 HTTPS URL。

方法 1: 下载 Zip (快速实现，单不推荐用于生产环境)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

第一个选项是使用远程 Git 提供商的 "Download Zip" 功能，它提供了一个 HTTPS 链接，用于压缩并下载存储库。
这样很快，但是 **不推荐**，因为它只允许你下载存储库分支的最新提交的 zip 文件。
要找到 GitHub URL，请在 `GitHub <https://github.com/>`_，选择一个分支，然后点击绿色的 "Code" 下拉按钮：

.. figure:: images/ray_repo.png
   :width: 500px

这会弹出一个菜单，提供三个选项："Clone" 提供了克隆存储库的 HTTPS/SSH 链接，"Open with GitHub Desktop" 以及 "Download ZIP."
点击 "Download Zip."
这会在你鼠标弹出一个 pop，选择“Copy Link Address”：

.. figure:: images/download_zip_url.png
   :width: 300px

现在你的 HTTPS 链接已经复制到剪贴板中。你可以将它粘贴到你的 ``runtime_env`` 字典中。

.. warning::

  使用远程 Git 提供商的 "Download as Zip" 功能的 HTTPS URL 不推荐，如果 URL 总是指向最新提交的话。
  例如，在 GitHub 上使用此方法会生成一个始终指向所选分支上的最新提交的链接。

  通过在 ``runtime_env`` 字典中指定此链接，你的 Ray 集群总是使用所选分支的最新提交。
  这会产生一致性风险：如果你在集群的节点拉取存储库内容时推送了更新，
  一些节点可能会拉取你推送之前的包版本，而另一些节点可能会拉取你推送之后的版本。
  为了保持一致性，最好指定一个特定的提交，这样所有节点都使用相同的包。
  参考 "方法 2: 手动创建 URL" 来创建一个指向特定提交的 URL。

方法 2: 手动创建 URL (较慢实现，但推荐用于生产)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

第二个选项是通过将你的特定用例与以下示例之一进行模式匹配，手动创建此 URL。
**建议这样做** 因为它可以更细粒度地控制在生成依赖项 zip 文件时要使用哪个存储库分支和提交。
这些选项可防止 Ray Clusters 上的一致性问题（有关更多信息，请参阅上面的警告）。
要创建 URL，请选择下面适合您用例的 URL 模板，并使用存储库中的特定值填写括号中的所有参数（例如 [用户名]、[存储库] 等）。
例如，假设你的 GitHub 用户名是 ``example_user``，存储库的名称是 ``example_repository``，并且所需的提交哈希是 ``abcdefg``。
如果 ``example_repository`` 是公共的，并且你想要检索 ``abcdefg`` 提交（与第一个示例用例匹配），URL 将是：

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/example_user/example_repository/archive/abcdefg.zip")}

以下是不同用例和对应 URL 的列表：

- 示例：从公共 GitHub 存储库上的特定提交哈希中检索包

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/[username]/[repository]/archive/[commit hash].zip")}

- 示例： **开发阶段** 使用个人接入认证从私有 GitHub 存储库中检索包。 **生产阶段** 请参考 :ref:`此文档 <runtime-env-auth>` 以了解如何安全地对私有依赖项进行身份验证。

.. testcode::

    runtime_env = {"working_dir": ("https://[username]:[personal access token]@github.com"
                                   "/[username]/[private repository]/archive/[commit hash].zip")}

- 示例：从公共 GitHub 存储库的最新提交中检索包

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/[username]/[repository]/archive/HEAD.zip")}

- 示例：从公共 Bitbucket 存储库上的特定提交哈希中检索包

.. testcode::

    runtime_env = {"working_dir": ("https://bitbucket.org"
                                   "/[owner]/[repository]/get/[commit hash].tar.gz")}

.. tip::

  建议指定特定提交，而不是始终使用最新提交。
  这可以防止多节点 Ray 集群出现一致性问题。
  有关更多信息，请参阅“选项 1：下载 Zip”下方的警告。

一旦你在 ``runtime_env`` 字典中指定了 URL，你就可以将字典传递给 ``ray.init()`` 或 ``.options()`` 调用。
恭喜！你现在已经在 GitHub 上远程托管了一个 ``runtime_env`` 依赖项！


调试
---------
如果 runtime_env 无法设置（例如，网络问题、下载失败等），Ray 将无法调度需要 runtime_env 的任务/ actor 。
如果你调用 ``ray.get``，它将引发 ``RuntimeEnvSetupError`` 并提供详细的错误消息。

.. testcode::

    import ray
    import time

    @ray.remote
    def f():
        pass

    @ray.remote
    class A:
        def f(self):
            pass

    start = time.time()
    bad_env = {"conda": {"dependencies": ["this_doesnt_exist"]}}

    # [Tasks] will raise `RuntimeEnvSetupError`.
    try:
      ray.get(f.options(runtime_env=bad_env).remote())
    except ray.exceptions.RuntimeEnvSetupError:
      print("Task fails with RuntimeEnvSetupError")

    # [Actors] will raise `RuntimeEnvSetupError`.
    a = A.options(runtime_env=bad_env).remote()
    try:
      ray.get(a.f.remote())
    except ray.exceptions.RuntimeEnvSetupError:
      print("Actor fails with RuntimeEnvSetupError")

.. testoutput::

  Task fails with RuntimeEnvSetupError
  Actor fails with RuntimeEnvSetupError


完整的日志可以在 ``runtime_env_setup-[job_id].log`` 文件中找到，用于每个 actor 、每个任务和每个作业的环境，或者
在使用 Ray Client 时用于每个作业的环境中的 ``runtime_env_setup-ray_client_server_[port].log`` 文件中找到。

 你也可以在每个节点上在启动 Ray 之前设置环境变量 ``RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED=1``，例如使用 Ray 集群配置文件中的 ``setup_commands`` (:ref:`reference <cluster-configuration-setup-commands>`)。
这会打印完整的 ``runtime_env`` 设置日志消息到驱动程序（调用 ``ray.init()`` 的脚本）。

示例日志输出：

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  ray.init(runtime_env={"pip": ["requests"]})

.. testoutput::
    :options: +MOCK

    (pid=runtime_env) 2022-02-28 14:12:33,653       INFO pip.py:188 -- Creating virtualenv at /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv, current python dir /Users/user/anaconda3/envs/ray-py38
    (pid=runtime_env) 2022-02-28 14:12:33,653       INFO utils.py:76 -- Run cmd[1] ['/Users/user/anaconda3/envs/ray-py38/bin/python', '-m', 'virtualenv', '--app-data', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv_app_data', '--reset-app-data', '--no-periodic-update', '--system-site-packages', '--no-download', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv']
    (pid=runtime_env) 2022-02-28 14:12:34,267       INFO utils.py:97 -- Output of cmd[1]: created virtual environment CPython3.8.11.final.0-64 in 473ms
    (pid=runtime_env)   creator CPython3Posix(dest=/private/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv, clear=False, no_vcs_ignore=False, global=True)
    (pid=runtime_env)   seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/private/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv_app_data)
    (pid=runtime_env)     added seed packages: pip==22.0.3, setuptools==60.6.0, wheel==0.37.1
    (pid=runtime_env)   activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:34,268       INFO utils.py:76 -- Run cmd[2] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-c', 'import ray; print(ray.__version__, ray.__path__[0])']
    (pid=runtime_env) 2022-02-28 14:12:35,118       INFO utils.py:97 -- Output of cmd[2]: 3.0.0.dev0 /Users/user/ray/python/ray
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:35,120       INFO pip.py:236 -- Installing python requirements to /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv
    (pid=runtime_env) 2022-02-28 14:12:35,122       INFO utils.py:76 -- Run cmd[3] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt']
    (pid=runtime_env) 2022-02-28 14:12:38,000       INFO utils.py:97 -- Output of cmd[3]: Requirement already satisfied: requests in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from -r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2.26.0)
    (pid=runtime_env) Requirement already satisfied: idna<4,>=2.5 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (3.2)
    (pid=runtime_env) Requirement already satisfied: certifi>=2017.4.17 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2021.10.8)
    (pid=runtime_env) Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (1.26.7)
    (pid=runtime_env) Requirement already satisfied: charset-normalizer~=2.0.0 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2.0.6)
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:38,001       INFO utils.py:76 -- Run cmd[4] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-c', 'import ray; print(ray.__version__, ray.__path__[0])']
    (pid=runtime_env) 2022-02-28 14:12:38,804       INFO utils.py:97 -- Output of cmd[4]: 3.0.0.dev0 /Users/user/ray/python/ray

参考 :ref:`Logging 目录结构 <logging-directory-structure>` 获取更多信息。
