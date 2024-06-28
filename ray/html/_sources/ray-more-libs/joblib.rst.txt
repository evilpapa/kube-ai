.. _ray-joblib:

分布式 Scikit-learn / Joblib
=================================

.. _`issue on GitHub`: https://github.com/ray-project/ray/issues

Ray 支持通过使用 `Ray Actors <actors.html>`__ 来实现 `joblib`` 的 Ray 后端，
从而支持在集群上运行分布式 `scikit-learn`_ 程序，而不是使用本地进程。
这使得将现有使用 scikit-learn 的应用程序从单节点扩展到集群变得容易。

.. note::

  这些 API 是新的，可能会在未来的 Ray 版本中进行修改。如果遇到任何错误，请在 GitHub 上提交 `issue`_。

.. _`joblib`: https://joblib.readthedocs.io
.. _`scikit-learn`: https://scikit-learn.org

快速入门
----------

要开始使用，首先 `安装 Ray <installation.html>`__，然后使用 ``from ray.util.joblib import register_ray`` 并运行 ``register_ray()``。
这将注册 Ray 作为 scikit-learn 使用的 joblib 后端。然后在 ``with joblib.parallel_backend('ray')`` 中运行原始 scikit-learn 代码。
查看下面的 `在集群上运行`_ 部分，以获取在多节点 Ray 集群上运行的说明。

.. code-block:: python

  import numpy as np
  from sklearn.datasets import load_digits
  from sklearn.model_selection import RandomizedSearchCV
  from sklearn.svm import SVC
  digits = load_digits()
  param_space = {
      'C': np.logspace(-6, 6, 30),
      'gamma': np.logspace(-8, 8, 30),
      'tol': np.logspace(-4, -1, 30),
      'class_weight': [None, 'balanced'],
  }
  model = SVC(kernel='rbf')
  search = RandomizedSearchCV(model, param_space, cv=5, n_iter=300, verbose=10)

  import joblib
  from ray.util.joblib import register_ray
  register_ray()
  with joblib.parallel_backend('ray'):
      search.fit(digits.data, digits.target)

你也可以在 ``parallel_backend`` 中设置 ``ray_remote_args`` 参数来 :func:`配置 Ray Actors <ray.remote>` 组成的池。
这可以用于例如 :ref:`为 Actor 分配资源，例如 GPU <actor-resource-guide>`。

.. code-block:: python

  # Allows to use GPU-enabled estimators, such as cuML
  with joblib.parallel_backend('ray', ray_remote_args=dict(num_gpus=1)):
      search.fit(digits.data, digits.target)

在集群上运行
----------------

本章节假设你已经有一个运行中的 Ray 集群。要启动一个 Ray 集群，请参考 `集群设置 <cluster/index.html>`__ 说明。

要在一个运行中的 Ray 集群上连接 scikit-learn，你需要通过设置 ``RAY_ADDRESS`` 环境变量来指定 head 节点的地址。

你也可以手动调用 ``ray.init()`` (使用任何支持的配置选项) 来在调用 ``with joblib.parallel_backend('ray')`` 之前启动 Ray。

.. warning::

    如果你不想设置 ``RAY_ADDRESS`` 环境变量并且不提供 ``address`` 参数给 ``ray.init(address=<address>)``，那么 scikit-learn 将在单个节点上运行！
