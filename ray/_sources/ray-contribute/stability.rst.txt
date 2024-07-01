.. _api-stability:

API 稳定性
=============

Ray 为 Ray 核心和库中的公共 API 提供稳定性保证，并进行相应的装饰/标记。

API 可以被标记为：

* :ref:`PublicAPI <public-api-def>`，这意味着 API 公开给最终用户。 PublicAPI 具有三个子级别（alpha、beta、stable），如下所述。
* :ref:`DeveloperAPI <developer-api-def>`这意味着 API 显式暴露给 *高级* Ray 用户和库开发人员
* :ref:`Deprecated <deprecated-api-def>`，可能会在 Ray 的未来版本中删除。

Ray 的 PublicAPI 稳定性定义基于 `Google 稳定性级别指南 <https://google.aip.dev/181>`_，有细微差别：

.. _api-stability-alpha:

Alpha
~~~~~

*alpha* 组件会与一组已知的用户进行快速迭代，这些用户
**必须** 能够容忍变化。用户数量 **应该** 是经过精心策划、
可管理的集合，以便可以单独与所有用户进行通信。

alpha 组件重大更改 **必须** 允许和预期，并且
用户 **必须** 不能期望稳定性。

.. _api-stability-beta:

测试版
~~~~

*beta* 组件应该尽可能稳定；然而，
beta 组件 **必须** 允许随时间变化。

Because users of beta components tend to have a lower tolerance of change, beta
components **should** be as stable as possible; however, the beta component
**must** be permitted to change over time. These changes **should** be minimal
but **may** include backwards-incompatible changes to beta components.

Backwards-incompatible changes **must** be made only after a reasonable
deprecation period to provide users with an opportunity to migrate their code.

.. _api-stability-stable:

Stable
~~~~~~

A *stable* component **must** be fully-supported over the lifetime of the major
API version. Because users expect such stability from components marked stable,
there **must** be no breaking changes to these components within a major version
(excluding extraordinary circumstances).

Docstrings
----------

.. _public-api-def:

.. autofunction:: ray.util.annotations.PublicAPI

.. _developer-api-def:

.. autofunction:: ray.util.annotations.DeveloperAPI

.. _deprecated-api-def:

.. autofunction:: ray.util.annotations.Deprecated

Undecorated functions can be generally assumed to not be part of the Ray public API.
