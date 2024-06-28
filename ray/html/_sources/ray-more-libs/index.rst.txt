更多 Ray ML 类库
=====================

.. TODO: we added the three Ray Core examples below, since they don't really belong there.
    Going forward, make sure that all "Ray Lightning" and XGBoost topics are in one document or group,
    and not next to each other.

Ray 有许多额外的生态系统库集成。

- :ref:`ray-joblib`
- :ref:`ray-multiprocessing`
- :ref:`ray-collective`
- :ref:`dask-on-ray`
- :ref:`spark-on-ray`
- :ref:`mars-on-ray`
- :ref:`modin-on-ray`

.. _air-ecosystem-map:

生态地图
-------------

以下地图可视化了 Ray 组件及其集成的组件和成熟度。实线表示 Ray 组件之间的集成；虚线表示与更广泛的 ML 生态系统的集成。

* **Stable**: 组件时稳定的。
* **Beta**: 组件仍在开发，并且 API 可能根据主题进行变更。
* **Alpha**: 组件处于早期开发阶段。
* **Community-Maintained**: 这些实现由社区维护，质量可能有所不同。

.. image:: /images/air-ecosystem.svg
