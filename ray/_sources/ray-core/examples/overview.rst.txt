.. _ray-core-examples-tutorial:

Ray 教程和示例
==========================


机器学习示例
-------------------------

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: /images/timeseries.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: automl_for_time_series

            使用 Ray 构建简单的时间序列 AutoML

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: batch_prediction

            使用 Ray 构建批量预测

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: batch_training

            使用 Ray 构建批量训练

    .. grid-item-card::
        :img-top: images/param_actor.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: plot_parameter_server

            使用 Ray 构建简单的参数服务器

    .. grid-item-card::
        :img-top: images/hyperparameter.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: plot_hyperparameter

            简单的并行模型选择

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: plot_example-lm

            Fairseq 容错训练


强化学习示例
-------------------------------

这些是向您展示如何利用 Ray Core 的简单示例。有关 Ray 的生产级强化学习库，请参阅 `RLlib <http://docs.ray.io/en/latest/rllib.html>`__。


.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top:  images/pong.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: plot_pong_example

            学打乒乓球

    .. grid-item-card::
        :img-top: images/a3c.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: plot_example-a3c

            异步高级 Actor Critic (A3C)


基础示例
--------------

.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: gentle_walkthrough

            通过示例温和地介绍 Ray Core

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: highly_parallel

            使用 Ray 处理高度可并行化的任务

    .. grid-item-card::
        :img-top: /ray-overview/images/ray_svg_logo.svg
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: map_reduce

            使用 Ray Core 运行简单的 MapReduce 示例
