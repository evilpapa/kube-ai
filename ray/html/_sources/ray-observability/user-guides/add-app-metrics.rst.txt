.. _application-level-metrics:

添加应用级指标
--------------------------------

Ray 在 :ref:`ray.util.metrics <custom-metric-api-ref>` 中提供了一个方便的 API ，用于定义和导出自定义指标，以便了解您的应用程序。
支持三种指标：计数器、仪表和直方图。
这些指标对应相同的 `Prometheus metric types <https://prometheus.io/docs/concepts/metric_types/>`_ 。
下面是一个使用这些 API 导出指标的 Actor 的简单示例：

.. literalinclude:: ../doc_code/metrics_example.py
   :language: python

脚本运行时，指标将导出到 ``localhost:8080`` （这是 Prometheus 配置为抓取的端点）。
在浏览器中打开它。您应该看到以下输出：

.. code-block:: none

  # HELP ray_request_latency Latencies of requests in ms.
  # TYPE ray_request_latency histogram
  ray_request_latency_bucket{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor",le="0.1"} 2.0
  ray_request_latency_bucket{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor",le="1.0"} 2.0
  ray_request_latency_bucket{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor",le="+Inf"} 2.0
  ray_request_latency_count{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor"} 2.0
  ray_request_latency_sum{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor"} 0.11992454528808594
  # HELP ray_curr_count Current count held by the actor. Goes up and down.
  # TYPE ray_curr_count gauge
  ray_curr_count{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor"} -15.0
  # HELP ray_num_requests_total Number of requests processed by the actor.
  # TYPE ray_num_requests_total counter
  ray_num_requests_total{Component="core_worker",Version="3.0.0.dev0",actor_name="my_actor"} 2.0

参阅 :ref:`ray.util.metrics <custom-metric-api-ref>` 了解更多详细信息。
