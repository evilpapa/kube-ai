.. _ray-tracing:

追踪
=======
为了帮助调试和监控 Ray 应用程序，Ray 与 OpenTelemetry 集成，以便于将跟踪导出到 Jaeger 等外部跟踪堆栈。


.. note::

    跟踪是一项实验性功能，正在积极开发中。 API 可能会发生变化。

安装
------------
首先，安装 OpenTelemetry：

.. code-block:: shell

    pip install opentelemetry-api==1.1.0
    pip install opentelemetry-sdk==1.1.0
    pip install opentelemetry-exporter-otlp==1.1.0

启动追踪钩子
--------------------
要启用跟踪，您必须提供一个跟踪启动挂钩，其中包含:ref:`Tracer Provider <tracer-provider>`\ :ref:`Remote Span Processors <remote-span-processors>` 以及 :ref:`Additional Instruments <additional-instruments>` 设置. 跟踪启动挂钩应该是一个不带 args 或 kwargs 调用的函数。该钩子需要在所有工作进程的 Python 环境中可用。

下面是一个示例跟踪启动挂钩，它设置默认跟踪提供程序，将 spans 文件导出到 ``/tmp/spans`` ，并且没有任何其他工具。

.. testcode::

  import ray
  import os
  from opentelemetry import trace
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import (
      ConsoleSpanExporter,
      SimpleSpanProcessor,
  )


  def setup_tracing() -> None:
      # Creates /tmp/spans folder
      os.makedirs("/tmp/spans", exist_ok=True)
      # Sets the tracer_provider. This is only allowed once per execution
      # context and will log a warning if attempted multiple times.
      trace.set_tracer_provider(TracerProvider())
      trace.get_tracer_provider().add_span_processor(
          SimpleSpanProcessor(
              ConsoleSpanExporter(
                  out=open(f"/tmp/spans/{os.getpid()}.json", "a")
                  )
          )
      )


对于想要尝试跟踪的开源用户，Ray 有一个默认的跟踪启动挂钩，可将 span 导出到文件夹/ ``/tmp/spans``。要使用此默认挂钩运行，请运行以下代码示例来设置跟踪并跟踪简单的 Ray 任务。

.. tab-set::

    .. tab-item:: ray start

        .. code-block:: shell

            $ ray start --head --tracing-startup-hook=ray.util.tracing.setup_local_tmp_tracing:setup_tracing
            $ python
                ray.init()
                @ray.remote
                def my_function():
                    return 1

                obj_ref = my_function.remote()

    .. tab-item:: ray.init()

        .. testcode::

            ray.init(_tracing_startup_hook="ray.util.tracing.setup_local_tmp_tracing:setup_tracing")

            @ray.remote
            def my_function():
                return 1

            obj_ref = my_function.remote()

如果您想提供自己的自定义跟踪启动挂钩，请在要运行的 ``setup_tracing`` 函数的 ``module:attribute`` 提供你的设置。

.. _tracer-provider:

Tracer 提供者
~~~~~~~~~~~~~~~
这配置了如何收集跟踪。参考 TracerProvider API `here <https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.html#opentelemetry.sdk.trace.TracerProvider>`__。

.. _remote-span-processors:

远程 span 处理
~~~~~~~~~~~~~~~~~~~~~~
这配置了将跟踪导出到的位置。参考 `此处的 <https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.html#opentelemetry.sdk.trace.SpanProcessor>`__  SpanProcessor API。

想要尝试跟踪的用户可以配置其远程跨度处理器以将跨度导出到本地 JSON 文件。本地开发的认真用户可以通过 `Jaeger exporter <https://opentelemetry-python.readthedocs.io/en/latest/exporter/jaeger/jaeger.html#module-opentelemetry.exporter.jaeger>`_ 将其跟踪推送到 Jaeger 容器。

.. _additional-instruments:

附加说明
~~~~~~~~~~~~~~~~~~~~~~
如果您使用的库具有内置跟踪支持，你提供的 ``setup_tracing`` 函数也应该修补这些库。 您可以在 `此处 <https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation>`_ 找到有关这些库的更多文档。

自定义跟踪
*************
在您的程序中添加自定义跟踪。在您的程序中，使用 ``trace.get_tracer(__name__)`` 获取跟踪器对象，并使用 ``tracer.start_as_current_span(...)`` 开始一个新的跨度。

请参阅下面的添加自定义跟踪的简单示例。

.. testcode::

  from opentelemetry import trace

  @ray.remote
  def my_func():
      tracer = trace.get_tracer(__name__)

      with tracer.start_as_current_span("foo"):
          print("Hello world from OpenTelemetry Python!")
