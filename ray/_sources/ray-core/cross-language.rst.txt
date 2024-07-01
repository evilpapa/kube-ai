.. _cross_language:

跨语言编程
==========================

本页将向您展示如何使用 Ray 的跨语言编程功能。

设置驱动
-----------------

我们需要在您的驱动程序中设置 :ref:`code_search_path`。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/cross_language.py
            :language: python
            :start-after: __crosslang_init_start__
            :end-before: __crosslang_init_end__

    .. tab-item:: Java

        .. code-block:: bash

            java -classpath <classpath> \
                -Dray.address=<address> \
                -Dray.job.code-search-path=/path/to/code/ \
                <classname> <args>

如果工作程序的 Python 和 Java 代码位于不同的目录中，您可能需要包含多个目录来加载它们。

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/cross_language.py
            :language: python
            :start-after: __crosslang_multidir_start__
            :end-before: __crosslang_multidir_end__

    .. tab-item:: Java

        .. code-block:: bash

            java -classpath <classpath> \
                -Dray.address=<address> \
                -Dray.job.code-search-path=/path/to/jars:/path/to/pys \
                <classname> <args>

Python 调用 Java
-------------------

假设我们有一个 Java 静态方法和一个 Java 类，如下所示：

.. code-block:: java

  package io.ray.demo;

  public class Math {

    public static int add(int a, int b) {
      return a + b;
    }
  }

.. code-block:: java

  package io.ray.demo;

  // A regular Java class.
  public class Counter {

    private int value = 0;

    public int increment() {
      this.value += 1;
      return this.value;
    }
  }

然后，在 Python 中，我们可以调用上述 Java 远程函数，或者从上述 Java 类创建一个 actor 。

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __python_call_java_start__
  :end-before: __python_call_java_end__

Java 调用 Python
-------------------

假设我们有一个如下的 Python 模块：

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __python_module_start__
  :end-before: __python_module_end__

.. note::

  * 该函数或类应该被 `@ray.remote` 装饰。

然后，在 Java 中，我们可以调用上述 Python 远程函数，或者从上述 Python 类创建一个 actor 。

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.PyActorHandle;
  import io.ray.api.Ray;
  import io.ray.api.function.PyActorClass;
  import io.ray.api.function.PyActorMethod;
  import io.ray.api.function.PyFunction;
  import org.testng.Assert;

  public class JavaCallPythonDemo {

    public static void main(String[] args) {
      // Set the code-search-path to the directory of your `ray_demo.py` file.
      System.setProperty("ray.job.code-search-path", "/path/to/the_dir/");
      Ray.init();

      // Define a Python class.
      PyActorClass actorClass = PyActorClass.of(
          "ray_demo", "Counter");

      // Create a Python actor and call actor method.
      PyActorHandle actor = Ray.actor(actorClass).remote();
      ObjectRef objRef1 = actor.task(
          PyActorMethod.of("increment", int.class)).remote();
      Assert.assertEquals(objRef1.get(), 1);
      ObjectRef objRef2 = actor.task(
          PyActorMethod.of("increment", int.class)).remote();
      Assert.assertEquals(objRef2.get(), 2);

      // Call the Python remote function.
      ObjectRef objRef3 = Ray.task(PyFunction.of(
          "ray_demo", "add", int.class), 1, 2).remote();
      Assert.assertEquals(objRef3.get(), 3);

      Ray.shutdown();
    }
  }

跨语言数据序列化
---------------------------------

以下类型 Ray 的参数及返回值的，可以被自动序列化和反序列化：

  - 原始数据类型
      ===========   =======  =======
      消息包   Python   Java
      ===========   =======  =======
      nil           None     null
      bool          bool     Boolean
      int           int      Short / Integer / Long / BigInteger
      float         float    Float / Double
      str           str      String
      bin           bytes    byte[]
      ===========   =======  =======

  - 基本容器类型
      ===========   =======  =======
      消息包   Python   Java
      ===========   =======  =======
      array         list     Array
      ===========   =======  =======

  - Ray 内置类型
      - ActorHandle

.. note::

  * 注意 Python 和 Java 之间的浮点/双精度。
    如果 Java 使用浮点类型接收输入参数，
    则 Python 的双精度数据将降低为 Java 中的浮点精度。
  * BigInteger 最大支持 2^64-1 的值，具体请参考：
    https://github.com/msgpack/msgpack/blob/master/spec.md#int-format-family。
    如果大于 2^64-1 则将该值传递给 Python 会抛出异常。

下面的示例展示了如何将这些类型作为参数传递以及如何返回这些类型

您可以编写一个返回输入数据的 Python 函数：

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __serialization_start__
  :end-before: __serialization_end__

然后，你可以将对象从 Java 传输到 Python，然后再从 Python 传输回 Java：

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.Ray;
  import io.ray.api.function.PyFunction;
  import java.math.BigInteger;
  import org.testng.Assert;

  public class SerializationDemo {

    public static void main(String[] args) {
      Ray.init();

      Object[] inputs = new Object[]{
          true,  // Boolean
          Byte.MAX_VALUE,  // Byte
          Short.MAX_VALUE,  // Short
          Integer.MAX_VALUE,  // Integer
          Long.MAX_VALUE,  // Long
          BigInteger.valueOf(Long.MAX_VALUE),  // BigInteger
          "Hello World!",  // String
          1.234f,  // Float
          1.234,  // Double
          "example binary".getBytes()};  // byte[]
      for (Object o : inputs) {
        ObjectRef res = Ray.task(
            PyFunction.of("ray_serialization", "py_return_input", o.getClass()),
            o).remote();
        Assert.assertEquals(res.get(), o);
      }

      Ray.shutdown();
    }
  }

跨语言异常堆栈
-------------------------------

假设我们有一个如下的 Java 包：

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.Ray;
  import io.ray.api.function.PyFunction;

  public class MyRayClass {

    public static int raiseExceptionFromPython() {
      PyFunction<Integer> raiseException = PyFunction.of(
          "ray_exception", "raise_exception", Integer.class);
      ObjectRef<Integer> refObj = Ray.task(raiseException).remote();
      return refObj.get();
    }
  }

以及如下的 Python 模块：

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __raise_exception_start__
  :end-before: __raise_exception_end__

然后运行以下代码：

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __raise_exception_demo_start__
  :end-before: __raise_exception_demo_end__

异常堆栈将是：

.. code-block:: text

  Traceback (most recent call last):
    File "ray_exception_demo.py", line 9, in <module>
      ray.get(obj_ref)  # <-- raise exception from here.
    File "ray/python/ray/_private/client_mode_hook.py", line 105, in wrapper
      return func(*args, **kwargs)
    File "ray/python/ray/_private/worker.py", line 2247, in get
      raise value
  ray.exceptions.CrossLanguageError: An exception raised from JAVA:
  io.ray.api.exception.RayTaskException: (pid=61894, ip=172.17.0.2) Error executing task c8ef45ccd0112571ffffffffffffffffffffffff01000000
          at io.ray.runtime.task.TaskExecutor.execute(TaskExecutor.java:186)
          at io.ray.runtime.RayNativeRuntime.nativeRunTaskExecutor(Native Method)
          at io.ray.runtime.RayNativeRuntime.run(RayNativeRuntime.java:231)
          at io.ray.runtime.runner.worker.DefaultWorker.main(DefaultWorker.java:15)
  Caused by: io.ray.api.exception.CrossLanguageException: An exception raised from PYTHON:
  ray.exceptions.RayTaskError: ray::raise_exception() (pid=62041, ip=172.17.0.2)
    File "ray_exception.py", line 7, in raise_exception
      1 / 0
  ZeroDivisionError: division by zero
