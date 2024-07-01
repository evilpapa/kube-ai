.. _generators:

生成器
==========

Python 生成器是一种类似迭代器的函数，每次迭代产生一个值。Ray 支持远程生成器用于两种情况：

1. 减少从远程函数返回多个值时的最大堆内存使用。
   请参阅 :ref:`设计模式指南 <generator-pattern>` 以获取示例。
2. 当远程函数动态设置返回值的数量时，Ray 支持远程生成器。

远程生成器可以在 actor 和非 actor 任务中使用。

.. _static-generators:

`num_returns` 由任务调用者设置
------------------------------------

如果可能，调用者应该使用 ``@ray.remote(num_returns=x)`` 或 ``foo.options(num_returns=x).remote()`` 来设置远程函数的返回值数量。
Ray 将会返回多个 ``ObjectRefs`` 给调用者。
远程函数应该返回相同数量的值，通常作为元组或列表。
对比动态设置返回值数量，这样做会减少用户代码的复杂性和性能开销，因为 Ray 会提前知道要返回给调用者多少个 ``ObjectRefs``。

除了调用者的语法不变外，我们还可以使用远程生成器函数来迭代地产生值。
语法生成器函数的语法与普通远程函数相同，只是它们使用 ``@ray.remote`` 装饰器的 ``num_returns`` 参数来指定返回值的数量。
如果生成器产生的值数量与调用者指定的数量不同，错误将会被抛出。

比如，我们可以将以下返回多个值的代码改为：

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_start__
    :end-before: __large_values_end__

针对这段代码，我们可以使用生成器函数：

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_generator_start__
    :end-before: __large_values_generator_end__

这样做的优点是生成器函数不需要立即将其所有返回值保存在内存中。
它可以逐个返回数组以减少内存压力。

.. _dynamic-generators:

任务执行器设置 `num_returns`
--------------------------------------

在一些情况下，调用者可能不知道远程函数的返回值数量。
比如，我们想要编写一个任务，将其参数分成相等大小的块并返回这些块。
我们可能不知道参数的大小，直到执行任务，因此我们不知道要期望的返回值数量。



在一些情况下，我们可以使用远程生成器函数，它返回 *动态* 数量的值。
我们可以使用 ``num_returns="dynamic"`` 特性来设置 ``@ray.remote`` 装饰器或远程函数的 ``.options()``。
然后，当调用远程函数时，Ray 将返回一个 *单个* ``ObjectRef``，当任务完成时，它将被填充为 ``ObjectRefGenerator``。
``ObjectRefGenerator`` 可以用于迭代包含任务返回的实际值的 ``ObjectRefs`` 列表。

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __dynamic_generator_start__
    :end-before: __dynamic_generator_end__

我们也可以将 ``num_returns="dynamic"`` 返回的 ``ObjectRef`` 传递给另一个任务。任务将接收 ``ObjectRefGenerator``，它可以用于迭代任务的返回值。同样，您也可以将 ``ObjectRefGenerator`` 作为任务参数传递。

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __dynamic_generator_pass_start__
    :end-before: __dynamic_generator_pass_end__

异常处理
------------------

如果一个生成器函数在产生所有值之前引发异常，那么已经存储的值仍然可以通过它们的 ``ObjectRefs`` 访问。
其余的 ``ObjectRefs`` 将包含引发的异常。
对于静态以及动态的 ``num_returns``，这一点都是 true 。
如果任务通过 ``num_returns="dynamic"`` 调用，异常将作为额外最终 ``ObjectRef`` 存储在 ``ObjectRefGenerator`` 。

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __generator_errors_start__
    :end-before: __generator_errors_end__

注意，目前已知的一个 bug 是，对于生成器产生的值数量超过预期的情况，异常不会被传播。这种情况可能发生在两种情况下：

1. 当调用者设置 ``num_returns`` 时，但生成器任务返回的值超过了这个值。
2. 当一个设置了 ``num_returns="dynamic"`` 的生成器任务被 :ref:`重新执行 <task-retries>` 时，如果重新执行的任务产生的值超过了原始执行的值，异常不会被传播。请注意，一般来说，如果任务是非确定性的，Ray 不保证任务重新执行的正确性，建议为这样的任务设置 ``@ray.remote(num_retries=0)``。

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __generator_errors_unsupported_start__
    :end-before: __generator_errors_unsupported_end__

限制
-----------

尽管生成器函数一次创建一个 ``ObjectRef``，但目前 Ray 不会在整个任务完成并且所有值都被创建之前调度依赖任务。这类似于返回列表的多值任务的语义。
