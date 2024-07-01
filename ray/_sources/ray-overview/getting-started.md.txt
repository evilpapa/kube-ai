(gentle-intro)=

# 入门
使用 Ray 扩展笔记本电脑或云上的应用程序。为您的任务选择正确的指南。
* 扩展 ML 工作负载：[Ray Libraries 快速入门](#ray-libraries-quickstart)
* 扩展通用 Python 应用程序：[Ray Core 快速入门](#ray-core-quickstart)
* 部署到云：[Ray Clusters 快速入门](#ray-cluster-quickstart)
* 调试和监视应用：[调试和监视快速入门](#debugging-and-monitoring-quickstart)


(libraries-quickstart)=
## Ray AI 库快速入门

使用单独的库来处理 ML 工作负载。单击下面适合您的工作负载的下拉菜单。

`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> 数据：用于 ML 的可扩展数据集
:animate: fade-in-slide-down

使用 [Ray Data](data_key_concepts) 扩展离线推理和训练摄取。--
一个专为 ML 设置的数据处理库。

要了解更多信息，请参阅 [离线批量推理](batch_inference_overview) 和
[用于 ML 训练的数据预处理和摄取](batch_inference_overview) 。

````{note}
要运行此示例，请安装 Ray Data：

```bash
pip install -U "ray[data]"
```
````

```{testcode}
from typing import Dict
import numpy as np
import ray

# 从本地磁盘文件，Python 对象，以及 s3 云存储 创建数据集
ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

# Apply functions to transform data. Ray Data executes transformations in parallel.
def compute_area(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    length = batch["petal length (cm)"]
    width = batch["petal width (cm)"]
    batch["petal area (cm^2)"] = length * width
    return batch

transformed_ds = ds.map_batches(compute_area)

# Iterate over batches of data.
for batch in transformed_ds.iter_batches(batch_size=4):
    print(batch)

# Save dataset contents to on-disk files or cloud storage.
transformed_ds.write_parquet("local:///tmp/iris/")
```

```{testoutput}
:hide:

...
```

```{button-ref}  ../data/data
:color: primary
:outline:
:expand:

了解更多关于 Ray Data 的信息
```
`````

``````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> 训练：分布式模型训练
:animate: fade-in-slide-down

Ray Train 抽象化了建立分布式训练系统的复杂性。

`````{tab-set}

````{tab-item} PyTorch

此示例展示了如何将 Ray Train 与 PyTorch 结合使用。

要运行此示例，请安装 Ray Train 和 PyTorch 软件包：

:::{note}
```bash
pip install -U "ray[train]" torch torchvision
```
:::

设置您的数据集和模型。

```{literalinclude} /../../python/ray/train/examples/pytorch/torch_quick_start.py
:language: python
:start-after: __torch_setup_begin__
:end-before: __torch_setup_end__
```

现在定义您的 single-worker PyTorch 训练函数。

```{literalinclude} /../../python/ray/train/examples/pytorch/torch_quick_start.py
:language: python
:start-after: __torch_single_begin__
:end-before: __torch_single_end__
```

该训练函数可以通过以下方式执行：

```{literalinclude} /../../python/ray/train/examples/pytorch/torch_quick_start.py
:language: python
:start-after: __torch_single_run_begin__
:end-before: __torch_single_run_end__
:dedent: 0
```

将其转换为分布式 multi-worker 训练函数。

使用 `ray.train.torch.prepare_model` 和 `ray.train.torch.prepare_data_loader` 实用函数设置模型和数据以进行分布式训练。
这会自动包装模型并将 `DistributedDataParallel` 其放置在正确的设备上，
然后添加 `DistributedSampler` 到 `DataLoaders` 中。

```{literalinclude} /../../python/ray/train/examples/pytorch/torch_quick_start.py
:language: python
:start-after: __torch_distributed_begin__
:end-before: __torch_distributed_end__
```

实例化一个有 4 个工作线程的 ``TorchTrainer``，
并使用他运行新的训练方法。

```{literalinclude} /../../python/ray/train/examples/pytorch/torch_quick_start.py
:language: python
:start-after: __torch_trainer_begin__
:end-before: __torch_trainer_end__
:dedent: 0
```
````

````{tab-item} TensorFlow

This example shows how you can use Ray Train to set up [Multi-worker training
with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).

要运行本示例需安装 Ray Train 以及 Tensorflow 包：

:::{note}
```bash
pip install -U "ray[train]" tensorflow
```
:::

设置数据集和模型。

```{literalinclude} /../../python/ray/train/examples/tf/tensorflow_quick_start.py
:language: python
:start-after: __tf_setup_begin__
:end-before: __tf_setup_end__
```

现在定义 single-worker TensorFlow 训练方法。

```{literalinclude} /../../python/ray/train/examples/tf/tensorflow_quick_start.py
:language: python
:start-after: __tf_single_begin__
:end-before: __tf_single_end__
```

此训练方法可通过以下执行：

```{literalinclude} /../../python/ray/train/examples/tf/tensorflow_quick_start.py
:language: python
:start-after: __tf_single_run_begin__
:end-before: __tf_single_run_end__
:dedent: 0
```

现在将其转换成分布式 multi-worker 训练方法。

1. 设置 *global* 批量大小 - each worker processes the same size
   batch as in the single-worker code.
2. Choose your TensorFlow distributed training strategy. This examples
   uses the ``MultiWorkerMirroredStrategy``.

```{literalinclude} /../../python/ray/train/examples/tf/tensorflow_quick_start.py
:language: python
:start-after: __tf_distributed_begin__
:end-before: __tf_distributed_end__
```

初始化一个 4 个工作现成的 ``TensorflowTrainer``，
并使用它运行新的训练方法。

```{literalinclude} /../../python/ray/train/examples/tf/tensorflow_quick_start.py
:language: python
:start-after: __tf_trainer_begin__
:end-before: __tf_trainer_end__
:dedent: 0
```

```{button-ref}  ../train/train
:color: primary
:outline:
:expand:

了解更多 Ray Train
```

````

`````

``````

`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Tune: Hyperparameter Tuning at Scale
:animate: fade-in-slide-down

[Tune](../tune/index.rst) 是用于任何规模的超参数调优的库。
借助 Tune，您可以用不到 10 行代码启动多节点分布式超参数扫描。
Tune 支持任何深度学习框架，包括 PyTorch、TensorFlow 和 Keras。

````{note}
要运行此示例，请安装 Ray Tune：

```bash
pip install -U "ray[tune]"
```
````

此示例使用迭代训练函数运行小型网格搜索。

```{literalinclude} ../../../python/ray/tune/tests/example.py
:end-before: __quick_start_end__
:language: python
:start-after: __quick_start_begin__
```

如果安装了 TensorBoard， automatically visualize all trial results:

```bash
tensorboard --logdir ~/ray_results
```

```{button-ref}  ../tune/index
:color: primary
:outline:
:expand:

了解更多 Ray Tune
```

`````


`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> 服务：可扩展模型服务
:animate: fade-in-slide-down

[Ray Serve](../serve/index) 是一个基于 Ray 构建的可扩展模型服务库。

````{note}
运行此示例，需安装 Ray Serve 和 scikit-learn：

```{code-block} bash
pip install -U "ray[serve]" scikit-learn
```
````

此示例运行服务于 scikit-learn 梯度增强分类器。

```{literalinclude} ../serve/doc_code/sklearn_quickstart.py
:language: python
:start-after: __serve_example_begin__
:end-before: __serve_example_end__
```

你将看到如示结果 `{"result": "versicolor"}`。

```{button-ref}  ../serve/index
:color: primary
:outline:
:expand:

了解更多 Ray Serve
```

`````


`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> RLlib: Industry-Grade Reinforcement Learning
:animate: fade-in-slide-down

[RLlib](../rllib/index.rst) 是一个基于 Ray 构建的工业级强化学习 (RL) 库。
RLlib 为各种行业和研究应用程序提供高可扩展性和统一的 API。

````{note}
运行此示例，请安装 `rllib` 以及 `tensorflow` 或 `pytorch`：

```bash
pip install -U "ray[rllib]" tensorflow  # or torch
```
````

```{literalinclude} ../../../rllib/examples/documentation/rllib_on_ray_readme.py
:end-before: __quick_start_end__
:language: python
:start-after: __quick_start_begin__
```

```{button-ref}  ../rllib/index
:color: primary
:outline:
:expand:

了解更多 Ray RLlib
```

`````

## Ray Core 入门

将类和函数轻松转为 Ray 任务和 actor，
适用于 Python 和 Java，具有用于构建和运行分布式应用程序的简单原语。


``````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Core: Parallelizing Functions with Ray Tasks
:animate: fade-in-slide-down

`````{tab-set}

````{tab-item} Python

:::{note}
要运行此示例，请安装 Ray Core：

```bash
pip install -U "ray"
```
:::

导入 Ray 并用以下命令初始化它 `ray.init()`。
使用 ``@ray.remote`` 装饰方法来定义要在远程运行的函数。
最后，通过调用方法的 ``.remote()`` 来替代常规调用。
这个远程调用会产生一个 future，一个 Ray _对象引用_，然后您可以使用 ``ray.get`` 获取它。

```{code-block} python

import ray
ray.init()

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures)) # [0, 1, 4, 9]
```

````

````{tab-item} Java

```{note}
要运行此示例，在你的项目添加 [ray-api](https://mvnrepository.com/artifact/io.ray/ray-api) 和 [ray-runtime](https://mvnrepository.com/artifact/io.ray/ray-runtime) 依赖。
```

使用 `Ray.init` 实例化 Ray 运行时。
使用 `Ray.task(...).remote()` 来转换任何 Java 静态方法为 Ray 任务。
该任务在远程工作进程中异步运行。`remote` 方法返回一个 ``ObjectRef``，
你可以通过 ``get`` 来获取实际的结果。

```{code-block} java

import io.ray.api.ObjectRef;
import io.ray.api.Ray;
import java.util.ArrayList;
import java.util.List;

public class RayDemo {

    public static int square(int x) {
        return x * x;
    }

    public static void main(String[] args) {
        // Intialize Ray runtime.
        Ray.init();
        List<ObjectRef<Integer>> objectRefList = new ArrayList<>();
        // Invoke the `square` method 4 times remotely as Ray tasks.
        // The tasks will run in parallel in the background.
        for (int i = 0; i < 4; i++) {
            objectRefList.add(Ray.task(RayDemo::square, i).remote());
        }
        // Get the actual results of the tasks.
        System.out.println(Ray.get(objectRefList));  // [0, 1, 4, 9]
    }
}
```

在上面的代码块中，我们定义了一些 Ray 任务。
虽然这些对于无状态操作非常有用，但有时您必须维护应用程序的状态。您可以使用 Ray Actors 来做到这一点。

```{button-ref}  ../ray-core/walkthrough
:color: primary
:outline:
:expand:

了解更多关于 Ray Core
```

````

`````

``````

``````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Core: Parallelizing Classes with Ray Actors
:animate: fade-in-slide-down

Ray 提供了 actor 来允许您并行化 Python 或 Java 中的类实例。
当您实例化作为 Ray actor 的类时，Ray 将在集群中启动该类的远程实例。
然后，该 actor 可以执行远程方法调用并维护其自己的内部状态。

`````{tab-set}

````{tab-item} Python

:::{note}
安装 Ray Core 运行示例：

```bash
pip install -U "ray"
```
:::

```{code-block} python

import ray
ray.init() # Only call this once.

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures)) # [1, 1, 1, 1]
```
````

````{tab-item} Java


```{note}
要运行示例，需在项目添加 [ray-api](https://mvnrepository.com/artifact/io.ray/ray-api) 和 [ray-runtime](https://mvnrepository.com/artifact/io.ray/ray-runtime) 依赖。
```

```{code-block} java

import io.ray.api.ActorHandle;
import io.ray.api.ObjectRef;
import io.ray.api.Ray;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class RayDemo {

    public static class Counter {

        private int value = 0;

        public void increment() {
            this.value += 1;
        }

        public int read() {
            return this.value;
        }
    }

    public static void main(String[] args) {
        // Intialize Ray runtime.
        Ray.init();
        List<ActorHandle<Counter>> counters = new ArrayList<>();
        // Create 4 actors from the `Counter` class.
        // They will run in remote worker processes.
        for (int i = 0; i < 4; i++) {
            counters.add(Ray.actor(Counter::new).remote());
        }

        // Invoke the `increment` method on each actor.
        // This will send an actor task to each remote actor.
        for (ActorHandle<Counter> counter : counters) {
            counter.task(Counter::increment).remote();
        }
        // Invoke the `read` method on each actor, and print the results.
        List<ObjectRef<Integer>> objectRefList = counters.stream()
            .map(counter -> counter.task(Counter::read).remote())
            .collect(Collectors.toList());
        System.out.println(Ray.get(objectRefList));  // [1, 1, 1, 1]
    }
}
```

```{button-ref}  ../ray-core/walkthrough
:color: primary
:outline:
:expand:

了解更多 Ray Core
```

````

`````

``````

## Ray Cluster 入门

在 Ray 集群部署你的应用，通常对现有代码进行最少的代码更改。

`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Clusters: Launching a Ray Cluster on AWS
:animate: fade-in-slide-down

Ray 程序可以在单台机器上运行，也可以无缝扩展到大型集群。
以这个等待各个节点加入集群的简单示例为例。

````{dropdown} example.py
:animate: fade-in-slide-down

```{literalinclude} ../../yarn/example.py
:language: python
```
````
你可以从我们的 [GitHub 仓库](https://github.com/ray-project/ray/blob/master/doc/yarn/example.py) 下载示例。
继续将其存储在本地一个名为 `example.py` 的文件中。

要在云端执行脚本，需要下载 [配置文件](https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-full.yaml)，
或者从这里拷贝：

````{dropdown} cluster.yaml
:animate: fade-in-slide-down

```{literalinclude} ../../../python/ray/autoscaler/aws/example-full.yaml
:language: yaml
```
````

假设您已将此配置存储在名为 `cluster.yaml` 的文件中，您现在可以按如下方式启动 AWS 集群：

```bash
ray submit cluster.yaml example.py --start
```

```{button-ref}  cluster-index
:color: primary
:outline:
:expand:

了解有关启动 Ray Cluster 的更多信息
```

`````

## 调试和监控快速入门

使用内置的可观察性工具来监视和调试 Ray 应用程序和集群。


`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Ray Dashboard: Web GUI to monitor and debug Ray
:animate: fade-in-slide-down

Ray仪表板提供可视化界面，显示实时系统指标、节点级资源监控、作业分析和任务可视化。该仪表板旨在帮助用户了解 Ray 应用程序的性能并识别潜在问题。

```{image} https://raw.githubusercontent.com/ray-project/Images/master/docs/new-dashboard/Dashboard-overview.png
:align: center
```

````{note}
要开始使用仪表板，请安装默认安装，如下所示：

```bash
pip install -U "ray[default]"
```
````
通过默认 URL 访问仪表板，http://localhost:8265。

```{button-ref}  observability-getting-started
:color: primary
:outline:
:expand:

了解更多 Ray Dashboard
```

`````

`````{dropdown} <img src="images/ray_svg_logo.svg" alt="ray" width="50px"> Ray State APIs: CLI to access cluster states
:animate: fade-in-slide-down

Ray 状态 API 允许用户通过 CLI 或 Python SDK 方便地访问 Ray 的当前状态（快照）。

````{note}
要开始使用状态 API，请参考默认安装，如下所示：

```bash
pip install -U "ray[default]"
```
````

运行以下代码。

```{code-block} python

    import ray
    import time

    ray.init(num_cpus=4)

    @ray.remote
    def task_running_300_seconds():
        print("Start!")
        time.sleep(300)

    @ray.remote
    class Actor:
        def __init__(self):
            print("Actor created")

    # Create 2 tasks
    tasks = [task_running_300_seconds.remote() for _ in range(2)]

    # Create 2 actors
    actors = [Actor.remote() for _ in range(2)]

    ray.get(tasks)

```

使用 ``ray summary tasks`` 查看 Ray 任务的汇总统计数据。

```{code-block} bash

    ray summary tasks

```

```{code-block} text

    ======== Tasks Summary: 2022-07-22 08:54:38.332537 ========
    Stats:
    ------------------------------------
    total_actor_scheduled: 2
    total_actor_tasks: 0
    total_tasks: 2


    Table (group by func_name):
    ------------------------------------
        FUNC_OR_CLASS_NAME        STATE_COUNTS    TYPE
    0   task_running_300_seconds  RUNNING: 2      NORMAL_TASK
    1   Actor.__init__            FINISHED: 2     ACTOR_CREATION_TASK

```

```{button-ref}  observability-programmatic
:color: primary
:outline:
:expand:

学习更多关于 Ray State APIs
```

`````

```{include} learn-more.md
```
