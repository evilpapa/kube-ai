.. _train-benchmarks:

Ray Train 基准测试
====================

下面我们记录常见 Ray Train 任务和工作流程的关键性能基准。

.. _pytorch_gpu_training_benchmark:

GPU 图像训练
------------------

此任务使用 TorchTrainer 模块通过 Pytorch ResNet 模型训练
不同数量的数据。

我们测试了不同集群大小和数据大小的性能。

- `GPU 图像训练脚本`_
- `GPU 训练小集群配置`_
- `GPU 训练大集群配置`_

.. note::

    对于多主机分布式训练，在 AWS 上我们需要确保 ec2 实例位于同一个 VPC 中
    并且所有端口都在安全组中打开。


.. list-table::

    * - **集群设置**
      - **数据大小**
      - **表现**
      - **命令**
    * - 1 g3.8xlarge 节点 (1 worker)
      - 1 GB (1623 张图片)
      - 79.76 秒 (2 epochs, 40.7 张图片/秒)
      - `python pytorch_training_e2e.py --data-size-gb=1`
    * - 1 g3.8xlarge 节点 (1 worker)
      - 20 GB (32460 张图片)
      - 1388.33 s (2 epochs, 46.76 张图片/秒)
      - `python pytorch_training_e2e.py --data-size-gb=20`
    * - 4 g3.16xlarge 节点 (16 workers)
      - 100 GB (162300 张图片)
      - 434.95 s (2 epochs, 746.29 张图片/秒)
      - `python pytorch_training_e2e.py --data-size-gb=100 --num-workers=16`

.. _pytorch-training-parity:

Pytorch 平价 Training 
-----------------------

此任务检查本机 Pytorch Distributed 和 Ray Train 的分布式 TorchTrainer 之间的性能一致性。

我们证明两个框架之间的性能相似（在 2.5\% 以内）。不同的模型、硬件和集群配置之间的性能可能会有很大差异。

报告的时间是原始训练时间。两种方法都有未报告的几秒钟的恒定设置开销，
对于较长的训练运行来说，这可以忽略不计。

- `Pytorch 对比训练脚本`_
- `Pytorch 比较 CPU 集群配置`_
- `Pytorch 比较 GPU 集群配置`_

.. list-table::

    * - **集群设置**
      - **数据集**
      - **表现**
      - **命令**
    * - 4 m5.2xlarge 节点 (4 workers)
      - FashionMNIST
      - 196.64 s (vs 194.90 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 4 --cpus-per-worker 8`
    * - 4 m5.2xlarge 节点 (16 workers)
      - FashionMNIST
      - 430.88 s (vs 475.97 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 2`
    * - 4 g4dn.12xlarge 节点 (16 workers)
      - FashionMNIST
      - 149.80 s (vs 146.46 s Pytorch)
      - `python workloads/torch_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 4 --use-gpu`


.. _tf-training-parity:

Tensorflow 平价 Training
--------------------------

此任务检查本机 Tensorflow Distributed 和 Ray Train 的
分布式 TensorflowTrainer 之间的性能一致性。

我们证明两个框架之间的性能相似（在 1\% 以内）。
不同的模型、硬件和集群配置之间的性能可能会有很大差异。

报告的时间是原始训练时间。
两种方法都有未报告的几秒钟的恒定设置开销，对于较长的训练运行来说，这可以忽略不计。

.. note:: GPU 基准的批次大小和时期数不同，导致运行时间更长。

- `Tensorflow 对比训练脚本`_
- `Tensorflow 比较 CPU 集群配置`_
- `Tensorflow 比较 GPU 集群配置`_

.. list-table::

    * - **集群设置**
      - **数据集**
      - **表现**
      - **命令**
    * - 4 m5.2xlarge 节点 (4 workers)
      - FashionMNIST
      - 78.81 s (vs 79.67 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 4 --cpus-per-worker 8`
    * - 4 m5.2xlarge 节点 (16 workers)
      - FashionMNIST
      - 64.57 s (vs 67.45 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 20 --num-workers 16 --cpus-per-worker 2`
    * - 4 g4dn.12xlarge 节点 (16 workers)
      - FashionMNIST
      - 465.16 s (vs 461.74 s Tensorflow)
      - `python workloads/tensorflow_benchmark.py run --num-runs 3 --num-epochs 200 --num-workers 16 --cpus-per-worker 4 --batch-size 64 --use-gpu`

.. _xgboost-benchmark:

XGBoost 训练
----------------

该任务使用 XGBoostTrainer 模块对不同大小的数据以不同的并行度进行训练。

对于 xgboost==1.6.1 此任务，XGBoost 参数保持为默认值。


- `XGBoost 训练脚本`_
- `XGBoost 集群配置`_

.. list-table::

    * - **集群设置**
      - **数据大小**
      - **表现**
      - **命令**
    * - 1 m5.4xlarge 节点 (1 actor)
      - 10 GB (26M rows)
      - 692 s
      - `python xgboost_benchmark.py --size 10GB`
    * - 10 m5.4xlarge 节点 (10 actors)
      - 100 GB (260M rows)
      - 693 s
      - `python xgboost_benchmark.py --size 100GB`

.. _`GPU 图像训练脚本`: https://github.com/ray-project/ray/blob/cec82a1ced631525a4d115e4dc0c283fa4275a7f/release/air_tests/air_benchmarks/workloads/pytorch_training_e2e.py#L95-L106
.. _`GPU 训练小集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_1_aws.yaml#L6-L24
.. _`GPU 训练大集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml#L5-L25
.. _`Pytorch 对比训练脚本`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/workloads/torch_benchmark.py
.. _`Pytorch 对比 CPU 集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_cpu_4_aws.yaml
.. _`Pytorch 对比 GPU 集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml
.. _`Tensorflow 对比训练脚本`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/workloads/tensorflow_benchmark.py
.. _`Tensorflow 对比 CPU 集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_cpu_4_aws.yaml
.. _`Tensorflow 对比 GPU 集群配置`: https://github.com/ray-project/ray/blob/master/release/air_tests/air_benchmarks/compute_gpu_4x4_aws.yaml
.. _`XGBoost 训练脚本`: https://github.com/ray-project/ray/blob/a241e6a0f5a630d6ed5b84cce30c51963834d15b/release/air_tests/air_benchmarks/workloads/xgboost_benchmark.py#L40-L58
.. _`XGBoost 集群配置`: https://github.com/ray-project/ray/blob/a241e6a0f5a630d6ed5b84cce30c51963834d15b/release/air_tests/air_benchmarks/xgboost_compute_tpl.yaml#L6-L24
