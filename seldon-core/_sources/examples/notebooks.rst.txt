=========
笔记本
=========

Seldon Core 设置
-----------------

.. toctree::
   :titlesonly:

   安装 Seldon Core <seldon_core_setup>
   安装 MinIO <minio_setup>



预包装推理服务示例
-------------------------------------

.. toctree::
   :titlesonly:

   发布二进制 Scikit-learn 模型 <../servers/sklearn.md>
   发布 Tensorflow 到处模型 <../servers/tensorflow.md>
   MLflow 预包装服务 A/B 测试 <mlflow_server_ab_test_ambassador>
   MLflow v2 协议点对点 Workflow (孵化中) <mlflow_v2_protocol_end_to_end>
   发布二进制 XGBoost 模型 <../servers/xgboost.md>
   发布集成集群 MinIO 的预包装模型服务 <minio-sklearn>
   自定义预包装 LightGBM 服务 <custom_server>

Python 语言封装示例
--------------------------------

.. toctree::
   :titlesonly:

   SKLearn Spacy NLP <sklearn_spacy_text_classifier_example>
   SKLearn 虹膜分类器 <iris>
   Sagemaker SKLearn 示例 <sagemaker_sklearn>
   TFserving MNIST <tfserving_mnist>
   Statsmodels Holt-Winter's 时序模型 <statsmodels>
   运行时指标及标签 <runtime_metrics_tags>
   Triton GPT2 示例 <triton_gpt2_example>

专门的框架示例
------------------------------

.. toctree::
   :titlesonly:

   NVIDIA TensorRT MNIST <tensorrt>
   OpenVINO ImageNet <openvino>
   OpenVINO ImageNet Ensemble <openvino_ensemble>
   Triton 示例 <triton_examples>

孵化项目示例
----------------------------

.. toctree::
   :titlesonly:

   Kubeflow Seldon E2E 管道 <kubeflow_seldon_e2e_pipeline>
   H2O Java MoJo <h2o_mojo>
   合并器异常值检测 <outlier_combiner>
   KNative Eventing 流处理 <knative_eventing_streaming>
   Kafka CIFAR10 <cifar10_kafka>
   Kafka SpaCy SKlearn NLP <kafka_spacy_sklearn>
   Kafka KEDA 自动缩放 <kafka_keda>
   CPP 单文件封装 <cpp_simple>
   CPP 高级自定义封装系统 <cpp_advanced>


基于云的基础示例
-----------------------

.. toctree::
   :titlesonly:

   AWS EKS Tensorflow Deep MNIST <aws_eks_deep_mnist>
   Azure AKS Tensorflow Deep MNIST <azure_aks_deep_mnist>
   GKE 的 GPU Tensorflow Deep MNIST <gpu_tensorflow_deep_mnist>
   Alibaba Cloud Tensorflow Deep MNIST <alibaba_ack_deep_mnist>
   Triton GPT2 Azure 示例 <triton_gpt2_example_azure>
   Azure 设置 Triton GPT2 示例  <triton_gpt2_example_azure_setup>

高级机器学习监控
---------------------------------------

.. toctree::
   :titlesonly:

   统计指标的实时监控 <feedback_reward_custom_metrics>
   模型解释器示例 <iris_explainer_poetry>
   模型 V2 协议解释器示例「孵化中」 <iris_anchor_tabular_explainer_v2>
   Outlier Detection on CIFAR10 <outlier_cifar10>
   Training Outlier Detector for CIFAR10 with Poetry <cifar10_od_poetry>

Seldon Core 批处理
---------------------------------

.. toctree::
   :titlesonly:

   集成 Argo Workflows 和 S3 / Minio 的批处理 <argo_workflows_batch>
   集成 Argo Workflows 和 HDFS 的批处理 <argo_workflows_hdfs_batch>
   集成 Kubeflow Pipelines 的批处理 <kubeflow_pipelines_batch>


MLOps: 扩展、监控和可观察性
-----------------------------------------------

.. toctree::
   :titlesonly:

   自动缩放示例 Example <autoscaling_example>
   KEDA 自动缩放示例 <keda>
   ELK 请求负载记录 <payload_logging>
   Grafana & Prometheus 自定义指标 <metrics>
   Jaeger 链路追踪 <tracing>
   CI / CD 集成 Jenkins Classic <jenkins_classic>
   CI / CD 集成 Jenkins X <jenkins_x>
   副本控制 <scale>

生产配置及实现
------------------------------------------

.. toctree::
   :titlesonly:

   Helm 部署示例 <helm_examples>
   最大 gRPC 消息大小 <max_grpc_msg_size>
   部署多 Seldon Core Operators <multiple_operators>
   协议示例 <protocol_examples>
   可配置超时 <timeouts>
   自定义 Protobuf Data 示例 <customdata_example>
   熔断示例 <pdbs_example>


AB 测试及渐进式部署
---------------------------------

.. toctree::
   :titlesonly:

   Istio AB 测试 <istio_canary>
   Ambassador AB 测试 <ambassador_canary>
   Seldon/Iter8 - 单次 Seldon Deployment 的渐进式 AB 测试 <iter8-single>
   Seldon/Iter8 - 多个 Seldon Deployments 的渐进式 AB 测试 <iter8-separate>



复杂图示例
----------------------

.. toctree::
   :titlesonly:
  
   Chainer MNIST 模型部署 <chainer_mnist>
   使用 V2 协议的自定义预处理器 <transformers-v2-protocol>

入口
-------

.. toctree::
   :titlesonly:
  
   Ambassador 金丝雀 <ambassador_canary>
   Ambassador 影子流量 <ambassador_shadow>
   Ambassador 头 <ambassador_headers>
   Ambassador 自定义配置 <ambassador_custom>
   Istio 示例 <istio_examples>
   Istio 金丝雀 <istio_canary>

基础设施
--------------

.. toctree::
   :titlesonly:
  
   1.2.0 版升级的补丁卷 <patch_1_2>
   

基准测试和负载测试
---------------------------

.. toctree::
   :titlesonly:
  
   服务编排器开销 <bench_svcOrch>
   Tensorflow 基准测试 <bench_tensorflow>
   Argo Workflows 基准测试 <vegeta_bench_argo_workflows>
   Python 序列化消耗基准测试 <python_serialization>
   KMP_AFFINITY 基准测试示例 <python_kmp_affinity>
   Kafka 负载记录 <kafka_logger>


升级示例
------------------

.. toctree::
   :titlesonly:

   RClone Storage Initializer - 测试新密钥格式 <rclone-upgrade>
   RClone Storage Initializer - 升级集群 (AWS S3 / MinIO) <global-rclone-upgrade>
