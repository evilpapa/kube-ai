# 基于注释的配置

您可以通过 SeldonDeployment 资源中的注释以及可选的 API OAuth 网关配置 Seldon 核心的各个方面。如果您想添加一些配置，请创建一个问题。

## SeldonDeployment 注释

### gRPC API 控制

 * ```seldon.io/grpc-max-message-size``` : 最大 gRPC 消息大小（字节）
   * 位置 : SeldonDeployment.spec.annotations
   * 默认为 MaxInt32
   * [gRPC 消息大小示例](model_rest_grpc_settings.md)
 * ```seldon.io/grpc-timeout``` : gRPC 超时（毫秒）
   * 位置 : SeldonDeployment.spec.annotations
   * 默认为不超时
   * [gRPC 超时示例](model_rest_grpc_settings.md)


### REST API 控制

.. Note:: 
   使用 REST API 时，超时仅适用于每个节点，而不适用于完整的推理图。
   因此，图中每个单独节点的每个子请求最多可能需要 ``seldon.io/rest-timeout`` 毫秒

* ```seldon.io/rest-timeout``` : REST 超时（毫秒）
  * 位置 : SeldonDeployment.spec.annotations
  * 默认没有整体超时，但将使用 GoLang 的默认传输设置，其中包括 30 秒的连接超时。
  * [REST 超时示例](model_rest_grpc_settings.md)


### 服务编排器

  * ```seldon.io/engine-separate-pod``` : 为服务编排器使用单独的 pod
    * 位置 : SeldonDeployment.spec.annotations
    * [Separate svc-orc pod 示例](model_svcorch_sep.md)
  * ```seldon.io/headless-svc``` : 将主端点作为无状态 Kubernetes 服务运行。这是通过 Ingress 进行 gRPC 负载平衡所必需的。
    * 位置 : SeldonDeployment.spec.annotations
    * [gRPC 无状态示例](grpc_load_balancing_ambassador.md)
  * ```seldon.io/executor-logger-queue-size``` : 请求记录工作队列的大小
    * 位置: SeldonDeployment.metadata.annotations, SeldonDeployment.spec.annotations
  * ```seldon.io/executor-logger-write-timeout-ms``` : 写入到日志工作队列的超时时间
    * 位置: SeldonDeployment.metadata.annotations, SeldonDeployment.spec.annotations


### Misc

 * ```seldon.io/svc-name``` : 预测器的自定义服务名称。您将负责它不会与部署的 SeldonDeployment 的命名空间中的任何现有服务名称发生冲突。
   * 位置 : SeldonDeployment.spec.predictors[].annotations
   * [自定义服务名称示例](custom_svc_name.md)

