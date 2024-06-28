# AB 测试和渐进式发布

## 简单的 AB 测试

Seldon Core 提供了使用 Istio 和 Ambassador 按照要求进行流量划分，轻松创建 AB 测试和影子流量部署的能力。

 * [Istio AB Test/Canary 示例](../examples/istio_canary.html)
 * [Ambassador AB Tests/Canary 示例](../examples/ambassador_canary.html)

使用 [Seldon Analytics 面板](../analytics/analytics.html)可在 AB 测试中使用 prometheus 为不同的预估器分析计算指标。


## 高级 AB 测试实验以及渐进式发布

更高级的用例我们推荐使用 [Iter8](https://iter8.tools) 进行集成，它提供了明确的候选实验目标以及清晰的候选模型优胜版。Iter8 还提供渐进式发布功能，以自动选择测试候选模型，如果候选模型的性能优于孵化模型，则将其推广到生产环境。

在 Seldon，我们提供了两个关于如何运行 Iter8 实验的当前示例。

 1. 在单个 Seldon Deployment 上的 Seldon/Iter8 实验。
 1. 在特定 Seldon Deployments 上的 Seldon/Iter8 实验。


## 在单个 Seldon Deployment 上的 Seldon - Iter8 实验

第一个选项是使用更新的 Seldon Deployment 候选模型创建 AB 实验，运行 Iter8 实验，并以一组指标标准逐步推出候选模型。架构如下：

![seldonIter8Single](seldon-iter8-single.png)


我们首先更新默认模型以启动 AB 测试，如下所示：

```yaml
apiVersion: v1
kind: Namespace
metadata:
    name: ns-production
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris
  namespace: ns-production
spec:
  predictors:
  - name: baseline
    traffic: 100    
    graph:
      name: classifier
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      implementation: SKLEARN_SERVER
  - name: candidate
    traffic: 0
    graph:
      name: classifier
      modelUri: gs://seldon-models/xgboost/iris
      implementation: XGBOOST_SERVER

```

这里我们有孵化的 SKLearn 模型和一个候选的 XGBoost 模型来替代它，当前流量为 0。

接下来，我们告诉 Iter8 它可用于 Iter8 指标自定义资源的指标。

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: iter8-seldon
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: 95th-percentile-tail-latency
  namespace: iter8-seldon
spec:
  description: 95th percentile tail latency
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      histogram_quantile(0.95, sum(rate(seldon_api_executor_client_requests_seconds_bucket{seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) by (le))
  provider: prometheus
  sampleSize: iter8-seldon/request-count
  type: Gauge
  units: milliseconds
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: error-count
  namespace: iter8-seldon
spec:
  description: Number of error responses
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      sum(increase(seldon_api_executor_server_requests_seconds_count{code!='200',seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0)
  provider: prometheus
  type: Counter
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query  
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: error-rate
  namespace: iter8-seldon
spec:
  description: Fraction of requests with error responses
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      (sum(increase(seldon_api_executor_server_requests_seconds_count{code!='200',seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0)) / (sum(increase(seldon_api_executor_server_requests_seconds_count{seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0))
  provider: prometheus
  sampleSize: iter8-seldon/request-count
  type: Gauge
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query    
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: mean-latency
  namespace: iter8-seldon
spec:
  description: Mean latency
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      (sum(increase(seldon_api_executor_client_requests_seconds_sum{seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0)) / (sum(increase(seldon_api_executor_client_requests_seconds_count{seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0))
  provider: prometheus
  sampleSize: iter8-seldon/request-count
  type: Gauge
  units: milliseconds
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query      
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: request-count
  namespace: iter8-seldon
spec:
  description: Number of requests
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      sum(increase(seldon_api_executor_client_requests_seconds_sum{seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0)
  provider: prometheus
  type: Counter
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query
---
apiVersion: iter8.tools/v2alpha2
kind: Metric
metadata:
  name: user-engagement
  namespace: iter8-seldon
spec:
  description: Number of feedback requests
  jqExpression: .data.result[0].value[1] | tonumber
  params:
  - name: query
    value: |
      sum(increase(seldon_api_executor_server_requests_seconds_count{service='feedback',seldon_deployment_id='$sid',predictor_name='$predictor',kubernetes_namespace='$ns'}[${elapsedTime}s])) or on() vector(0)
  provider: prometheus
  type: Gauge
  urlTemplate: http://seldon-core-analytics-prometheus-seldon.seldon-system/api/v1/query

```

这创建了一组指标，用于与其相应的 Prometheus 查询语言表达式的实验。这些指标是参数化的，可用于不同的实验。

```
NAME                           TYPE      DESCRIPTION
95th-percentile-tail-latency   Gauge     95th percentile tail latency
error-count                    Counter   Number of error responses
error-rate                     Gauge     Fraction of requests with error responses
mean-latency                   Gauge     Mean latency
request-count                  Counter   Number of requests
user-engagement                Gauge     Number of feedback requests
```

然后，这些指标可用于实验定义奖励给对比模型和服务级目标模型来决策出运行成功的目标。

一旦指标定义，就可以按照 Iter8 实验 CRD 标识开始实验：

```yaml
apiVersion: iter8.tools/v2alpha2
kind: Experiment
metadata:
  name: quickstart-exp
spec:
  target: iris
  strategy:
    testingPattern: A/B
    deploymentPattern: Progressive
    actions:
      # when the experiment completes, promote the winning version using kubectl apply
      finish:
      - task: common/exec
        with:
          cmd: /bin/bash
          args: [ "-c", "kubectl apply -f {{ .promote }}" ]
  criteria:
    requestCount: iter8-seldon/request-count
    rewards: # Business rewards
    - metric: iter8-seldon/user-engagement
      preferredDirection: High # maximize user engagement
    objectives:
    - metric: iter8-seldon/mean-latency
      upperLimit: 2000
    - metric: iter8-seldon/95th-percentile-tail-latency
      upperLimit: 5000
    - metric: iter8-seldon/error-rate
      upperLimit: "0.01"
  duration:
    intervalSeconds: 10
    iterationsPerLoop: 15
  versionInfo:
    # information about model versions used in this experiment
    baseline:
      name: iris-v1
      weightObjRef:
        apiVersion: machinelearning.seldon.io/v1
        kind: SeldonDeployment
        name: iris
        namespace: ns-production
        fieldPath: .spec.predictors[0].traffic
      variables:
      - name: ns
        value: ns-production
      - name: sid
        value: iris
      - name: predictor
        value: baseline
      - name: promote
        value: https://raw.githubusercontent.com/SeldonIO/seldon-core/master/examples/iter8/progressive_rollout/single_sdep/promote-v1.yaml
    candidates:
    - name: iris-v2
      weightObjRef:
        apiVersion: machinelearning.seldon.io/v1
        kind: SeldonDeployment
        name: iris
        namespace: ns-production
        fieldPath: .spec.predictors[1].traffic
      variables:
      - name: ns
        value: ns-production
      - name: sid
        value: iris
      - name: predictor
        value: candidate
      - name: promote
        value: https://raw.githubusercontent.com/SeldonIO/seldon-core/master/examples/iter8/progressive_rollout/single_sdep/promote-v2.yaml

```

这有几个关键部分：

 * Strategy: 策略：运行的实验类型和完成的操作。
 * Criteria: 标准：奖励和服务目标的关键指标。
 * Duration: 持续时间：运行实验的时间。
 * VersionInfo: 版本信息：要比较的各种候选模型的详细信息。

实验启动后，流量将根据定义的奖励和目标移动到各个候选者。

随着实验的进行，可以使用 iter8 工具 `iter8ctl` 跟踪状态：

```
****** Overview ******
Experiment name: quickstart-exp
Experiment namespace: seldon
Target: iris
Testing pattern: A/B
Deployment pattern: Progressive

****** Progress Summary ******
Experiment stage: Running
Number of completed iterations: 6

****** Winner Assessment ******
App versions in this experiment: [iris-v1 iris-v2]
Winning version: iris-v2
Version recommended for promotion: iris-v2

****** Objective Assessment ******
> Identifies whether or not the experiment objectives are satisfied by the most recently observed metrics values for each version.
+-------------------------------------------+---------+---------+
|                 OBJECTIVE                 | IRIS-V1 | IRIS-V2 |
+-------------------------------------------+---------+---------+
| iter8-seldon/mean-latency <=              | true    | true    |
|                                  2000.000 |         |         |
+-------------------------------------------+---------+---------+
| iter8-seldon/95th-percentile-tail-latency | true    | true    |
| <= 5000.000                               |         |         |
+-------------------------------------------+---------+---------+
| iter8-seldon/error-rate <=                | true    | true    |
|                                     0.010 |         |         |
+-------------------------------------------+---------+---------+

****** Metrics Assessment ******
> Most recently read values of experiment metrics for each version.
+-------------------------------------------+---------+---------+
|                  METRIC                   | IRIS-V1 | IRIS-V2 |
+-------------------------------------------+---------+---------+
| iter8-seldon/request-count                |   5.256 |   1.655 |
+-------------------------------------------+---------+---------+
| iter8-seldon/user-engagement              |  49.867 |  68.240 |
+-------------------------------------------+---------+---------+
| iter8-seldon/mean-latency                 |   0.016 |   0.016 |
| (milliseconds)                            |         |         |
+-------------------------------------------+---------+---------+
| iter8-seldon/95th-percentile-tail-latency |   0.025 |   0.045 |
| (milliseconds)                            |         |         |
+-------------------------------------------+---------+---------+
| iter8-seldon/error-rate                   |   0.000 |   0.000 |
+-------------------------------------------+---------+---------+


```

我们也可以通过 kubectl 检查实验的状态：

```bash
kubectl get experiment
NAME             TYPE   TARGET   STAGE       COMPLETED ITERATIONS   MESSAGE
quickstart-exp   A/B    iris     Completed   15                     ExperimentCompleted: Experiment Completed

```

在上述示例中，为成功候选版本定义了最后操作行为，来更新新的默认的 Seldon deployment。

下一步 [运行示例 notebook](../examples/iter8-single.html)。

## 运行特定 Seldon Deployments 的 Seldon/Iter8 实验

我们还可以在特定的部署上运行实验。这需要通过在创建服务网格是选择路由流量规则可被 Iter8 修改并推送流量到各个 Seldon Deployment。

此类实验的架构如下所示：

![seldonIter8Separate](seldon-iter8-separate.png)

不同之处在与我们有两个 Seldon Deployments。一个基线为：

```yaml
apiVersion: v1
kind: Namespace
metadata:
    name: ns-baseline
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris
  namespace: ns-baseline
spec:
  predictors:
  - name: default
    graph:
      name: classifier
      modelUri: gs://seldon-models/v1.10.0-dev/sklearn/iris
      implementation: SKLEARN_SERVER
```

一个候选模型：

```yaml
apiVersion: v1
kind: Namespace
metadata:
    name: ns-candidate
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris
  namespace: ns-candidate
spec:
  predictors:
  - name: default
    graph:
      name: classifier
      modelUri: gs://seldon-models/xgboost/iris
      implementation: XGBOOST_SERVER
```

然后，我们需要 Istio 定义路由规则来为上面两个划分流量：


```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: routing-rule
  namespace: default
spec:
  gateways:
  - istio-system/seldon-gateway
  hosts:
  - iris.example.com
  http:
  - route:
    - destination:
        host: iris-default.ns-baseline.svc.cluster.local
        port:
          number: 8000
      headers:
        response:
          set:
            version: iris-v1
      weight: 100
    - destination:
        host: iris-default.ns-candidate.svc.cluster.local
        port:
          number: 8000
      headers:
        response:
          set:
            version: iris-v2
      weight: 0

```

指标定义同上面的节点相同。与上面实验非常相似，但具有不同的 VersionInfo 指向到 Istio VirtualService 来修改切换流量：

```yaml
apiVersion: iter8.tools/v2alpha2
kind: Experiment
metadata:
  name: quickstart-exp
spec:
  target: iris
  strategy:
    testingPattern: A/B
    deploymentPattern: Progressive
    actions:
      # when the experiment completes, promote the winning version using kubectl apply
      finish:
      - task: common/exec
        with:
          cmd: /bin/bash
          args: [ "-c", "kubectl apply -f {{ .promote }}" ]
  criteria:
    requestCount: iter8-seldon/request-count
    rewards: # Business rewards
    - metric: iter8-seldon/user-engagement
      preferredDirection: High # maximize user engagement
    objectives:
    - metric: iter8-seldon/mean-latency
      upperLimit: 2000
    - metric: iter8-seldon/95th-percentile-tail-latency
      upperLimit: 5000
    - metric: iter8-seldon/error-rate
      upperLimit: "0.01"
  duration:
    intervalSeconds: 10
    iterationsPerLoop: 10
  versionInfo:
    # information about model versions used in this experiment
    baseline:
      name: iris-v1
      weightObjRef:
        apiVersion: networking.istio.io/v1alpha3
        kind: VirtualService
        name: routing-rule
        namespace: default
        fieldPath: .spec.http[0].route[0].weight      
      variables:
      - name: ns
        value: ns-baseline
      - name: sid
        value: iris
      - name: predictor
        value: default
      - name: promote
        value: https://raw.githubusercontent.com/SeldonIO/seldon-core/master/examples/iter8/progressive_rollout/separate_sdeps/promote-v1.yaml
    candidates:
    - name: iris-v2
      weightObjRef:
        apiVersion: networking.istio.io/v1alpha3
        kind: VirtualService
        name: routing-rule
        namespace: default
        fieldPath: .spec.http[0].route[1].weight      
      variables:
      - name: ns
        value: ns-candidate
      - name: sid
        value: iris
      - name: predictor
        value: default
      - name: promote
        value: https://raw.githubusercontent.com/SeldonIO/seldon-core/master/examples/iter8/progressive_rollout/separate_sdeps/promote-v2.yaml

```

实验的进展与在这种情况下，将最佳模型推广到现有默认基线上相似。

下一步骤 [运行示例 notebook](../examples/iter8-separate.html)。