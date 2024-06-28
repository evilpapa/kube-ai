# 扩缩容

## 副本设置

副本设置可在几个优先级别上提供，从最一般到最具体，如下所示：

  * `.spec.replicas`
  * `.spec.predictors[].replicas`
  * `.spec.predictors[].componentSpecs[].replicas`

如果使用了 `seldon.io/engine-separate-pod` 注解，你也可以为服务编排设置指定数量的副本：

 * `.spec.predictors[].svcOrchSpec.replicas`

如下所示，下面显示了各种选项的示例：

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: test-replicas
spec:
  replicas: 1
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier2
      replicas: 3
    graph:
      endpoint:
        type: REST
      name: classifier
      type: MODEL
      children:
      - name: classifier2
        type: MODEL
        endpoint:
          type: REST
    name: example
    replicas: 2
    traffic: 50
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier3
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier3
      type: MODEL
    name: example2
    traffic: 50

```

 * classfier 在 predictors 定义下有 2 个副本 
 * classifier2 在 componentSpec 定义下有 3 个部署的副本
 * classifier3 在 `.spec.replicas` 值定义下只有 1 个副本

更多信息查看 [可工作副本设置示例](../examples/scale.html)。

## 副本调度

有能力使用 `kubectl scale` 命令设置 SeldonDeployment 的 `replicas` 值。在单预估图中可轻松进行扩缩容。比如：

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-scale
spec:
  replicas: 1  
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: example
```

通过以下命令为 Seldon Deployment 扩容

```console
kubectl scale --replicas=2 sdep/seldon-scale
```

更多信息参考 [扩缩容配置示例](../examples/scale.html)。

## 自动扩容 Seldon Deployments

要自动为 Seldon Deployment 资源扩容，你可以添加水平的 Horizontal Pod 模板定义。创建模板定义分三步执行：

  1. 如果是标准指标（如 cpu 或内存），请确保您对要扩展的指标有资源请求。
  1. 添加 HPA 定义关联到部署。（我们当前提供 v1beta1 版本 k8s HPA 指标定义）

为了说明这一点，我们有一个例子 Seldon 部署如下：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - hpaSpec:
        maxReplicas: 3
        minReplicas: 1
        metrics:
        - resource:
            name: cpu
            targetAverageUtilization: 70
          type: Resource
      spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          imagePullPolicy: IfNotPresent
          name: classifier
          resources:
            requests:
              cpu: '0.5'
        terminationGracePeriodSeconds: 1
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: example
```

这里的要点是：

 * 我们定义了容器的 CPU 请求。这需要允许我们在 k8s 为 cpu 自动扩展。
 * 我们定义了与组件相关的 HPA Spec，当平均 CPU 高于 70% 时，该组件在 CPU 上缩放，最多可复制 3 个副本。

一旦发布，HPA 资源会花几分钟来启动。要检查 HPA 资源状态，`kubectl describe hpa -n <podname>` 将会有用。


参考示例 [notebook](../examples/autoscaling_example.html)。
