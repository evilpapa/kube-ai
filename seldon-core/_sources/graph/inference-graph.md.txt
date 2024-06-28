# 推理图

Seldon Core 使用 k8s 自定义资源 SeldonDeployment 扩展 Kubernetes，您可以在其中定义由 Seldon 管理的模型和其他组件组成的运行时序推理图。

一个 SeldonDeployment 为 JSON 或 YAML 文件，它允许你定义一张组件镜像图，以及运行在（Kubernetes PodTemplateSpec）模式下的镜像的资源。如下示例：

![](./inf-graph.png)

最小单元的 YAML 模型示例
```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: seldonio/mock_classifier:1.0
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: example
    replicas: 1
```

关键组件包括：

  * 预估列表，每个都包含特定数量的副本
     * 每个都包含相应的推理图及部署信息。想在生产和 Canary 之间区分和控制流量是，多个推理图就会至关重要
  * 预估器的 `componentSpecs`。每个 `componentSpec` 都是 Kubernetes 的`PodTemplateSpec`，Seldon 将构建到 Kubernetes Deployment。在这里贴出推理图所需镜像、依赖，如：Volumes, ImagePullSecrets, Resources Requests 等。
  * 描述组件的组合方式

## 带有前置和后置处理的推理图示例

面我们展示了一个具有稍微复杂的图形结构的示例。我们正在定义预处理器和后处理器组件

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - name: step_one
          image: seldonio/step_one:1.0
        - name: step_two
          image: seldonio/step_two:1.0
        - name: step_three
          image: seldonio/step_three:1.0
    graph:
      name: step_one
      endpoint:
        type: REST
      type: MODEL
      children:
          name: step_two
          endpoint:
            type: REST
          type: MODEL
          children:
              name: step_three
              endpoint:
                type: REST
              type: MODEL
              children: []
    name: example
    replicas: 1
```

## 更复杂的推理图

可以使用 ROUTERS, COMBINERS 等其他组件构造更复杂的推理图，可在[示例章节](../examples/notebooks.xhtml)找到这些特殊用例。

## 通过 GoLang 源码 学习所有类型

根据 [Kubernetes Seldon Deployment GoLang 类型文件](../reference/seldon-deployment.xhtml) 学习 SeldonDeployment YAML 定义。


## 镜像 UserIds

我们提供一个环境变量 `DEFAULT_USER_ID` (通过 helm chart `.Values.defaultUserID` 设置) ，它允许镜像在此 User Id 下运行。默认 8888。如果想在特定的 Pod/Container 下（非全局），可通过以下方式设置：

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier_rest:1.3
          name: classifier
          securityContext:
            runAsUser: 1000
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    name: example
    replicas: 1
```

上述示例使传统容器以用户 Id 1000 运行。我们建议所有容器都使用非 root。在 Openshift 集群中，通常会强制自动执行此操作。

