# 熔断

高可用性是运行生产系统的一个重要方面。
为此，您可以将 Pod 熔断规范添加到您创建的 Pod 模板规范中。
根据相应地定义来让应用程序处理熔断。

Seldon Deployment 熔断配置实例：

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  replicas: 2
  predictors:
  - componentSpecs:
    - pdbSpec:
        minAvailable: 90%
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

这个例子确保我们的服务能力不会下降超过10%。
