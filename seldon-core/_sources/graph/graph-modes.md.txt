# 图部署选项

在 Seldon core 中不同模式范围的不同容器化模型和 Seldon core 组件的推理图有着不同的能力。
推理图的每个节点都是 Kubernetes 集群中的一个容器。推理图节点可以封装在
单个或者多个 kubernetes pods中。Seldon core 的外部组件是那些包含一个或多个组件并且
定义在推理图 `spec.componentSpecs.graph` 中构建的多个预估器。

## 模式一: 单 pod 部署

以下示例是 Seldon core 推理图仅有 
一个预估。
```bash
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: linear-pipeline-single-pod
spec:
  name: linear-pipeline
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.0
          name: node-one
        - image: seldonio/mock_classifier:1.0
          name: node-two
        - image: seldonio/mock_classifier:1.0
          name: node-three
    graph:
      name: node-one
      type: MODEL
      children:
      - name: node-two
        type: MODEL
        children:
        - name: node-three
          type: MODEL
          children: []
    name: example
```

所有图部署的节点都是在一个单独的pod：

```bash
kubectl get pods

NAME                                                       READY   STATUS    RESTARTS   AGE
seldon-c71cc2d950d44db1bc6afbeb0194c1da-5d8dddb8cb-xx4gv   5/5     Running   0          6m59s
```

## 模式二： 单独的pod部署

另一种部署方式是在单独的预测器中实现推理图的每个节点，这将导致每个推理图节点都
有单独的 pod。

```bash
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: linear-pipeline-separate-pods
spec:
  name: linear-pipeline
  annotations:
    seldon.io/engine-separate-pod: "true"
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.0 
          name: node-one
          imagePullPolicy: Always
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.0
          name: node-two
          imagePullPolicy: Always
    - spec:
        containers:
          - image: seldonio/mock_classifier:1.0
            name: node-three
            imagePullPolicy: Always
    graph:
      name: node-one
      type: MODEL
      children:
      - name: node-two
        type: MODEL
        children:
        - name: node-three
          type: MODEL
          children: []
    name: example
```

这一次它将导致每个容器都有单独的 pod。

```bash
kubectl get pods
NAME                                                              READY   STATUS    RESTARTS   AGE
linear-pipeline-separate-pods-example-0-node-one-6954fbbd5m7pcp   1/1     Running   0          4m33s
linear-pipeline-separate-pods-example-1-node-two-c4f55f689gxkkr   1/1     Running   0          4m33s
linear-pipeline-separate-pods-example-2-node-three-99667dcmg9kg   1/1     Running   0          4m33s
linear-pipeline-separate-pods-example-svc-orch-656c6bdf59-6m6nc   1/1     Running   0          4m33s
```
Kubernetes 中最基本的单元是 Pod。该模型将支持模型界别的[缩放](scaling.md)。换言之，你可以
您可以单独缩放每个模型，而另一方面将它们放在一个 pod 中会改变缩放到整个图的粒度。然而，
另一方面，单个 pod 部署将只需要一个[边车 istio 容器](../ingress/istio.md)
它需要更少的来自 sidecar 容器的资源请求。另一个潜在的区别是单 pod 模式下的通信开销更少，因为它们总是
被调度在同一个 Kubernetes 节点上。
