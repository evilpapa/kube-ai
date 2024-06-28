# Tempo 服务

[Tempo](https://github.com/SeldonIO/tempo) 是个 MLOps python SDK，它允许你打包自定义的 python 服务，并协调多个 python 模型。Tempo python SDK 允许将自定义代码打包成 conda-pack 环境 tar 包和 Cloudpickle artifacts。它有个支持 Tempo artifacts 运行在 Seldon Core 运行时下的核心。

更多信息请查看 [Tempo 文档](https://tempo.readthedocs.io/en/latest/)。

一个使用 Tempo 模型的配置文件示例：

```python
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  annotations:
    seldon.io/tempo-description: ''
    seldon.io/tempo-model: '{"model_details": {"name": "numpyro-divorce", "local_folder":
      "/home/clive/work/mlops/fork-tempo/docs/examples/custom-model/artifacts", "uri":
      "s3://tempo/divorce", "platform": "custom", "inputs": {"args": [{"ty": "numpy.ndarray",
      "name": "marriage"}, {"ty": "numpy.ndarray", "name": "age"}]}, "outputs": {"args":
      [{"ty": "numpy.ndarray", "name": null}]}, "description": ""}, "protocol": "tempo.kfserving.protocol.KFServingV2Protocol",
      "runtime_options": {"runtime": "tempo.seldon.SeldonKubernetesRuntime", "docker_options":
      {"defaultRuntime": "tempo.seldon.SeldonDockerRuntime"}, "k8s_options": {"replicas":
      1, "minReplicas": null, "maxReplicas": null, "authSecretName": "minio-secret",
      "serviceAccountName": null, "defaultRuntime": "tempo.seldon.SeldonKubernetesRuntime",
      "namespace": "production"}, "ingress_options": {"ingress": "tempo.ingress.istio.IstioIngress",
      "ssl": false, "verify_ssl": true}}}'
  labels:
    seldon.io/tempo: 'true'
  name: numpyro-divorce
  namespace: production
spec:
  predictors:
  - graph:
      envSecretRefName: minio-secret
      implementation: TEMPO_SERVER
      modelUri: s3://tempo/divorce
      name: numpyro-divorce
      serviceAccountName: tempo-pipeline
      type: MODEL
    name: default
    replicas: 1
  protocol: kfserving
```
