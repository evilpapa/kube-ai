# HuggingFace Server

感谢我们的合作伙伴 HuggingFace 团队，你可以轻易的使用 [HuggingFace Hub](https://huggingface.co/models) 部署模型到 Seldon Core。

我们还支持 [Transformer Optimum 框架](https://huggingface.co/docs/optimum/index) 提供的高性能优化。

## Pipeline 参数

可供您配置的参数包括：

| 名称 | 描述 |
| ---- | ----------- |
| `task` | 传输管道任务 |
| `pretrained_model` | Hub 中预训练模型的名称 |
| `pretrained_tokenizer` | Hub 中的传输名称（如果与模型提供的名称不同） |
| `optimum_model` | 使用 Optimum 框架启用加载模型的布尔值 |

## 简单示例

你可以发布一个 HuggingFace 模型通过提供参数到 [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)。

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: gpt2-model
spec:
  protocol: v2
  predictors:
  - graph:
      name: transformer
      implementation: HUGGINGFACE_SERVER
      parameters:
      - name: task
        type: STRING
        value: text-generation
      - name: pretrained_model
        type: STRING
        value: distilgpt2
    name: default
    replicas: 1
```

## 使用 Optimum 量化及优化的模型

您可以使用 `optimum_model` 参数部署使用 Optimum 库加载的 HuggingFace 模型

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: gpt2-model
spec:
  protocol: v2
  predictors:
  - graph:
      name: transformer
      implementation: HUGGINGFACE_SERVER
      parameters:
      - name: task
        type: STRING
        value: text-generation
      - name: pretrained_model
        type: STRING
        value: distilgpt2
      - name: optimum_model
        type: BOOL
        value: true
    name: default
    replicas: 1
```

