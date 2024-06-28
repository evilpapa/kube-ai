# 日志和日志等级

开箱即用，在日志记录方面，您的 Seldon 部署将预先配置为一组合理的默认值。
这些设置涉及日志记录级别和日志消息的结构。

这些设置可以基于每个组件进行更改。

## 日志等级

默认情况下，您的 Seldon 部署中的所有组件都将 `INFO` 作为默认日志级别。

您可以使用 `SELDON_LOG_LEVEL` 环境变量更改日志级别。
一般来说，这个变量可以设置为以下日志级别（从多到少）：

- `DEBUG`
- `INFO`
- `WARNING` 
- `ERROR`

### Python 预估服务器

.. Note:: 
   在 Python 封装上设置 ``SELDON_LOG_LEVEL`` 为 ``WARNING`` 
   和以上会禁用服务器的请求日志，
   推荐 ``INFO`` 等级。

使用 [Python wrapper](../python/index) （包括 [MLflow](../servers/mlflow)，[SKLearn](../servers/sklearn) 和 [XGBoost](../servers/xgboost) 预封装服务器）时，
您可以使用 `SELDON_LOG_LEVEL` 环境变量控制日志级别。
请注意，
`SELDON_LOG_LEVEL` 必须在推理图中的 **相应容器中** 设置变量。

例如，要在使用 python 封装器运行的每个容器中设置它，您可以通过添加 `SELDON_LOG_LEVEL` 环境变量到由 python 封装器运行镜像容器容器：

```javascript
"spec": {
  // ...
  "predictors": [
    {
      "componentSpecs": [
        {
          "spec": {
            "containers": [
                { 
                    "name": "mymodel",
                    "image": "x.y:123",
                    "env": [
                        {
                            "name": "SELDON_LOG_LEVEL",
                            "value": "DEBUG"
                        }
                    ]
                }
            ]
          }
        }
      ]
    }
  ]
  // ...
}
```

一旦设置完成，就可以在封装器代码中使用日志，如下所示：

```python
import logging

log = logging.getLogger()
log.debug(...)
```

### 服务编排器中的日志级别

要在服务编排器中更改日志级别，您可以在 `SeldonDeployment` CRD 部分的 `svcOrchSpec` 设置 `SELDON_LOG_LEVEL` 环境变量：

```javascript
"spec": {
  // ...
  "predictors": [
    {
      "svcOrchSpec": {
          "env": [
              {
                  "name": "SELDON_LOG_LEVEL",
                  "value": "DEBUG"
              }
          ]
      }
    }
  ]
  // ...
}
```

## 日志格式和采样

默认情况下，Seldon 的服务编排器和操作器会将日志消息序列化为 JSON 并启用日志采样。
可以通过将 `SELDON_DEBUG` 变量设置为 `true` 来禁用此行为。
注意这将 **开启 "debug mode"**，这也会产生其他副作用。

例如，要在服务编排器上更改此设置，您可以执行以下操作：

```javascript
"spec": {
  // ...
  "predictors": [
    {
      "svcOrchSpec": {
          "env": [
              {
                  "name": "SELDON_DEBUG",
                  "value": "true"
              }
          ]
      }
    }
  ]
  // ...
}
```
