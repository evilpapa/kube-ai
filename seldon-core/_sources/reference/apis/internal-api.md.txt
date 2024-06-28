# 内部微服务 API

![graph](./graph.png)

要将微服务组件添加到运行时预测图中，用户需要创建遵循内部 API 规范的服务。API 为系统内的每种类型的组件提供默认服务：

 * [Model](#model)
 * [Router](#router)
 * [Combiner](#combiner)
 * [Transformer](#transformer)
 * [Output_Transformer](#output_transformer)

查看完整的 [proto 定义](./prediction.md#proto-buffer-and-grpc-definition)。

## Model

一个返回预估的服务。

### REST API

 | | |
 | - |- |
 | Endpoint | POST /predict |
 | Request | JSON representation of SeldonMessage
 | Response | JSON representation of SeldonMessage

请求 payload 示例：

```json
{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}
```

响应 payload 示例


### gRPC

```protobuf
service Model {
  rpc Predict(SeldonMessage) returns (SeldonMessage) {};
  rpc SendFeedback(Feedback) returns (SeldonMessage) {};
  rpc Metadata(google.protobuf.Empty) returns (SeldonModelMetadata) {};
}
```

## 路由

一种将请求路由到它的一个子服务并为他们接收反馈奖励的服务。

### REST API


 | | |
 | - |- |
 | Endpoint | POST /route |
 | Request | SeldonMessage JSON 表示
 | Response | SeldonMessage JSON 表示

请求负载示例：

```json
{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}
```

响应负载示例：

```json
{"data":{"ndarray":[1]}}
```

### gRPC

```protobuf
service Router {
  rpc Route(SeldonMessage) returns (SeldonMessage) {};
 }
```


## 发送反馈

 | | |
 | - |- |
 | Endpoint | POST /send-feedback |
 | Request | Feedback 的 JSON 表示
 | Response | SeldonMessage 的 JSON 表示

请求有效负载示例：

```json
{
    "request": {
        "data": {
            "names": ["a", "b"],
            "tensor": {
                "shape": [1, 2],
                "values": [0, 1]
            }
        }
    },
    "response": {
        "data": {
            "names": ["a", "b"],
            "tensor": {
                "shape": [1, 1],
                "values": [0.9]
            }
        }
    },
    "reward": 1.0
}
```


### gRPC

```protobuf
service Router {
  rpc SendFeedback(Feedback) returns (SeldonMessage) {};
 }
```

## Combiner

将来自其子级的响应组合成单个响应的服务。

### REST API


 | | |
 | - |- |
 | Endpoint | POST /combine |
 | Request | SeldonMessageList 的 JSON 表示
 | Response | SeldonMessage 的 JSON 表示


### gRPC

```protobuf
service Combiner {
  rpc Aggregate(SeldonMessageList) returns (SeldonMessage) {};
}
```


## Transformer

转换输入的服务。

### REST API


 | | |
 | - |- |
 | Endpoint | POST /transform-input |
 | Request | SeldonMessage 的 JSON 表示
 | Response | SeldonMessage 的 JSON 表示

请求负载示例：

```json
{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}
```

### gRPC

```protobuf
service Transformer {
  rpc TransformInput(SeldonMessage) returns (SeldonMessage) {};
}
```


## Output_Transformer

一种服务，用于转换来自其子级的响应。

### REST API

 | | |
 | - |- |
 | Endpoint | POST /transform-output |
 | Request | JSON representation of SeldonMessage
 | Response | JSON representation of SeldonMessage

请求负载示例：

```json
{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[0,0,1,1]}}}
```

### gRPC

```protobuf
service OutputTransformer {
  rpc TransformOutput(SeldonMessage) returns (SeldonMessage) {};
}
```
