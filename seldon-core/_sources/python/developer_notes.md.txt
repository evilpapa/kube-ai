# 开发技巧


## 在本地运行以进行测试

有时，无需使用 s2i 或 docker 构建映像即可在本地测试模型很有用。

可通过安装的 `seldon-core` 命令行开启微服务。

假设我们有一个简单的模型存储在 `MyModel.py` 文件：
```python
class MyModel:

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called - will run identity function")
        return X
```

我们可以启动 Seldon Core 微服务
```bash
seldon-core-microservice MyModel --service-type MODEL
```

然后在其他终端中，我们可以发送 curl 请求来测试 REST 节点：
```bash
curl http://localhost:9000/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{"data": {"names": ["input"], "ndarray": ["data"]}}'
```


假设 `seldon-core` 代码存放在 `${SELDON_CORE_DIR}` 我们可以使用 `grpcurl` 发送 gRPC 请求：
```bash
cd ${SELDON_CORE_DIR}/executor/proto && grpcurl \
    -d '{"data": {"names": ["input"], "ndarray": ["data"]}}' \
    -plaintext -proto ./prediction.proto  0.0.0.0:5000 seldon.protos.Seldon/Predict
```

`grpcurl` 工具可以使用 [GitHub](https://github.com/fullstorydev/grpcurl) 或者使用 [asdf-vm](https://github.com/asdf-vm/asdf-plugins)。

查看 [Python 服务器](./python_server.html#configuration) 文档获取配置选项。
