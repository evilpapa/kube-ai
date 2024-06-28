# 使用本地私有仓库构建

## 要求

* 本地 Docker
* 接入 Kubernetes 集群
* 设置了集群可访问的本地私有仓库
    * 使用 [k8s-local-docker-registry](https://github.com/SeldonIO/k8s-local-docker-registry) 项目
    * "127.0.0.1:5000" 将作为镜像地址 url 使用

## 要求检查

确保要求已经就位，端口设置正确。

```bash
# 检测本地私有仓库正确工作
(set -x && curl -X GET http://127.0.0.1:5000/v2/_catalog && \
        docker pull busybox && docker tag busybox 127.0.0.1:5000/busybox && \
        docker push 127.0.0.1:5000/busybox)
```

## 更新组件并重新发布到集群

测试集群中代码如何更改的基本过程。

1. 停止运行中的 seldon core。
1. 如有必要，构建并推送已更新的组件或所有组件。
1. 启动 seldon core。
1. 发布模型。

以下是实现此目标的详细信息。

### 构建所有组件

构建所有图像并推送至本地私有存储库。

```bash
./build-all-private-repo
./push-all-private-repo
```

### 开始/停止 Seldon Core

```bash
./start-seldon-core-private-repo
./stop-seldon-core-private-repo
```

### 构建单个组件

```bash
./cluster-manager/build-private-repo
./cluster-manager/push-private-repo

./api-frontend/build-private-repo
./api-frontend/push-private-repo

./engine/build-private-repo
./engine/push-private-repo
```

