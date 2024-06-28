# 开发者

我们欢迎新的贡献者。
请阅读[行为守则](https://github.com/SeldonIO/seldon-core/blob/master/CODE_OF_CONDUCT.md)和[贡献指引](contributing.rst)。

## Operator 开发

管理 SeldonDeployment CRD 的 Operator 包含在 `/operator` 目录。它由 [kubebuilder](https://book.kubebuilder.io/) 创建。

在本地开发中，我们使用 [kind](https://kind.sigs.k8s.io/) 创建集群。

```console
kind create cluster
```

安装 cert-manager

```console
make install-cert-manager
```

构建并加载当前控制器镜像到 Kind 集群：

```console
make kind-image-install
```

运行以下命令安装 Operator：

```console
make deploy
```

如果在集群外准备控制器和安装 Operator，比如在 IDE 中（我们用的是 GoLand）运行。尽在本地 Kind 集群中测试通过：

```console
make deploy-local
```

当以上全部运行，在 seldon-system 空间中删除运行中的 seldon-controller-manager 即可完成本地化。

下一步，下载本地 webhook 证书（通过 cert-manager 创建）：

```console
make tls-extract
```

在本地运行 manager，需要在启动时设置 webhook-port，例如：

```console
go run ./main.go --webhook-port=9000
```

如果在 IDE 中运行 Kind，请确保设置好 KUBECONFIG 环境变量。

## 用到的工具

- [github-changelog-generator](https://github.com/skywinder/github-changelog-generator)
- [Grip - 本地 Markdown 预览](https://github.com/joeyespo/grip)

## 构建 Seldon Core

- [使用私有仓库构建](build-using-private-repo.md)

