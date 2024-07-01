(kuberay-tls)=

# TLS 身份验证

ay 可以配置为在其 gRPC 通道上使用 TLS。这意味着连接到 Ray head 将需要一组适当的凭据，并且各个进程（客户端、head、workers）之间交换的数据将被加密([Ray 文档](https://docs.ray.io/en/latest/ray-core/configure.html?highlight=tls#tls-authentication)）。

本文档提供了生成用于配置 KubeRay 的公私密钥对和 CA 证书的详细说明。

> Warning: 由于相互身份验证和加密的额外开销，
启用 TLS 将导致性能下降。测试表明，对于小型工作负载，此开销较大，而对于大型工作负载，此开销相对较小。
确切的开销取决于您的工作负载的性质。

# 先决条件

为了充分理解本文档，强烈建议您充分理解以下概念：

* 私钥/公钥
* CA （证书颁发机构）
* CSR （证书签名请求）
* 自签名证书

这个 [YouTube 视频](https://youtu.be/T4Df5_cojAs) 是一个好的开始。

# TL;DR

> 请注意，本文档旨在支持 KubeRay 0.5.0 或更高版本。如果您使用的是较旧版本的 KubeRay，某些说明或配置可能不适用或可能需要额外修改。

> 警告：请注意， `ray-cluster.tls.yaml` 文件仅用于演示目的。 **请勿** 将 CA 私钥
存储在生产环境中的 Kubernetes Secret 中，这一点至关重要。

```sh
# Install v1.0.0-rc.0 KubeRay operator
# `ray-cluster.tls.yaml` will cover from 步骤 1 to 步骤 3

# Download `ray-cluster.tls.yaml`
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.tls.yaml

# Create a RayCluster
kubectl apply -f ray-cluster.tls.yaml

# Jump to 步骤 4 "Verify TLS authentication" to verify the connection.
```

`ray-cluster.tls.yaml` 将创建：

* 包含 CA 私钥 ( `ca.key`) 和自签名证书 ( `ca.crt`) 的 Kubernetes Secret (**步骤 1**)
* 包含 `gencert_head.sh` 和 `gencert_worker.sh` 脚本的 Kubernetes ConfigMap，允许 Ray Pods 生成私钥 ( `tls.key`) 和自签名证书 ( `tls.crt`) (**步骤 2**)
* 具有正确 TLS 环境变量配置的 RayCluster (**步骤 3**)

Ray Pod 的证书 ( `tls.crt`) 使用 CA 的私钥 ( `ca.key`) 进行加密。此外，所有 Ray Pod 都包含在 中的 CA 公钥 `ca.crt`，这使得它们能够解密其他 Ray Pod 的证书。

# 步骤 1: 为CA生成私钥和自签名证书

在本文档中，使用自签名证书，但用户还可以选择公开信任的证书颁发机构 (CA) 进行 TLS 身份验证。

```sh
# 步骤 1-1: Generate a self-signed certificate and a new private key file for CA.
openssl req -x509 \
            -sha256 -days 3650 \
            -nodes \
            -newkey rsa:2048 \
            -subj "/CN=*.kuberay.com/C=US/L=San Francisco" \
            -keyout ca.key -out ca.crt

# 步骤 1-2: Check the CA's public key from the self-signed certificate.
openssl x509 -in ca.crt -noout -text

# 步骤 1-3
# Method 1: Use `cat $FILENAME | base64` to encode `ca.key` and `ca.crt`.
#           Then, paste the encoding strings to the Kubernetes Secret in `ray-cluster.tls.yaml`.

# Method 2: Use kubectl to encode the certifcate as Kubernetes Secret automatically.
#           (Note: You should comment out the Kubernetes Secret in `ray-cluster.tls.yaml`.)
kubectl create secret generic ca-tls --from-file=ca.key --from-file=ca.crt
```

* `ca.key`: CA的私钥
* `ca.crt`: CA的自签名证书

此步骤是可选的，因为 `ca.key` 和 `ca.crt` 文件已包含在 [ray-cluster.tls.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.tls.yaml) 中指定的 Kubernetes Secret 中。

# 步骤 2: 为 Ray Pods 创建单独的私钥和自签名证书

在 [ray-cluster.tls.yaml](https://github.com/ray-project/kuberay/blob/v1.0.0-rc.0/ray-operator/config/samples/ray-cluster.tls.yaml) 中，
每个 Ray Pod（头和工作线程）在其 init 容器中生成自己的私钥文件 ( `tls.key`) 和自签名证书文件 ( `tls.crt`)。
我们为每个 Pod 生成单独的文件，因为 worker Pod 没有确定性的 DNS 名称，并且我们无法在不同的 Pod 之间使用相同的证书。

在 YAML 文件中，您将找到一个名为 `tls` 的 ConfigMap 包含两个 shell 脚本：
`gencert_head.sh` 和 `gencert_worker.sh`。 这些脚本用于为 
Ray 头和 worker Pod生成私钥和自签名证书文件 (`tls.key` 和 `tls.crt`)。
用户的另一种方法是将 shell 脚本直接打包到 init 容器使用的 docker 映像中，而不是依赖 ConfigMap。

请在下面找到每个脚本中发生的情况的简要说明：
1. 生成 2048 位 RSA 私钥并保存为 `/etc/ray/tls/tls.key`.
2. 使用私钥文件 ( `tls.key`) 和 `csr.conf` 配置文件生成证书签名请求 (CSR)。
3. 使用证书颁发机构 ( `ca.key` ) 的私钥和之前生成的 CSR 生成自签名证书 ( `tls.crt`)。

`gencert_head.sh` 和 `gencert_worker.sh` 的唯一不同是在 `csr.conf` 和 `cert.conf` 的 `[ alt_names ]` 部分。
Worker Pod 使用头 Kubernetes Service 的完全限定域名 (FQDN) 与 Head Pod 建立连接。
因此， Head Pod 的 `[alt_names]` 部分需要包含头 Kubernetes Service 的 FQDN。顺便说一句， Head Pod 用 `$POD_IP` 来与 worker Pod 进行通信。

```sh
# gencert_head.sh
[alt_names]
DNS.1 = localhost
DNS.2 = $FQ_RAY_IP
IP.1 = 127.0.0.1
IP.2 = $POD_IP

# gencert_worker.sh
[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = $POD_IP
```

在 [Kubernetes 网络模型](https://github.com/kubernetes/design-proposals-archive/blob/main/network/networking.md#pod-to-pod) 中，Pod 认为自己的 IP 与其他 Pod 认为的 IP 相同。这就是 Ray Pods 可以自行注册证书的原因。

# 步骤 3: 配置 Ray TLS 身份验证的环境变量

要在 Ray 集群中启用 TLS 身份验证，请设置以下环境变量：

- `RAY_USE_TLS`: 1 或 0 表示使用/不使用 TLS。如果将此设置为 1，则必须设置以下所有环境变量。默认值：0。
- `RAY_TLS_SERVER_CERT`: 向其他端点提供的证书文件的位置，以实现相互身份验证（即  `tls.crt` ）。
- `RAY_TLS_SERVER_KEY`: 私钥文件的位置，它是向其他端点证明您是给定证书的授权用户的加密手段（即 `tls.key`）。
- `RAY_TLS_CA_CERT`: CA 证书文件的位置，允许 TLS 确定端点的证书是否已由正确的机构签名（即 `ca.crt`）。

有关如何使用 TLS 身份验证配置 Ray 的更多信息，请参阅 [Ray 文档](https://docs.ray.io/en/latest/ray-core/configure.html#tls-authentication)。

# 步骤 4: 验证 TLS 身份验证

```sh
# Log in to the worker Pod
kubectl exec -it ${WORKER_POD} -- bash

# Since the head Pod has the certificate of $FQ_RAY_IP, the connection to the worker Pods
# will be established successfully, and the exit code of the ray health-check command
# should be 0.
ray health-check --address $FQ_RAY_IP:6379
echo $? # 0

# Since the head Pod has the certificate of $RAY_IP, the connection will fail and an error
# message similar to the following will be displayed: "Peer name raycluster-tls-head-svc is
# not in peer certificate".
ray health-check --address $RAY_IP:6379

# If you add `DNS.3 = $RAY_IP` to the [alt_names] section in `gencert_head.sh`,
# the head Pod will generate the certificate of $RAY_IP.
#
# For KubeRay versions prior to 0.5.0, this step is necessary because Ray workers in earlier
# versions use $RAY_IP to connect with Ray head.
```
