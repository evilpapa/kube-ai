(runtime-env-auth)=
# 在 runtime_env 验证远程 URI

本节可帮助您：

* 避免在 `runtime_env` 泄露远程 URI 凭据
* 在 KubeRay 中安全地提供凭证
* 了解验证远程 URI 的最佳实践

## 验证远程 URI

你可以使用 [remote URIs](remote-uris) 在 `runtime_env` 添加依赖。 对于公开托管的文件，这很简单，因为您只需将公共 URI 粘贴到您的 `runtime_env` 中即可：

```python
runtime_env = {"working_dir": (
        "https://github.com/"
        "username/repo/archive/refs/heads/master.zip"
    )
}
```

但是，私有托管的依赖项（例如私有 GitHub 存储库）需要身份验证。一种常见的身份验证方法是将凭据插入 URI 本身：

```python
runtime_env = {"working_dir": (
        "https://username:personal_access_token@github.com/"
        "username/repo/archive/refs/heads/master.zip"
    )
}
```

在此示例中， `personal_access_token` 是用于验证此 URI 的秘密凭证。虽然 Ray 可以使用经过验证的 URI 成功访问您的依赖项，但 **您不应在 URI 中包含秘密凭证** ，原因有二：

1. Ray 可能会记录您在 `runtime_env` 中使用的 URI，这意味着 Ray 日志可能包含您的凭据。
2. Ray 将您的远程依赖包存储在本地目录中，并使用远程 URI 的解析版本（包括您的凭据）作为目录的名称。

简而言之，您的远程 URI 不被视为机密，因此它不应包含机密信息。相反，请使用文件 `netrc` 文件。

## 在虚拟机上运行：netrc 文件

[netrc 文件](https://www.gnu.org/software/inetutils/manual/html_node/The-_002enetrc-file.html) 包含Ray 用于自动登录远程服务器的凭据。在此文件中（而不是在远程 URI 中）设置您的凭据：

```bash
# "$HOME/.netrc"

machine github.com
login username
password personal_access_token
```

在此示例中， `machine github.com` 行指定了任何访问 `github.com` 都会使用提供的 `login` 和 `password` 进行认证。

:::{note}
 Unix 上，将 `netrc` 文件命名为 `.netrc`。在 Windows 命名文件为 `_netrc`。
:::

`netrc` 文件需要所有者读/写访问权限，因此请确保在创建文件后运行 `chmod` 命令：

```bash
chmod 600 "$HOME/.netrc"
```

添加 `netrc` 文件到虚拟机主目录，这样 Ray 就可以访问 `runtime_env` 的私有远程 URI，及时他们不包含任何凭据。

## 在 KubeRay 上运行：使用 netrc 的 Secrets

[KubeRay](kuberay-index) 也可以从 `netrc` 文件中获取远程 URI 的凭证。 `netrc` file 按照以下步骤使用 Kubernetes secret 和 volume 按照这些步骤应用：

1\. 启动您的 Kubernetes 集群。

2\. 在您的主目录中本地创建该 `netrc` 文件。

3\. 将 `netrc` 文件的内容作为 Kubernetes secret 存储在您的集群上：

```bash
kubectl create secret generic netrc-secret --from-file=.netrc="$HOME/.netrc"
```

4\. 使用已挂载的卷将密钥公开给您的 KubeRay 应用程序，并更新 `NETRC` 环境变量以指向该 `netrc` 文件。在您的 KubeRay 配置中包含以下 YAML。

```yaml
headGroupSpec:
    ...
    containers:
        - name: ...
          image: rayproject/ray:latest
          ...
          volumeMounts:
            - mountPath: "/home/ray/netrcvolume/"
              name: netrc-kuberay
              readOnly: true
          env:
            - name: NETRC
              value: "/home/ray/netrcvolume/.netrc"
    volumes:
        - name: netrc-kuberay
          secret:
            secretName: netrc-secret

workerGroupSpecs:
    ...
    containers:
        - name: ...
          image: rayproject/ray:latest
          ...
          volumeMounts:
            - mountPath: "/home/ray/netrcvolume/"
              name: netrc-kuberay
              readOnly: true
          env:
            - name: NETRC
              value: "/home/ray/netrcvolume/.netrc"
    volumes:
        - name: netrc-kuberay
          secret:
            secretName: netrc-secret
```

5\. 应用您的 KubeRay 配置。

您的 KubeRay 应用程序可以使用 `netrc` 文件访问私有远程 URI，即使它们不包含凭据。
