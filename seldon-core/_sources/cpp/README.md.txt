# 为 Seldon Core 封装 C++ 框架/模型

本章中，我们使用 Seldon CPP 封装你的 CPP 模型。

快速开始请参考：

* [简单的 CPP 单文件示例](../examples/cpp_simple)
* [高级 CPP 构建系统示例](../examples/cpp_advanced)

CPP 封装通过 C++ 打包利用 [Python Inference Server](../python) 进行核心 CPP 原生组件通信。

如果你对 s2i 不熟悉，请参考 [s2i 使用说明](../wrappers/s2i.md) 并跟随下面的步骤。

## 步骤 1 - 安装 s2i

[下载安装 s2i](https://github.com/openshift/source-to-image#installation)

 * 使用 s2i 准备工作
   * Docker
   * Git（如果使用远程 git 仓库）

所有工作就绪，可执行以下

```bash
s2i usage seldonio/s2i-cpp-build:0.0.1
```

## 步骤 2（可选） - 安装 Seldon Core C++ 包

本步骤有两点：

* 安装 Seldon Core C++ 包便于开发
* 不要安装包，而是通过 CMakeLists.txt 文件导入子目录

### （可选 1） 安装 C++ 包

针对此可选项，切到 `incubating/wrappers/s2i/wrappers/cpp/` 使用 CMAKE 进行安装。

与 cmake 一样，您能够为操作系统构建相应的构建组件 - 对于 Linux：

```bash
cmake . -Bbuild

make

make install
```

在 CMakeLists.txt 你可以添加

```
find_package(seldon REQUIRED)
```

### （可选 1）以子目录导入

通过 CMAKE 你可以导入子目录作为工程。像这样：

```
add_subdirectory(
    ../../../incubating/wrappers/s2i/cpp/
    ${CMAKE_CURRENT_BINARY_DIR}/seldon_dir})
```

现在，导入了目录。如果您确实采用此方法，您需要记住，当您运行 s2i 命令时，这将执行图构建，这意味着你仍需要在 if(...) 判断中添加 find_package 命令。

### 步骤 3 - 添加源码

在 C++ 代码和 Seldon 间的核心接口是一个具有 `predict` 接口功能的封装类。

为了简化与 Seldon 的交互，我们提供了一个包含相关实用程序的 C++ 模块，例如要继承的父封装类以及 protobuf 类型。

上面的示例提供了有关可以创建的类的结构的注解。下面是一个包含所有核心元素的示例类。

```cpp
#include "seldon/SeldonModel.hpp"

class ModelClass : public seldon::SeldonModelBase {

    seldon::protos::SeldonMessage predict(seldon::protos::SeldonMessage &data) override {
        return data;
    }
};

SELDON_DEFAULT_BIND_MODULE()
```

我们将分解每个部分，并提供有关可用作替代品的扩展和选项的进一步见解。

#### Seldon C++ 包含模块

主要包含文件 `seldon/SeldonModel.hpp` 提供了一组核心实用程序。

```cpp
#include "seldon/SeldonModel.hpp"
```

导入的核心组件是：

* SeldonModel<proto> 和 SeldonModelBase 类
* SELDON_{X}_BIND_MODULE(...) 宏定义
* seldon::protos::SeldonMessage

#### SeldonModel 基础类

下一个组件是 SeldonModel 类。正如你在上面的例子中看到的，我们继承了这个类，如下所示：

```cpp
class ModelClass : public seldon::SeldonModelBase {

    seldon::protos::SeldonMessage predict(seldon::protos::SeldonMessage &data) override {
        return data;
    }
};
```

SeldonModel 类提供了两个关键组件：

* 它提供了一个 `public virtual abstract` 方法 `predict(proto)` 用户可以覆盖添加自己的逻辑
* 在幕后，SeldonModel 还实现了一个 `public virtual` 方法`predictRaw(py::bytes)`，该方法基本上接收原始字节，将它们转换为相关的 proto 并将其传递给函数 `predict(proto)`。

值得一提的是， SeldonModel<proto> 实际上提供了一个可以启用任何 protos 的模板类。这当然受到服务编排器的限制，但提供了进一步的灵活性。

更具体地说，我们上面使用的 SeldonModelBase 类实际上是一个模板实现 `using SeldonModelBase = SeldonModel<seldon::protos::SeldonMessage>;`。

#### 绑定宏

最后我们有最后一步，即我们的绑定宏。这就是告诉 Seldon 使用我们上面提供的类。默认情况下，Selon 需要 `ModelClass` 类名转换和 `SeldonPackage` 包名本身。

上面我们使用了默认绑定，即以下行：

```cpp
SELDON_DEFAULT_BIND_MODULE()
```

上面的宏相当于定义了下面的宏：

```cpp
SELDON_BIND_MODULE(SeldonPackage, ModelClass)
```

如果更改任何一个名称，则需要确保执行相关覆盖。更具体地说，所需的更改如下。

如果您更改 ModelClass 的名称：

* 使用宏 SELDON_BIND_MODULE 注册它
* 在 MODEL env var 中将其指定为您的 s2i 参数（在下面的可选步骤中有更多相关信息）

如果您更改 SeldonPackage 的名称：
* 使用宏 SELDON_BIND_MODULE 注册它
* 在 MODEL env var 中将其指定为您的 s2i 参数（下一步将详细介绍）
* 覆盖构建系统中的包名称（在下面的可选步骤中有更多内容）

## 可选步骤

上述步骤是可选的，仅当您需要更高级的功能或想要更改命名约定时才需要。

### 步骤 4 (可选) - 指定您的 seldon 环境变量

为了指定您的 seldon 环境变量，您可以通过以下方式之一进行：

* 当运行 `s2i` 组件时在 `.s2i/environment` 文件指定
* 通过 `-E` 指定变量值到 `s2i` 命令

您可以覆盖的主要变量是：

* MODEL=SeldonPackage.ModelClass

正如您可以假设的那样，这是确保为 Seldon 封装器正确设置并能相关 C++ 包/类的环境变量。

### 步骤 5 (可选) - 覆盖构建系统

最后，您还可以通过高级配置覆盖构建系统。

我们使用的构建系统是 CMAKE，主要是因为它支持灵活和模块化的开发。

这意味着您将能够将 CMakeLists.txt 文件添加到您的项目文件夹中。

该文件的内容如下：

```
cmake_minimum_required(VERSION 3.4.1)
project(seldon_custom_model VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)

find_package(seldon REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    SeldonPackage
    ModelClass.cpp)

target_link_libraries(
    CustomSeldonPackage PRIVATE
    seldon::seldon)
```

上面的大纲是您可以创建的最简单的 cmake 文件。它具有以下组件：

* find_package(seldon,...) - 获取 seldon 包
* find_package(pybind11, ...) - 这会获取所需的绑定
* pybind11_add_module(SeldonPackage, ModelClass.cpp) - 这将使用所需包名和所有相关的 C++ 源文件创建您的模块。
* target_link_libraries(CustomSeldonPackage PRIVATE seldon::seldon) 绑定 Seldon 共享/静态库

除此之外，您还可以为您的环境配置任何您想要的东西。

还可以通过扩展 docker 镜像库来进一步设置，可查看仓库模块 `incubating/wrappers/s2i/cpp/`。





