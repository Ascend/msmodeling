# msModeling 安装指南

<br>

## 1. 安装说明

请确认运行环境满足以下条件：

- Python 版本 ≥ 3.10，建议使用独立虚拟环境。
- 网络可正常访问 GitCode 及 Python 包源。
- 如需拉取 Hugging Face 模型配置，需确保网络可访问 Hugging Face。

本工具支持以下三种安装方式：[在线安装](#21-在线安装)、[离线安装](#22-离线安装)、[源码安装](#23-源码安装)。

## 2. 安装方式

### 2.1 在线安装

**Linux / macOS 环境**

若您的设备具备互联网访问能力，可通过一条命令自动完成工具的下载与安装。请参见昇腾社区 MindStudio [下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择产品版本，选择“推理开发”使用场景，选择 msModeling 工具，并在安装方式中选择“在线安装”，系统将引导您完成后续操作。

**Windows 环境**

请依次执行以下命令完成安装（whl 包链接可从 [发行版页面](https://gitcode.com/Ascend/msmodeling/releases) 中获取）：

```bash
# 1. 创建并激活虚拟环境（必须使用虚拟环境，以防止破坏执行环境）
python -m venv .venv
.venv\Scripts\activate
# 2. 安装 msmodeling（以下为示例链接）
pip install https://gitcode.com/Ascend/msmodeling/releases/download/tag_MindStudio_26.1.0.B100_002/msmodeling-26.1.0-py3-none-any.whl
```

### 2.2 离线安装

对处于企业内网等无外网环境的设备，请先在可联网的机器上下载完整的离线安装包，再将其传输至目标设备进行安装。

**Linux / macOS 环境**

请参见昇腾社区 MindStudio [下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择产品版本，选择“推理开发”使用场景，选择 msModeling 工具，并在安装方式中选择“离线安装”，获取对应的安装包及操作指引。

**Windows 环境**

请依次执行以下命令完成安装（whl 包链接可从 [发行版页面](https://gitcode.com/Ascend/msmodeling/releases) 中获取）：

```bash
# 1. 创建并激活虚拟环境（必须使用虚拟环境，以防止破坏执行环境）
python -m venv .venv
.venv\Scripts\activate
# 2. 安装 msmodeling（以下为示例文件名）
pip install msmodeling-26.1.0-py3-none-any.whl
```

### 2.3 源码安装

如需使用最新代码的功能，或对源码进行修改以增强功能，可下载本仓库代码，自行编译、打包工具并完成安装。

#### 2.3.1 环境准备

源码编译统一使用 MindStudio 标准构建环境。

- 日常开发或使用已发布镜像，请参考《[MindStudio 工具开发环境安装指导](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/dev_env_setup.md)》。
- 需要从基础操作系统复现环境、执行源码构建验证或单元测试验证时，必须参考《[MindStudio 统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》，从 openEuler 基础镜像现场构建环境镜像。

本文档后续的源码编译和单元测试命令，均在上述指定镜像容器或现场构建的环境镜像的容器中执行，CANN 软件包版本、GCC 版本和 Python 版本以统一镜像制作指南为准，本仓库不重复维护。

镜像构建完成后，必须使用统一镜像制作指南第 7 章给出的 `ctr_in.py` 命令，在交互式终端中启动并进入容器。不得使用普通 `docker run` 创建容器，也不得使用 `docker exec <容器名> bash -c '<命令>'` 替代交互式环境；否则可能跳过 Python、GCC 和 CANN 环境初始化。

进入 `ctr_in.py` 打开的交互式容器 Shell 后，执行如下命令克隆本仓库：

```bash
cd ~
git clone https://gitcode.com/Ascend/msmodeling.git
```

#### 2.3.2 执行编译

保持在 `ctr_in.py` 打开的同一个交互式容器 Shell 中，在仓库根目录执行以下命令，自动完成依赖下载与构建：

```bash
cd ~/msmodeling
python3 build.py
```

构建成功后，安装包将生成在 `artifacts/` 目录下。

#### 2.3.3 执行单元测试（可选）

此步骤非安装必需。如需验证代码基本功能，可执行单元测试：

```bash
cd ~/msmodeling
python3 build.py test
```

命令返回码为 0，且测试用例均无失败，表示单元测试通过。

#### 2.3.4 安装

将 whl 包拷贝到运行环境中（本机安装无需拷贝），执行如下安装操作：

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install msmodeling-xxxxx.whl
```

## 3. 验证安装

安装完成后，在已激活的 Python 环境中执行以下命令：

```bash
msmodeling --help
```

若命令执行无报错，且能显示帮助信息，则表明安装成功。

## 4. 卸载

### 4.1 通过昇腾社区方式安装

1. 下载脚本。

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.1.0/ms_install.py
   ```

   > [!NOTE]
   >
   > - 需要联网环境才能下载，若环境不允许联网或处于离线状态，请先在可联网的环境下载该脚本后拷贝到目标设备。
   > - 若执行命令无响应或出现连接失败、SSL证书错误等问题，请参见[FAQ](https://www.hiascend.com/developer/blog/details/02176213671719317003)。

2. 执行卸载。

   ```bash
   python ms_install.py uninstall msmodeling
   ```
   
   卸载成功打印如下信息：

   ```text
   Successfully uninstalled 1 tool (msmodeling)
   ```

### 4.2 通过 pip install 命令安装

激活安装 msmodeling 的 Python 虚拟环境（venv），执行以下命令：

```bash
pip uninstall msmodeling
```

> [!NOTE]
> 卸载前请确认当前终端环境是安装 msModeling 的 Python 环境，避免卸载到其他环境中的同名包。

## 5. 升级

升级即“先卸后装”。直接执行安装命令，工具将自动卸载旧版本，并引导您完成覆盖安装。 升级版本时需要关注版本配套关系，请参见《[版本说明](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md)》。
