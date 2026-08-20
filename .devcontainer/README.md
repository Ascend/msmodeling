# msmodeling Dev Container 快速指南

> **真正开箱即用**：零手动配置！容器创建后自动完成 Python 3.11、Git 身份、开发依赖与 pre-commit 初始化，进入即可开发、构建与单测。

## 🛠️ 前置准备

无需手动配置复杂工具链，请根据场景选择以下任一基础环境：

| 方案 | VS Code 安装 | Docker 服务 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **远程服务器（推荐）** | [VS Code](https://code.visualstudio.com/) + `Dev Containers` + `Remote - SSH` 插件 | Linux 服务器已启用 Docker 服务 | 高性能计算、释放本地资源 |
| **本地 PC** | [VS Code](https://code.visualstudio.com/) + `Dev Containers` 插件 | [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Linux 模式） | 单机离线开发 |

> 前后置条件：需能访问 `swr.cn-north-4.myhuaweicloud.com`（预构建镜像）、Python 包源与 GitCode。
>
> ⚠️ **安全提示**：容器仅挂载代码目录与缓存目录，不挂载宿主机完整 Home、SSH 目录或凭据文件；Git 身份只同步 `user.name` 与 `user.email`。请在可信环境中使用。

### 可选：配置 SSH 免密登录（仅远程服务器方案）

> ⚠️ **免责声明**：以下免密登录配置用于 VS Code 连接远程服务器（宿主机侧操作），与 Dev Container 的凭据隔离策略无关（容器内不挂载 SSH 目录或私钥）。生成 ed25519 密钥对属可选操作，仅为免去重复输入密码的便利，请确认在自有可信账户下执行并妥善保管私钥。

为避免频繁输入密码，可在 Windows PowerShell 中粘贴执行以下脚本，按提示操作即可自动完成配置：

```powershell
$ip = Read-Host "请输入远程服务器的IP地址"
$user = Read-Host "请输入远程服务器的用户名"
$sshDir = "$env:USERPROFILE\.ssh"
$pubKeyPath = "$sshDir\id_ed25519.pub"

if (-not (Test-Path $pubKeyPath)) {
    Write-Host "未检测到本地公钥，正在生成 ed25519 密钥对..." -ForegroundColor Yellow
    ssh-keygen -t ed25519 -C "msmodeling_devcontainer" -f "$sshDir\id_ed25519" -N '""'
    Write-Host "密钥对生成完毕。" -ForegroundColor Green
}

Get-Content $pubKeyPath | ssh "${user}@${ip}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
Write-Host "公钥上传完成，免密登录配置成功！" -ForegroundColor Green
```

## 🚀 3 步闪电开工

1. **打开项目**：在 VS Code 中打开本项目代码目录。
2. **加载容器**：点击右下角弹出的 **`Reopen in Container`** 提示（或通过 `F1` 执行同名命令）。
3. **进入开发**：待容器环境初始化完成后，即可直接进行编码、构建、单元测试与调试。

## ⏱️ 自动化流程与耗时说明

启动 Dev Container 后，系统将**全自动完成以下环境配置**，期间无需任何人工干预：

| 阶段 | 自动化任务 | 说明 |
| :--- | :--- | :--- |
| **1. 环境拉取** | 拉取预构建的 MindStudio 镜像并部署 VS Code Server | 全程无感 |
| **2. 身份与挂载** | 挂载代码目录（`/workspace`），同步宿主 Git 身份（只取 `user.name` / `user.email`） | 不挂载完整 Home / SSH / 凭据 |
| **3. 工具链加载** | 自动安装 uv、激活 Python 3.11、准备开发依赖、安装 pre-commit Hook | 开箱即用 |

> **镜像说明**：本方案内置预构建镜像 `swr.cn-north-4.myhuaweicloud.com/mindstudio-image/mindstudio-build`。若需了解镜像细节，可参考《[MindStudio 统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》。

## 🔨 构建与单元测试

环境就绪后，通过 VS Code 菜单栏 **`Terminal`** > **`Run Task`** 即可调用预设的自动化任务：

| 任务名称 | 功能说明 |
| :--- | :--- |
| `Build: Download Dependencies` | 准备 IDE 开发依赖，不构建 wheel |
| `Build: Release Mode` | 构建 wheel，产物输出至 `artifacts` 目录 |
| `Test: Run Unit Tests` | 执行全量单元测试 |
| `Clean: Python Workspace` | 清理工作区内的构建缓存与临时文件 |

> *也可直接在终端执行仓库统一入口 `python3 build.py`，功能与上述 Build/Test 任务一致：`python3 build.py -e only_down_deps=true` 准备依赖、`python3 build.py` 构建 wheel、`python3 build.py test` 执行全量单测。*

## ♻️ 环境复原：毁坏无忧

若开发过程中容器环境搞乱或损坏，无需重新搭建：只需按 `F1` 键选择 **Dev Containers: Rebuild Container**，即可瞬间获得一个全新的纯净环境！

## ❓ FAQ

### 1. VS Code 远程连接卡在“Waiting for port forwarding...”？

**原因分析**：VS Code 远程开发依赖 SSH 端口转发。若服务端 `sshd_config` 限制过严，或远程 VS Code Server 组件异常，均会导致连接挂起。

**解决方案**：

1. **检查服务端 SSH 配置**（需 root 权限，编辑 `/etc/ssh/sshd_config`），确保以下参数已启用：

   ```bash
   AllowTcpForwarding yes
   GatewayPorts yes
   X11Forwarding yes
   ```

   **关键检查**：确认不存在 `PermitOpen none` 配置，若有请注释掉（`#PermitOpen none`），否则将禁用所有端口转发。

2. **清理远程 VS Code Server**：在远程服务器执行 `rm -rf ~/.vscode-server`，重新发起连接，VS Code 将自动重新部署匹配的 Server 组件。

### 2. Python 依赖异常或缺失？

在容器内重新执行仓库统一入口准备依赖：

```bash
python3 build.py -e only_down_deps=true
```

包源访问慢或不可达时，先配置国内镜像后重试：

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
python3 build.py -e only_down_deps=true
```

### 3. Git 身份缺失？

先在宿主机配置 `git config --global user.name` 与 `user.email`，再重建容器；也可在容器内直接执行 `git config --global` 配置。

### 4. post-create 初始化步骤失败？

单个初始化步骤失败不会阻止进入容器。可重复执行初始化脚本并检查日志：

```bash
bash .devcontainer/post-create.sh
```

先修复对应失败项（日志中以 `[post-create] WARN:` 标出）再进行对应开发操作。
