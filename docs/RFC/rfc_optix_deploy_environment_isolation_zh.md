# RFC: OptiX 部署子进程环境隔离

| 项目 | 内容 |
|:-----|:-----|
| **状态** | Draft |
| **作者** | msmodeling community |
| **创建日期** | 2026-07-21 |
| **更新日期** | 2026-07-21 |
| **相关 PR** | [#490](https://gitcode.com/Ascend/msmodeling/pull/490)、[#577](https://gitcode.com/Ascend/msmodeling/pull/577) |

---

## 1. 概述

### 1.1 简介

本提案为 OptiX 建立统一的部署子进程环境隔离机制。msmodeling 及 TensorCast 依赖安装在独立虚拟环境中，vLLM、MindIE 和 benchmark 工具继续使用系统侧已验证的部署栈；OptiX 在启动服务或测评子进程前，从继承环境中剥离 msmodeling 虚拟环境路径，解析并校验实际部署命令。

该机制集中在 `optix/deploy_env.py`，统一负责运行上下文识别、环境构建、部署路径覆盖、命令物化和启动前校验，避免不同 simulator、benchmark 或插件分别实现路径解析而产生行为分叉。

### 1.2 动机

msmodeling 安装会引入 `torch`、`transformers` 等仿真依赖，而真机部署栈通常已经固定了与 vLLM、MindIE、CANN 等组件匹配的依赖版本。如果把两类依赖安装在同一 Python 环境中，可能出现以下问题：

- msmodeling 安装覆盖系统部署栈依赖，导致服务启动或推理失败；
- OptiX 在 msmodeling venv 中运行时，`PATH` 优先命中 venv 内误装的 `vllm` 或 benchmark 命令；
- 子进程继承 `PYTHONPATH`、`LD_LIBRARY_PATH` 后加载到仿真侧包或动态库；
- 各插件各自调用 `shutil.which()`，校验环境与实际运行环境不一致；
- 配置特殊部署目录时缺少统一优先级和可诊断的失败信息。

因此需要明确两套环境的责任边界，并保证“启动前校验”和“子进程实际运行”使用同一份解析结果。

### 1.3 目标与非目标

目标：

1. 识别 msmodeling 当前所在的 venv、legacy virtualenv 或非 base Conda 环境。
2. 构建不包含 msmodeling 虚拟环境路径的部署子进程环境。
3. 支持通过 `OPTIX_DEPLOY_PATH` 或 `[deploy].path_prefix` 指定部署根目录。
4. 在开始寻优前校验内置 engine 和 benchmark 命令，尽早暴露环境错误。
5. 将校验通过的运行上下文和环境传递给实际子进程，避免重复解析造成不一致。
6. 为自定义插件提供可复用的环境构建和命令物化 API。

非目标：

1. 不负责安装或升级 vLLM、MindIE、AISBench 等部署组件。
2. 不创建或管理部署栈专用虚拟环境。
3. 不修改 PSO、Scheduler、benchmark 协议或服务配置语义。
4. 不保证任意第三方插件的自定义命令都能由内置 registry 自动识别；插件仍需声明或自行校验外部依赖。
5. 不尝试修复系统部署栈本身的依赖冲突。

## 2. 用例分析

### 2.1 标准部署栈

用户在仓库中执行 `uv sync`，并从 `.venv` 运行：

```bash
uv run msmodeling optix -e vllm -b ais_bench
```

OptiX 应从子进程环境的 `PATH`、`PYTHONPATH` 和 `LD_LIBRARY_PATH` 中移除 `.venv` 下的路径，再从系统 `PATH` 解析 `vllm` 和 `ais_bench`。解析结果落在 msmodeling venv 时必须拒绝启动。

### 2.2 自定义部署根目录

当部署命令不在系统 `PATH`，或机器上存在多套部署栈时，用户可以设置：

```bash
export OPTIX_DEPLOY_PATH=/opt/deploy-stack
```

或者：

```toml
[deploy]
path_prefix = "/opt/deploy-stack"
```

OptiX 将 `<path_prefix>/bin` 放在隔离后 `PATH` 的首位。环境变量的优先级高于配置文件，未配置时使用隔离后的系统 `PATH`。

### 2.3 MindIE 固定安装路径

MindIE 优先使用 `${MIES_INSTALL_PATH:-/usr/local/Ascend/mindie/latest/mindie-service}/bin/mindieservice_daemon`。该文件不存在时，从隔离后的 `PATH` 查找 `mindie_llm_server`。`MIES_INSTALL_PATH` 属于部署信息，构建子进程环境时必须保留。

### 2.4 自定义插件

自定义 simulator 或 benchmark 若自行创建 `subprocess`，必须使用 OptiX 提供的部署环境，不得直接把 `os.environ` 传给部署子进程。插件命令依赖 `PATH` 时，应声明 `required_executable`；显式命令路径应在启动前调用 `materialize_command()`。

### 2.5 Conda 与嵌套环境

Conda base 不是 msmodeling 隔离根目录，不能无条件删除其激活上下文。对于同时存在 PEP 405 venv 和外层 Conda 的场景，应优先隔离实际运行 Python 所在的 venv，并保留外层 Conda 信息。

## 3. 方案设计

### 3.1 总体流程

```text
msmodeling optix
    │
    ├─ detect_runtime_context()
    │      └─ 识别 msmodeling 隔离根目录
    │
    ├─ resolve_deploy_path_prefix()
    │      └─ OPTIX_DEPLOY_PATH > [deploy].path_prefix > 系统 PATH
    │
    ├─ build_deploy_env()
    │      └─ 剥离 venv 路径并构建部署环境
    │
    ├─ validate_deploy_stack()
    │      └─ 校验 engine 与 benchmark 可执行文件
    │
    └─ 将同一 RuntimeContext + deploy_env 传给 Simulator/Benchmark
           ├─ materialize_command()
           └─ subprocess(..., env=deploy_env)
```

CLI 入口只解析一次运行上下文和部署环境。校验通过后，将同一份 `RuntimeContext` 和 `deploy_env` 注入 simulator 与 benchmark，避免配置热加载、测试夹具或进程环境变化导致“校验环境”和“运行环境”不一致。

### 3.2 隔离根目录识别

隔离根目录按照以下优先级解析：

1. `VIRTUAL_ENV`；
2. 存在 `sys.real_prefix` 的 legacy virtualenv，使用 `sys.prefix`；
3. `sys.prefix != sys.base_prefix` 的 PEP 405 venv，使用 `sys.prefix`；
4. `CONDA_PREFIX` 存在且 `CONDA_DEFAULT_ENV != "base"` 的 Conda 环境；
5. 以上均不满足时，不设置隔离根目录。

`detect_runtime_context()` 与 `build_deploy_env()` 的默认解析路径必须复用同一套规则。调用方显式传入 `isolation_root` 时，以显式值为准。

### 3.3 子进程环境构建

`build_deploy_env()` 复制父环境，不原地修改调用方传入的 mapping。处理规则如下：

| 变量 | 处理方式 |
|------|----------|
| `VIRTUAL_ENV` | 从子进程环境移除 |
| `PYTHONHOME` | 从子进程环境移除 |
| `CONDA_PREFIX`、`CONDA_DEFAULT_ENV` | 仅当当前隔离根确实是非 base Conda 环境时移除 |
| `OPTIX_DEPLOY_PATH` | 解析完成后从子进程环境移除，避免向下游泄漏 OptiX 控制变量 |
| `PATH` | 删除位于隔离根目录下的路径段 |
| `PYTHONPATH` | 删除位于隔离根目录下的路径段 |
| `LD_LIBRARY_PATH` | 删除位于隔离根目录下的路径段 |
| `MIES_INSTALL_PATH`、Ascend/NPU 变量及其他变量 | 原样保留 |

路径判断使用解析后的真实路径和目录包含关系，不使用字符串前缀匹配，避免 `/opt/.venv2` 被误认为位于 `/opt/.venv` 下。

如果设置了部署根目录，则把 `<deploy_path_prefix>/bin` 插入隔离后 `PATH` 的最前面。配置值表示部署根目录，而不是 `bin` 目录本身。

### 3.4 命令解析与 fail-fast

内置部署命令的处理规则：

| 场景 | 处理方式 |
|------|----------|
| `vllm`、`ais_bench` 裸命令 | 使用隔离环境的 `PATH` 调用 `shutil.which()`，物化为绝对路径 |
| MindIE daemon | 优先使用 `MIES_INSTALL_PATH` 或默认安装路径下的 daemon |
| MindIE fallback | 从隔离环境的 `PATH` 解析 `mindie_llm_server` 并返回绝对路径 |
| 绝对命令路径 | 解析真实路径，若落在 msmodeling venv 中则拒绝 |
| 带 `/` 或 `\` 的相对命令路径 | 相对实际子进程 `cwd` 解析；文件存在时物化为绝对路径并执行 venv 归属校验 |
| 不含路径分隔符的其他裸命令 | 保持原命令，由插件的 `required_executable` 或自定义逻辑负责校验 |

`validate_deploy_stack()` 在创建寻优对象前校验内置 engine 与 benchmark。找不到命令、部署路径无效、命令落在 msmodeling venv 等情况抛出 `OptixDeployEnvError`，错误信息统一使用 `[optix/env]` 前缀并给出修复建议。

### 3.5 配置优先级

部署路径优先级从高到低为：

1. 环境变量 `OPTIX_DEPLOY_PATH`；
2. `config.toml` 中的 `[deploy].path_prefix`；
3. 隔离后的系统 `PATH`。

配置路径会执行 `expanduser()` 和 `resolve()`，且必须指向已存在的目录；否则在启动寻优前失败。

### 3.6 API 与插件集成

核心 API：

```python
def detect_runtime_context() -> RuntimeContext: ...

def build_deploy_env(
    parent: Mapping[str, str],
    *,
    deploy_path_prefix: str | None,
    isolation_root: Path | None = None,
) -> dict[str, str]: ...

def resolve_deploy_context() -> tuple[RuntimeContext, dict[str, str]]: ...

def materialize_command(
    argv: list[str],
    env: Mapping[str, str],
    ctx: RuntimeContext,
    *,
    cwd: str | Path | None = None,
) -> list[str]: ...
```

插件自行启动部署子进程时，推荐复用完整上下文：

```python
import subprocess

from optix.deploy_env import materialize_command, resolve_deploy_context

ctx, deploy_env = resolve_deploy_context()
command = materialize_command(
    ["/opt/custom/bin/server", "--port", "8000"],
    deploy_env,
    ctx,
    cwd="/opt/custom",
)
subprocess.Popen(command, env=deploy_env, cwd="/opt/custom")
```

`CustomProcess` 构造函数中的 `runtime_ctx` 与 `deploy_env` 必须同时提供或同时省略。框架主流程负责注入已经校验的上下文；独立使用 `CustomProcess` 时允许其自行调用 `resolve_deploy_context()`。

插件类的 `required_executable` 校验也应基于同一份 `deploy_env`，不能再次使用进程原始 `PATH`。否则命令只存在于 `OPTIX_DEPLOY_PATH` 或 `[deploy].path_prefix` 时，会出现部署栈预检通过、插件策略校验失败的不一致。

### 3.7 未采用方案

| 方案 | 不采用原因 |
|------|------------|
| 在 msmodeling venv 中同时安装 vLLM/MindIE | 仿真依赖与部署依赖版本容易冲突，无法保证系统部署栈稳定 |
| 为部署栈自动创建第二个 venv | OptiX 不掌握用户已有部署方式，会增加安装、权限和运维复杂度 |
| 启动子进程时使用 `shell=True` 激活系统环境 | 引入 shell 注入与跨平台差异，且难以精确控制实际可执行文件 |
| 每个插件自行过滤环境和解析命令 | 容易形成不一致的优先级、错误信息和安全边界 |
| 只在 CLI 入口校验、不向子进程传递解析结果 | 校验和运行可能使用不同环境，fail-fast 失去约束力 |

### 3.8 安全、隐私与 DFX

| 属性 | 设计 |
|------|------|
| 安全性 | 不使用 `shell=True`；显式路径进行真实路径解析和 venv 归属校验；部署路径必须是有效目录 |
| 可靠性 | 启动寻优前 fail-fast；校验和运行复用同一上下文；缺少命令时给出确定性错误 |
| 兼容性 | 支持 venv、legacy virtualenv、PEP 405、非 base Conda；保留 Conda base 和部署相关环境变量 |
| 可维护性 | 环境识别、过滤、命令解析集中在 `optix/deploy_env.py`，插件不重复实现 |
| 可测试性 | 环境构建函数接收 mapping；路径、上下文和 `shutil.which()` 均可通过临时目录或 mock 覆盖 |
| 可观测性 | 统一 `[optix/env]` 日志前缀，记录最终部署命令绝对路径 |
| 性能 | 每次 OptiX 任务只在启动阶段解析和校验；相对服务启动与 benchmark 耗时可忽略 |

## 4. 测试设计

### 4.1 单元测试

1. 识别 `VIRTUAL_ENV`、`sys.real_prefix`、`sys.prefix != sys.base_prefix` 和非 base Conda。
2. 验证 venv 优先于外层 Conda，Conda base 上下文不被误删。
3. 过滤 `PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH` 中位于隔离根下的路径段。
4. 保留路径名称相似但不属于隔离根的目录。
5. 验证 `OPTIX_DEPLOY_PATH` 高于配置文件，非法目录立即报错。
6. 验证 vLLM、MindIE、AISBench 和 vLLM benchmark 的成功与缺失路径。
7. 拒绝指向 msmodeling venv 的绝对路径和相对路径命令。
8. 验证 `CustomProcess` 必须成对接收 `runtime_ctx` 与 `deploy_env`，且不会重新解析已注入上下文。

### 4.2 集成与冒烟测试

1. 在新的 `.venv` 中执行 `uv sync`，验证 `msmodeling`、`msmodeling optix` 和 inference CLI 入口可用。
2. 在 venv 中调用 `resolve_deploy_context()`，确认 `VIRTUAL_ENV` 已从部署子进程环境移除。
3. 在具备真实部署栈的 Linux/NPU 环境中验证解析出的 vLLM/MindIE/benchmark 命令位于系统部署路径。
4. 使用错误的 `[deploy].path_prefix`、缺失命令和 venv 内命令验证启动前失败，不进入 baseline 或 PSO。
5. 命令仅存在于 `OPTIX_DEPLOY_PATH/bin` 时，部署栈预检、`required_executable` 校验和插件构造均成功。

## 5. 缺点和风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 系统部署栈依赖 venv 中的某个辅助路径 | 过滤后命令可能无法启动 | 将部署依赖安装到系统部署栈，或通过 `[deploy].path_prefix` 指定完整部署根目录 |
| 自定义插件绕过公共 API 直接使用 `os.environ` | 仍可能加载 msmodeling venv 依赖 | 插件指南明确约束；内置插件统一继承 `CustomProcess` 环境 |
| 未知裸命令不由 `materialize_command()` 自动解析 | 第三方命令可能延迟到运行时失败 | 插件声明 `required_executable`，或在插件初始化阶段显式调用解析 API |
| 符号链接跨目录 | 命令表面在系统路径、真实文件位于 venv | 使用 `Path.resolve()` 后再做目录归属判断 |
| Windows 与 POSIX 路径分隔符不同 | 路径过滤不一致 | 使用 `os.pathsep`、`os.sep` 和 `os.altsep`，并保留跨平台单元测试 |

## 6. 未解决问题

当前没有阻塞本提案落地的开放问题。后续若要让第三方插件命令自动加入统一的 deploy registry，应单独讨论插件元数据和兼容策略。

---

## 附录

### 参考资料

- [OptiX 服务参数寻优设计](../design/optix-service-parameter-optimizer-design.md)
- [OptiX 用户指南](../zh/user_guide/optix_user_guide.md)
- [OptiX 插件开发指导](../zh/user_guide/optix_plugin_user_guide.md)
- [msmodeling 环境安装 RFC](rfc_msmodeling_env_installer_skill_zh.md)

### 术语表

| 术语 | 含义 |
|------|------|
| msmodeling venv | 安装 msmodeling、TensorCast 及其 Python 依赖的虚拟环境 |
| 部署栈 | vLLM、MindIE、CANN 和 benchmark 工具等真机运行组件 |
| 隔离根目录 | 需要从部署子进程路径变量中剥离的 msmodeling 虚拟环境根目录 |
| 命令物化 | 将命令首元素解析为实际可执行文件绝对路径并执行归属校验 |
| fail-fast | 在 baseline 或 PSO 开始前发现配置或部署命令错误并终止 |

### 文档更新计划

| 文档 | 变更 |
|------|------|
| OptiX 总体设计文档 | 补充部署环境层、配置、插件契约、测试项和本 RFC 变更记录 |
| OptiX 用户指南 | 维护安装、配置优先级、日志和排障操作说明 |
| OptiX 插件开发指导 | 维护自定义 subprocess 的环境构建与命令物化约定 |
