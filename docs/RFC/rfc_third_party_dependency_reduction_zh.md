# RFC: msmodeling 三方库非必要依赖消减

## 1. Overview（概述）

Status（状态）: Draft

Author(s)（作者）: @h7star

Created（创建日期）: 2026-07-31

Updated（更新日期）: 2026-08-01

Related Issue/PR（相关 Issue/PR）:
[Issue #237](https://gitcode.com/Ascend/msmodeling/issues/237) /
[PR #635](https://gitcode.com/Ascend/msmodeling/merge_requests/635)

---

### 1.1 Summary（简介）

本 RFC 提议对 msmodeling 的主动三方依赖进行最小化治理：移除经静态检索和人工审视确认不属于项目直接运行依赖的包；将唯一直接使用 `requests` 的 OptiX 健康检查改为 Python 标准库实现；将 pytest 相关工具从 pip 默认运行依赖中拆分为独立的测试依赖文件。

该方案不改变已有公开 API 和核心仿真逻辑，目标是降低默认安装的解析、下载和供应链维护成本，同时通过依赖声明契约测试、OptiX 回归测试及 CI gate 看护，保证依赖清理不影响现有功能。

### 1.2 Motivation（动机）

专项要求对组件在运行、打包和构建过程中的主动依赖进行分析，清理无关依赖，降低用户安装和上手难度。Issue #237 的分析结果显示，默认依赖中存在两类可治理项：

1. 项目源码没有直接使用，但长期作为主动依赖声明的包；
2. 仅被局部简单逻辑使用，可由 Python 标准库等价替代的包。

保留这些依赖会带来以下问题：

| 问题 | 影响 |
| --- | --- |
| 默认安装范围大于实际运行需要 | 增加解析、下载、安装时间和失败概率 |
| 主动依赖与真实代码使用关系不清晰 | 升级和安全漏洞处置时需要维护无效依赖 |
| 测试工具混入 pip 默认运行依赖 | 只运行 msmodeling 的用户也会安装 pytest 工具链 |
| `requests` 仅支撑一次简单健康检查 | 为单一 HTTP GET 场景引入额外直接依赖 |
| `pyproject.toml`、`requirements.txt`、`uv.lock` 缺少消减契约 | 后续修改可能无意重新引入已清理依赖 |

如果不进行治理，默认安装负担和依赖维护面会持续扩大，且无法形成可自动看护的依赖边界。

### 1.3 Goals（目标）

#### Goals

- 从主动运行依赖中移除 `filelock`、`Pillow`、`requests`、`scikit-learn`。
- 使用 Python 标准库 `urllib` 替代 OptiX simulator 健康检查中的 `requests`，保持原有状态语义。
- 将 `pytest`、`pytest-cov`、`pytest-xdist`、`parameterized` 从 pip 默认运行依赖拆分到 `requirements-ci.txt`。
- 保持 uv 原有 `ci` dependency group 及 `uv sync --group ci` 开发流程不变。
- 同步更新 `requirements.txt`、`pyproject.toml` 和 `uv.lock`，避免声明与锁文件不一致。
- 增加自动化契约测试，防止已消减依赖被无意重新声明。
- 保证测试版本看护不受依赖清理影响。

#### Non-goals

- 不移除仍被运行路径使用或由上游生态要求的依赖。
- 不移除 `greenlet`；ServingCast 初始化 `salabim.Environment()` 时，`salabim` 会在运行时导入该依赖。
- 不移除 `optree`；PyTorch 的 `torch.utils._cxx_pytree` 在运行时要求 `optree>=0.13.0`，TensorCast 导入链会触发该检查。
- 不在本期拆分 TensorCast、ServingCast、Web UI、OptiX 等功能的 optional extras。
- 不移除 `torchvision`、`modelscope`，也不改变其安装方式。
- 不承诺从 `uv.lock` 中删除所有同名传递依赖；某个包被其他直接依赖需要时，仍可作为传递依赖存在。
- 不修改 uv 已有 `ci` dependency group 的包集合和命令语义。
- 不调整本次人工评估后决定保留的其他三方依赖。
- 不改变公开 Python API、CLI 参数或模型仿真算法。

## 2. Use Case Analysis（用例分析）

### 2.1 运行环境默认安装

用户只需要运行 msmodeling，不执行项目测试：

```bash
uv sync
```

pip 兼容安装路径为：

```bash
pip install -r requirements.txt
```

两种默认安装路径均不再主动安装本 RFC 清理的四个包；pip 路径同时不再安装 pytest 工具链。若上游包仍传递依赖其中某个包，包管理器可按解析结果安装，该情况不违反“消减主动依赖”的目标。

### 2.2 开发和 CI 测试环境

使用 uv 的开发者继续执行：

```bash
uv sync --group ci
```

使用 pip 兼容路径的开发者或 CI 执行：

```bash
pip install -r requirements-ci.txt
```

`requirements-ci.txt` 通过 `-r requirements.txt` 继承运行依赖，并额外安装：

- `pytest`
- `pytest-cov`
- `pytest-xdist`
- `parameterized`

因此运行环境与测试工具链的职责分开，同时保持测试环境一次安装完成。

### 2.3 OptiX simulator 健康检查

OptiX 启动 simulator 后会请求其健康检查地址。替换前后必须保持以下行为：

| 场景 | 预期状态 |
| --- | --- |
| HTTP 200 | `Stage.running` |
| 非 200 响应 | `Stage.error`，保留状态码和响应文本 |
| 启动阶段发生网络异常 | `Stage.start`，允许外层流程继续等待 |
| 运行阶段发生网络异常 | `Stage.error` |
| HTTP 协议错误 | `Stage.error`，保留状态码和响应文本 |

该路径只需要同步 HTTP GET、超时、状态码和响应体处理，不需要 `requests` 的会话、认证、重试或连接池能力，因此适合使用标准库。

## 3. Design（方案设计）

### 3.1 Overall Design（总体方案）

依赖治理分为“识别、分类、修改、锁定、看护”五步：

```text
依赖声明与源码使用分析
        ↓
人工确认直接依赖 / 传递依赖 / 测试依赖
        ↓
移除无直接使用依赖，替换可由标准库覆盖的局部使用
        ↓
同步 pyproject.toml、requirements*.txt、uv.lock
        ↓
契约测试 + 功能回归 + CI gate 持续看护
```

#### 3.1.1 主动依赖消减

| 依赖 | 处理方式 | 设计依据 |
| --- | --- | --- |
| `filelock` | 从主动运行依赖移除 | 项目源码无直接使用；如上游需要，由其传递依赖声明负责 |
| `greenlet` | 保留主动运行依赖 | ServingCast 初始化 `salabim.Environment()` 时，`salabim` 会在运行时导入 `greenlet` |
| `optree` | 保留主动运行依赖 | TensorCast 导入 `torch.utils._cxx_pytree`，PyTorch 要求运行环境安装 `optree>=0.13.0` |
| `Pillow` | 从主动运行依赖移除 | 项目源码无直接使用；不禁止上游包传递安装 |
| `requests` | 标准库替换后移除 | 唯一直接使用场景为 OptiX HTTP 健康检查 |
| `scikit-learn` | 从主动运行依赖移除 | 项目源码无直接使用 |

主动依赖同时从以下位置删除：

- `pyproject.toml` 的 `[project].dependencies`
- pip 兼容声明 `requirements.txt`
- `uv.lock` 中 msmodeling 根包的直接依赖引用

锁文件只删除解析后不再被任何依赖需要的包记录。`filelock`、`Pillow`、`requests` 等包若仍由其他包传递依赖，允许继续出现在 `uv.lock` 中。

#### 3.1.2 `requests` 标准库替换

`optix/optimizer/interfaces/simulator.py` 使用 `urllib.request.urlopen()` 发起同步请求：

- 超时继续使用 10 秒；
- 通过 context manager 关闭响应；
- HTTP 200 返回运行状态；
- 非 200 响应读取并以 UTF-8 解码，非法字节使用替换策略；
- `HTTPError` 单独处理，以保留 HTTP 状态码和响应体；
- `URLError`、`TimeoutError`、`OSError` 作为网络或系统异常处理；
- 异常发生时继续依据前一进程阶段区分 `Stage.start` 和 `Stage.error`。

替换不增加重试，避免改变既有健康检查频率和进程状态机行为。

#### 3.1.3 测试依赖分层

依赖声明职责如下：

| 文件或配置 | 职责 |
| --- | --- |
| `pyproject.toml` `[project].dependencies` | uv/打包场景的运行依赖 |
| `pyproject.toml` `ci` dependency group | uv 开发与 CI 测试工具 |
| `requirements.txt` | pip 兼容的运行依赖 |
| `requirements-ci.txt` | pip 兼容的运行依赖与测试工具 |
| `uv.lock` | uv 可复现解析结果 |

本 RFC 只新增 pip 侧的测试依赖分层。uv 侧 `ci` group 在需求实施前已经存在，因此 `uv sync --group ci` 不是新增命令，也不改变原有行为。

#### 3.1.4 防回退契约

新增依赖声明契约测试，验证：

1. 四个已消减包不再作为项目主动运行依赖出现；
2. 四个测试工具不再出现在 `requirements.txt`；
3. 四个测试工具仍存在于 `requirements-ci.txt`；
4. `requirements-ci.txt` 继续引用 `requirements.txt`；
5. CI gate 将新增测试文件纳入测试映射和策略看护。

### 3.2 Technology Selection（技术选型）

| 方案 | 优点 | 缺点 | 决策 |
| --- | --- | --- | --- |
| 保留 `requests` | API 熟悉，扩展能力强 | 为单一 GET 健康检查保留直接依赖 | 不采用 |
| 使用 `http.client` | 完全使用标准库 | URL 解析、HTTPS 和响应管理代码更复杂 | 不采用 |
| 使用 `urllib.request` | 标准库内置，支持 URL/HTTPS，满足同步 GET 和异常处理 | API 较底层，需要显式处理 `HTTPError` | 采用 |
| 将 pytest 工具继续放在默认 `requirements.txt` | 单文件安装简单 | 运行用户承担不必要测试依赖 | 不采用 |
| 新增 `requirements-ci.txt` | pip 运行/测试职责清晰，与 uv group 对齐 | 增加一个需维护的兼容文件 | 采用，并用契约测试看护 |

### 3.3 Security, Privacy, and DFX Design（安全隐私与 DFX 设计）

| 属性 | 设计 |
| --- | --- |
| 安全性 | 减少主动供应链依赖面；不新增网络端点和凭据处理 |
| 隐私 | 健康检查仍访问原有本地或配置地址，不新增数据采集 |
| 兼容性 | 不改变公开 API、CLI 和健康检查状态语义 |
| 可维护性 | 明确运行依赖、测试依赖和传递依赖边界 |
| 可测试性 | 使用 mock 覆盖 200、非 200、HTTPError 和网络异常 |
| 可靠性 | 保留 10 秒超时和启动/运行阶段差异化异常处理 |
| 可诊断性 | 非成功响应继续输出状态码和响应文本 |

### 3.4 Programming and Integration Design（编程与调用设计）

#### 3.4.1 Basic Programming Model Design（编程模型基本设计）

本 RFC 不新增公开编程接口。开发者只需根据用途选择安装命令：

```bash
# 默认运行环境
uv sync

# 开发/测试环境
uv sync --group ci

# pip 兼容运行环境
pip install -r requirements.txt

# pip 兼容测试环境
pip install -r requirements-ci.txt
```

开发约束：

- 新增运行时 import 时，依赖必须声明在 `pyproject.toml`，并同步评估 pip 兼容声明；
- 仅测试代码使用的包应进入 `ci` group 和 `requirements-ci.txt`，不得加入默认运行依赖；
- 不得仅为了“可能使用”而新增主动依赖；
- 修改依赖声明后必须更新并检查 `uv.lock`，区分直接依赖变化和传递依赖变化。

#### 3.4.2 API Definition and Design（接口定义与设计）

无公开 Python API 变化。

内部 `SimulatorInterface.health()` 的返回类型、阶段枚举和错误信息格式保持兼容。实现由 `requests.get()` 切换为 `urlopen()`，调用方无需修改。

#### 3.4.3 Usage Instructions（使用说明）

- 普通用户无需安装测试工具。
- 开发者使用 uv 时继续通过 `uv sync --group ci` 安装测试工具。
- 使用 pip 的测试环境改用 `requirements-ci.txt`，不能只安装 `requirements.txt` 后假设 pytest 已存在。
- 不应通过手工安装已消减包来掩盖缺失的真实直接依赖；若发现运行路径确实直接 import 某个包，应补充使用证据、声明和回归测试。

## 4. Test Design（测试设计）

| 测试层级 | 验证内容 |
| --- | --- |
| 依赖契约测试 | 已消减依赖、测试依赖分层及继承关系符合设计 |
| OptiX 单元/回归测试 | HTTP 200、非 200、`HTTPError`、启动阶段异常、运行阶段异常 |
| CI gate 测试 | 新增契约测试进入 gate policy 和测试映射 |
| 语法检查 | `simulator.py` 可编译 |
| 差异检查 | `git diff --check` 无空白符错误 |
| 版本看护 | 现有 smoke、regression 和 nightly 流水线继续执行 |

本需求定向回归应至少覆盖依赖声明和 OptiX 健康检查。已实施变更的定向测试结果为 `73 passed`；同步最新 `master` 后再次执行关键测试，结果为 `54 passed`。

验收标准：

1. 默认运行依赖不再主动声明四个已确认可消减包；
2. pip 默认安装不再包含四个测试工具；
3. uv 和 pip 的测试环境均有明确的一步安装命令；
4. OptiX 健康检查状态语义不变；
5. 测试版本看护不因依赖清理出现回退；
6. `uv.lock` 的变化仅反映本需求及其解析结果。

## 5. Drawbacks and Risks（缺点和风险）

| 风险 | 缓解措施 |
| --- | --- |
| 静态分析遗漏动态 import | 结合人工审视、运行测试和版本看护；契约允许基于新证据恢复真实直接依赖 |
| 上游包仍传递安装已消减包，安装体积下降不明显 | 验收聚焦主动依赖消减；传递依赖由上游功能需要决定 |
| pip 用户仍使用旧的 `requirements.txt` 执行测试 | 安装文档明确指向 `requirements-ci.txt` |
| `urllib` 异常层次与 `requests` 不同 | 显式覆盖 `HTTPError`、`URLError`、超时和系统异常 |
| 依赖文件双轨再次漂移 | 契约测试同时验证 `pyproject.toml`、`requirements*.txt` 和 CI 映射 |
| 后续代码重新引入被移除包 | CI 契约测试阻止无声明或无评估地恢复依赖 |

## 6. Existing Technology（现有技术）

本 RFC 是 [uv 依赖与环境管理 RFC](./rfc_uv_dependency_management_en.md) 的后续治理：

- uv RFC 定义 `pyproject.toml`、dependency groups 和 `uv.lock` 的总体依赖管理方式；
- 本 RFC 在该基线之上缩小默认主动依赖集合，并补齐 pip 兼容路径的运行/测试分层；
- 两份 RFC 均保留 `requirements.txt` 作为 pip 兼容入口，以 `pyproject.toml` 和 `uv.lock` 作为 uv 工作流的声明与可复现解析基础。

## 7. Resolved Decisions（已决策事项）

1. 本期仅处理人工确认可消减的四个主动依赖，不扩大到其他已评估依赖。
2. `greenlet` 经干净环境 UT 验证为 ServingCast 初始化 `salabim.Environment()` 时的必需运行依赖，继续主动声明。
3. `optree` 经干净环境 UT 验证为 PyTorch `_cxx_pytree` 的必需运行依赖，继续主动声明。
4. `requests` 使用标准库替换，而不是作为 OptiX optional extra 保留。
5. 测试依赖从 pip 默认文件拆分；uv 继续复用原有 `ci` group。
6. `uv.lock` 允许保留同名传递依赖，不以“锁文件完全无该包”为验收条件。
7. `torchvision`、`modelscope` 本期保留，不拆 optional extras。
8. 本期不进行按组件的 extras 重构，避免扩大改动和验证范围。

### 7.1 Open Questions（开放问题）

当前没有阻塞本 RFC 落地的开放问题。若后续推进按组件拆分 optional extras，应单独提交 RFC，并分别定义 TensorCast、ServingCast、Web UI 和 OptiX 的安装与测试矩阵。

---

## Appendix（附录）

### References（参考资料）

- [三方依赖梳理和消减 Issue #237](https://gitcode.com/Ascend/msmodeling/issues/237)
- [依赖消减实现 PR #635](https://gitcode.com/Ascend/msmodeling/merge_requests/635)
- [uv 依赖与环境管理 RFC](./rfc_uv_dependency_management_en.md)
- [Python `urllib.request` 文档](https://docs.python.org/3/library/urllib.request.html)

### Glossary（术语表）

| 术语 | 含义 |
| --- | --- |
| 主动依赖 | 项目在自身运行依赖清单中直接声明的包 |
| 传递依赖 | 由某个直接依赖继续声明并由解析器安装的包 |
| 测试依赖 | 仅用于测试、覆盖率或测试并行执行的工具 |
| dependency group | uv 中按用途组织、默认不属于运行安装集合的依赖组 |
| 依赖契约测试 | 自动验证依赖声明边界、防止已治理依赖回退的测试 |

### Documentation Update Plan（文档更新计划）

- [x] 新增本 RFC
- [x] 更新中英文安装指南中的运行/测试依赖安装方式
- [x] 新增 `requirements-ci.txt`
- [x] 更新 CI gate 策略和测试映射
- [x] 增加依赖声明契约测试
