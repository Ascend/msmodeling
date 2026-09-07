# RFC: CLI 与 Web UI 参数系统解耦重构

## 元数据

| 项目 | 内容 |
|:-----|:--------|
| **状态** | Draft |
| **作者** | msmodeling community |
| **创建日期** | 2026-08-07 |
| **最后更新** | 2026-08-31 |

---

## 1. 概述

### 1.1 简介

CLI（`text_generate` / `video_generate` / `throughput_optimizer` / `image_generate`）与 Web UI 的参数字段当前分成两套独立维护：CLI 使用 argparse 声明参数，前端使用 TypeScript form config 声明参数，后端 runner 再做参数转发与类型强转。新增一个参数需要在 4–8 处文件中同步修改，参数类型、默认值、枚举选项、校验规则均存在重复定义与漂移风险。

本提案引入**声明式参数注册表（Parameter Registry）**，将参数定义与校验规则从 CLI 中抽取为共享的 Python 声明，放置在 `cli/registry/` 下。`cli/registry/` 只包含 CLI 关注点（类型、约束、choices、help、#704 CLI 别名/短选项/双态开关等 escape hatch），不含任何 Web UI 专属内容。Web UI 的字段基础信息（参数列表、类型、默认值、约束）从 `cli/registry/` 注入，仅在 `web_ui/backend/services/ui_props/` 中按字段 id 维护 UI 特有属性（i18n label、tooltip、placeholder、互斥/条件禁用等）。`json_adapter` 以 `Param` 为基础注入 UI 属性生成前端 form JSON，`argparse_adapter` 直接使用 `Param` 并通过 `cli/spec_cli.py` 公共基础设施（#704 MindStudio CLI 规范：kebab-case 长选项、隐藏废弃别名、metavar 约定、Description/Usage/Required/Optional/Examples 帮助布局）生成 CLI parser，实现四套模块（text_generate / video_generate / throughput_optimizer / image_generate）的参数归一维护，同时保持 CLI 与 Web UI 的完全解耦。

> **范围说明**：本 RFC 的"四模块归一"指 CLI/registry 侧统一。image_generate 已接入 registry（见 §8），但其 Web UI runner adapter 暂未实现、`json_adapter.BUNDLED_MODULES` 不含 image_generate，因此 Web UI 表单不在本 PR 暴露（后续阶段）。本文所有 Web UI 相关的 UT、回归与风险描述中的"三个模块"特指已暴露 Web UI 的前三模块（text_generate / video_generate / throughput_optimizer）。

### 1.2 动机

#### 当前痛点

| 痛点 | 具体表现 |
|------|----------|
| **重复维护** | 新增 `mxfp4_group_size` 参数需修改 6 个文件（CLI argparse × 2 模块、前端 form config × 2 模块、`_validators.ts`、`user_config.py`） |
| **枚举漂移** | `QuantizeLinearAction`（9 个值）在 Python Enum 与 TypeScript 常量 `QUANTIZE_LINEAR_OPTIONS` 中各维护一份，新增量化模式需手动同步 |
| **校验重复** | `prefix_cache_hit_rate ∈ [0,1)` 在 CLI 用 `check_prefix_cache_hit_rate()`、在前端用 `prefixCacheRate()` 各实现一遍 |
| **不一致** | `video_generate` 不使用 `get_common_argparser()`，其参数定义与其他两个模块完全独立；`reserved_memory_gb` 通过 `get_common_argparser(reserved_memory_gb_default=...)` 的参数化默认值区分（text_generate=0.0, throughput_optimizer=10.0），但 video_generate 完全缺失该参数 |
| **后端类型强转** | `throughput_optimizer` runner 的 `_build_namespace()` 需 ~120 行手工代码做 null/空字符串/默认值的类型转换 |

#### 不做的影响

- 每次新增参数的同步成本持续存在，遗漏概率随参数量线性增长
- 三模块间 ~18 个共享参数始终重复定义，无法统一修改默认值或约束
- 前端 TypeScript validator 与 Python validator 的行为差异难以发现和修复

### 1.3 目标与非目标

**目标**：

1. 参数定义（名称、类型、默认值、取值范围、枚举选项）归一维护于 `cli/registry/`；UI 属性（i18n label、tooltip、placeholder、互斥/条件禁用）归一维护于 `web_ui/backend/services/ui_props/`
2. 校验规则分层：声明式约束（min/max/required/choices）自动生成；复杂交叉校验 Python canonical、后端直接复用
3. 新增参数只需编辑 2 个文件：`cli/registry/modules/`（参数定义）+ `web_ui/backend/services/ui_props/`（UI 属性），JSON 自动生成
4. 向后兼容：可逐模块迁移，不影响已运行的 job

**非目标**：

- 不改变 `UserInputConfig` / `ModelRunner` 的核心仿真逻辑
- 不改变前端 form 组件（`SchemaForm.vue` / `SchemaFormItem.vue`）的渲染引擎
- 不改变后端 runner 的执行流程（`_run_one_case` 等）
- 前端不维护复杂 validator 代码（L2 校验全部后端执行，见 §3.2）
- 不涉及 CLI 处理流程的输出解耦（`instruction.md` 中的问题 3，另行 RFC）

---

## 2. 用例分析

### 2.1 新增一个参数

**场景**：为 `text_generate` 新增 `--new-feature-flag`（boolean, default=False）。

| 步骤 | 当前流程 | 重构后流程 |
|------|----------|------------|
| 1 | `cli/inference/text_generate.py` 添加 `add_argument()` | `cli/registry/modules/text_generate.py` 添加 `Param(...)`（纯 CLI 定义） |
| 2 | `web_ui/frontend/src/config/forms/text_generate.ts` 添加字段 | `web_ui/backend/services/ui_props/text_generate.py` 添加 `UIFieldProps(...)`（UI 属性） |
| 3 | — | 运行 `json_adapter --check` 确认 JSON 已更新 |
| 4 | `web_ui/backend/runners/text_generate.py` 确认参数转发 | 无需修改（runner 接收 params dict，不变） |
| 5 | 若需校验：`_validators.ts` 添加 JS 函数 | 若需校验：`cli/registry/validators.py` 添加 Python 函数 |
| **编辑文件数** | **3–5 个** | **CLI 参数 1 个 + UI 属性 1 个** |

### 2.2 修改共享参数默认值

**场景**：将 `quantize-linear-action` 的默认值从 `W8A8_DYNAMIC` 改为 `FP8`。

| 当前 | 重构后 |
|------|--------|
| 修改 `cli/inference/text_generate.py`、`cli/inference/throughput_optimizer.py`、`_validators.ts`、两个 `.ts` form config（共 5 处） | 修改 `cli/registry/shared.py` 中 `QUANTIZE_LINEAR_ACTION` 的 `default`（1 处）；如需更新 UI 选项显示，改 `ui_props/` 中对应条目 |

### 2.3 跨模块共享参数

**场景**：`text_generate` 和 `throughput_optimizer` 共享 ~18 个参数（model-id, device, num-devices, quantize-linear-action 等）。

| 当前 | 重构后 |
|------|--------|
| 两个 CLI 文件各写一遍 `add_argument()`；两个 TS 文件各写一遍字段定义 | `cli/registry/shared.py` 定义一次；各模块 `override()` 差异部分（如 `reserved-memory-gb` 默认值） |

---

## 3. 方案设计

### 3.1 总体架构

```text
┌───────────────────────────────┐  ┌──────────────────────────────────┐
│  cli/registry/（CLI 侧）       │  │  web_ui/backend/services/（UI 侧）│
│                               │  │                                  │
│  ├── datatypes.py             │  │  ├── ui_props/                   │
│  │   Param, ModuleSpec,       │  │  │   __init__.py                 │
│  │   ValidatorRef             │  │  │   I18nText, UIFieldProps      │
│  ├── shared.py                │  │  │   text_generate.py  ◄── UI 属性│
│  │   MODEL_ID, DEVICE, ...    │  │  │   video_generate.py           │
│  ├── validators.py            │  │  │   throughput_optimizer.py     │
│  │   Python 校验函数           │  │  ├── json_adapter.py             │
│  ├── modules/                 │  │  │   merge(Param+UI) → JSON      │
│  │   text_generate.py         │  │  └── schema_registry.py（不变）  │
│  │   video_generate.py        │  │                                  │
│  │   throughput_optimizer.py  │  │                                  │
│  └── argparse_adapter.py      │  │                                  │
│      → ArgumentParser         │  │                                  │
└──────────────┬────────────────┘  └───────────────┬──────────────────┘
               │                                   │
               │  import cli.registry.*             │
               └──────────────►                    │
                              ┌────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │   json_adapter    │
                    │ merge(Param+UI)   │ → backend/var/config/forms/*.json
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ schema_registry   │
                    │ hash-pin → SQLite │
                    └───────────────────┘

CLI 独立使用：
  cli/registry/ → argparse_adapter → ArgumentParser → CLI 执行

依赖方向：web_ui/backend ──import──► cli/registry（单向）
```

**解耦要点**：

- `cli/registry/` 只含 CLI 关注点（类型、约束、choices、help、group），**无任何 i18n / placeholder / conditions / hidden / multi_values / 控件覆盖**
- `web_ui/backend/services/ui_props/` 按 `Param.name`（kebab-case）维护 UI 特有属性（`dict[str, UIFieldProps]`），字段基础信息从 `cli/registry/` 注入
- `json_adapter` = `Param`（基础）+ `UIFieldProps`（补充）→ 前端 form JSON；`argparse_adapter` 只用 `Param`
- 两侧独立演进：CLI 新增参数只需改 `cli/registry/modules/`；Web UI 调整 UI 只改 `ui_props/`
- `cli/registry/` 不依赖 `torch` 等重型推理模块，确保 web_ui 后端可安全 import

### 3.2 校验规则分层设计

将 ~35 个 validator 按本质分为两层，**每层用最简单的方式处理**：

| 层级 | 本质 | 数量 | 处理方式 | 前端实时反馈 |
|------|------|------|----------|:---:|
| **L1：声明式约束** | data | ~15 | `Param` 的 `min/max/required/choices/pattern/max_length` 属性；argparse adapter 自动生成 type 函数，json adapter 自动生成 async-validator rules | ✅ 即时 |
| **L2：交叉校验** | code | ~20 | Python 函数存于 `cli/registry/validators.py`，注册到 `ModuleSpec.validators`；CLI `main()` 在 `parse_args()` 完成后调用 + 后端 `validate_module()` 统一执行 | ❌ 提交时校验 |

> **设计原则**：
>
> - L1 是声明式约束，前端可实时反馈（输入时校验）
> - L2 是 Python 代码校验，统一在 `parse_args()` 完成后执行——所有字段均已解析，可以访问完整 params dict
> - 不再区分"字段级"和"交叉"校验——所有 Python 校验器都在同一时机执行，避免 argparse `type=` 回调阶段的复杂性
>
> **Validator 归属**：所有 Python 校验器**统一注册**到 `ModuleSpec.validators`（通过 `ValidatorRef`），
> 后端 `validate_module()` 遍历 `spec.validators` 依次执行，从 `ui_props.VALIDATOR_UI` 读取 `fields` 列表将错误映射到具体字段。
> 前端不执行任何 L2 校验——提交后后端返回 `errors[]`，前端映射到字段标红。

**校验架构**：

前端不维护任何复杂 validator 代码。校验分两层，**每层都在前端和后端执行**（前端实时反馈，后端权威校验）：

| 层级 | 前端（实时反馈） | 后端（权威校验，防绕过） | 来源 |
|------|----------|------|------|
| **L1 声明式约束** | async-validator rules（输入时） | `validate_module()` 内的 `_check_declarative()`（提交时） | `Param` 的 `min/max/exclusive_min/exclusive_max/required/choices/pattern/max_length` |
| **L2 交叉校验** | ❌ 不执行 | `validate_module()` 内的 `spec.validators`（提交时） | `cli/registry/validators.py` 中的 Python 函数 |

> **关键设计**：L1 声明式约束**前端实时反馈 + 后端权威校验双重执行**。前端 L1 可被绕过（直接构造 HTTP 请求），因此后端必须独立执行 L1，防止越界数值、非法 choices、不匹配 pattern 的参数进入仿真逻辑。CLI 端通过 argparse type 函数天然执行 L1；Web 后端通过 `validate_module()` 执行 L1 + L2，确保 CLI 与 Web UI 的参数接受范围一致。
>
> **Web 后端校验粒度：以 case 为单位**。多值字段（`multi_values=True`）在 API 层先展开笛卡尔积，生成若干单 case 标量 params dict。L1 和 L2 均对**每个 case** 独立执行——每个 case 的字段值为标量，L2 交叉校验（如 `tp_size × dp_size × pp_size == num_devices`）语义明确。任一 case 校验失败则整个 job 返回 400。这保证 CLI subprocess 在 job 调度前就不会接收到非法参数组合。

前端对 L2 校验无感知——提交后后端返回字段级错误，前端直接映射到对应字段标红。`_validators.ts` 中的 validator 函数全部删除，仅保留 option 常量（也由 registry 自动生成）。

### 3.3 核心数据结构

**设计原则**：

- `cli/registry/` 定义 `Param`（CLI 关注点），是字段的**唯一权威来源**
- `web_ui/backend/` 维护每个字段的 **UI 特有属性**（label i18n、tooltip、placeholder、conditions 等），按 `Param.name`（kebab-case）索引
- `json_adapter` 以 `Param` 为基础，注入 UI 属性，生成前端 form JSON
- Web UI 不需要维护与 `Param` 平行的完整字段列表——字段列表从 `cli/registry/` 注入

```python
# cli/registry/datatypes.py — 纯 CLI 关注点

@dataclass(frozen=True)
class Param:
    """参数的唯一权威定义。

    frozen=True：共享单例（如 MODEL_ID）不可被意外修改（如 MODEL_ID.default = x
    会抛 FrozenInstanceError）。需要覆盖时使用 override() 方法创建副本。

    CLI argparse_adapter 读取全部字段生成 parser；
    json_adapter 读取 id/data_type/default/required/min/max/choices/pattern/
    max_length/group/nargs/cli_help 生成前端 form JSON。
    """

    # ── 标识 ──────────────────────────────────────────────────────
    name: str
    ## 字段唯一标识，kebab-case（如 "tp-size"、"chrome-trace-file"）。
    ## 直接作为：
    ##   - CLI flag 名（--tp-size、--chrome-trace-file）
    ##   - form-schema JSON field id (== Param.name, kebab-case)
    ##   - API params dict key
    ##   - validator dict key（如 params['num-devices']）

    data_type: str
    ## 数据类型，决定 CLI 解析方式和前端控件推断。
    ## 可选值："string" | "integer" | "number" | "boolean"
    ##         | "string[]" | "integer[]" | "number[]"
    ## 数组类型（*[]）与 nargs 配合，前端根据 dataType 自动推断控件
    ## （multi-select 或 text 逗号分隔输入）。

    # ── 默认值与必填 ──────────────────────────────────────────────
    default: Any = None
    ## 参数默认值。None 表示无默认值。
    ## CLI: parser 的 default；前端: form 初始值。

    required: bool = False
    ## 是否为必填参数。
    ## CLI: argparse required=True；前端: 生成 {"rule":"required"} 校验。

    # ── 声明式约束（L1 校验，两端自动生成校验规则） ────────────────
    min: int | float | None = None
    ## 最小值（含）。CLI 生成 range checker；前端生成 {"rule":"min"} 校验。

    max: int | float | None = None
    ## 最大值（含）。CLI 生成 range checker；前端生成 {"rule":"max"} 校验。

    exclusive_min: int | float | None = None
    ## 最小值（不含，开区间）。用于表达 (lo, ...] 范围。
    ## CLI: 在 type 函数中检查 n > exclusive_min；
    ## 前端: 生成 {"rule":"gt","value":exclusive_min} 校验。
    ## 示例: prefix-cache-hit-rate 用 exclusive_max=1 表达 [0, 1)。

    exclusive_max: int | float | None = None
    ## 最大值（不含，开区间）。用于表达 [..., hi) 范围。
    ## CLI: 在 type 函数中检查 n < exclusive_max；
    ## 前端: 生成 {"rule":"lt","value":exclusive_max} 校验。
    ## 示例: prefix-cache-hit-rate 的 [0, 1) 用 min=0 + exclusive_max=1 表达，
    ## 避免 max=0.9999 截断（0.99995 等合法值会被误拒）。

    choices: list | None = None
    ## 允许的枚举值列表。
    ## CLI: argparse choices；前端: 生成 options 列表。
    ## 可直接传 Python Enum 的 list(EnumClass)。

    pattern: str | None = None
    ## 正则约束（仅 string 类型）。
    ## CLI: 在 type 函数中检查；前端: 生成 {"rule":"pattern"} 校验。

    max_length: int | None = None
    ## 字符串最大长度（仅 string 类型）。
    ## CLI: 在 type 函数中检查；前端: 生成 {"rule":"max","type":"length"} 校验。

    # ── 分组 ──────────────────────────────────────────────────────
    group: str | None = None
    ## 参数分组名（英文），如 "LLM Options"、"Quantization Options"。
    ## CLI: 作为 argparse add_argument_group() 的名称，组织 --help 输出。
    ## 前端: 通过 ui_props 的 group_labels 字典查找 i18n 翻译。

    # ── 多值参数 ──────────────────────────────────────────────────
    nargs: str | int | None = None
    ## argparse nargs 参数，指定 CLI 端接受多个值的方式。
    ## "*" = 0 或多个，"+" = 1 或多个，"?" = 0 或 1，整数 = 精确个数。
    ## 前端不受影响（前端通过 multi_values 控制多值行为）。

    # ── CLI 专属 escape hatch（前端不使用） ───────────────────────
    cli_type: Callable | None = None
    ## 自定义 argparse type 函数（覆盖自动推断）。
    ## 用于复杂类型转换，如 QuantizeLinearAction 枚举、check_positive_integer 等。
    ## 为 None 时根据 data_type + min/max 自动生成。

    cli_action: str = "store"
    ## argparse action，直接映射到 add_argument(action=...)。
    ##
    ## 自动推断：data_type="boolean" 时默认为 "store_true"（无需显式设置）。
    ## Escape hatch：当需要非默认 action 时手动指定。
    ##
    ## 映射表：
    ## | cli_action     | argparse 行为                              | 示例                          |
    ## |----------------|-------------------------------------------|-------------------------------|
    ## | "store"        | 普通赋值（boolean 自动推断为 store_true）   | --compile (data_type=boolean) |
    ## | "store_true"   | boolean flag，无参数，出现即为 True         | --decode                      |
    ## | "append"       | 多次指定追加到列表                          | --performance-model analytic  |
    ## |                |                                           |   --performance-model profiling |
    ##
    ## 示例：
    ##   # 自动推断：boolean → store_true
    ##   Param(name="compile-allow-graph-break", data_type="boolean")
    ##   → add_argument("--compile-allow-graph-break", action="store_true")
    ##
    ##   # Escape hatch：多次指定追加到列表
    ##   Param(id="performance_model", data_type="string", cli_action="append",
    ##         choices=["analytic", "profiling"])
    ##   → add_argument("--performance-model", action="append", choices=[...])

    cli_positional: bool = False
    ## 是否为位置参数（如 model_id）。
    ## True: argparse 生成位置参数（无 -- 前缀）。
    ## False（默认）: 生成 --flag 可选参数。
    ## 注意：位置参数由显式声明控制，不使用 heuristic（如 `not p.default`），
    ## 避免有默认值的参数被误判为可选 flag。
    ##
    ## Post-#704 扩展：model_id 用 `nargs="?"` positional +
    ## `--model-path`/`--model-id` same-dest flag + runtime require_model_id
    ## 校验。通过 cli_positional=True + cli_flag="model-path" 表达
    ## （见 §3.5.1 model_id 特殊案例）。

    cli_flag: str | None = None
    ## 仅用于 positional+flag 组合（如 model_id）。覆盖 formal --flag 名。
    ## 常规 flag 参数此字段 MUST 为 None（name IS the flag name）。
    ## 当前只有 MODEL_ID 使用：name="model-id"（位置参数）+
    ## cli_flag="model-path"（formal flag 名）。

    cli_flag_help: str | None = None
    ## positional+flag 组合的 formal --flag 帮助文本。
    ## None 则使用通用兜底（"Model source. ... Equivalent to the positional
    ## model id."）。各模块可 override 为领域特定措辞
    ## （video: "Diffusers model source..."、image: "Image model source..."）。

    cli_aliases: tuple[str, ...] = ()
    ## 隐藏兼容废弃别名（不带 -- 前缀），如 ("model_id",)。
    ## #704 保留旧名（可解析 + 一次性 stderr deprecation warning，
    ## 不出现在 --help）。argparse_adapter 通过 spec_cli.add_option(
    ## aliases=...) 注册，CLI 接受性与 #704 前完全一致。
    ## 别名日落周期遵循 #704 RFC §7 unresolved state；注册表不决策 ——
    ## 删除 tuple 中某项即可，无需改 adapter。

    cli_extra_flags: tuple[str, ...] = ()
    ## 额外 FORMAL 选项名（不带 -- 前缀），如 ("devices",)。
    ## 与 cli_aliases 不同：这些是非废弃的公开名，与主 flag 一起显示
    ## 在 --help，接受时不带 warning。用于 legacy add_argument(
    ## "--a", "--b") 模式。

    cli_short: str | None = None
    ## 单字符短选项（不带 - 前缀），如 throughput jobs → "j"（-j/--jobs）。
    ## #704 词汇约束：短选项跨命令语义固定（-h/-V/-v/-q/-o/-j/-c）；
    ## 模块特定短选项只允许在此词汇表内，否则 None。

    cli_off_flag: str | None = None
    ## 双态 off-switch（不带 -- 前缀）。仅 data_type="boolean" +
    ## cli_action="store_true" 有效。#704 spec 4.3.1 要求 boolean 对
    ## `--name` / `--no-name`。argparse_adapter 自动生成 on/off 对。

    cli_off_help: str | None = None
    ## off-flag 帮助文本。None 则自动生成 "Enable <off-flag-name> (default)."。

    cli_metavar: str | None = None
    ## argparse metavar，如 "<N>"/"<FILE>"/"{col,row}"。
    ## None 则回退到 spec_cli 约定（METAVAR_* 常量 + _metavar_for）。

    cli_help: str | None = None
    ## argparse --help 帮助文本（英文）。
    ## 前端: tooltip/label 未设置时 fallback 到此值（中英文相同）。

    cli_dest: str | None = None
    ## argparse ``dest`` 覆盖（snake_case）。None 时 dest = py_name。
    ## 仅当 #704 重命名保留了不同的内部属性名时设置
    ## （如 --ttft-limit 解析到 args.ttft_limits 因为 serving_cast
    ## 读复数属性；参见 d79e7342）。罕见 —— 除非 legacy CLI 显式
    ## 传过 ``dest=``，否则保持 None。

    # ── 方法 ──────────────────────────────────────────────────────
    @property
    def py_name(self) -> str:
        """Python 属性名（snake_case）：name 中连字符 → 下划线。
        用作 argparse ``dest``，使得 args.chrome_trace_file 可访问，
        尽管 CLI flag 和 JSON field id 用 kebab-case（``chrome-trace-file``）。
        """
        return self.name.replace("-", "_")

    @property
    def dest_name(self) -> str:
        """有效 argparse dest：``cli_dest`` 覆盖或 ``py_name``。
        CLI 入口和 Web UI runner 的 ``_build_namespace`` 都必须用此属性
        ``getattr(args, ...)``。
        """
        return self.cli_dest or self.py_name

    # 允许覆盖的字段白名单（防止意外修改关键字段如 name/data_type）。
    # group 允许覆盖：某些模块（如 image_generate baseline 把所有参数注册
    # 在裸 parser 上）需要 override(group=None) 解除共享分组。
    _OVERRIDE_ALLOWED = frozenset({
        "default", "required", "min", "max", "exclusive_min", "exclusive_max",
        "choices", "nargs", "group", "cli_help", "cli_action", "cli_positional",
        "cli_flag", "cli_flag_help", "cli_aliases", "cli_extra_flags", "cli_short",
        "cli_off_flag", "cli_off_help", "cli_metavar", "cli_dest",
    })

    def override(self, **kwargs) -> "Param":
        """创建带覆盖的副本。用于跨模块复用共享参数时修改个别属性。
        示例: RESERVED_MEMORY_GB.override(default=10.0)

        注意：只能覆盖 _OVERRIDE_ALLOWED 中的字段，
        不可覆盖 name、data_type 等关键字段。
        """
        forbidden = set(kwargs) - self._OVERRIDE_ALLOWED
        if forbidden:
            raise ValueError(f"Cannot override fields: {forbidden}")
        return dataclasses.replace(self, **kwargs)


@dataclass
class ValidatorRef:
    """校验器引用。

    CLI 端: main() 解析参数后遍历 validators 调用 fn()，fn() 返回
            True（通过）或 str（错误信息），直接 print 到 stderr。
            CLI 只使用 fn，不使用 name。
    UI 端:  name 作为关联 key，连接 cli/registry ↔ ui_props ↔ 前端。

    name 的数据流：
    ┌─────────────────────────┐
    │ cli/registry/ValidatorRef│  name = "productEqNumDevices"
    │   fn = python_fn        │  ← CLI 只使用 fn
    └───────────┬─────────────┘
                │ name 作为 key
                ▼
    ┌─────────────────────────┐
    │ ui_props/VALIDATOR_UI   │  VALIDATOR_UI["productEqNumDevices"] =
    │                         │    { fields: [...], message: I18nText(...) }
    └───────────┬─────────────┘
                │ json_adapter 输出到 formValidation[]
                ▼
    ┌─────────────────────────┐
    │ 前端 form JSON          │  formValidation: [{ rule: "validator",
    │                         │    value: "productEqNumDevices", message: {...} }]
    └─────────────────────────┘

    CLI 不直接使用 name，但 name 是 cli/registry 与 ui_props 的关联标识符
    （类似 Param.name 在 CLI 生成 flag name、在 UI 作为 field id）。
    """

    name: str
    ## 校验器唯一标识，camelCase（如 "productEqNumDevices"）。
    ## CLI 不直接使用（CLI 只调用 fn），但 name 是以下三方的关联 key：
    ##   1. cli/registry/ValidatorRef.name → 标识校验器
    ##   2. ui_props/VALIDATOR_UI[name] → 查找 fields 和 i18n message
    ##   3. 前端 formValidation[].value → 引用校验器名称
    ## 示例: "productEqNumDevices"、"sharedExpertMutex"、"draftMtpMutex"。

    fn: Callable[..., str | None]
    ## 校验函数。接收整个表单的字段值 dict，返回:
    ##   None       — 校验通过（Python 函数默认返回 None，无需显式 return）
    ##   str        — 校验失败，str 为英文错误信息（CLI 直接使用）
    ##
    ## 调用约定（lenient，由 wants_provided 决定 1 或 2 个位置参数）:
    ##   fn(params: dict) -> str | None                     # wants_provided=False
    ##   fn(params: dict, provided: set[str]) -> str | None # wants_provided=True
    ##
    ## params: 以 kebab-case Param.name 为 key 的字段值 dict
    ## provided: 显式在命令行或 form 中设置的 kebab-case 字段名集合
    ##   （用于区分"显式传了默认值" vs "未传"——params 本身无法表达此差异。
    ##    CLI 侧从 argv + parser._actions 反推；Web UI 从 form payload keys 取）
    ##
    ## UI 侧的错误映射（i18n message、字段列表）在 ui_props.VALIDATOR_UI 中
    ## （见 §3.6）。

    wants_provided: bool = False
    ## 是否使用 2-arg 调用约定。False（默认）保留经典 fn(params) 合约，
    ## 适配已有所有校验器；True 用于需要区分"显式默认值 vs 未传"的场景
    ## （如 draft_dependents_require_method、draft_mtp_mutex —— G2/G3
    ## 必须忽略未显式传入的默认值，否则会误报）。


@dataclass
class ModuleSpec:
    """CLI 侧模块定义。一个 ModuleSpec 对应一个 CLI 模块 / 前端表单。

    不含 version（schema 版本由 Web UI 的 ui_props 维护，用于 schema_registry 版本 pinning）。
    不含 i18n message（由 ui_props.VALIDATOR_UI 维护）。
    """

    module_id: str
    ## 模块标识，如 "text_generate"、"video_generate"、"throughput_optimizer"。
    ## 必须与 CLI 模块名和前端 moduleId 一致。

    title: str
    ## 模块英文标题，如 "Text Generation"。CLI --help 使用。
    ## 前端 i18n 标题在 ui_props.TITLE 中维护。

    fields: list[Param]
    ## 该模块的所有参数定义列表。
    ## 顺序决定 CLI --help 中的参数出现顺序。

    validators: list[ValidatorRef] = field(default_factory=list)
    ## 该模块的交叉校验器列表。
    ## CLI: parse_module_args() 解析后依次调用每个 validator 的 fn()。
    ## 前端: 通过 json_adapter 输出到 formValidation[]（仅用 name，不含 fn）。

    # ── #704 CLI 元数据 ───────────────────────────────────────────
    cli_prog: str | None = None
    ## SpecArgumentParser 的 prog，如 "msmodeling inference text-generate"。

    cli_description: str | None = None
    ## --help Description 段落。None 则回退到 title。

    cli_examples: str | None = None
    ## --help Examples 段落（至少 1 个可运行命令 + # 注释 — #704 spec 4.4）。

    cli_output_help: str | None = None
    ## --help Output 段落（描述磁盘产物，可选）。

    log_options: bool = True
    ## 是否注入 spec_cli.add_log_options()（--log-level/-v/-q 等）。
    ## 日志组是公共基础设施，不是业务 Param（见 §3.5.1）。

    log_options_after_group: str | None = None
    ## log options 在该 group 之后插入。None 默认为 "General Options"。

    log_options_after_field: str | None = None
    ## log options 在该 Param py_name 之后插入（更细粒度）。
    ## 用于 image_generate 等 baseline 把所有参数注册在裸 parser 上，
    ## 并在特定位置调用 add_log_options() 的场景。优先于 log_options_after_group。

    requires_model_id: bool = True
    ## 是否在 parse 后调用 cli.utils.require_model_id()（位置参数 +
    ## --model-path/--model-id 双入口延迟必填校验）。所有三个已暴露 Web UI 的仿真模块为 True。
```

```python
# web_ui/backend/services/ui_props/__init__.py — Web UI 侧：基础类型与每个字段的 UI 特有属性

@dataclass(frozen=True)
class I18nText:
    """双语文本。用于所有需要中英文的 UI 属性。"""

    zh: str
    ## 中文文本。

    en: str
    ## 英文文本。


@dataclass
class UIFieldProps:
    """单个字段的前端 UI 特有属性。

    与 cli/registry 的 Param 通过 `Param.name`（kebab-case）关联（dict key = Param.name）。
    Param 已提供的属性（data_type/default/required/min/max/choices/group）
    由 json_adapter 直接注入前端 JSON，无需在此重复。

    属性级兜底（UIFieldProps 条目内，单个属性未设置时）：
    - label 未设置     → fallback 到 I18nText(Param.name, Param.name)
    - tooltip 未设置   → fallback 到 I18nText(Param.cli_help, Param.cli_help)
    - data_type 未覆盖 → 使用 Param.data_type
    - choices 未覆盖   → 使用 Param.choices
    - control 不在此类 → 前端根据 data_type 自动推断
    - group 不在此类   → 从 Param.group 通过 GROUP_LABELS 查找 i18n

    注意：UT 要求每个字段必须有 UIFieldProps 条目（即使为空），
    因此"字段级兜底"（字段不在 UI dict 中）在生产环境不应触发。
    此处的兜底仅针对"条目内属性未设置"的情况。

    覆盖规则：
    - data_type/choices 覆盖用于前端与 CLI 行为不同的场景
      （如 chrome-trace-file: CLI 是文件路径 → 前端是下载开关；
       performance_model: CLI 全量选项 → 前端限制为子集）
    """

    label: I18nText | None = None
    ## 字段显示名称（中英文）。
    ## 未设置时 fallback 到 Param.name。
    ## 示例: I18nText("模型 ID", "Model ID")

    tooltip: I18nText | None = None
    ## 字段说明文本（中英文），前端显示为 hover 提示。
    ## 未设置时 fallback 到 Param.cli_help（中英文相同）。
    ## 示例: I18nText("待仿真模型的 HuggingFace 名称", "HuggingFace model name")

    placeholder: I18nText | None = None
    ## 输入框占位提示文本（中英文）。
    ## 示例: I18nText("如 Qwen/Qwen3-32B", "e.g. Qwen/Qwen3-32B")

    option_source: dict | None = None
    ## 选项数据来源（用于 select/multi-select 控件）。
    ## 动态选项: {"type": "dynamic", "name": "devices"}  → 从 /api/options/devices 获取
    ## 前端根据 data_type + option_source 自动推断控件类型为 select/multi-select。

    disabled: bool = False
    ## 静态禁用（始终禁用，不可编辑）。
    ## 用于始终由系统决定的参数（如 compile 开关）。

    conditions: dict | None = None
    ## 动态条件控制（结构化 predicate，复用前端 usePredicate.ts 引擎）。
    ## 格式: {"enabled": <predicate>}，当 enabled 条件为 true 时字段启用。
    ## 用于依赖/互斥场景（互斥通过 not 取反实现）。
    ##
    ## 支持操作符: eq, ne, include, exclude, contains, notContains,
    ##            gt, gte, lt, lte, notEmpty, empty, isTrue, isFalse, present, absent
    ## 支持组合器: and, or, not
    ##
    ## 示例: {"enabled": {"not": {"field": "enable-shared-expert-tp", "op": "isTrue"}}}
    ##       → 互斥：A 开启则 B 禁用（A=true → enabled=false）
    ## 示例: {"enabled": {"field": "quantize-linear-action", "op": "contains", "value": "mxfp4"}}
    ##       → 依赖：选择了 MXFP4 则启用（MXFP4 选中 → enabled=true）
    ##
    ## 注意：conditions 使用结构化 JSON 格式，与前端 useFieldConditions.ts 完全兼容，
    ## 无需新增解析器。后端 ValidatorRef.fn() 作为权威兜底，防止绕过前端的非法提交。

    hidden: bool = False
    ## 字段是否在前端 UI 中隐藏（不渲染）。
    ## True: 字段不出现在前端表单中（CLI-only 字段）。
    ## False（默认）: 字段在前端正常渲染。
    ## 与 disabled 的区别：
    ##   disabled → 灰显不可编辑，字段可见（用户知道参数存在）
    ##   hidden   → 完全隐藏，字段不可见（用户不知道参数存在）

    multi_values: bool = False
    ## 标记该字段在后端 runner 中参与多用例笛卡尔积展开。
    ## 此属性仅控制 Runner 展开行为，不影响前端控件渲染
    ## （前端控件由 data_type + option_source + control 决定）。
    ## 后端 runner 从此标记动态收集展开字段（替代硬编码 _MULTI_CASE_FIELDS），
    ## 对多值字段做笛卡尔积展开，每个组合独立运行一个 case。
    ## 示例: device、num-queries、quantize-linear-action、quantize-attention-action、tp-size。
    ##
    ## Runner 动态收集逻辑：
    ##   1. 遍历 spec.fields，找到 UIFieldProps.multi_values=True 的字段
    ##   2. 解析函数由 Param.data_type 推断（无需额外配置）：
    ##      - "string"  → 字符串列表（_as_list）
    ##      - "integer" → int 列表（_parse_int_list）
    ##      - "number"  → float 列表（_parse_float_list）
    ##   3. 对多值字段做笛卡尔积展开
    ##
    ## 控件推断补充规则：当 multi_values=True 且无 option_source/choices 时，
    ## 前端使用 control="text"（接受逗号分隔输入如 "1,2,4"），而非 data_type 推断的控件。

    # ── 覆盖 Param 默认行为（用于前端与 CLI 差异场景） ──────────

    data_type: str | None = None
    ## 覆盖 Param.data_type。用于前端与 CLI 数据类型不同的场景。
    ## 示例: chrome-trace-file — CLI 是 str（文件路径），前端是 boolean（是否下载）
    ##   Param: data_type="string"
    ##   UIFieldProps: data_type="boolean"
    ## 未设置时使用 Param.data_type。

    choices: list | None = None
    ## 覆盖 Param.choices。用于前端限制为 CLI 选项子集的场景。
    ## 示例: performance_model — CLI 支持 ['analytic','profiling']，前端只显示 ['analytic']
    ##   Param: choices=["analytic", "profiling"]
    ##   UIFieldProps: choices=["analytic"]
    ## 约束: UIFieldProps.choices 必须是 Param.choices 的子集。
    ## json_adapter --check 或 CI 校验应验证此约束，防止前端出现 CLI 不接受的选项。
    ## 未设置时使用 Param.choices。

    control: str | None = None
    ## 显式指定前端控件类型，覆盖自动推断。
    ## 可选值: "text" | "number" | "select" | "multi-select" | "switch"
    ## 通常不需要设置（前端根据 data_type 自动推断），仅在需要强制指定时使用。
    ## 示例: chrome-trace-file 覆盖为 boolean 后，自动推断为 switch，无需显式设置 control。
    ## 示例: 强制某个 integer 字段使用 text 输入（而非 number spinner）。

    clearable: bool = False
    ## select/multi-select 控件是否显示清除按钮（重置为空）。
    ## True: 用户可清除选择（值置为 None/空）。
    ## False（默认）: 无清除按钮，用户必须选择有效选项。
    ## 仅对 select/multi-select 控件有效。
    ## 示例: speculative-method — 用户可能需要在启用后禁用。

    # ── 覆盖 Param 默认行为（扩展） ──────────────────────────────

    default: Any = _UNSET
    ## 覆盖 Param.default。用于前端与 CLI 默认值不同的场景。
    ## 使用哨兵 _UNSET（非 None）区分"未设置"与"显式 None"。
    ## 示例: device — CLI 默认 "TEST_DEVICE"（单值），前端需数组默认
    ##   Param: default="TEST_DEVICE" (data_type="string")
    ##   UIFieldProps: default=["ATLAS_350_425T_112G"], data_type="string[]"
    ## _UNSET（默认）表示使用 Param.default。

    required: bool | None = None
    ## 覆盖 Param.required。用于前端 UX / CLI required 语义分歧。
    ## CLI 有默认值的参数通常 required=False，但前端 UX 可能需要显式选择。
    ## 示例: device / log_level — CLI 有默认值（required=False），
    ##   但前端 UX 需要显式选择，UIFieldProps: required=True
    ## None（默认）表示使用 Param.required。

    group: dict | None = None
    ## i18n group 覆盖（用于 convention field 和 baseline 对齐调整）。
    ## 形状: {"zh": ..., "en": ...}。常规字段通过 Param.group +
    ## GROUP_LABELS 映射；convention field 无 Param.group，需此字段直接设置。

    # ── 方法 ──────────────────────────────────────────────────────

    def override(self, **kwargs: Any) -> UIFieldProps:
        """创建修改后的副本，类似 Param.override()。

        用于共享 UI 常量（如 LOG_LEVEL_UI）表达模块间差异
        （default/group/option_source），无需重复完整定义。
        """
```

**`conditions` 结构化 predicate**（复用前端 `usePredicate.ts` 引擎，119 行成熟实现）：

`conditions` 使用结构化 JSON 格式，与前端 `usePredicate.ts` 完全兼容：

```python
# 示例 1：互斥 — A 开启则 B 禁用
conditions = {"enabled": {"not": {"field": "enable-shared-expert-tp", "op": "isTrue"}}}
# → enable-shared-expert-tp=true → enabled=false → B 禁用

# 示例 2：依赖 — 选择了 MXFP4 则启用
conditions = {"enabled": {"field": "quantize-linear-action", "op": "contains", "value": "mxfp4"}}
# → 选中 MXFP4 → enabled=true → 字段启用

# 示例 3：组合条件 — compile 开启且 device 为 GPU 时启用
conditions = {"enabled": {"and": [
    {"field": "compile", "op": "isTrue"},
    {"field": "device", "op": "eq", "value": "GPU"}
]}}
# → compile=true 且 device=GPU → enabled=true → 字段启用
```

**支持的操作符**（16 种）：

- 比较：`eq`, `ne`, `gt`, `gte`, `lt`, `lte`
- 成员：`include`（标量 ∈ 数组）, `exclude`, `contains`（数组 ∋ 标量）, `notContains`
- 状态：`isTrue`, `isFalse`, `empty`, `notEmpty`, `present`, `absent`

**支持的组合器**（3 种）：`and`, `or`, `not`

**优势**：

- 无需新增解析器，直接复用 `usePredicate.ts`
- 支持任意复杂条件（嵌套、组合）
- 与现有 form config 的 `conditions.enabled` 格式一致，迁移无缝

**安全约束**：`usePredicate.ts` 是白名单解析器，仅接受预定义操作符，**不使用 `eval()` 或动态属性访问**。

**两侧数据的关系**：

```text
cli/registry/Param（权威来源）     web_ui UIFieldProps / ui_props 模块文件
  id ───────────────────────────►  dict key（按 Param.name 索引）
  data_type ────────────────────►   dataType（UIFieldProps.data_type 可覆盖）
  default ──────────────────────►   default（直接使用）
  required / min / max ─────────►   validation rules（自动生成）
  choices ──────────────────────►   options（UIFieldProps.choices 可覆盖为子集）
  cli_help ─────────────────────►   tooltip fallback（zh=cli_help, en=cli_help）
  group ────────────────────────►   group_labels[group] 查找 i18n
                                   label ◄── UIFieldProps（未设置 fallback 到 Param.name）
                                   placeholder ◄── UIFieldProps
                                   control ◄── UIFieldProps（显式覆盖自动推断）
                                   hidden ◄── UIFieldProps（不渲染该字段）
                                   conditions ◄── UIFieldProps（动态禁用/启用）
                                   multi_values ◄── UIFieldProps（多用例笛卡尔积展开）
```

**兜底逻辑**（防御性机制，但 UT 强制要求显式配置）：

- 字段不在 `UI` dict 中 → `json_adapter` 用 `Param` 数据直接生成 field（**技术兜底**）
- **但** `test_all_fields_have_ui_props` 要求所有字段必须在 `UI` dict 中有条目（**流程约束**）
- 兜底逻辑的实际用途：
  1. 开发阶段的临时状态（新增字段后尚未配置 UI 属性时，前端仍可渲染）
  2. 防止历史代码因缺少 UI 配置而崩溃
  3. 未来如果放宽 UT 要求，兜底逻辑自动生效
- 生产环境中，兜底逻辑不应被触发（UT 会阻断）
- `tooltip` 未设置 → fallback 到 `I18nText(cli_help, cli_help)`
- `label` 未设置 → fallback 到 `I18nText(param.name, param.name)`
- `group` → 从 `Param.group` 通过 `group_labels` 查找 i18n；未命中则用英文原名
- `control` → 前端根据 `dataType` 自动推断

`json_adapter` = `Param`（基础） + `UIFieldProps`（补充） → 前端 form JSON。
`argparse_adapter` = 只用 `Param`，完全独立。

### 3.4 共享参数与模块覆盖

```python
# cli/registry/shared.py — 跨模块共享参数（纯 CLI 关注点）

# ── Common positional: model_id ─────────────────────────────────────
# Used by text_generate, throughput_optimizer, video_generate, image_generate.
# Post-#704: positional with nargs="?" + --model-path/--model-id same-dest
# formal flags + runtime require_model_id validation. argparse_adapter
# handles this pattern when cli_positional=True + cli_flag="model-path"
# (see §3.5.1). cli_aliases are DEPRECATED names (one-shot warning);
# --model-id is a formal sibling registered separately by the adapter.
MODEL_ID = Param(
    name="model-id",
    data_type="string",
    required=True,
    nargs="?",
    pattern=r"^[a-zA-Z0-9_/.-]+$",
    max_length=256,
    cli_positional=True,
    cli_flag="model-path",        # positional+flag combo: formal --model-path flag
    cli_aliases=("model_id",),    # no -- prefix; deprecated underscore form
    group="General Options",
    cli_help=(
        "Model source. Recommended safe mode: a reviewed absolute local model path. "
        "Model id mode also accepts Hugging Face or ModelScope ids, but may execute "
        "remote Python code through trust_remote_code=True and is not security-guaranteed. "
        "Equivalent to --model-id."
    ),
)

# ── Common: device ──────────────────────────────────────────────────
# choices placeholder None; real choices injected at runtime from
# get_device_choices() via argparse_adapter's device-id special handling.
# Module override examples:
#   throughput_optimizer: DEVICES = Param(name="device", data_type="string[]",
#       nargs="+", cli_extra_flags=("devices",), ...) — new module-level Param
#   text_generate / video_generate: no override (single-value default)
DEVICE = Param(
    name="device",
    data_type="string",
    default="TEST_DEVICE",
    group="General Options",
    cli_help=(
        "Specifies the target device profile to use for benchmarking and simulation. "
        "Must be a valid device name as defined in DeviceProfile. "
        "The default device 'TEST_DEVICE' is used for standard simulation runs."
    ),
)


def get_device_choices() -> list[str]:
    """Runtime device profile names.

    Deferred import: tensor_cast.device_profiles may not be importable at module
    load time (e.g. in tests without the sim stack). At build_argparser() time
    the profiles are expected to be registered.

    Falls back to ["TEST_DEVICE"] if no profiles are available — keeps CLI
    usable in degraded environments (matches legacy get_common_argparser
    behavior for --device choices).
    """
    try:
        import tensor_cast.device_profiles  # noqa: F401  (registers built-in profiles)
        from tensor_cast.device import DeviceProfile
        names = list(DeviceProfile.all_device_profiles.keys())
        return names or ["TEST_DEVICE"]
    except Exception:
        return ["TEST_DEVICE"]


RESERVED_MEMORY_GB = Param(
    name="reserved-memory-gb",
    data_type="number",
    default=0.0,
    min=0,
    group="General Options",
    cli_help=(
        "Amount of device memory (in gigabytes) reserved for system usage "
        "and unavailable for application. "
        "Set to 0 to disable memory reservation."
    ),
)

# ── 日志等级：不是 Param ───────────────────────────────────────────
# log_level 由 spec_cli.add_log_options() 作为公共基础设施提供
# （--log-level / -v / -q / --debug / --log-file）。见 §3.5.1。
# ModuleSpec.log_options=True（默认）时，argparse_adapter 自动注入这些选项。
# Web UI 把 log_level 作为 convention field 暴露在 ui_props 中（不是 Param，
# 因为 cli/ 不知道 UI 概念）；LOG_LEVEL_UI / LOG_LEVEL_UI_FULL 共享常量
# 在 ui_props/__init__.py 中维护，choices 来自 cli.spec_cli.STANDARD_LOG_LEVELS，
# 各模块通过 .override(group=..., default=...) 表达差异。

QUANTIZE_LINEAR_ACTION = Param(
    name="quantize-linear-action",
    data_type="string",
    default=QuantizeLinearAction.W8A8_DYNAMIC,
    choices=list(QuantizeLinearAction),
    group="Quantization Options",
    cli_help="Quantize all linear layers (symmetric quant).",
)

# ── Unified speculative decoding (mtp/dflash/dspark) ───────────────
# Shared by text_generate (6 params) and throughput_optimizer (adds
# ACCEPTANCE_LENGTH via override + overrides NUM_SPECULATIVE_TOKENS to
# multi-value for search). Registered in both modules' fields;
# G2/G3 cross-field checks live in cli/registry/validators.py
# (draft_dependents_require_method, draft_mtp_mutex, etc.).

SPECULATIVE_METHOD = Param(
    name="speculative-method",
    data_type="string",
    default=None,
    choices=["mtp", "dflash", "dspark"],
    group="Request",
    cli_metavar="{mtp,dflash,dspark}",
    cli_help="Enable speculative decoding: mtp, dflash, or dspark. "
    "Mutually exclusive with the legacy MTP entry (--num-mtp-tokens). "
    "Required before speculative-dependent options.",
)

NUM_SPECULATIVE_TOKENS = Param(
    name="num-speculative-tokens",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method. Number of speculative tokens excluding anchor/bonus "
    "(vLLM-aligned). When >= 1, internal block_size = n + 1. "
    "Omitting keeps builtin block_size for dflash/dspark. "
    "mtp always requires an explicit value; explicit 0 with --speculative-method is rejected.",
)

ACCEPTANCE_LENGTH = Param(
    name="acceptance-length",
    data_type="number",
    default=5.0,
    group="Model & Quantization Options",
    cli_help="Requires --speculative-method. Decode fold scalar. "
    "Clamped to num_speculative_tokens (n) for all methods.",
)

NUM_DRAFT_LAYERS = Param(
    name="num-draft-layers",
    data_type="integer",
    default=0,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method dflash or dspark. Override draft num_hidden_layers from builtin/config. "
    "0 = use config default. Not allowed with mtp.",
)

DRAFT_MODEL_CONFIG_PATH = Param(
    name="draft-model-config-path",
    data_type="string",
    default=None,
    group="Request",
    cli_help="Requires --speculative-method dflash or dspark. Optional path to override builtin draft config.json. "
    "Not allowed with mtp.",
)

DSPARK_MARKOV_RANK = Param(
    name="dspark-markov-rank",
    data_type="integer",
    default=256,
    min=0,
    group="Request",
    cli_metavar="<N>",
    cli_help="Requires --speculative-method dspark. Markov embedding rank (0 disables MarkovHead). Default: 256.",
)

DSPARK_MARKOV_HEAD = Param(
    name="dspark-markov-head",
    data_type="string",
    default="vanilla",
    choices=["vanilla", "gated", "rnn"],
    group="Request",
    cli_metavar="{vanilla,gated,rnn}",
    cli_help="Requires --speculative-method dspark. Markov head type: vanilla (default), gated, or rnn.",
)
```

```python
# cli/registry/modules/text_generate.py — 纯参数定义，无 Web UI 内容
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from cli.registry import shared as S
from cli.registry import validators as V

# Module-specific parameters (kebab-case name; only CLI-focused attrs)
NUM_QUERIES = Param(
    name="num-queries",
    data_type="integer",
    required=True,
    min=1,
    group="Request",
    cli_metavar="<N>",
    cli_help="Number of parallel inference queries to execute in a single batch.",
)

QUERY_LENGTH = Param(
    name="query-length",
    data_type="integer",
    required=True,
    min=1,
    group="Request",
    cli_metavar="<N>",
    cli_help="Length (in tokens) of new input sequence for each query.",
)

PREFIX_CACHE_HIT_RATE = Param(
    name="prefix-cache-hit-rate",
    data_type="number",
    default=0.0,
    min=0,
    exclusive_max=1,              # [0, 1) 开区间，精确表达
    group="Request",
    cli_help="Prefix cache hit rate for prefill token reuse in [0, 1). [default: 0.0]",
)

DISABLE_REPETITION = Param(
    name="no-repetition",
    data_type="boolean",
    default=False,
    cli_action="store_true",
    cli_aliases=("disable-repetition",),
    cli_dest="disable_repetition",  # Backward compat: downstream code uses this attr name
    group="Request",
    cli_help="Do not reuse repeated transformer layers to save runtime cost. [default: off]",
)
# ... 其余 ~50 个 module-specific Params

SPEC = ModuleSpec(
    module_id="text_generate",
    title="Text Generation",
    cli_prog="msmodeling inference text-generate",
    cli_description="Run a simulated LLM inference pass and dump the perf result.",
    cli_examples=(
        "# Prefill one query of 128 tokens\n"
        "msmodeling inference text-generate Qwen/Qwen3-32B --num-queries 1 --query-length 128 --device TEST_DEVICE\n"
        "# Decode with tensor parallel\n"
        "msmodeling inference text-generate Qwen/Qwen3-32B --num-queries 8 --query-length 1 --context-length 4096 --decode --tp-size 8"
    ),
    cli_output_help="Metrics table on stdout. Optional chrome trace via --chrome-trace-file.",
    log_options=True,
    requires_model_id=True,
    fields=[
        # General Options (shared)
        S.MODEL_ID,
        S.DEVICE,
        S.NUM_DEVICES,
        S.RESERVED_MEMORY_GB,
        # log_level 不在 fields 中 — 由 log_options=True 自动注入（见 §3.5.1）
        # Request
        NUM_QUERIES,
        QUERY_LENGTH,
        PREFIX_CACHE_HIT_RATE,
        # Quantization
        S.QUANTIZE_LINEAR_ACTION,
        S.QUANTIZE_NON_EXPERT_LINEAR_ACTION,
        S.MXFP4_GROUP_SIZE,
        S.QUANTIZE_ATTENTION_ACTION,
        # Draft speculative decoding — G2/G3 validators 需要 wants_provided=True
        S.SPECULATIVE_METHOD,
        S.NUM_SPECULATIVE_TOKENS,
        S.NUM_DRAFT_LAYERS,
        S.DRAFT_MODEL_CONFIG_PATH,
        S.DSPARK_MARKOV_RANK,
        S.DSPARK_MARKOV_HEAD,
        DISABLE_REPETITION,  # cli_dest="disable_repetition" (backward compat)
        # ... 其余 ~40 个字段（Optimization / Debug / Parallelism / Expert / MultiModal）
    ],
    validators=[
        ValidatorRef(name="productEqNumDevices", fn=V.product_eq_num_devices),
        ValidatorRef(name="moeProductEqNumDevices", fn=V.moe_product_eq_num_devices),
        ValidatorRef(name="perLayerProductEqNumDevices", fn=V.per_layer_product_eq_num_devices),
        ValidatorRef(name="sharedExpertMutex", fn=V.shared_expert_mutex),
        ValidatorRef(name="effectiveLenGe1", fn=V.effective_len_ge1),
        # G2/G3: 需要区分"显式传默认值" vs "未传"，用 wants_provided=True
        ValidatorRef(name="draftDependentsRequireMethod", fn=V.draft_dependents_require_method, wants_provided=True),
        ValidatorRef(name="draftMtpMutex", fn=V.draft_mtp_mutex, wants_provided=True),
        ValidatorRef(name="draftLegacyMtpMutex", fn=V.draft_legacy_mtp_mutex, wants_provided=True),
        ValidatorRef(name="mtpRequiresNumSpeculativeTokens", fn=V.mtp_requires_num_speculative_tokens, wants_provided=True),
        ValidatorRef(name="mtpNoDraftLayers", fn=V.mtp_no_draft_layers),
        ValidatorRef(name="speculativeTokensPositive", fn=V.speculative_tokens_positive, wants_provided=True),
    ],
)
```

```python
# web_ui/backend/services/ui_props/text_generate.py
# Web UI 侧：UI 特有属性 + CLI 不关心的元数据
from web_ui.backend.services.ui_props import I18nText, UIFieldProps

# ── 模块级元数据（CLI 不使用） ──
VERSION = "2.0.0"                              # schema 版本（schema_registry pinning）
# 注意：修改 cli/registry 的 fields 或 validators 后，必须 bump 此 VERSION，
# 否则 schema_registry 会因 hash 变化但版本不变而拒绝注册。
TITLE = I18nText("文本生成", "Text Generation")
RUNNER = "ModelRunner"

GROUP_LABELS: dict[str, I18nText] = {
    # key 与 Param.group 对齐（CLI 英文分组名），value 为 i18n 翻译
    "General Options":       I18nText("通用", "General"),
    "LLM Options":           I18nText("LLM", "LLM"),
    "Optimization Options":  I18nText("优化", "Optimization"),
    "Quantization Options":  I18nText("量化", "Quantization"),
    "Parallelism Options":   I18nText("并行", "Parallelism"),
    "Expert Options":        I18nText("专家并行", "Expert"),
    "Advanced Parallelism":  I18nText("高级并行", "Advanced Parallelism"),
    "MultiModal Options":    I18nText("多模态", "Multimodal"),
    "Debugging Options":     I18nText("调试", "Debug"),
    "Model Options":         I18nText("模型", "Model"),
}

# GROUPS 仅声明折叠行为，label 从 GROUP_LABELS 查找（避免 i18n 重复维护）。
# json_adapter 在输出 JSON 时，将 group_key 通过 GROUP_LABELS 解析为 i18n 文本。
# 兜底：Param.group 未在 GROUP_LABELS 中找到时，直接使用英文原名（前端显示英文）。
# 建议通过 UT 检查所有 Param.group 值是否都有 GROUP_LABELS 条目，开发阶段发现遗漏。
GROUPS = [
    {"group_key": "Expert Options",        "defaultCollapsed": True},
    {"group_key": "Advanced Parallelism",  "defaultCollapsed": True},
    {"group_key": "MultiModal Options",    "defaultCollapsed": True},
    {"group_key": "Debugging Options",     "defaultCollapsed": True},
]

OPTION_SOURCE_REGISTRY = {
    "devices": {"endpoint": "/api/options/devices", "cache": "session"},
}

# ── 交叉校验的 i18n message ──
# CLI 侧 ValidatorRef 只提供函数引用；这里补充前端的 i18n 消息和字段映射
# fields: 校验失败时标红的字段列表（UI 关注点，不在 cli/registry/ 中）
# dependsOn 不需要——前端提交到后端统一校验，不需要知道字段联动关系
VALIDATOR_UI = {
    "productEqNumDevices": {
        "fields": ["tp-size", "dp-size", "pp-size", "num-devices"],
        "message": I18nText(
            "tp_size × dp_size × pp_size 必须等于 num_devices",
            "tp_size × dp_size × pp_size must equal num_devices"),
    },
    "moeProductEqNumDevices": {
        "fields": ["moe-tp-size", "moe-dp-size", "ep-size", "num-devices"],
        "message": I18nText(
            "moe_tp × moe_dp × ep 必须等于 num_devices",
            "moe_tp × moe_dp × ep must equal num_devices"),
    },
}

# ── 每个字段的 UI 特有属性 ──
# 未列入的字段：label fallback 到 Param.name，tooltip fallback 到 Param.cli_help，
# control 由 _infer_control() 根据 data_type 自动推断
UI: dict[str, UIFieldProps] = {
    "model-id": UIFieldProps(
        label=I18nText("模型 ID", "Model ID"),
        tooltip=I18nText("待仿真模型的 HuggingFace 名称或本地路径。",
                         "Standard HuggingFace model name or local path."),
        placeholder=I18nText("如 Qwen/Qwen3-32B", "e.g. Qwen/Qwen3-32B")),
    "device": UIFieldProps(
        label=I18nText("目标设备", "Target Device"),
        option_source={"type": "dynamic", "name": "devices"},
        multi_values=True),                    # ← 多用例对比
    "num-queries": UIFieldProps(
        label=I18nText("查询数量", "Number of Queries"),
        multi_values=True),                    # ← 多用例对比
    # ── 互斥/依赖条件示例 ──
    "enable-shared-expert-tp": UIFieldProps(
        label=I18nText("Enable Shared Expert TP", "Enable Shared Expert TP")),
    "host-external-shared-experts": UIFieldProps(
        label=I18nText("Host External Shared Experts", "Host External Shared Experts"),
        conditions={"enabled": {"not": {"field": "enable-shared-expert-tp", "op": "isTrue"}}}),      # ← 互斥：A 开则 B 禁用
    "mxfp4-group-size": UIFieldProps(
        label=I18nText("MXFP4 Group Size", "MXFP4 Group Size"),
        conditions={"enabled": {"field": "quantize-linear-action", "op": "contains", "value": "mxfp4"}}),  # ← 依赖：选了 MXFP4 才启用
    # ── 覆盖示例：前端与 CLI 行为不同 ──
    "chrome-trace-file": UIFieldProps(
        data_type="boolean",                  # ← 覆盖：CLI 是 str(路径)，前端是 bool(是否下载)
        label=I18nText("Chrome Trace", "Chrome Trace"),
        tooltip=I18nText("启用后自动下载 trace 文件", "Enable to download trace file")),
    "performance-model": UIFieldProps(
        choices=["analytic"],                 # ← 覆盖：CLI 有 [analytic,profiling]，前端限制为子集
        label=I18nText("性能模型", "Performance Model")),
    # ── Convention field: log_level（不是 Param，纯 UI 侧注入） ──
    # log_level 在 CLI 由 spec_cli.add_log_options() 注入，不是业务 Param。
    # 这里用 LOG_LEVEL_UI.override(group=...) 表达模块特定的 default/group 差异。
    "log-level": LOG_LEVEL_UI.override(
        group={"zh": "通用", "en": "General Options"}),
    # ... 所有字段都必须在此 dict 中有条目（UT 强制要求）
    # CLI-only 字段设置 hidden=True，不需要定制的字段使用空 UIFieldProps()
}
```

**关键设计**：

- `cli/registry/` 的 `ModuleSpec` 不含 `version`、不含 i18n `message`
- `version` 在 `ui_props` 侧维护（schema_registry 的版本 pinning）
- 交叉校验的 i18n `message` 在 `ui_props.VALIDATOR_UI` 中维护；前端提交到后端统一校验，无需 `dependsOn`
- CLI 侧的 `ValidatorRef.fn` 返回的错误字符串直接用于 CLI 输出
- `UI: dict[str, UIFieldProps]` 按 `Param.name`（kebab-case）索引，只写 CLI 不关心的 UI 属性
- **属性级兜底**：`UIFieldProps` 条目内的属性（如 `label`、`tooltip`）未设置时，fallback 到 `Param` 默认值（见 §3.3）。**注意**：UT 要求所有字段必须有 `UIFieldProps` 条目（即使为空），因此"字段级兜底"（字段不在 `UI` dict 中）仅用于开发阶段的临时状态
- **互斥/依赖条件**：通过 `UIFieldProps.conditions` 声明，复用前端 `usePredicate.ts` 引擎即时评估启用状态；互斥通过 `not` 取反实现；后端 `ValidatorRef.fn` 作为权威兜底，防止绕过前端的非法提交。**注意**：conditions 中的 field 名使用 kebab-case `Param.name`（如 `"enable-shared-expert-tp"`），与 Param 定义保持一致
- **多用例对比**：通过 `UIFieldProps.multi_values` 标记（如 `device`、`num-queries`、`quantize-linear-action`）。前端支持多选，后端 runner 从 `multi_values=True` 动态收集展开字段（替代当前硬编码的 `_MULTI_CASE_FIELDS`），做笛卡尔积展开
- **前后端字段差异**：`UI` dict 包含所有字段（UT 强制要求），CLI-only 字段设置 `hidden=True`。前端可通过 `UIFieldProps.data_type` / `choices` / `control` / `default` / `required` 覆盖 Param 的默认行为（如 `chrome-trace-file` 从路径改为下载开关，`performance-model` 限制为选项子集，`device` 在前端默认为数组）
- **Convention field**：CLI 公共基础设施（如 `log_level`，由 `spec_cli.add_log_options()` 注入，不是 Param）在 UI 侧通过 `LOG_LEVEL_UI` / `LOG_LEVEL_UI_FULL` 共享常量 + `.override(group=..., default=...)` 表达模块差异。`json_adapter` 识别 `cfg.ui` 中"不在 spec.fields 内"的条目，调用 `_build_convention_field()` 追加到 form JSON（见 §3.5.2）

```python
# cli/registry/modules/throughput_optimizer.py — 复用共享参数
from cli.registry.datatypes import ModuleSpec, Param, ValidatorRef
from cli.registry import shared as S
from cli.registry import validators as V

# Module-specific override: device 在该模块为 data_type="string[]"、nargs="+"
# 的多值参数，且带一个额外的 formal flag --devices（baseline parity）。
# 注意：这不是 DEVICE.override()（data_type 不能 override），而是新建 module-level Param。
DEVICES = Param(
    name="device",
    data_type="string[]",
    nargs="+",
    default=None,
    group="Optional arguments",
    cli_extra_flags=("devices",),
    cli_metavar="<NAME>",
    cli_help="Device profile(s) to evaluate. Multiple values enable cross-hardware summaries.",
)

SPEC = ModuleSpec(
    module_id="throughput_optimizer",
    title="Throughput Optimizer",
    cli_prog="msmodeling inference throughput-optimizer",
    cli_description="Get best throughput for given input/output sequence length and SLO limits ...",
    log_options=True,
    log_options_after_group="General Options",
    requires_model_id=True,
    fields=[
        S.MODEL_ID,
        S.NUM_DEVICES,
        S.RESERVED_MEMORY_GB.override(default=10.0),   # ← 仅覆盖默认值（该模块默认 10GB）
        DEVICES,                                        # ← module-specific 多值 device
        # ... ~40 个字段（Request / Model & Quantization / Performance Model / Service / PD Ratio / ...）
    ],
    validators=[
        ValidatorRef(name="validParallelCombo", fn=V.valid_parallel_combo),
        ValidatorRef(name="effectiveLenGe1", fn=V.effective_len_ge1_to),
        ValidatorRef(name="pdRatioMutexDisagg", fn=V.pd_ratio_mutex_disagg),
        ValidatorRef(name="mtpTokensVsAcceptanceRate", fn=V.mtp_tokens_vs_acceptance_rate),
        ValidatorRef(name="draftDependentsRequireMethod", fn=V.draft_dependents_require_method, wants_provided=True),
        ValidatorRef(name="draftMtpMutex", fn=V.draft_mtp_mutex, wants_provided=True),
        ValidatorRef(name="draftLegacyMtpMutex", fn=V.draft_legacy_mtp_mutex, wants_provided=True),
        ValidatorRef(name="mtpRequiresNumSpeculativeTokens", fn=V.mtp_requires_num_speculative_tokens, wants_provided=True),
        ValidatorRef(name="mtpNoDraftLayers", fn=V.mtp_no_draft_layers),
        ValidatorRef(name="speculativeTokensPositive", fn=V.speculative_tokens_positive, wants_provided=True),
        ValidatorRef(name="mtpNoLegacyAcceptanceRate", fn=V.mtp_no_legacy_acceptance_rate, wants_provided=True),
        ValidatorRef(name="legacyMtpNoAcceptanceLength", fn=V.legacy_mtp_no_acceptance_length, wants_provided=True),
    ],
)
```

### 3.5 适配器

#### 3.5.1 spec_cli — CLI 公共基础设施层（`cli/spec_cli.py`）

`spec_cli.py` 是 #704 MindStudio CLI 规范的公共基础设施层。`argparse_adapter` 是 `spec_cli` 的下游——它把 `Param` 翻译为 `spec_cli` 调用，而不是直接操作 argparse。`spec_cli` 提供：

| 设施 | 用途 |
|------|------|
| `SpecArgumentParser` | 自定义 `ArgumentParser`，重写 `format_help()` 输出 Description/Usage/Required/Optional/Examples/Output 段落布局 |
| `add_option(target, *option_strings, aliases=(), **kw)` | 注册公开选项 + 隐藏废弃别名（一次性 stderr warning） |
| `add_log_options(parser)` | 注入公共日志选项：`--log-level`（含 `--log_level` 别名）/ `-v` / `-q` |
| `add_version_option(parser)` | 注入 `-V/--version`（显示 logo + 版本 + git hash） |
| `parse_args(parser, argv)` | 封装 `parser.parse_args()` + alias deprecation 警告 + `resolve_log_level()` |
| `make_enum_type(enum_cls, option_name)` | 生成 Enum 解析器（接受原生值和 kebab-case 拼写）+ metavar |
| `make_token_type(canonical_values, option_name, store_canonical=...)` | 生成 token 解析器（kebab/snake 都解析，返回规范形式） |
| `STANDARD_LOG_LEVELS` | `("debug", "info", "warning", "error", "critical")` — UI option_source 的事实源 |
| `METAVAR_*` 常量 | `<DIR>` / `<FILE>` / `<N>` / `<FLOAT>` / `<NAME>` / `<SEC>` / `<ID>` / `<RANGE>` |
| `to_kebab(value)` | 通用 kebab-case 转换 |

```python
# cli/spec_cli.py — 关键 API 节选

STANDARD_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")

def add_option(target, *option_strings, aliases=(), **kwargs):
    """Register a public option plus hidden compatibility aliases.

    aliases: tuple of option strings (e.g. ("--model_id",)) without -- prefix
    in Param.cli_aliases; registered with help=SUPPRESS + one-shot deprecation
    warning to stderr. Canonical form is the first --option.
    """
    action = target.add_argument(*option_strings, **kwargs)
    if not aliases:
        return action
    dest = kwargs.get("dest", action.dest)
    alias_kwargs = dict(kwargs)
    alias_kwargs["dest"] = dest
    alias_kwargs["help"] = argparse.SUPPRESS
    alias_kwargs.pop("required", None)
    parser = _root_parser(target)
    canonical = next((opt for opt in option_strings if opt.startswith("--")), option_strings[0])
    for alias in aliases:
        alias_action = target.add_argument(alias, **alias_kwargs)
        alias_action.spec_replacement = canonical
        # Wrap __call__ to emit one-shot deprecation warning on first use
        ...
    return action


class SpecArgumentParser(argparse.ArgumentParser):
    """Spec-compliant help layout:
    Description / Usage / Required arguments / Optional arguments / Examples / Output
    """
    def __init__(self, *args, examples=None, output_help=None, **kwargs):
        kwargs.setdefault("formatter_class", SpecHelpFormatter)
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.output_help = output_help


def make_enum_type(enum_cls, option_name):
    """Parse native enum values; also accept kebab-case spellings.
    Returns (parse_fn, metavar).
    """
    members = list(enum_cls)
    kebab_map = {to_kebab(member.value): member for member in members}
    def parser(value: str):
        if isinstance(value, enum_cls):
            return value
        kebab = to_kebab(value)
        member = kebab_map.get(kebab)
        if member is None:
            allowed = ", ".join(str(item.value) for item in members)
            raise argparse.ArgumentTypeError(f"invalid choice {value!r} for {option_name}")
        return member
    return parser, native_choice_metavar(member.value for member in members)


def parse_args(parser, args=None):
    """parse_args + warn deprecated aliases + resolve_log_level."""
    namespace = parser.parse_args(args)
    warn_deprecated_from_argv(parser, args)
    resolve_log_level(namespace, argv=list(sys.argv[1:] if args is None else args))
    return namespace
```

**设计要点**：

- `argparse_adapter` 不直接调用 `argparse.ArgumentParser.add_argument()`，而是通过 `spec_cli.add_option()` 注册，自动获得 alias deprecation 机制
- `log_level` 不是 Param，而是 `spec_cli.add_log_options()` 注入的公共基础设施。`ModuleSpec.log_options=True`（默认）时由 `build_argparser()` 自动注入，位置通过 `log_options_after_group` / `log_options_after_field` 精细控制（匹配各 baseline 的 --help 顺序）
- `SpecArgumentParser.format_help()` 输出固定段落布局（Description/Usage/Required/Optional/Examples/Output），与 #704 MindStudio CLI 规范一致
- `make_enum_type()` 让 Enum 选项同时接受原生值（如 `"W8A8_DYNAMIC"`）和 kebab-case（`"w8a8-dynamic"`），返回 Enum 实例
- `STANDARD_LOG_LEVELS` 是 CLI 和 Web UI option_source 的唯一事实源（ui_props 的 `_log_level_options()` 派生）

#### 3.5.2 argparse 适配器（CLI 侧，`cli/registry/argparse_adapter.py`）

```python
# cli/registry/argparse_adapter.py
# 薄层：把 Param 翻译为 cli.spec_cli 调用，不直接操作 argparse

def build_argparser(spec: ModuleSpec) -> spec_cli.SpecArgumentParser:
    """ModuleSpec → SpecArgumentParser（spec_cli 规范布局）。"""
    parser = spec_cli.SpecArgumentParser(
        prog=spec.cli_prog,
        description=spec.cli_description or spec.title,
        examples=spec.cli_examples,
        output_help=spec.cli_output_help,
    )
    spec_cli.add_version_option(parser)
    groups: dict[str, argparse._ArgumentGroup] = {}
    log_options_added = False
    log_options_after = spec.log_options_after_group or "General Options"

    for p in spec.fields:
        grp_name = p.group or "Optional arguments"
        # 字段级 log_options 插入点（image_generate baseline 把 log 选项放在特定 flag 之后）
        if (spec.log_options and not log_options_added
                and spec.log_options_after_field is not None
                and p.py_name == spec.log_options_after_field):
            grp = groups.setdefault(grp_name, parser.add_argument_group(grp_name))
            _register_param(grp, p)
            spec_cli.add_log_options(parser)
            log_options_added = True
            continue
        # 组级 log_options 插入点（text_generate / throughput_optimizer / video_generate）
        if (spec.log_options and not log_options_added
                and grp_name != log_options_after
                and log_options_after in groups):
            spec_cli.add_log_options(parser)
            log_options_added = True
        grp = groups.setdefault(grp_name, parser.add_argument_group(grp_name))
        _register_param(grp, p)

    if spec.log_options and not log_options_added:
        spec_cli.add_log_options(parser)
    return parser


def _register_param(grp, p: Param) -> None:
    """Dispatch: positional (model_id 模式) vs regular flag."""
    if p.cli_positional:
        _register_positional(grp, p)
    else:
        _register_flag(grp, p)


def _register_positional(grp, p: Param) -> None:
    """Post-#704 model_id 模式：
      1. 注册 nargs="?" 位置参数
      2. 若 cli_flag 设置：注册 formal --flag(s) 同 dest
         - --model-path（来自 cli_flag，隐藏 --help）
         - --model-id（硬编码兄弟项，可见 --help）
      3. cli_aliases 为废弃名（一次性 warning）
      4. required 不传给 argparse；parse_module_args() 调 require_model_id() 延迟校验
    """
    positional_kw = _build_kwargs(p, is_positional=True)
    positional_name = f"{p.py_name}_positional" if p.cli_flag else p.dest_name
    if p.nargs == "?":
        positional_kw["default"] = argparse.SUPPRESS  # 防止位置默认值覆盖 flag 值
    grp.add_argument(positional_name, **positional_kw)

    if p.cli_flag:
        alias_flags = tuple(f"--{a}" for a in p.cli_aliases)
        flag_kw = _build_kwargs(p, is_positional=False)
        flag_kw.pop("required", None)
        flag_kw.pop("nargs", None)
        if p.cli_flag == "model-path":
            # --model-path 隐藏；--model-id 可见（附 deprecated --model_id 别名）
            hidden_kw = dict(flag_kw)
            hidden_kw["help"] = argparse.SUPPRESS
            spec_cli.add_option(grp, "--model-path", dest=p.dest_name, **hidden_kw)
            visible_help = getattr(p, "cli_flag_help", None) or (
                "Model source. Recommended safe mode: a reviewed absolute local model path. "
                "Equivalent to the positional model id.")
            visible_kw = dict(flag_kw)
            visible_kw["help"] = visible_help
            spec_cli.add_option(grp, "--model-id", dest=p.dest_name,
                                aliases=alias_flags, **visible_kw)
        else:
            spec_cli.add_option(grp, f"--{p.cli_flag}", dest=p.dest_name,
                                aliases=alias_flags, **flag_kw)


def _register_flag(grp, p: Param) -> None:
    """常规 --flag 参数（常见情况）。"""
    formal_flag = f"--{p.name}"
    extra_formal_flags = tuple(f"--{a}" for a in p.cli_extra_flags)
    alias_flags = tuple(f"--{a}" for a in p.cli_aliases)
    kw = _build_kwargs(p, is_positional=False)
    spec_cli.add_option(
        grp, formal_flag,
        *extra_formal_flags,
        *([f"-{p.cli_short}"] if p.cli_short else []),
        dest=p.dest_name,
        aliases=alias_flags,
        **kw,
    )
    # Boolean 双态切换：--no-X (formal) + --X (off-flag)
    if p.cli_off_flag:
        off_help = p.cli_off_help or f"Enable {p.cli_off_flag.replace('-', ' ')} (default)."
        grp.add_argument(f"--{p.cli_off_flag}", dest=p.dest_name,
                         action="store_false", default=argparse.SUPPRESS, help=off_help)


def parse_module_args(spec, parser, argv=None):
    """统一解析入口：spec_cli.parse_args + spec.validators + require_model_id。
    CLI main() 调此函数而不是 parser.parse_args()。
    """
    from cli.utils import require_model_id

    args = spec_cli.parse_args(parser, argv)

    # RFC §3.7: 跨字段校验器。dict 以 kebab-case Param.name 为 key。
    # ``provided`` = argv 中显式出现的 kebab-case 名集合（用于 wants_provided=True 校验器）
    if spec.validators:
        params = {p.name: getattr(args, p.dest_name, None) for p in spec.fields}
        provided = _explicitly_provided(spec, parser, args, argv)
        for v in spec.validators:
            msg = v.fn(params, provided) if v.wants_provided else v.fn(params)
            if msg is not None:
                parser.error(msg)  # print stderr + exit(2) — argparse 风格

    if spec.requires_model_id:
        require_model_id(parser, args)
    return args


def _build_kwargs(p: Param, *, is_positional: bool) -> dict:
    """Param → add_argument() kwargs。"""
    kw: dict = {}

    # type: 自定义 > Enum/make_enum_type > string[]/make_token_type > 从 data_type 自动推断
    if p.cli_type:
        kw["type"] = p.cli_type
    elif p.choices and _is_enum_list(p.choices):
        enum_cls = type(p.choices[0])
        parse_fn, metavar = spec_cli.make_enum_type(enum_cls, f"--{p.name}")
        kw["type"] = parse_fn
        kw["metavar"] = metavar
    elif p.choices and p.data_type == "string[]":
        parse_fn, metavar = spec_cli.make_token_type(
            list(p.choices), f"--{p.name}", store_canonical="snake")
        kw["type"] = parse_fn
        kw["metavar"] = metavar
    elif p.data_type == "integer":
        if any(v is not None for v in (p.min, p.max, p.exclusive_min, p.exclusive_max)):
            kw["type"] = _int_type(p.min, p.max, p.exclusive_min, p.exclusive_max)
        else:
            kw["type"] = int
    elif p.data_type in ("integer[]", "number[]", "number"):
        # 数组类型：对每个元素应用 int/float 转换
        ...
    elif p.data_type == "string":
        if p.pattern is not None or p.max_length is not None:
            kw["type"] = _string_type(p.pattern, p.max_length)

    # choices: DEVICE 特殊处理（运行时 lookup）
    if p.name == "device" and p.data_type == "string":
        from .shared import get_device_choices
        kw["choices"] = get_device_choices()
    elif p.choices and not (p.data_type == "string[]" and not _is_enum_list(p.choices)):
        kw["choices"] = list(p.choices) if not _is_enum_list(p.choices) else _enum_values(p.choices)

    if p.nargs is not None:
        kw["nargs"] = p.nargs

    # action / default / required
    if p.data_type == "boolean":
        kw["action"] = p.cli_action if p.cli_action != "store" else "store_true"
        if not is_positional and p.default is not None:
            kw["default"] = p.default
    else:
        if p.cli_action != "store":
            kw["action"] = _resolve_custom_action(p.cli_action) if p.cli_action in ("batch_range",) else p.cli_action
        if not is_positional:
            if p.required:
                kw["required"] = True
            if p.default is not None:
                kw["default"] = p.default

    if p.cli_metavar:
        kw["metavar"] = p.cli_metavar
    if p.cli_help:
        kw["help"] = p.cli_help
    return kw
```

#### 3.5.3 JSON 适配器（Web UI 侧，`web_ui/backend/services/json_adapter.py`）

以 `Param` 为基础，注入 `UIFieldProps` 和 Web UI 属性，生成前端 form JSON。
另外识别 `cfg.ui` 中"不在 spec.fields 内"的 convention field（如 `log-level`），
通过 `_build_convention_field()` 追加到 form JSON（与 baseline form 对齐）。

```python
# web_ui/backend/services/json_adapter.py
from cli.registry.datatypes import ModuleSpec, Param
from web_ui.backend.services.ui_props import UIFieldProps, I18nText, _UNSET


# ── 模块级配置（来自 ui_props/*.py） ────────────────────────────

@dataclass
class FormConfig:
    """单个模块的 Web UI 配置，由 ui_props 模块文件组装后传入。"""
    version: str                                        # schema 版本
    title: I18nText                                     # 模块标题 i18n
    runner: str                                         # 前端 runner 类名
    ui: dict[str, UIFieldProps] = field(default_factory=dict)
    group_labels: dict[str, I18nText] = field(default_factory=dict)
    groups: list[dict] = field(default_factory=list)
    option_source_registry: dict = field(default_factory=dict)
    validator_ui: dict[str, dict] = field(default_factory=dict)


# ── 主入口 ──────────────────────────────────────────────────────

def export_form_json(spec: ModuleSpec, cfg: FormConfig) -> dict:
    """ModuleSpec + FormConfig → 前端 form JSON。

    fields 来自 spec.fields（SSOT）；此外 cfg.ui 中不在 spec.fields 内的条目
    （convention fields，如 log_level — 由 spec_cli.add_log_options 提供的 CLI
    基础设施，不是注册表 Param）会通过 _build_convention_field() 追加，
    与 baseline form 对齐。
    """
    fields = [_build_field(p, cfg) for p in spec.fields]
    spec_names = {p.name for p in spec.fields}
    for fid, ui in cfg.ui.items():
        if fid in spec_names:
            continue  # 常规字段，上面已经从 Param 构建
        f = _build_convention_field(fid, ui, cfg)
        if f:
            fields.append(f)

    return {
        "$schema": "form-schema/v1",
        "moduleId": spec.module_id,
        "title": _i18n(cfg.title),
        "version": cfg.version,
        "runner": cfg.runner,
        "optionSourceRegistry": cfg.option_source_registry,
        "formValidation": _build_form_validation(spec, cfg),
        "groups": cfg.groups,
        "fields": fields,
    }


# ── 内部构建函数 ────────────────────────────────────────────────

def _build_form_validation(spec: ModuleSpec, cfg: FormConfig) -> list[dict]:
    """交叉校验 → formValidation[]。"""
    result = []
    for v in spec.validators:
        vu = cfg.validator_ui.get(v.name, {})
        msg = vu.get("message")
        # msg 是 I18nText 则转换为 dict；否则用 v.name 作为 fallback
        message = _i18n(msg) if isinstance(msg, I18nText) else {"zh": v.name, "en": v.name}
        result.append({
            "rule": "validator",
            "value": v.name,
            "message": message
        })
    return result


def _infer_control(data_type: str, choices, option_source) -> str:
    """根据 data_type 和选项自动推断前端控件。
    boolean → "switch"
    string[] with choices/option_source → "multi-select"
    string[] without → "text" (逗号分隔输入)
    integer[]/number[] → "text"
    string/integer/number with choices/option_source → "select"
    integer/number without → "number" (numeric spinner)
    string without → "text"
    """
    if data_type == "boolean":
        return "switch"
    if data_type.endswith("[]"):
        return "multi-select" if (choices or option_source) else "text"
    if choices or option_source:
        return "select"
    if data_type in ("integer", "number"):
        return "number"
    return "text"


def _build_field(p: Param, cfg: FormConfig) -> dict:
    """Param + UIFieldProps → 前端 field dict。"""
    ui = cfg.ui.get(p.name, UIFieldProps())

    # data_type / choices / default / required 可能被 UIFieldProps 覆盖
    data_type = ui.data_type or p.data_type
    choices = ui.choices if ui.choices is not None else p.choices
    default = p.default if ui.default is _UNSET else ui.default
    required = ui.required if ui.required is not None else p.required

    field: dict = {"id": p.name, "dataType": data_type, "default": default}

    # i18n 属性（带 fallback）
    _set_if(field, "label",       _i18n_or(ui.label,       I18nText(p.name, p.name)))
    _set_if(field, "tooltip",     _i18n_or(ui.tooltip,     I18nText(p.cli_help, p.cli_help) if p.cli_help else None))
    _set_if(field, "placeholder", _i18n(ui.placeholder) if ui.placeholder else None)
    # group: 字段级 ui.group 覆盖优先（baseline parity — 同一个 Param group key
    # 在不同模块可能需要不同的 i18n 标签）；否则通过 GROUP_LABELS 映射 Param.group
    if ui.group:
        field["group"] = ui.group
    else:
        _set_if(field, "group", _resolve_group(p.group, cfg.group_labels))

    # UI 控制属性（control 自动从 data_type 推断；ui.control 显式覆盖）
    control = ui.control or _infer_control(data_type, choices, ui.option_source)
    field["control"] = control
    _set_if(field, "optionSource", ui.option_source)
    _set_if(field, "disabled",     True if ui.disabled else None)
    _set_if(field, "hidden",       True if ui.hidden else None)
    _set_if(field, "conditions",   ui.conditions)
    _set_if(field, "multiValues",  True if ui.multi_values else None)
    _set_if(field, "clearable",    True if ui.clearable else None)

    if choices:
        field["options"] = _build_options(choices)

    # 声明式约束 → async-validator rules（带 type 推断）
    label_zh = field.get("label", {}).get("zh", p.name)
    label_en = field.get("label", {}).get("en", p.name)
    field["validation"] = _build_validation(p, label_zh, label_en, required, data_type)
    return field


def _build_validation(p: Param, label_zh: str, label_en: str, required: bool = False, data_type: str = "string") -> list[dict]:
    """Param 的声明式约束 → async-validator rules（双语 message）。
    required 可被 UIFieldProps 覆盖。
    data_type 驱动 async-validator 的 `type`：不指定的话 min/max 会比较字符串长度
    （min:1 on num_queries=0 会通过，因为 len("0")==1）。
    """
    # 数值范围规则需要 type: 'number'（接受整数和浮点）
    range_type = "number" if data_type in ("integer", "number") else None
    if data_type.endswith("[]"):
        range_type = "array"
    rules = []
    if required:
        rule: dict = {"rule": "required", "message": {"zh": f"{label_zh}为必填项", "en": f"{label_en} is required"}}
        if data_type.endswith("[]"):
            rule["type"] = "array"
        # NOTE: required 规则不添加 type:"number" — 前端 text 控件存储字符串，
        # async-validator 的 type:"number" 会对字符串 "4" 报类型不匹配。
        # min/max/gt/lt 规则仍使用 range_type（async-validator 对 range 规则做 coercion）。
        rules.append(rule)
    if p.min is not None:
        _r = {"rule": "min", "value": p.min, "message": {"zh": f"必须 ≥ {p.min}", "en": f"must be ≥ {p.min}"}}
        if range_type:
            _r["type"] = range_type
        rules.append(_r)
    if p.max is not None:
        _r = {"rule": "max", "value": p.max, "message": {"zh": f"必须 ≤ {p.max}", "en": f"must be ≤ {p.max}"}}
        if range_type:
            _r["type"] = range_type
        rules.append(_r)
    if p.exclusive_min is not None:
        _g = {"rule": "gt", "value": p.exclusive_min, "message": {"zh": f"必须 > {p.exclusive_min}", "en": f"must be > {p.exclusive_min}"}}
        if range_type and range_type != "array":
            _g["type"] = range_type
        rules.append(_g)
    if p.exclusive_max is not None:
        _l = {"rule": "lt", "value": p.exclusive_max, "message": {"zh": f"必须 < {p.exclusive_max}", "en": f"must be < {p.exclusive_max}"}}
        if range_type and range_type != "array":
            _l["type"] = range_type
        rules.append(_l)
    if p.pattern:
        rules.append({"rule": "pattern", "value": p.pattern, "message": {"zh": "格式不匹配", "en": "format mismatch"}})
    if p.max_length is not None:
        # async-validator 没有 'length' rule type — 字符串最大长度用 type:'string' + max (字符数)
        rules.append({"rule": "max", "type": "string", "value": p.max_length,
                       "message": {"zh": f"长度不能超过 {p.max_length} 字符",
                                   "en": f"length must be ≤ {p.max_length} characters"}})
    return rules

# 注：gt/lt 是本次 RFC 新增的声明式 rule，前端 useFormValidation.ts 已扩展支持。
# 原 RFC 预估的前端改动点之一（见 §3.8 构建流程）。


# ── 工具函数 ────────────────────────────────────────────────────

def _build_options(choices: list) -> list[dict]:
    """choices → [{value, label}] 格式（RFC §3.8.1 kebab decision 落地）。

    处理三种情况：
    1. Enum 成员 → value=to_kebab(member.value), label=member.name
    2. 预格式化 dict → 直接透传（如 {"value": "w8a8-dynamic", "label": "W8A8_DYNAMIC"}）
    3. 纯字符串 → value=label=字符串本身

    前端 SchemaFormItem.vue 渲染 select/multi-select 使用此格式。
    """
    opts = []
    for c in choices:
        if isinstance(c, dict):
            opts.append(c)  # Pre-formatted — pass through
        elif hasattr(c, "value") and hasattr(c, "name"):
            # Enum member
            opts.append({"value": to_kebab(c.value), "label": c.name})
        else:
            opts.append({"value": c, "label": str(c)})
    return opts


def _i18n(t: I18nText | None) -> dict | None:
    """I18nText → {"zh": ..., "en": ...}。"""
    return {"zh": t.zh, "en": t.en} if t else None

def _i18n_or(t: I18nText | None, fallback: I18nText | None) -> dict | None:
    """I18nText → dict，未设置时用 fallback。"""
    return _i18n(t) or _i18n(fallback)

def _resolve_group(group: str | None, labels: dict[str, I18nText]) -> dict | None:
    """CLI 英文分组名 → i18n dict；未在 labels 中则用英文原名。"""
    if not group:
        return None
    i18n = labels.get(group)
    return _i18n(i18n) if i18n else {"zh": group, "en": group}

def _set_if(d: dict, key: str, value) -> None:
    """仅当 value 非 None 时写入 dict，避免输出 null 字段。"""
    if value is not None:
        d[key] = value
```

**调用示例**（`ui_props/text_generate.py` 中组装配置）：

```python
# web_ui/backend/services/ui_props/text_generate.py

from cli.registry.modules import get_spec
from web_ui.backend.services.json_adapter import FormConfig, export_form_json

spec = get_spec("text_generate")

config = FormConfig(
    version="2.0.0",
    title=I18nText("文本生成", "Text Generation"),
    runner="ModelRunner",
    ui=UI,                          # dict[str, UIFieldProps]
    group_labels=GROUP_LABELS,      # dict[str, I18nText]
    groups=GROUPS,
    option_source_registry=OPTION_SOURCE_REGISTRY,
    validator_ui=VALIDATOR_UI,
)

form_json = export_form_json(spec, config)
```

### 3.6 后端校验

提交时后端执行 **L1 声明式约束 + L2 交叉校验**，均以 **case** 为单位：

```text
Web UI 校验流程：

1. 前端提交 params（可能含多值字段，如 device=["A","B"]、num-queries=[1,8]）
2. API 层展开笛卡尔积 → cases[]（每个 case 为标量 params dict）
3. 计算 provided set（job 级别，所有 case 共享）
4. 对每个 case：
   a. 类型转换（string → int/float，对齐 Param.data_type）
   b. L1：_check_declarative(spec, case_params) — 声明式约束检查
   c. L2：遍历 spec.validators，调用 v.fn(case_params, provided) 或 v.fn(case_params)
   d. 任一校验失败 → 返回 str 错误消息 → 400 响应
5. 全部 case 通过 → 继续创建 job

CLI 流程（对称设计，天然 case 级别——CLI 无多值展开）：
1. main() 调用 parse_module_args()
2. parse_module_args() 内部：
   a. spec_cli.parse_args()（argparse type 函数天然执行 L1）
   b. spec.validators（L2 交叉校验）
3. 校验失败 → parser.error(msg) → print stderr + exit(2)
```

**`provided` 集合计算**：

对于 `wants_provided=True` 的校验器（G2/G3 投机解码约束），需要区分"显式传了默认值" vs "未传"：

- **CLI 侧**（`argparse_adapter._explicitly_provided()`）：扫描 argv tokens，精确判断哪些 flag 在命令行中出现
- **Web UI 侧**（`jobs.py._compute_provided_set()`）：**job 级别**计算，所有 case 共享同一份。优先使用前端发送的 `explicitly_touched` 列表（用户交互了哪些字段）；fallback 到 `val != field.default` 比较（取原始多值 params，任一值与默认不同即视为显式设置）。正常场景下逻辑正确；边界场景（前端发送类型不匹配值如 `"0"` vs `0`）有假阳性风险，待观察前端实际序列化行为

**多值字段校验**：API 层先展开笛卡尔积，对**每个 case** 独立执行 L1 + L2。任一 case 失败则返回 400。L1 和 L2 接收的均为标量值——L2 交叉校验（如 `tp_size × dp_size × pp_size == num_devices`）在标量间进行，语义明确。

**完整性校验 UT**：`ValidatorRef.name` 与 `VALIDATOR_UI` 的一致性通过单元测试保证，放在 `tests/test_registry/test_integrity.py`：

```python
# tests/test_registry/test_integrity.py
"""校验 cli/registry 与 ui_props 的一致性。

防止以下静默失败场景：
- cli/registry 新增字段/校验器但 ui_props 漏写对应条目
- name typo（如 "productEqNumDevice" vs "productEqNumDevices"）
- VALIDATOR_UI.fields 引用不存在的字段

CI 通过 pytest 自动执行，失败即阻断。
"""
import pytest
from cli.registry.modules import get_spec
from web_ui.backend.services.ui_props import get_validator_ui, get_ui


# 注意：parametrize 仅覆盖三个已暴露 Web UI 的模块。
# image_generate 已接入 registry（CLI 侧），但 Web UI runner adapter 暂未实现，
# BUNDLED_MODULES 不含 image_generate，因此 UI 完整性测试不适用。
# 待 image_generate Web UI 表单暴露后，将其加入 parametrize 列表。

@pytest.mark.parametrize("module_id", [
    "text_generate", "video_generate", "throughput_optimizer",
])
def test_all_fields_have_ui_props(module_id):
    """spec.fields 中每个字段必须在 UI dict 中有 UIFieldProps 条目。

    **目的**：强制开发者显式考虑每个字段的 UI 属性（i18n label/tooltip 等），
    防止新增字段后忘记配置，导致前端显示 snake_case 原始 id 作为 label。

    **注意**：
    - 不需要 UI 展示的字段设置 hidden=True（仍需条目）
    - 即使字段不需要任何 UI 定制，也需添加空条目 `UIFieldProps()`
    - json_adapter 有兜底逻辑（字段不在 UI dict 时使用 Param 默认值），
      但此 UT 强制要求显式配置，兜底仅用于开发阶段的临时状态
    """
    spec = get_spec(module_id)
    ui = get_ui(module_id)
    for p in spec.fields:
        assert p.name in ui, (
            f"Field '{p.name}' in {module_id} has no UIFieldProps entry. "
            f"Add it to UI dict (set hidden=True if CLI-only, or use empty UIFieldProps() if no customization needed)."
        )


@pytest.mark.parametrize("module_id", [
    "text_generate", "video_generate", "throughput_optimizer",
])
def test_validator_has_ui_entry(module_id):
    """每个 ValidatorRef 必须在 VALIDATOR_UI 中有对应条目。"""
    spec = get_spec(module_id)
    validator_ui = get_validator_ui(module_id)
    for v in spec.validators:
        assert v.name in validator_ui, (
            f"ValidatorRef '{v.name}' has no VALIDATOR_UI entry in {module_id}"
        )


@pytest.mark.parametrize("module_id", [
    "text_generate", "video_generate", "throughput_optimizer",
])
def test_validator_ui_fields_exist(module_id):
    """VALIDATOR_UI.fields 引用的字段必须存在于 spec.fields 中。"""
    spec = get_spec(module_id)
    validator_ui = get_validator_ui(module_id)
    field_ids = {p.name for p in spec.fields}
    for v in spec.validators:
        vu = validator_ui.get(v.name)
        if not vu:
            continue  # 由 test_validator_has_ui_entry 捕获
        for f in vu.get("fields", []):
            assert f in field_ids, (
                f"VALIDATOR_UI['{v.name}'].fields references unknown field '{f}' in {module_id}"
            )


@pytest.mark.parametrize("module_id", [
    "text_generate", "video_generate", "throughput_optimizer",
])
def test_ui_choices_is_subset_of_param_choices(module_id):
    """UIFieldProps.choices 必须是 Param.choices 的子集。

    防止前端出现 CLI 不接受的选项，导致提交 400 错误。
    """
    spec = get_spec(module_id)
    ui = get_ui(module_id)
    for p in spec.fields:
        u = ui.get(p.name)
        if u and u.choices is not None and p.choices is not None:
            # 提取 Enum 的 value 进行比较
            param_vals = {getattr(c, "value", c) for c in p.choices}
            ui_vals = {getattr(c, "value", c) for c in u.choices}
            assert ui_vals <= param_vals, (
                f"UIFieldProps.choices for '{p.name}' contains values "
                f"not in Param.choices: {ui_vals - param_vals}"
            )
```

### 3.7 CLI 集成变更

```python
# cli/inference/text_generate.py — 重构前（~200 行 argparse 定义）
def main():
    common_parser = get_common_argparser()
    parser = argparse.ArgumentParser(parents=[common_parser])
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument("--num-queries", type=check_positive_integer, required=True, ...)
    llm_group.add_argument("--query-length", type=check_positive_integer, required=True, ...)
    # ... 50+ 行 add_argument()

# cli/inference/text_generate.py — 重构后（~5 行）
from cli.registry.modules import get_spec
from cli.registry.argparse_adapter import build_argparser, parse_module_args

def main(argv=None):
    spec = get_spec("text_generate")
    parser = build_argparser(spec)
    args = parse_module_args(spec, parser, argv)
    # parse_module_args 内部：spec_cli.parse_args (alias warning + log resolve)
    #   → spec.validators (跨字段校验, parser.error on fail)
    #   → cli.utils.require_model_id (延迟必填校验)
    # ... 后续逻辑不变
```

### 3.8 构建流程

```text
当前：  TS source → gen-form-schemas.mjs (esbuild+node) → JSON

重构后：cli/registry/ → web_ui/backend/services/json_adapter.py → JSON
        CI check: python -m web_ui.backend.services.json_adapter --check
```

- `gen-form-schemas.mjs` 废弃，`.ts` form config 文件删除
- `_validators.ts` 中的 **validator 函数全部删除**（~300 行）；option 常量迁移到 registry 自动生成
- 前端 `SchemaForm.vue` / `SchemaFormItem.vue` 渲染组件**不变**（已支持声明式规则）
- 前端 `useFormValidation.ts` 需**小幅扩展**：新增 `gt`/`lt` rule 分支（用于 `exclusive_min`/`exclusive_max` 声明式约束），现有 `min`/`max`/`pattern`/`required` 分支不变

### 3.9 特殊字段处理策略

部分字段在 CLI 和 Web UI 中语义不同（如 `chrome-trace-file` CLI 是文件路径，前端是下载开关），或需要仿真后的额外处理（如 `_extract_op_breakdown`、`export-empirical-metrics-file`）。本节说明这类字段在重构后的处理方式。

**核心原则**：不引入额外抽象层（如 PostProcessor 基类），各字段的特殊逻辑保留在 Runner 中，按本质分类处理。

#### 3.9.1 字段分类

| 类型 | 示例 | 本质 | 处理方式 |
|------|------|------|----------|
| **仿真配置开关** | `chrome-trace-file`、`dump-input-shapes`、`dump-op-bound-results` | 影响仿真内部行为，数据在仿真过程中生成 | Runner 合成路径或直接传入 `params`，ModelRunner 内部处理 |
| **结果格式化** | `_extract_op_breakdown()` | 把 metrics 内部结构转为前端可消费的 JSON | Runner 的 `_run_one_case` 中直接做 |
| **仿真后导出** | `export-empirical-metrics-file` | 仿真结束后从 metrics 导出额外数据 | CLI `main()` 或 Runner 直接处理 |

**判断标准**：数据在仿真过程中已收集并包含在 metrics 中 → 不是后处理，只是格式化或配置。不需要统一的 PostProcessor 抽象。

#### 3.9.2 chrome-trace-file 处理

**CLI 行为**：`--chrome-trace-file PATH` — 用户指定文件路径，ModelRunner 内部生成 trace 文件。

**Web UI 行为**：前端 `chrome-trace-file` 为 boolean 开关。用户开启后，Runner 自动合成路径，前端提供下载。

**处理流程（Web UI）**：

```text
前端提交 chrome-trace-file: true
         │
         ▼
Runner._synth_trace_path()
  → 将 True 替换为 legacy_hash_path(job_id, case_hash)
         │
         ▼
ModelRunner 内部生成 trace 文件到合成路径
         │
         ▼
前端 record 中包含下载链接 → 渲染下载按钮
```

`UIFieldProps` 覆盖 `data_type`：

```python
# ui_props/text_generate.py
"chrome-trace-file": UIFieldProps(
    data_type="boolean",    # CLI 是 str(路径)，前端是 bool(开关)
    label=I18nText("Chrome Trace", "Chrome Trace"),
    tooltip=I18nText("启用后自动生成 trace 文件供下载",
                     "Generate trace file for download")),
```

**三 Runner 现状**：

| Runner | 当前处理方式 | 是否需要改动 |
|--------|-------------|:---:|
| `text_generate` | `_synth_trace_path()` → ModelRunner 内部写文件 | 否（保留） |
| `throughput_optimizer` | 同上 | 否（保留） |
| `video_generate` | 仿真后直接调用 `runtime.export_chrome_trace(path)` | 否（保留） |

三种方式均能将 trace 文件写到正确路径，前端下载链接不受影响，无需统一。

> **注意：video_generate 的 chrome-trace-file 处理存在已知差异**
>
> `text_generate` / `throughput_optimizer` 使用 `_synth_trace_path()`：前端发送 `chrome-trace-file: true` 时，Runner 将其替换为 `legacy_hash_path(job_id, case_hash)` 合成路径。
>
> `video_generate` runner 未使用 `_synth_trace_path()`，直接将 `chrome-trace-file` 值转为字符串路径（`str(params["chrome-trace-file"])`）。若前端发送 `True`（boolean），`str(True)` 会产生 `"True"` 作为文件路径——这是一个**已存在的 pre-existing 问题**，不是本次重构引入的。
>
> **处理决策**：当前 video_generate 的 Web UI 表单中 `chrome-trace-file` 字段未被暴露（前端未启用该开关），因此该问题暂不影响实际使用。Phase 3 迁移 video_generate 时，可选择：
>
> 1. 为 video_generate runner 补充 `_synth_trace_path()` 逻辑（与 text_generate 统一）
> 2. 或在前端不为 video_generate 暴露 chrome-trace-file 开关（保持现状）

#### 3.9.3 dump-input-shapes / dump-op-bound-results

**CLI 行为**：`--dump-input-shapes` / `--dump-op-bound-results` — `store_true` 开关，控制结果表格的列。

**Web UI 行为**：前端同样为 boolean 开关。

**处理流程**：

```text
params.get("dump-input-shapes")    ──┐
params.get("dump-op-bound-results") ─┤
                                      ▼
            _extract_op_breakdown(rt, perf_model, dump_input_shapes, dump_op_bound_results)
                                      │
                          ┌───────────┴───────────┐
                          │                       │
              group_by_input_shapes=True   dump_op_bound_results=True
              → 按 input_shapes 分组       → 增加 bound_pct 列
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                          envelope["op_breakdown"] = [...]
                          envelope["dump_input_shapes"] = bool
                          envelope["dump_op_bound_results"] = bool
                                      │
                                      ▼
                          前端根据两个 flag 决定表格列显隐
```

**三 Runner 现状**：

| Runner | 处理方式 | 是否需要改动 |
|--------|---------|:---:|
| `text_generate` | `_extract_op_breakdown()` 接收两个 flag → 生成 op_breakdown | 否（保留） |
| `throughput_optimizer` | 复用 text_generate 的 `_extract_op_breakdown()` | 否（保留） |
| `video_generate` | 无 op_breakdown 表格（视频模型不需要） | 否（不涉及） |

本质：纯仿真配置开关 + 结果格式化。flag 传入 `_extract_op_breakdown()`，控制输出结构；flag 值同时写入 envelope，前端据此决定表格列。不涉及文件输出，无需路径合成。

#### 3.9.4 _extract_op_breakdown()

**本质**：结果格式化函数，不是独立字段。把 `Runtime` 内部聚合数据（`_aggregate_average_table_data`）转为前端可消费的 `list[dict]`。

**输入**：`rt`（Runtime 对象）、`perf_model`（性能模型名）、两个 flag。

**输出**：

```python
[
    {
        "name": "aten.mm",
        "bound": "compute",           # key.bound
        "input_shapes": "(128, 64)",  # key.input_shapes（仅 dump_input_shapes=True）
        "total_s": 0.012,
        "avg_s": 0.003,
        "calls": 4,
        "bound_pct": {                # 仅 dump_op_bound_results=True
            "memory": 30.0,
            "comm": 10.0,
            "mma": 50.0,
            "gp": 10.0,
        },
    },
    ...
]
```

**位置**：`web_ui/backend/runners/text_generate.py:89`。`_run_one_case()` 中直接调用，不需要独立抽象。

#### 3.9.5 export-empirical-metrics-file

**CLI 行为**：`--export-empirical-metrics PATH` — 仿真结束后，从 `EmpiricalPerformanceModel.op_records` 收集 M1-M5 指标，导出 JSON 文件到指定路径。要求 `--performance-model profiling`。

**Web UI 行为**：当前未实现。

**处理流程（CLI，`main()` 中）**：

```text
仿真完成
    │
    ▼
遍历 model_runner.perf_models
    │ 找到 EmpiricalPerformanceModel 实例
    ▼
MetricsCollector().collect_from_records(pm.op_records)
    │
    ▼
collector.export_hit_miss_report(output_path=Path(args.export_empirical_metrics_file))
```

**Web UI 待实现时**：在 Runner 的 `_run_one_case()` 中，仿真完成后添加同样的逻辑：

```python
if params.get("export-empirical-metrics-file"):
    from tensor_cast.performance_model.metrics_collector import MetricsCollector
    for pm in runner.perf_models:
        if isinstance(pm, EmpiricalPerformanceModel):
            collector = MetricsCollector()
            collector.collect_from_records(pm.op_records)
            collector.export_hit_miss_report(output_path=...)
            break
```

本质：仿真后数据导出。依赖仿真过程中的 `op_records`，必须在仿真完成后立即处理。直接在 Runner 或 CLI `main()` 中实现，无需独立抽象。

#### 3.9.6 新增特殊字段的标准流程

1. **`cli/registry/`** 添加 `Param`（定义 CLI 行为）
2. **`ui_props/`** 添加 `UIFieldProps`（如需覆盖 `data_type` / `choices` 等前端行为）
3. **如需校验**：在 `cli/registry/validators.py` 添加校验函数，并在模块文件中注册到 `ModuleSpec.validators`；同时在 `ui_props/VALIDATOR_UI` 中补充 `fields` 和 `message`
4. **Runner** 在 `_run_one_case()` 中添加处理逻辑（仿真配置 → 传入 params；结果格式化 → 直接处理 metrics；仿真后导出 → 在仿真完成后处理）

---

## 4. 测试设计

| 测试类型 | 范围 | 方法 |
|----------|------|------|
| **Registry 单元测试** | `cli/registry/` 中 `Param` 创建/override/序列化 | pytest 参数化 |
| **argparse 适配器测试** | `cli/registry/argparse_adapter.py` 生成 parser → `parse_args()` 与旧 CLI 结果逐字段对比 |  golden file 对比 |
| **JSON 适配器测试** | `web_ui/backend/services/json_adapter.py` 生成 JSON 与现有 TS→JSON 输出字段级 diff | 逐字段等价断言 |
| **validator 测试** | `cli/registry/validators.py` 每个函数的正常/异常/边界 | pytest 参数化 |
| **CLI 回归测试** | 三个已暴露 Web UI 的 CLI 的 `--help` 输出对比、`parse_args` 结果对比 | subprocess 调用 + diff |
| **Web UI E2E 测试** | 表单渲染、提交、校验错误显示 | Playwright |
| **跨模块共享测试** | throughput_optimizer 的共享参数与 text_generate 一致性 | registry 级别断言 |
| **schema version 测试** | `schema_registry.py` 对同版本 hash 变更的拒绝行为 | API 级别测试 |
| **跨适配器一致性测试** | 同一 `Param` 经 argparse_adapter 和 json_adapter 处理后语义一致：枚举值、默认值、min/max 约束、required 行为等价 | 参数化测试：遍历所有 `Param`，对比两端输出 |
| **validator 完整性测试** | `spec.fields` 每个字段都有 `UIFieldProps` 条目；`ValidatorRef.name` 在 `VALIDATOR_UI` 中有对应条目；`fields` 引用的字段存在于 `spec.fields` 中；`UIFieldProps.choices ⊆ Param.choices` | pytest 参数化：`tests/test_registry/test_integrity.py`（见 §3.6） |
| **L1 后端声明式校验测试** | `validate_module()` 的 `_check_declarative()` 对越界数值、非法 choices、不匹配 pattern、超长字符串的拒绝行为；确保绕过前端 L1 的非法提交在后端被拦截 | pytest 参数化：构造非法 params dict，断言返回 errors[] |
| **case 级别校验测试** | L1 + L2 校验以 case 为单位执行：多值字段展开后每个 case 独立校验；L2 交叉校验接收标量值（非数组）；任一 case 失败返回 400 | pytest 参数化：构造多值 params → 展开 → 断言每个 case 独立校验 |

---

## 5. 缺点与风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|----------|
| **Schema hash 变化** | 旧版本 job 无法重开 | 高 | 迁移时 bump version（1.x→2.0），旧版本保留在 SQLite 中 |
| **argparse `--help` 输出格式变化** | 用户习惯受影响 | 中 | 迁移期 golden file 对比，精确控制 `formatter_class` |
| **复杂 validator 前端无实时反馈** | L2 validator（如 `lteNumDevices`、`dividesNumDevices`）从前端实时校验移到后端提交时校验，用户体验确定性降低 | 确定性 | L1 声明式约束仍有即时反馈；L2 提交后返回字段级错误并标红；`conditions` 结构化 predicate 覆盖部分场景的前端实时启用/禁用控制 |
| **`_validators.ts` 删除后前端构建变化** | 前端需调整 import | 低 | `_validators.ts` 的 option 常量由 registry 生成 JSON 提供，前端改为读取 JSON |
| **`_build_namespace` 重构** | throughput_optimizer runner 的 ~120 行类型转换需重写 | 中 | 从 `Param.data_type` + `default` 自动生成类型转换函数 |
| **团队适应新抽象** | 初期学习成本 | 中 | 先迁移 text_generate 做 demo；文档覆盖 |

---

## 6. 迁移策略

分 4 个阶段，逐模块迁移，每阶段可独立交付和验证：

**阶段依赖**：Phase 4（清理）必须在 Phase 2 和 Phase 3 完成后才能开始。Phase 1-3 期间旧文件保留，各模块可独立运行在新旧系统上，互不影响。

| 阶段 | 内容 | 预计代码改动 | 预计工期 |
|------|------|:---:|:---:|
| **Phase 1** | 基础设施：`cli/registry/` 核心 + `argparse_adapter` + `web_ui/backend/services/ui_props/` + `json_adapter` + text_generate 迁移 | ~1200 行新增 / ~300 行删除 | 2 周 |
| **Phase 2** | throughput_optimizer 迁移（复用 Phase 1 的 18 个共享参数） | ~600 行新增 / ~200 行删除 | 1.5 周 |
| **Phase 3** | video_generate 迁移（多数专有参数，验证灵活性） | ~400 行新增 / ~150 行删除 | 1 周 |
| **Phase 4** | 清理：删除旧 TS form configs、`_validators.ts` 函数、`gen-form-schemas.mjs`、`get_common_argparser` | 0 行新增 / ~400 行删除 | 0.5 周 |

**回滚策略**：Phase 1-3 期间旧文件保留，如某阶段上生产后发现问题，可直接 revert 该阶段 commit 回滚。Phase 4 删除旧文件后不可回滚，因此 Phase 4 应在 Phase 1-3 稳定运行至少 1 周后执行。

---

## 7. 工作量估算

### 7.1 代码改动量

| 类别 | 新增 | 删除 | 净增 |
|------|:---:|:---:|:---:|
| 基础设施（`cli/registry/` + `argparse_adapter` + `ui_props/` + `json_adapter`） | ~700 | 0 | +700 |
| Phase 1（text_generate） | ~1200 | ~300 | +900 |
| Phase 2（throughput_optimizer） | ~600 | ~200 | +400 |
| Phase 3（video_generate） | ~400 | ~150 | +250 |
| Phase 4（清理） | 0 | ~400 | -400 |
| **合计** | **~2900** | **~1050** | **+1850** |

### 7.2 验证工作量

| 验证维度 | 预计工期 |
|----------|:---:|
| Registry + adapter 单元测试 | 1 天 |
| argparse 端到端对比测试 | 1.5 天 |
| JSON schema 字段级等价测试 | 1.5 天 |
| Validator 测试 | 1 天 |
| CLI 回归测试 | 1 天 |
| Web UI 前端 E2E 验证 | 2 天 |
| Web UI 后端验证 | 1 天 |
| 跨模块共享验证 | 0.5 天 |
| **验证合计** | **~9.5 天** |

### 7.3 整体汇总

| 项目 | 人天 |
|------|:---:|
| 方案设计与评审 | 1 |
| Phase 1：基础设施 + text_generate | 5 |
| Phase 2：throughput_optimizer | 3 |
| Phase 3：video_generate | 2 |
| Phase 4：旧代码清理 | 1 |
| 验证（单元 + 集成 + E2E） | 9.5 |
| 文档更新 | 1 |
| **总计** | **~22.5 人天（约 4.5 周）** |

### 7.4 降低风险的先决验证（Phase 0）

建议在全面铺开前先做 **Phase 0 可行性验证（2 天）**：

- 实现 `cli/registry/` 核心（`Param` + `ModuleSpec` + `datatypes.py`）— 不含任何 UI 字段
- 实现 `cli/registry/argparse_adapter.py`（CLI 侧）
- 实现 `web_ui/backend/services/ui_props/` + `json_adapter.py`（Web UI 侧）
- 对 text_generate 的 3 个共享参数（model_id/device/log_level）做端到端：
  - CLI `parse_args()` 结果与旧版字段级等价（相同 default、相同 type 解析、相同 choices 约束）
  - CLI `--help` 输出的参数列表和默认值与旧版一致（允许 formatter 差异）
  - `json_adapter` 以 Param 为基础注入 UIFieldProps，生成 JSON 与现有 TS→JSON 输出字段级一致
  - 前端能正常渲染

"字段级等价"定义：对同一组命令行输入，新旧 parser 产生的 `Namespace` 对象中每个字段值相同。`--help` 一致性定义：参数名、默认值、choices 列表逐行相同（标题行和格式空白允许差异）。

Phase 0 通过后，再投入 Phase 1–4。

---

## 8. 未解决问题 / 实施记录

| 问题 | 状态 / 讨论 |
|------|--------|
| **`cli/registry/` 的 import 依赖边界** | ~~`cli/registry/` 是否允许依赖 `tensor_cast`？~~ **已解决**：`cli/registry/` 不直接 import `tensor_cast` 重型模块。`DEVICE` 不设 choices，`argparse_adapter` 在 `build_argparser()` 时调用 `cli.registry.shared.get_device_choices()` 延迟获取（内部 import `DeviceProfile`，带 fallback 到 `["TEST_DEVICE"]`）。`QuantizeLinearAction` 等纯 `StrEnum` 类型可安全直接 import |
| **前端 `_validators.ts` 的 option 常量** | **已解决**：`QUANTIZE_LINEAR_OPTIONS` / `QUANTIZE_ATTENTION_OPTIONS` / `REMOTE_SOURCE_OPTIONS` / `CLI_LOG_LEVEL_OPTIONS` 均已在 `Param.choices` 中定义，`json_adapter` 生成 form JSON 时自动输出为 `field.options`（Enum → `{value: to_kebab(member.value), label: member.name}`）。前端直接从 form JSON 读取 |
| **schema version 策略** | **已解决**：bump 1.x→2.0；1.x snapshot 保留 SQLite；按 moduleId 独立版本号（已实现）；`json_adapter` 输出与现有 TS 格式一致 |
| **runner 中硬编码的 `_MULTI_CASE_FIELDS`** | **已解决**：Runner 从 `spec.fields` + `UIFieldProps.multi_values=True` 动态收集展开字段 |
| **Param.id (snake_case) vs Param.name (kebab-case)** | **已解决（落地）**：最终选择 `name` 字段直接用 kebab-case（如 `"tp-size"`、`"chrome-trace-file"`）。CLI flag / JSON field id / API params key / validator dict key 完全统一，无需转换；Python 属性访问通过 `py_name` property（`name.replace("-", "_")`）或 `dest_name`（考虑 `cli_dest` 覆盖）。`cli_name` property 移除（不再需要） |
| **spec_cli 集成** | **已解决（落地）**：`argparse_adapter` 是 `cli/spec_cli.py`（#704 CLI 规范）的下游——通过 `spec_cli.add_option()`、`spec_cli.make_enum_type()`、`spec_cli.make_token_type()` 注册选项，自动获得 alias deprecation、kebab-case Enum 解析、metavar 约定。`SpecArgumentParser` 重写 `format_help()` 输出 Description/Usage/Required/Optional/Examples/Output 段落布局 |
| **log_level 处理** | **已解决（落地）**：`log_level` 不是 Param——是 `spec_cli.add_log_options()` 公共基础设施（`--log-level` / `-v` / `-q`）。CLI 通过 `ModuleSpec.log_options=True`（默认）自动注入；Web UI 通过 `LOG_LEVEL_UI` / `LOG_LEVEL_UI_FULL` 共享常量 + `.override(group=..., default=...)` 表达模块差异。`json_adapter` 识别 `cfg.ui` 中"不在 spec.fields 内"的条目，通过 `_build_convention_field()` 追加到 form JSON |
| **model_id positional+flag 拆分** | **已解决（落地）**：Post-#704 模式：`nargs="?"` 位置参数 + `--model-path`（隐藏 formal flag）+ `--model-id`（可见 formal flag）+ deprecated `--model_id` alias（挂在 `--model-id` 上）。位置参数用独立 dest（`<py_name>_positional`）+ `default=argparse.SUPPRESS` 防止 clobber flag 值。延迟必填校验通过 `cli.utils.require_model_id()` 在 `parse_module_args()` 中执行 |
| **boolean dual-toggle (--X / --no-X)** | **已解决（落地）**：`cli_off_flag` 字段指定 off-switch 名（如 `no-repetition` 的 `cli_off_flag="repetition"`）。`argparse_adapter` 自动注册 `--repetition` 为 `action="store_false"` + `default=argparse.SUPPRESS`，与 `--no-repetition` 的 `store_true` 共享 dest |
| **validator 调用约定（kebab-case dict + provided set）** | **已解决（落地）**：validator dict 以 kebab-case `Param.name` 为 key（如 `params.get('num-devices')`），与 argparse dest snake_case 区分（`getattr(args, 'num_devices')`）。`wants_provided=True` 校验器额外接收 `provided: set[str]`——argv 中显式出现的 kebab-case 名集合，用于区分"显式传了默认值" vs "未传"（draft_dependents_require_method / draft_mtp_mutex 等 G2/G3 校验器需要此信息避免误报） |
| **UIFieldProps 新增 default / required / group 覆盖** | **已解决（落地）**：`default` 用 `_UNSET` sentinel 区分"未设置" vs "显式 None"（`device` 在 CLI 是单字符串默认，前端需要数组默认）；`required` 用于 CLI 有默认但前端 UX 需要强制选择的场景（如 `device` / `log-level`）；`group`（i18n dict 形状）用于 convention field 或 baseline parity 调整 |
| **json_adapter --check CI 漂移检测** | **已解决（落地）**：`python -m web_ui.backend.services.json_adapter --check` 对比 bundled JSON（`var/config/forms/*.json`）与即时生成结果，CI 失败阻断。`schema_registry.generate_form_configs()` 作为生产生成路径，避免双模块名（`services.*` vs `web_ui.backend.services.*`）引起的 `_UNSET` sentinel 不一致 |
| **image_generate CLI 接入** | **已落地**：image_generate 已作为第四模块接入 registry（23 Params + ui_props），但 Web UI runner adapter 暂未实现——`json_adapter.BUNDLED_MODULES` 不包含 image_generate，避免暴露无法执行的 form |
| **`cli_dest` 覆盖（backward compat）** | **已解决（落地）**：`Param.cli_dest` 用于 #704 重命名后保持旧属性名兼容。典型场景：`Param(name="no-repetition")` → 默认 `py_name="no_repetition"` → 但下游代码使用 `args.disable_repetition`。设置 `cli_dest="disable_repetition"` 使 `dest_name` 指向旧属性名。`argparse_adapter._register_flag()` 将 `dest=p.dest_name` 传给 argparse，`parse_module_args()` 的 validator dict 也用 `dest_name` 读值。同理 `graph-log-path` → `cli_dest="graph_log_url"` 使下游代码仍可用 `args.graph_log_url`（源码同步更新 `args.graph_log_path` → `args.graph_log_url`）。使用约束：仅当 #704 重命名保留不同内部属性名时设置；除非 legacy CLI 显式传过 `dest=`，否则保持 `cli_dest=None` |
| **统一投机解码 G2/G3 校验器** | **已解决（落地）**：`speculative-method` 引入 mtp/dflash/dspark 三种方法后，需要新增 8 个 G2/G3 校验器约束参数组合。其中 6 个使用 `wants_provided=True` 区分"显式传默认值" vs "未传"：`draftDependentsRequireMethod`（draft 相关选项依赖 method 已指定）、`draftMtpMutex`（draft 选项与 method=mtp 互斥）、`draftLegacyMtpMutex`（unified method 不能与 legacy `--num-mtp-tokens` 混用）、`mtpRequiresNumSpeculativeTokens`（method=mtp 必须提供 `--num-speculative-tokens`）、`mtpNoDraftLayers`（method=mtp 不能使用 draft 专有选项 `--num-draft-layers`）、`speculativeTokensPositive`（method + num-speculative-tokens=0 被拒绝）、`mtpNoLegacyAcceptanceRate`（method=mtp 不能使用 legacy `--mtp-acceptance-rates`）、`legacyMtpNoAcceptanceLength`（legacy `--num-mtp-tokens` 不能使用 `--acceptance-length`）。text_generate 注册 6 个（除 `mtpNoLegacyAcceptanceRate` 和 `legacyMtpNoAcceptanceLength`）；throughput_optimizer 注册全部 8 个（含 legacy 相关 2 个） |
| **Web UI `provided` 集合计算** | **已修复**：CLI 通过扫描 argv tokens 精确计算 `_explicitly_provided()`；Web UI 在 **job 级别**计算（所有 case 共享），优先使用前端发送的 `explicitly_touched` 列表，fallback 到 `val != field.default` 比较 + **类型强制转换**（integer/number 类型的 string 值先转为 int/float）+ 空字符串归一化为 None。消除了类型不匹配导致的假阳性 |
| **UIFieldProps `clearable` 字段** | **已落地**：`clearable: bool = False` 控制 select/multi-select 是否显示清除按钮。RFC §3.3 UIFieldProps 代码块已补充。`json_adapter._build_field()` 输出 `clearable` 属性到前端 JSON |
| **L1 声明式约束后端校验** | **待实现**：`_run_spec_validators` 当前仅执行 L2 交叉校验，缺少 L1 声明式约束（min/max/choices/pattern/max_length/required）的后端校验。直接构造 HTTP 请求可绕过前端 L1，提交越界数值等。需新增 `_check_declarative()` 函数，在 L2 之前执行 L1 检查。详见 `docs/design/l1_declarative_validation.md` |
| **case 级别校验粒度** | **待实现**：当前 `_run_spec_validators` 在 `_expand_job_cases` 之后被调用，但使用原始多值 `request.params`（数组），L2 validator 收到数组做交叉校验（如 `tp_size × dp_size × pp_size == num_devices`）语义不正确。需改为以 case 为单位执行 L1 + L2——展开笛卡尔积后对每个标量 case 独立校验，任一 case 失败返回 400。详见 `docs/design/l1_declarative_validation.md` |

---

## 附录

### A. 关键文件清单

| 文件 | 当前状态 | 重构后 |
|------|----------|--------|
| `cli/spec_cli.py`（#704） | ~520 行（CLI 规范基础设施：SpecArgumentParser、add_option、make_enum_type、make_token_type、add_log_options、STANDARD_LOG_LEVELS、METAVAR_*） | 不变（argparse_adapter 的下游依赖） |
| `cli/inference/text_generate.py` | ~407 行（含 ~50 行 argparse） | ~50 行（`build_argparser(spec)` + `parse_module_args()` + main 逻辑） |
| `cli/inference/video_generate.py` | ~496 行（含 ~24 个 add_argument） | ~30 行 |
| `cli/inference/throughput_optimizer.py` | ~545 行（含 ~38 个 add_argument） | ~50 行 |
| `cli/inference/image_generate.py` | ~N 行（含 ~23 个 add_argument） | ~30 行 |
| `cli/utils.py:get_common_argparser` | 函数 ~60 行（文件 194 行） | 删除（被 `cli/registry/argparse_adapter.py` 替代） |
| `web_ui/frontend/src/config/forms/*.ts` | ~1450 行 TS（4 文件合计 1827 行） | 删除（被 `json_adapter.py` 生成的 JSON 替代） |
| `web_ui/frontend/src/config/forms/_validators.ts` | 310 行 | 删除函数部分，option 常量由 JSON 自动生成 |
| `web_ui/frontend/scripts/gen-form-schemas.mjs` | 70 行 | 删除（被 `web_ui/backend/services/json_adapter.py` 替代） |
| `web_ui/backend/runners/throughput_optimizer.py:_build_namespace` | ~120 行（含 _int/_num/_num_or_none helper） | ~50 行（从 Param 元数据自动生成类型转换） |
| `cli/registry/`（新增） | 不存在 | datatypes.py (~250 行) + shared.py (~350 行) + validators.py (~560 行) + modules/*.py (~3×~300 行) + argparse_adapter.py (~500 行) |
| `web_ui/backend/services/ui_props/`（新增） | 不存在 | __init__.py (~315 行, 含 LOG_LEVEL_UI 共享常量 + UIFieldProps.override) + 4 模块 UI 属性 dict |
| `web_ui/backend/services/json_adapter.py`（新增） | 不存在 | ~480 行（Param + UIFieldProps → JSON，含 _infer_control、_build_convention_field、--check 漂移检测 CLI） |
| `web_ui/backend/var/config/forms/*.json`（新增） | 不存在 | 每个模块一份 form JSON（schema_registry 管理，json_adapter --check 校验漂移） |

### B. 模块参数重叠分析（四模块）

| 参数（Param.name，kebab-case） | text_generate | video_generate | throughput_optimizer | image_generate |
|:------|:---:|:---:|:---:|:---:|
| model-id（位置+--model-path/--model-id） | ✓ | ✓ | ✓ | ✓ |
| device | ✓ (单值) | ✓ (单值) | ✓ (**nargs="+"**，多值，附加 --devices) | ✓ |
| num-devices / world-size | ✓ | ✓ (world-size) | ✓ | ✓ |
| reserved-memory-gb | ✓ (default=0.0) | — | ✓ (default=10.0) | ✓ |
| log-level（convention field，非 Param） | ✓ (default="error") | ✓ (default="info") ⚠️ | ✓ (default="error") | ✓ (default="error", 5 levels) |
| quantize-linear-action | ✓ | ✓ | ✓ | ✓ |
| quantize-attention-action | ✓ | ✓ | ✓ | ✓ |
| mxfp4-group-size | ✓ | — (仅函数参数，无 CLI flag) | ✓ | ✓ |
| compile | ✓ | ✓ | ✓ | ✓ |
| compilation-config | ✓ | — | ✓ | — |
| prefix-cache-hit-rate | ✓ | — | ✓ | — |
| chrome-trace-file | ✓ | ✓ | ✓ | ✓ |
| image-batch-size/height/width | ✓ | — (video 使用 height/width/frame-num) | ✓ | — |
| **共享参数合计** | **~13** | **~9** | **~13** | **~10** |
| **模块专有参数** | **~45**（含大量并行参数） | **~15** | **~35** | **~15** |

> **默认值差异说明**：
>
> - `reserved-memory-gb`：text_generate=0.0, throughput_optimizer=10.0，video_generate 缺失该参数。重构后通过 `RESERVED_MEMORY_GB.override(default=...)` 处理。
> - `log-level`：**不是 Param**——由 `spec_cli.add_log_options()` 公共基础设施注入。video_generate 默认 `"info"`，其他模块默认 `"error"`。Web UI 通过 `LOG_LEVEL_UI.override(default="info")`（vi）/ `LOG_LEVEL_UI.override()`（tg/to，默认 "error"）/ `LOG_LEVEL_UI_FULL.override()`（ig，5 个 level）表达差异；CLI 端 video_generate 通过 `resolve_log_level()` 在 `spec_cli.parse_args()` 后处理。
>
> **CLI 行为差异说明**：
>
> - `device`：text_generate / video_generate 的 CLI `--device` 接受单值；throughput_optimizer 在 registry 中不使用 `DEVICE.override(nargs="+")`（data_type 不能 override），而是定义 module-level `DEVICES = Param(name="device", data_type="string[]", nargs="+", cli_extra_flags=("devices",), ...)` 覆盖共享 DEVICE。text_generate / video_generate 的 Web UI 多设备对比通过 `UIFieldProps.multi_values=True` + Runner 笛卡尔积展开实现（CLI 语义不变）。
