# RFC: msmodeling 公开命令行统一规范化

## 元数据

| 项目 | 内容 |
|:---|:---|
| **状态** | Draft |
| **作者** | eveyin1 |
| **创建日期** | 2026-08-16 |
| **更新日期** | 2026-08-18 |
| **相关链接** | 依据《MindStudio 工具链命令行统一规范化设计方案》§4.7；本修订以当前公开 CLI 代码为准 |

---

## 1. Overview (概述)

### 1.1 Summary (简介)

本 RFC 描述 msmodeling **已经落地**的公开控制台规范：长选项 kebab-case、短选项单字符且语义固定、`--version/-V` 与统一日志分级、`--help` 按固定段落输出。并行度短名（`--tp-size`）与量化原取值（`W8A8_DYNAMIC`）保持正式接口。

实现集中在 `cli/spec_cli.py`。旧参数不删除，降为隐藏兼容别名：继续解析、stderr 一次性弃用提示、不出现在 `--help`。内部 argparse dest 与 `UserInputConfig` / ServingCast / OptiX 字段名不变。

**本修订对齐当前代码。** 公共日志词表只实现 `--log-level`、`-v/--verbose`、`-q/--quiet`；**不提供** `--debug` 与 `--log-file`。调试级别用 `--log-level debug` 或 `-v`。

### 1.2 Motivation (动机)

整改前，公开入口混用路径后缀（`--chrome-trace` 与文件语义不符）、optix 残留 `--load_breakpoint` 与多字符短选项 `-lb`，且缺少统一的 `--version/-V` 与 `--log-level`。跨工具记忆成本高，Agent 难以从 `--help` 推断参数形态。

不做此提案的影响：存量脚本仍能跑，但新用户与 AI 工作流会持续接触两套写法，后续增量参数会继续发散，§4.7 验收无法闭环。

### 1.3 Goals (目标)

**目标（与当前实现一致）**

- 公开入口：`msmodeling`、`inference text-generate` / `throughput-optimizer` / `model-adapter` / `video-generate` / `image-generate`、`optix`。
- 公共词表中 **§4.7 验收点名** 且已实现的标准写法：`--log-level`、`--verbose/-v`、`--quiet/-q`、`--version/-V`、`--jobs/-j`（仅 throughput-optimizer）、`--config/-c`（仅 optix）、`--output-file/-o`（model-adapter 子命令）。
- `--help` 固定输出 `Description` / `Usage` / `Commands`（有子命令时）/ `Required arguments` / `Optional arguments` / `Examples`；声明了 `output_help` 的入口再附加 `Output`。
- 带值参数打印语义化 metavar。量化与注册表原值（`W8A8_DYNAMIC`、`ais_bench`）在 help 中保持原样。
- 被改名的旧参数可解析；使用旧名时 stderr 一次性告警。

**非目标 / 当前代码明确不做**

- **不注册 `--debug`、`--log-file`。** 二者不在 §4.7 对本工具的验收点名范围内。`-v/--verbose` 在未写 `--log-level` 时等价 `--log-level debug`。日志只走 stderr / stdlib `logging.basicConfig`（inference）或 optix loguru stderr handler，没有公共文件 handler。
- 不整改 `tools/`、`serving_cast/main.py`、测试辅助脚本等非公开控制台。
- 不实现 wrapper 类 `-- <prog> [args]`（§4.7 第 7 条针对 mssanitizer 等 launcher）。
- 不把 `--device` 改成规范词表中的 `cpu/npu`；取值仍是 TensorCast `DeviceProfile` 名称。
- 不重命名内部 dest（`tp_size`、`disagg`、`chrome_trace`、`graph_log_url` 等）。
- 不把 `--model-id` 改成 `--model-path`。text-generate / optimizer / video-generate / image-generate / model-adapter 均支持位置参数与 `--model-id`（分 dest）；adapter 额外把 `--model_id` 作为正式名展示，其它入口将 `--model_id` 作为隐藏别名。
- 不在本 RFC 同步改 Web UI 命令拼装、Skill 文档中的旧参数示例。

### 1.4 方案要求改什么（简表）

依据《MindStudio 工具链命令行统一规范化设计方案》：整改覆盖 **参数名 (K)、参数值 (V)、帮助信息 (解释 KV)**，外加兼容别名。

| 块 | 规范章节 | 当前实现 |
|:---|:---|:---|
| 参数名 (K) | 4.2 | 长选项 kebab-case；短选项单字符；文件 `-file`、目录 `-path`；并行度短名保持正式接口 |
| 参数值 (V) | 4.3 | 布尔用 flag / `--no-*`；多值用复数名 + nargs；量化等枚举保持原取值，kebab 拼写可解析 |
| 帮助与版本 | 4.4–4.5 | `SpecArgumentParser.format_help()`；metavar；`[default: xxx]`；`--version/-V` |
| 兼容 | 4.1、4.6、4.7.11–12 | 旧写法可解析、不进 help、stderr 一次性弃用提示 |

内部 dest 不改。§4.7 第 7 条 wrapper 不适用。

#### 1.4.1 参数改动（对照当前代码）

落地范围只覆盖必须改和值得改。dest 一律不改；被改名的旧写法可解析并告警。

**第一类：改名字（旧名留下当隐藏别名）**

路径与产物后缀（文件 `-file`，目录 `-path`）：

| 正式接口 | 隐藏别名 | dest | 范围 | 当前行为 |
|:---|:---|:---|:---|:---|
| `--chrome-trace-file` | `--chrome-trace` | `chrome_trace` | text-generate、throughput-optimizer、video-generate、image-generate | Chrome trace JSON，metavar `<FILE>` |
| `--graph-log-path` | `--graph-log-url`、`--graph-log-file` | `graph_log_url` | 仅 text-generate | 编译图 dump **目录**，`<DIR>`。内部交给 `GraphTransformObserver(log_url=...)` |
| `--profiling-database-path` | `--profiling-database` | `profiling_database` | text-generate、throughput-optimizer、model-adapter verify 相关路径 | profiling CSV 目录，`<DIR>` |
| `--export-empirical-metrics-file` | `--export-empirical-metrics` | `export_empirical_metrics` | text-generate | M1–M5 JSON；须 `--performance-model profiling` |
| `-o, --output-file` | `--output` | `output` | model-adapter doctor / verify / export-evidence | JSON 或 evidence YAML |
| `--profile-draft-output-file` | `--profile-draft-output` | `profile_draft_output` | model-adapter doctor | ModelProfile 草稿文件 |
| `--st-case-output-path` | `--st-case-output` | `st_case_output` | model-adapter verify | ST case 输出；help 写文件或目录，metavar 为 `<FILE>` |
| `--doctor-report-file` | `--doctor-report` | `doctor_report` | model-adapter export-evidence | 输入 doctor JSON；缺省时 handler 报错（别名不能 `required=True`） |

其它改名：

| 正式接口 | 隐藏别名 | dest | 范围 | 当前行为 |
|:---|:---|:---|:---|:---|
| `--num-devices` | `--world-size` | `world_size` | video-generate、image-generate | 与其它入口语义对齐；二者 **不用** `get_common_argparser` |
| `--ttft-limit` | `--ttft-limits` | `ttft_limits` | throughput-optimizer | 单个 TTFT 约束，`<FLOAT>` |
| `--tpot-limit` | `--tpot-limits` | `tpot_limits` | throughput-optimizer | 单个 TPOT 约束 |
| `--mtp-acceptance-rates` | `--mtp-acceptance-rate` | `mtp_acceptance_rate` | throughput-optimizer | `nargs=+`，默认 `[0.9, 0.6, 0.4, 0.2]` |
| `--no-repetition` | `--disable-repetition` | `disable_repetition` | text-generate、model-adapter | `store_true`。不另增 `--repetition` |
| `--ignore-existing-profiles` | `--ignore-existing-profile` | `ignore_existing_profile` | model-adapter doctor | `action=append` |
| `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` | optix | `store_true` |
| `--benchmark-policy` | `--benchmark_policy` | `benchmark_policy` | optix | 正式短选项仍是 `-b`；取值见第三类 |
| `--log-level` | `--log_level` | `log_level` | 挂了 `add_log_options` 的入口 | 见第二类 |

**第二类：补公共参数**

实现：`add_version_option` / `add_log_options`（`cli/spec_cli.py`）。

| 正式接口 | dest / 行为 | 挂载位置（当前代码） |
|:---|:---|:---|
| `-V, --version` | `VersionAction`，打印后 `parser.exit(0)` | 顶层 `msmodeling`、`inference` 父 parser、text-generate / throughput-optimizer（经 `get_common_argparser`）、video-generate、image-generate、model-adapter 及其子命令、optix |
| `--log-level {debug,info,warning,error,critical}` | `log_level`，argparse 默认 `"error"` | `get_common_argparser`、video-generate、image-generate、model-adapter **doctor/verify**、optix。**顶层 `msmodeling` 与 `model-adapter export-evidence` 不挂日志选项** |
| `-v, --verbose` | `verbose`，`store_true` | 与 `--log-level` 同挂载 |
| `-q, --quiet` | `quiet`，`store_true` | 同上 |
| `-j, --jobs` | 默认 `8`，`<N>` | 仅 throughput-optimizer；寻优进程并发，不是模型 TP/DP |
| `-c, --config` | `config`，`<FILE>` | 仅 optix；TOML |

**未实现（有意为之，不是漏改）：**

| 词表项 | 现状 |
|:---|:---|
| `--debug` | 不注册。未知参数。请用 `-v` 或 `--log-level debug` |
| `--log-file <FILE>` | 不注册。未知参数。无公共文件 handler |

日志冲突裁决（`cli/spec_cli.py` 的 `resolve_log_level`）：

1. argv 中出现 `--log-level` 或 `--log_level` → 以该值为准。
2. 否则 `--verbose` / `-v` → `debug`。
3. 否则 `--quiet` / `-q` → `error`。
4. 都没有 → `error`（与 argparse 默认一致）。

`parse_args` 会把裁决结果写回 `args.log_level`。

消费路径：

- TensorCast 公开入口：`configure_std_logging` → stdlib `logging.basicConfig`。
- optix：若 argv 出现 `--log-level`/`--log_level`，或 `-v`/`-q`，则 `set_log_level(args.log_level)` 接到 loguru stderr。**若以上都未出现**，则 `set_log_level(resolve_optix_env_log_level())`，读取 `OPTIX_LOG_LEVEL`（或遗留 `MODELEVALSTATE_LEVEL`），而不是强制 `error`。

**第三类：取值与 metavar（参数名大多不动）**

- 量化 / 编译开关 / attention backend：help 展示原值（`W8A8_DYNAMIC`、`enable_multistream` 等）；kebab 拼写可解析，不告警。实现：`make_enum_type` / `make_token_type`。
- `--log-level` 含 `critical`。
- optix `--engine` / `--benchmark-policy`：注册表固定名（默认引擎 `vllm`，测评 `ais_bench` / `vllm_benchmark`），`registered_names=True`，help 不改成 kebab。
- throughput-optimizer：`--device` 与 `--devices` 都是**正式公开长选项**（同一 dest `device`，`nargs=+`），`--devices` **不是**弃用别名、不告警。覆盖 `get_common_argparser` 里单值 `--device`。
- `--remote-source` / `--performance-model` 补 metavar。text-generate 的 `--performance-model` 为可重复 `action=append`。
- 布尔 help：`[default: off]` / `[default: on]`，不写 True/False。
- 多值 metavar：`nargs=+` → `<N> [<N> ...]`，`nargs=*` → `[<N> ...]`（`SpecArgumentParser` 按 nargs 展开）。

**第四类：帮助与版本**

| 项 | 当前行为 |
|:---|:---|
| 帮助段落 | `SpecArgumentParser.format_help()` 重写全文。argparse 的 `add_argument_group` 标题（如 General Options）**不出现在**最终 `--help` |
| 必填/可选 | 靠 Required / Optional 分段，无行内 `<Required>` |
| 默认值 | 行尾 `[default: xxx]`；无默认不写；禁止 `(default: None)`。枚举默认展示原值 |
| metavar | `<N>` / `<FILE>` / `<DIR>` / `<FLOAT>` / `<NAME>` / `<RANGE>` / `{a,b,c}` |
| `--help` 文案 | `Show help message.` |
| 示例 | 各公开 parser 的 `examples=` |
| 版本 | Logo、`{tool} {version} ({7 位 git})`、版权、Mulan PSL v2、Repo。`-v` 不是 version |

#### 1.4.2 旧接口是否还能用

能用。旧参数不删除，只是降为隐藏别名：

1. **能解析**：`--chrome-trace out.json` 与 `--chrome-trace-file out.json` 效果相同。
2. **stderr 打一次弃用提示**：`WARNING: --chrome-trace is deprecated; use --chrome-trace-file instead.`
3. **不出现在 `--help`**。

存量脚本、CI 与现有 UT/ST 不必先改参数名。内部 dest 也不改。

例外（不是隐藏别名）：

- `--model-id` 为公开正式接口（位置参数与选项分 dest）。adapter 的 `--model_id` 也是正式名；其它入口 `--model_id` 为隐藏别名。
- `--device` 与 `--devices`（optimizer）都是正式名。

---

## 2. Use Case Analysis (用例分析)

| 用例 | 行为 | DFX |
|:---|:---|:---|
| 新用户查看帮助 | `msmodeling --help` 与各子命令 `--help` 只展示标准名、metavar、默认值与示例 | 可学习、可被 Agent 解析 |
| 查询版本 | 挂了 `add_version_option` 的入口 `--version` / `-V` | `-v` 是 verbose，不是 version |
| 调节日志 | `--log-level` / `-v` / `-q`；**没有** `--debug`、`--log-file` | inference 默认 error；optix 无 CLI 日志开关时走环境变量 |
| 新脚本 | `--tp-size 8`、`--chrome-trace-file`、`--disagg`、`--load-breakpoint` | 正式接口无多字符短选项 |
| 存量脚本 | `--chrome-trace`、`--load_breakpoint`、`-lb`、`--ttft-limits` 仍可解析 | 兼容；stderr 引导迁移 |
| 模型标识 | 位置参数或 `--model-id`；adapter 另以 `--model_id` 为正式名 | 不引入 `--model-path` |
| 多硬件寻优 | optimizer `--device` / `--devices` 均可 | `--devices` 不告警 |
| 适配器导出 | `-o` / `--output-file`；`--output` 为隐藏别名 | `-o` 符合词表 |

使用约束：

- `--device` 取值是已注册 DeviceProfile 名，不是 `cpu`/`npu`。
- 量化正式值为 `W8A8_DYNAMIC`；`w8a8-dynamic` 仍可解析。
- `--compilation-config` help 展示内部 snake_case 选项名（`make_token_type(..., store_canonical="snake")`）。
- `msmodeling inference <cmd> ...` 与 `python -m cli.inference.<module> ...` 等价；顶层用 `parse_known_args` 把剩余 argv 转给子模块 `main()`。

---

## 3. Design (方案设计)

### 3.1 Overall Design (总体方案)

规范内核在 `cli/spec_cli.py`。顶层只做子命令分发，真正的参数表在各模块 parser。

```text
msmodeling (cli/main.py, SpecArgumentParser)
├── -V / --version          # 无 --log-level
├── inference               # 仅 version + Commands；剩余 argv 转交
│   ├── text-generate       # parents=get_common_argparser()；inherit_deprecated
│   ├── throughput-optimizer
│   ├── model-adapter {doctor, verify, export-evidence}
│   ├── video-generate      # 独立 parser，不走 common_parser
│   └── image-generate      # 独立 parser，不走 common_parser
└── optix                   # optix/optimizer/optimizer.py，独立 SpecArgumentParser

cli/spec_cli.py
├── SpecArgumentParser / SpecHelpFormatter
├── add_option(..., aliases=...)
├── add_version_option / add_log_options
├── make_enum_type / make_token_type
├── inherit_deprecated
└── parse_args → warn_deprecated_from_argv + resolve_log_level
```

核心逻辑：

1. 公开长选项 kebab-case；标准短选项仅单字符（`-h/-V/-v/-q/-o/-j/-c/-e/-b` 等）。
2. 兼容别名经 `add_option(..., aliases=)` 注册，`help=argparse.SUPPRESS`，命中时一次性 `WARNING`。
3. dest 不变：公开 `--tp-size` 的 dest 仍是 `tp_size`。
4. help 由 `SpecArgumentParser.format_help()` 重写，不依赖 argparse 默认分组标题。
5. 日志：`resolve_log_level` 写回 `args.log_level`；inference 用 `configure_std_logging`；optix 用 `set_log_level`（CLI 优先，否则环境变量）。

### 3.2 Technology Selection (技术选型)

| 方案 | 结论 | 理由 |
|:---|:---|:---|
| `cli/spec_cli.py` 作为规范内核 | 采用 | 各入口共享 formatter、version、别名与枚举解析 |
| 删除旧参数，只保留新名 | 不采用 | 违反兼容别名与存量脚本零中断 |
| 同步重命名 dest / UserInputConfig | 不采用 | 改动面会进入仿真内核 |
| `--device` 改为 cpu/npu | 不采用 | 仿真目标是 DeviceProfile |
| 补 `--debug` / `--log-file` | 不采用 | §4.7 未点名为必须项；debug 已由 `-v` / `--log-level debug` 覆盖；落盘不是本工具公共日志需求 |
| 每个子命令手写 help | 不采用 | 体例无法跨命令一致 |
| 顶层 parser 展开全部子命令参数 | 不采用 | `parse_known_args` + 模块 `main()` 避免重复维护两套参数表 |

### 3.3 Security, Privacy, and DFX Design (安全隐私与 DFX 设计)

| 属性 | 设计 |
|:---|:---|
| 兼容性 | 旧长选项、旧枚举 kebab 取值、`-lb` 可解析；dest 与默认值语义不变 |
| 可维护性 | 别名、metavar 常量、help 段落集中在 `spec_cli.py` |
| 可测试性 | `tests/regression/cli/test_spec_cli.py` 扫描 help（无多余 snake_case 长选项、无多字符短选项、无 `(default: None)`），并断言别名告警与日志裁决 |
| 可靠性 | 别名告警按 `old->new` 去重；非法枚举走 `ArgumentTypeError` |
| 安全 | `--version` 不打印 Token 或本机路径；模型路径仍走 `check_string_valid`；optix 保留 root 运行告警 |

### 3.4 Programming and Integration Design (编程与调用设计)

#### 3.4.1 Basic Programming Model Design (编程模型基本设计)

- 语言：Python 3.10+，stdlib `argparse`。
- 新增公开命令必须用 `SpecArgumentParser`，经 `add_option` / `add_version_option` / `add_log_options` 注册，用 `cli.spec_cli.parse_args` 收尾（不要只用 `parser.parse_args`，否则别名告警和日志裁决不会跑）。
- `parents=[get_common_argparser()]` 时必须 `inherit_deprecated(child, common)`，否则父级 `--log_level` 等别名告警不生效。
- `get_common_argparser()` 本身是普通 `ArgumentParser(add_help=False)`，help 由子级 `SpecArgumentParser` 渲染。
- 验收：对照 §4.7 与 `tests/regression/cli/test_spec_cli.py`。

#### 3.4.2 API Definition and Design (接口定义与设计)

##### `add_option`

- **原型**：`add_option(target, *option_strings, aliases=(), **kwargs) -> argparse.Action`
- **约束**：`aliases` 的 help 为 `SUPPRESS`，并从 kwargs 去掉 `required`，避免「必填 + 别名」在 argparse 中失效。必填在 handler 内校验（如 `--doctor-report-file`）。

##### `parse_args` / `resolve_log_level` / `configure_std_logging`

- `parse_args`：`parser.parse_args` → `warn_deprecated_from_argv` → `resolve_log_level`。
- `resolve_log_level`：见 1.4.1 第二类。`argv is None` 时无法区分「用户没写 `--log-level`」与「用了 argparse 默认值」，此时若 `log_level` 有值则直接采用（测试里 `Namespace(log_level="warning", verbose=True)` 会得到 `warning`）。
- `configure_std_logging`：再解析一次级别后 `logging.basicConfig`。当前实现用 `sys.argv[1:]` 判断是否出现 `--log-level`，模块入口在改写 `sys.argv` 后调用即可。
- **不**根据任何 `log_file` 字段添加 `FileHandler`。

##### `make_enum_type` / `make_token_type`

- help 展示传入的原取值；kebab 拼写可解析，不告警。
- 枚举返回成员实例；token 可 `store_canonical="snake"`；optix 注册表键用 `registered_names=True`。

##### 公开参数映射（节选）

| 概念 | 正式接口 | 隐藏别名 | dest |
|:---|:---|:---|:---|
| 张量并行 | `--tp-size` | 无（本来就是正式名） | `tp_size` |
| 分离部署 | `--disagg` | 无 | `disagg` |
| Trace | `--chrome-trace-file` | `--chrome-trace` | `chrome_trace` |
| Profiling 库 | `--profiling-database-path` | `--profiling-database` | `profiling_database` |
| 编译图 dump | `--graph-log-path` | `--graph-log-url`、`--graph-log-file` | `graph_log_url` |
| 输出文件 | `-o, --output-file` | `--output` | `output` |
| 断点续跑 | `--load-breakpoint` | `--load_breakpoint`、`-lb` | `load_breakpoint` |
| 反向开关 | `--no-repetition` | `--disable-repetition` | `disable_repetition` |
| TTFT | `--ttft-limit` | `--ttft-limits` | `ttft_limits` |
| 模型源 | `--model-id`（及位置参数） | `--model_id`（adapter 为正式名） | `model_id` / `model_id_positional` |
| 多硬件 | `--device`、`--devices` | 无（双正式名，仅 optimizer） | `device` |
| 调试级别 | `--log-level debug` 或 `-v` | `--log_level` | `log_level` |
| `--debug` / `--log-file` | **不存在** | — | — |

#### 3.4.3 Usage Instructions (使用说明)

```bash
msmodeling inference text-generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 128 --device TEST_DEVICE \
  --tp-size 8 --chrome-trace-file trace.json

msmodeling inference throughput-optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE --num-devices 8 \
  --input-length 1024 --output-length 512 --disagg

msmodeling inference model-adapter doctor --model-id Qwen/Qwen3-32B -o doctor.json
msmodeling optix -e vllm -b ais_bench --config ./config.toml
```

约束：

- `-v` 只表示 verbose；版本只用 `-V`。
- 需要 debug 日志时写 `--log-level debug` 或 `-v`，不要写 `--debug`。
- 需要落盘日志时由调用方重定向 stderr，不要写 `--log-file`。
- 旧写法可用，会告警，且不出现在 `--help`。
- `throughput-optimizer` 的 `--jobs/-j` 是寻优进程并发。

---

## 4. Test Design (测试设计)

本节面向手工 / ST 验收。开发回归见 `tests/regression/cli/test_spec_cli.py`。

### 4.1 验收范围

- 公开入口：`msmodeling`、`inference text-generate` / `throughput-optimizer` / `model-adapter` / `video-generate` / `image-generate`、`optix`
- `--help`、`--version/-V`、已实现的日志开关、正式参数名与取值
- 旧参数名仍能跑通，stderr 有弃用提示
- `--device` 仍是 DeviceProfile 名
- `--debug`、`--log-file` 应被 argparse 判为未知参数（这是预期，不是缺陷）

### 4.2 环境

仓库根目录、依赖已安装。两组入口等价：

```bash
msmodeling inference text-generate --help
python -m cli.inference.text_generate --help
```

optix：`msmodeling optix --help`。若缺 `pydantic_settings` 直接报错，记为环境问题。

### 4.3 验收步骤

#### A. 帮助与版本

对下列命令执行 `--help`，并对顶层及至少一个子命令执行 `-V` / `--version`：

- `msmodeling --help`、`msmodeling -V`
- `python -m cli.inference.text_generate --help`
- `python -m cli.inference.throughput_optimizer --help`
- `python -m cli.inference.video_generate --help`
- `python -m cli.inference.image_generate --help`
- `python -m cli.inference.model_adapter doctor --help`（`verify` / `export-evidence` 抽一个）
- `msmodeling optix --help`

**通过标准：**

1. `--help` 含 Description、Usage、Examples；有子命令时含 Commands；text-generate 含 Required / Optional。
2. `-V` / `--version` 含 MindStudio / msmodeling 与 Mulan PSL v2。不要用 `-v` 查版本。
3. 挂了 `add_log_options` 的入口 help 含 `--log-level {debug,info,warning,error,critical}`、`-v`、`-q`。**不含** `--debug`、`--log-file`。顶层 `msmodeling --help` 可以没有日志选项。
4. 量化正式取值仍是 `W8A8_DYNAMIC` 等；kebab 拼写也能解析。
5. 布尔用 `[default: off]` / `[default: on]`。
6. 下列旧名**不应作为正式选项出现在 `--help`**：无 `-file` 的 `--chrome-trace`、`--load_breakpoint`、`-lb`、adapter 的 `--output`、`--ttft-limits`。`--tp-size`、`--disagg` **应**出现。
7. optix help 测评工具取值仍是 `ais_bench`、`vllm_benchmark`，不要验收成 `ais-bench`。
8. throughput-optimizer help 能看到 `--jobs` / `-j`，以及 `--device` / `--devices`。
9. model-adapter doctor/verify help 同时出现 `--model-id` 与 `--model_id`。text-generate / throughput-optimizer / video-generate / image-generate help 出现 `--model-id`，不把 `--model_id` 列为正式选项。

#### B. 新写法功能抽测

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 128 --device TEST_DEVICE \
  --tp-size 1 --log-level info

python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device TEST_DEVICE --num-devices 2 \
  --input-length 128 --output-length 16 \
  --tp-sizes 1 2 --disagg --jobs 2
```

有实测环境再抽测 optix；否则确认 `--help` 与 `-e` / `-b` / `-c` 即可。

**通过标准：** 命令能解析并进入原有流程；加上 `--chrome-trace-file trace.json` 时应生成文件。

#### C. 旧写法兼容抽测（必做）

| 场景 | 旧写法 | 正式写法 | 期望 |
|:---|:---|:---|:---|
| Trace | `--chrome-trace out.json` | `--chrome-trace-file out.json` | 都能写出；旧名 stderr deprecated |
| TTFT | `--ttft-limits 2000` | `--ttft-limit 2000` | 同上 |
| adapter 输出 | `doctor ... --output a.json` | `-o a.json` 或 `--output-file a.json` | 都能写出报告 |
| optix 断点 | `--load_breakpoint` 或 `-lb` | `--load-breakpoint` | 都能解析；`--help` 无 `-lb` |
| 量化 kebab | `--quantize-linear-action w8a8-dynamic` | `W8A8_DYNAMIC` | 都能解析；help 展示大写正式值 |

#### D. 日志开关抽测

| 操作 | 期望 |
|:---|:---|
| inference 入口不指定日志参数 | `log_level` 为 `error` |
| `--verbose` 或 `-v` | 等价 debug |
| `--quiet` 或 `-q` | 等价 error |
| `--log-level warning` 与 `-v` 同时出现 | 以 `--log-level` 为准（warning） |
| `--log-level critical` | 成功 |
| `--debug` 或 `--log-file out.log` | **失败**：unrecognized arguments |
| optix 不写任何日志 CLI 开关 | 使用 `OPTIX_LOG_LEVEL` / `MODELEVALSTATE_LEVEL`，不是本 helper 的默认 `error` |

#### E. 文档抽测

中英文 TensorCast / 吞吐优化 / 快速入门示例使用正式名。指南应写明不提供 `--debug` / `--log-file`。OptiX 指南 `-b` 取值仍是 `ais_bench` / `vllm_benchmark`。

### 4.4 判定为不通过的典型现象

- `--help` 仍把 `-lb`、`--load_breakpoint`、无 `-file` 的 `--chrome-trace` 列成正式选项。
- `--tp-size 2` 报 unrecognized arguments。
- `-v` 打印版本而不是详细日志。
- optix help 把测评工具写成 `ais-bench` / `vllm-benchmark`。
- `--device cpu` 被当成合法设备类型。
- 把「没有 `--debug` / `--log-file`」当成实现遗漏（除非规范后续把它们列为必须项并改 RFC）。

### 4.5 自动化

```bash
python -m pytest tests/regression/cli/test_spec_cli.py tests/regression/cli/test_export.py
```

全量结论以 CI 为准。

---

## 5. Drawbacks and Risks (缺点和风险)

| 风险 | 影响 | 应对 |
|:---|:---|:---|
| dest 与公开选项名不一致 | 按 dest 猜 CLI 会出错 | RFC 与 help 只承诺 option string |
| Web UI / Skill 仍拼旧路径名 | `--chrome-trace` 等会告警 | 别名保留；后续单独改命令拼装 |
| `--device` 与规范词表不完全同义 | 可能被误解为 cpu/npu | help 写明 DeviceProfile |
| 公共词表还有 `--debug` / `--log-file` | 跨工具用户可能以为本 CLI 也有 | RFC、user guide、本验收条款明确不提供；debug 走 `-v` |
| dest `log_level` 默认 `"error"` 与 optix 无 CLI 开关时的环境变量路径不一致 | 只影响 optix | 代码按「有 CLI 日志开关才覆盖环境变量」处理 |
| 本机缺 `pydantic_settings` 时 optix `--help` 不可用 | 测试跳过 | CI 完整依赖环境仍覆盖 optix |

---

## 6. Existing Technology (现有技术)

参考 POSIX / IEEE Std 1003.1 短选项惯例，以及 argparse 生态对 kebab-case 与 nargs 多值的预期。规范相对 POSIX Guideline 8（逗号单参数）选择 nargs / 可重复选项，本实现与之一致。

与仓库内 `text-generate-executor` 等 Skill RFC 的关系：那些 RFC 描述如何**生成** CLI 命令；本 RFC 定义命令**表面**。Skill 文档中的旧参数示例应在后续演进中改为正式名。

---

## 7. Unresolved Questions (未解决问题)

- 是否在后续版本把 `--device` 拆成设备类型 + `--device-profile`，需产品确认，不在本实现范围。
- Web UI 命令构造与各 Skill `references/*-params.md` 何时切换到新参数名。
- 兼容别名的下线周期（规范未规定删除时间点）。
- 若部门规范后续把 `--debug` / `--log-file` 列为必须项，再单独开改动：注册选项、明确与 `--log-level`/`-v` 的优先级、为 stdlib logging 与 optix loguru 增加 file handler，并补入口解析测试。当前代码与本 RFC **不以该项为缺口**。

---

## 附录

### 修改文件

| 文件 | 说明 |
|:---|:---|
| `cli/spec_cli.py` | 规范内核：formatter、version、别名、日志裁决、枚举解析 |
| `cli/utils.py` | `get_common_argparser`：version、log、`add_model_id_source`、`--device` |
| `cli/main.py` | 顶层 help / version / Commands；`parse_known_args` 分发 |
| `cli/inference/text_generate.py` | 路径后缀、`--no-repetition`、log/version |
| `cli/inference/throughput_optimizer.py` | `--ttft-limit`、`--jobs/-j`、`--device/--devices`、路径后缀 |
| `cli/inference/model_adapter.py` | 子命令 help、`--output-file/-o`、双正式名 `--model-id/--model_id` |
| `cli/inference/video_generate.py` | `--num-devices`（dest `world_size`）、log/version；`--ulysses-size` 保持正式名 |
| `cli/inference/image_generate.py` | `--num-devices`（dest `world_size`）、`--chrome-trace-file`、`--model-id`、log/version；独立 parser |
| `optix/optimizer/optimizer.py` | kebab 选项、`--load-breakpoint`、引擎/benchmark 取值、CLI/环境日志分流 |
| `optix/logging.py` | `set_log_level`；无 CLI 开关时 `OPTIX_LOG_LEVEL` |
| `tests/regression/cli/test_spec_cli.py` | 4.7 回归 |

### References (参考资料)

- 《MindStudio 工具链命令行统一规范化设计方案》§4.2–4.7
- [RFC 模板](rfc_template.md)
- [text-generate-executor Skill RFC](rfc_text_generate_executor_skill_zh.md)
- [throughput-optimizer-executor Skill RFC](rfc_throughput_optimizer_executor_skill_zh.md)

### Glossary (术语表)

| 术语 | 含义 |
|:---|:---|
| 正式接口 | 出现在 `--help` 中的选项名 |
| 隐藏别名 | 可解析但不进 help 的旧选项名 |
| dest | argparse 写入 `Namespace` 的内部字段名 |
| DeviceProfile | TensorCast 设备画像，`--device` 的取值空间 |

### Documentation Update Plan (文档更新计划)

| 文档 | 变更 |
|:---|:---|
| 本 RFC | 以当前代码为准记录公开 CLI 表面、日志词表边界与兼容策略 |
| 中英文 user guide / quick start | 示例与参数表使用正式名；写明不提供 `--debug` / `--log-file` |
| Web UI 命令拼装 | 后续改为生成 `--chrome-trace-file` 等新名 |
