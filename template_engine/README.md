# 模型模板图使用指南

模板图用于减少吞吐寻优和 `text_generate` 单点仿真中执行模型 forward、compile 获取计算图的耗时。可以通过 Skill 自动制作并接入，也可以通过 CLI 手动采集样本、构建模板和生成目标计算图。

新构建流程按固定算子序列和张量引用关系组织结构桶；目标 B/S 用于计算桶内各算子的 shape 和参数。交付范围为随项目提供的模板，以及本版本正式 CLI／Skill 生成的模板。

模板文件默认采用紧凑 JSON + XZ 压缩，重复公式和字段定义只保存一份；CLI 和 Skill 均自动使用，无需额外参数。压缩不改变覆盖范围、算子序和张量信息，文件大小随模型及覆盖配置数量变化。

模板命中时不会调用 `torch.compile()`；`--compile` 用于匹配模板的采集模式。未覆盖请求回退时才启用官方编译路径。同一工作进程内的 runner 共享已加载的模板；多进程寻优时，各进程分别加载一次相同版本，文件更新后新建的 runner 会重新加载。

模板仅负责生成目标计算图；后续统一进入官方 `ModelRunner.run_inference`，由官方 Runtime、性能模型和内存统计产生结果。模板侧不单独计算性能指标。

性能仿真验收检查算子序、张量 shape/dtype、算子参数及张量引用关系；stride 差异单列为诊断，不作为模板命中的限制条件。

方案设计见 [Issue #591](https://gitcode.com/Ascend/msmodeling/issues/591)。

## 1. 使用前准备

本文命令面向 Linux Bash，在项目根目录、已激活的项目 Python 环境中执行。文中的 `python` 指该环境的 Python 解释器。环境安装见项目 [README](../README.md)。使用 Skill 时需要完整源码工作副本，包含 `.git` 和 `.agents/skills/`，仅安装 wheel 不包含完整 Skill 工作流。

本文以 `deepseek-ai/DeepSeek-V4-Flash` 为例，可替换为当前项目已支持的其他模型。模型配置应能从本地缓存或模型服务读取。本次交付只考虑 compile-on 模板，采集和生成时均显式指定 `--compile`。默认使用模型原生层数；示例采集是项目的 CPU/Meta 性能仿真取图，不需要真实 NPU 推理。

```bash
python -m cli.inference.template_graph --help
```

查看统一入口。本文主要使用 `capture`（采集样本）、`build`（构建模板）、`generate`（生成目标图）。

## 2. 使用 Skill 自动生成并接入寻优

### 2.1 通过 AI 助手执行

在项目中加载 [msmodeling-template-workflow](../.agents/skills/msmodeling-template-workflow/SKILL.md)，输入：

> 使用 msmodeling-template-workflow，为 deepseek-ai/DeepSeek-V4-Flash 生成模型模板图。使用默认典型覆盖范围，保留原生层数，先执行一个配置单元，验证后自动注册进寻优，输出报告和未覆盖配置。

Skill 会执行规划、采集、构建、独立验证和合格模板注册。需要指定范围时，可在请求中补充 B、S、并行配置、Prefill/Decode、量化、MTP、DFC 和执行预算。

B 对应全局并发请求数 `--num-queries`，不是单卡批量。也可以直接给出输入、输出长度，例如：

> 为 deepseek-ai/DeepSeek-V4-Pro 制作模板。32 卡、EP=32、TP=1/2/4/8/16、DCP=1，W8A8_DYNAMIC/C8，B=1～255，输入 16384、输出 1024，DFC 关闭、MTP=0。列出独立比较通过率和未覆盖配置。

普通 Decode 下，这组长度对应 Prefill 的 S=16384、Decode 的代表 S=16897；输出 1024 不代表 Decode 一次输入 1024 个 query token。

随附的 DSV4-Pro 模板包含上述 32 卡、W8A8_DYNAMIC/C8 配置的已验证子集，并保留原有配置单元。
其中 Decode 收录 B=1～255 对应的全部本地批量；部分极短 Prefill 分支按精确锚点及等价 DP 请求覆盖。
具体 B/S 仍须通过模板覆盖判定；范围内未覆盖的请求回退原生仿真，不表示整个参数笛卡尔积均可命中。

随附 GLM-5.3 模板的配置范围如下，使用寻优时自动发现，无需指定模板文件名：

| 参数 | 范围 |
|---|---|
| 模型 | `zai-org/GLM-5.3`，原生 78 层 |
| 卡数、EP | 均为 32 |
| TP / 对应 DP | `1/32`、`2/16`、`4/8`、`8/4`、`16/2` |
| DCP、MOE_DP | 均为 1 |
| 全局 B | 1～255 |
| 输入 / 输出长度 | 输入 1～16384，输出 1～1024 |
| 阶段 | Prefill、Decode；普通 Decode 的 query-length 为 1 |
| 量化 | 线性 `W8A8_DYNAMIC`，Attention `INT8`（C8） |
| 其他 | compile 开启、DFC 关闭、PP=1；Decode MTP=0 |

按官方寻优代表请求形状，Prefill 的模板 S 为输入长度，Decode 的 S 为输入长度加输出长度的一半（向下取整）再加 1。
此处覆盖普通请求路径；prefix cache、chunked prefill 及表外配置不在这份范围内。

### 2.2 手动启动同一工作流

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py preflight
```

检查所用 Python、项目路径和 CLI 是否可加载，不启动模型采集。

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py plan --model-id deepseek-ai/DeepSeek-V4-Flash --run-dir ./template_runs/flash-r1
```

创建采集计划，输出 `plan.json`、`model-config.json` 和 `epoch.json`。运行目录必须尚不存在；此步骤不采集计算图。

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py plan \
  --model-id deepseek-ai/DeepSeek-V4-Flash \
  --input-length 16384 --output-length 1024 \
  --decode-batch-coverage all \
  --run-dir ./template_runs/flash-long-request
```

在默认典型离散配置范围内，为请求的整个 B 区间准备 Decode 本地批量样本，并增加 16K/1K 长请求独立目标。先检查计划数量再执行；采样覆盖和实际通过率以报告为准。模型上下文不足时直接报告，不降低长度。

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py run --run-dir ./template_runs/flash-r1 --max-cells 1 --max-captures 16 --max-seconds 1800
```

先执行一个配置单元，最多尝试采集 16 次，累计 CLI 执行时间预算为 1800 秒。完成后输出模板、比较报告，并自动注册通过验证的部分；其余计划未执行时报告为 `partial`。

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py run --run-dir ./template_runs/flash-r1 --resume --max-captures 512 --max-seconds 10800
```

继续已有计划并复用核验通过的结果。预算按同一运行目录累计，不随恢复清零；达到预算后保留已完成结果，不保证全部配置都能覆盖。

只生成和验证、不注册时，在 `run` 命令后添加 `--no-register`。该选项不会撤销已经安装的模板。

### 2.3 指定覆盖范围

默认范围文件为 [scope.json](../.agents/skills/msmodeling-template-workflow/references/scope.json)。如需调整，先复制一份：

```bash
cp .agents/skills/msmodeling-template-workflow/references/scope.json ./template_scope.json
```

编辑副本中的范围和典型配置，保留完整 JSON。常用字段如下：

| 字段 | 用途 |
|---|---|
| `requested.B`、`requested.S` | 请求数量和序列长度的上下限 |
| `requested.world_size` | 总卡数范围 |
| `requested.phase` | `prefill`、`decode` |
| `requested.MTP`、`requested.DFC` | MTP 数量列表和 DFC 开关列表 |
| `requested.linear_families` | 线性量化家族，如 `DISABLED`、`W8A8` |
| `requested.attention_actions` | 注意力量化动作列表，如 `["DISABLED"]` |
| `requested.compile` | 本工作流固定为 `true`，只制作 compile-on 模板 |
| `typical_moe_layouts`、`typical_dense_layouts` | 典型并行配置，每项顺序为 `[TP, EP, MOE_DP, DCP]` |
| `sampling` | 训练和独立验证的序列长度选择 |
| `execution` | 采集次数、执行时间和单命令超时预算 |

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py plan --model-id deepseek-ai/DeepSeek-V4-Flash --scope ./template_scope.json --run-dir ./template_runs/flash-custom-r1
```

使用修改后的范围创建新计划，然后按 2.2 节执行 `run`。修改范围不保证模型支持所有组合，以计划和实际报告为准。

默认 profile 在需求范围中选取典型配置，不遍历全部组合。最终执行清单见 `plan.json`，实际通过数量见 `report.json`，计划数量不代表命中数量。

### 2.4 查看结果

运行以下命令查看结果摘要：

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("./template_runs/flash-r1/report.json").read_text(encoding="utf-8"))
fields = ("status", "planned_cells", "independent_targets_passed", "optimizer_template_ready", "registration")
print(json.dumps({key: report.get(key) for key in fields}, ensure_ascii=False, indent=2))
PY
```

查看总体完成情况、独立目标通过数，以及注册状态、安装路径和备份路径。

| 字段/状态 | 使用者应如何理解 |
|---|---|
| `status=passed` | 本计划全部独立目标通过，且无流程错误或预算停止 |
| `status=partial` | 有未执行、拒绝、失配、流程错误或预算停止；查看 `targets`、`errors` 和 `stop` |
| `targets[].status=compared_pass` | 该目标已通过独立图比较 |
| `registration.status=registered` | 合格部分已安装到本项目的模板目录 |
| `optimizer_template_ready=true` | 本地寻优能够发现已注册模板，不表示全部范围都覆盖 |

`run` 退出码为 `0` 表示计划全部通过，`2` 表示 `partial`。先跑一个单元时出现 `partial` 属于正常的未完成状态，应同时检查错误字段。

## 3. 使用 CLI 手动制作模板

手动流程为：采集训练样本 → 构建模板 → 生成目标图。若需要自动独立验证和注册，推荐使用第 2 节 Skill 工作流。

### 默认目录与简化命令

`capture` 可以省略 `--cache-dir`，样本默认保存在当前工作目录的 `./template_runs/captures/` 下，并按完整模型标识隔离子目录。计算图文件名由请求参数自动生成，不需要手动命名。

```bash
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 4 --query-length 512 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED
```

采集一个 Prefill 样本到默认目录。context 长度默认 0，MOE_DP/DCP 默认 1，MTP 默认 0，层数默认采用模型原生值。线性量化的官方默认值为 `W8A8_DYNAMIC`，此处显式关闭；`--compile` 仍须显式指定。构建前按需要变换 B/S 采集多个训练样本。

```bash
python -m cli.inference.template_graph build
```

默认目录只有一个模型的样本时，自动识别该模型及样本编译状态，直接在项目的 `template_engine/templates/` 生成 `<模型简称>_cmp_on.json.xz` 等模板文件，供寻优自动发现；终端 JSON 的 `built` 字段给出实际路径。默认目录包含多个模型时，必须用 `--cache-dir` 指向目标模型子目录；同一模型混合 compile-on/off 样本时，必须用 `--compile-mode on` 或 `off` 筛选。当前交付使用 compile-on。

```bash
python -m cli.inference.template_graph build --cache-dir ./template_runs/manual-training --compile-mode on
```

也可从自定义样本目录自动识别模型并命名，输出仍位于项目的 `template_engine/templates/`。目录中的样本须具有一致、可读取的模型身份元数据；无法确定模型时会拒绝自动命名。

显式 `--cache-dir` 和 `--template` 继续有效，以下各节展示自定义路径的完整用法。默认输出路径已位于寻优自动查找目录，构建成功后新启动的寻优进程即可发现；重复构建会更新同一路径的模板。直接 `build` 不包含独立图比较，正式使用前应验证目标配置；需要自动独立验证流程时使用第 2 节 Skill。`generate` 仍通过 `--template` 指定要读取的模板。

### 3.1 采集一个 Prefill 样本

```bash
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 4 --query-length 512 --context-length 0 --tp-size 1 --ep-size 2 --moe-dp-size 1 --dcp-size 1 --compile --quantize-linear-action DISABLED --quantize-attention-action DISABLED --cache-dir ./template_runs/manual-prefill
```

采集 B=4、S=512 的 Prefill 图，启用编译、关闭量化。终端返回样本信息，图文件及同路径附加 `.meta` 的元数据文件保存到指定目录下；两者应一起保留。

### 3.2 采集一个 Decode 样本

```bash
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 8 --query-length 2 --context-length 1022 --decode --num-mtp-tokens 1 --tp-size 1 --ep-size 2 --moe-dp-size 1 --dcp-size 1 --compile --quantize-linear-action DISABLED --quantize-attention-action DISABLED --cache-dir ./template_runs/manual-decode
```

采集 B=8、MTP=1、总长度 S=1024 的 Decode 图。本轮 query 长度为 MTP+1=2，context 长度为 1022；具体模型和 MTP 组合是否支持，以实际采集结果为准。

### 3.3 准备训练样本并构建模板

```bash
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 4 --query-length 512 --context-length 0 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED --cache-dir ./template_runs/manual-training
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 4 --query-length 1024 --context-length 0 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED --cache-dir ./template_runs/manual-training
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 8 --query-length 512 --context-length 0 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED --cache-dir ./template_runs/manual-training
python -m cli.inference.template_graph capture --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 8 --query-length 1024 --context-length 0 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED --cache-dir ./template_runs/manual-training
```

四条命令采集 B=4/8、S=512/1024 的 Prefill 训练样本，其余参数保持一致。训练目录只放本模型、本轮构建的训练图；不同模型、层数或编译状态不要混放。

```bash
python -m cli.inference.template_graph build --cache-dir ./template_runs/manual-training --compile-mode on --template ./template_runs/manual-template.json.xz
```

使用训练目录构建模板，输出 `manual-template.json.xz` 和终端构建统计。上述四点是操作示例，不保证足以覆盖所有新 B/S；构建成功后仍需检查目标能否生成并验证。

### 3.4 从模板生成目标计算图

```bash
python -m cli.inference.template_graph generate --model-id deepseek-ai/DeepSeek-V4-Flash --num-queries 6 --query-length 768 --context-length 0 --tp-size 1 --ep-size 2 --compile --quantize-linear-action DISABLED --template ./template_runs/manual-template.json.xz --output ./template_runs/candidate.json.xz
```

尝试生成未用于上述训练的 B=6、S=768 计算图，保存到 `candidate.json.xz`。目标未覆盖时返回拒绝原因和非零退出码，不自动实跑补答案。默认仅生成图，不进行性能预测。

安装项目命令行工具后，也可使用 `msmodeling inference template-graph capture/build/generate`，参数与上述模块命令一致。

### 3.5 常用 CLI 参数

| 参数 | 使用说明 |
|---|---|
| `--model-id` | 模型标识；采集和生成时使用同一模型 |
| `--num-queries` | 全局请求数量 B |
| `--query-length`、`--context-length` | S 为两者之和；Prefill 的 context 为 0，Decode 的 query 为 MTP+1 |
| `--tp-size`、`--ep-size`、`--moe-dp-size`、`--dcp-size` | 并行配置，默认均为 1；必须使用项目支持的合法组合 |
| `--decode` | 启用 Decode；不传时为 Prefill |
| `--compile` | 本流程必须指定，生成时须与模板一致 |
| `--num-mtp-tokens` | MTP 数量，默认 0 |
| `--quantize-linear-action` | 线性量化动作；默认 `W8A8_DYNAMIC`，关闭时显式传 `DISABLED` |
| `--quantize-attention-action` | 注意力量化动作，默认 `DISABLED` |
| `--compilation-config enable_dispatch_ffn_combine` | 开启 DFC；当前模板入口不支持任意其他编译特性 |
| `--cache-dir` | 采集输出或构建输入目录；默认 `./template_runs/captures/`，构建时须能确定唯一模型 |
| `--template` | 构建时可省略，自动输出到项目的 `template_engine/templates/`；生成时须指定输入模板 |
| `--compile-mode on` | 本流程使用开启编译采集的样本构建模板 |
| `--output` | 生成图的保存路径，父目录须已存在 |

```bash
python -m cli.inference.template_graph capture --help --model-id deepseek-ai/DeepSeek-V4-Flash --quantize-linear-action DISABLED
python -m cli.inference.template_graph build --help
python -m cli.inference.template_graph generate --help
```

查看当前版本的完整参数和量化选项，不启动采集或构建。

## 4. 产物与寻优使用

Skill 的主要产物均保存在 `--run-dir` 指定的目录中：

| 文件/目录 | 用途 |
|---|---|
| `plan.json` | 本轮模型、范围和采集/验证计划 |
| `model-config.json`、`epoch.json` | 模型配置与运行版本记录 |
| `captures/` | 样本计算图及元数据 |
| `template.json.xz` | 本轮构建的模板，是否通过验证以报告为准 |
| `candidates/`、`comparisons/` | 目标生成图与独立比较结果 |
| `report.json` | 实际通过、拒绝、未执行、错误及注册情况 |
| `receipts/`、`logs/` | 命令执行记录和排障日志 |
| `registration/validated-template.json.xz` | 本轮通过验证并用于安装的模板部分 |
| `registration/backups/` | 覆盖旧注册前保存的备份 |

Skill 默认把合格部分安装到当前项目的 `template_engine/templates/`。实际安装文件以 `report.json.registration.path` 为准。

注册成功后，在同一工作副本或安装环境运行原有 `throughput_optimizer` 命令，保持模板优化启用即可；已经启动的 worker 需要重新加载。模板命中时使用生成图，未覆盖配置回退原生仿真。`partial` 不妨碍已注册部分使用，但不表示整个请求范围都能命中。

`build` 省略 `--template` 时直接输出到寻优自动查找目录；显式指定其他输出路径时，需要将合格模板安装到对应环境。`generate` 仅生成目标图，不安装模板。需要制作、独立验证和安装一并完成时使用 Skill。

### 4.1 查看模板路由日志

在原有寻优命令中添加 `--log-level info`（或 `debug`），可查看模板路由记录。保存日志时用 `2>&1 | tee <日志路径>`，同时收集标准输出和标准错误；完整命令见下一节。

```bash
grep -E '\[template (HIT|FALLBACK)\]' ./template_runs/optimizer-compare/template-enabled.log
```

从已保存的寻优日志中筛选模板记录。`[template HIT]` 表示请求已成功完成模板重放；`[template FALLBACK]` 表示选择原生仿真路径，具体原因见 `reason`。

| 日志字段 | 含义 |
|---|---|
| `scope=request` | 针对具体请求作出的路由决定，包含阶段、B、S 及并行配置 |
| `scope=runner` | 构造 runner 时的策略决定，尚无具体 B/S；阶段可能为 `unknown`，B/S 为 `None` |
| `phase`、`B`、`S` | Prefill/Decode、请求数量、序列长度；Decode 的 S 为总长度 |
| `TP`、`EP`、`MOE_DP`、`DCP` | 当前并行配置 |
| `compile`、`MTP`、`DFC`、`linear`、`attention` | 编译、MTP、DFC 和量化配置 |
| `ops` | 模板成功重放的算子数；回退记录中通常为 `None` |
| `reason` | 命中说明或回退原因，例如未注册匹配模板、覆盖拒绝、缺少 Decode 语义标记 |

主动关闭模板时，原因是 `template_optimization_disabled`；未发现匹配注册模板时为 `no_matching_registered_template`。这些策略级记录使用 `scope=runner`。日志记录次数不等于不同配置数量：同一配置可能被多次评估，策略级记录也不是逐请求记录。核对覆盖情况时应查看具体请求配置及独立比较报告，不能直接用日志行数统计命中配置数。

### 4.2 跳过模板进行原生仿真对照

`--no-template-graph` 跳过注册模板和工作负载复用，使用原生 Meta 仿真取图；不会删除或修改已注册模板。不传时默认启用模板优化，未覆盖配置仍会回退。

先设置模型和设备画像，定义两轮共用的参数：

```bash
set -o pipefail
MODEL_ID='deepseek-ai/DeepSeek-V4-Flash'
: "${DEVICE:?请先用 export DEVICE=设备画像名称 设置已注册且容量足够的设备画像}"
LOG_DIR='./template_runs/optimizer-compare'
mkdir -p "$LOG_DIR"

common_args=(
  --model-id "$MODEL_ID" --device "$DEVICE" --num-devices 2
  --input-length 768 --output-length 1
  --tp-sizes 1 --ep-sizes 2 --moe-dp-sizes 1 --batch-range 2 8
  --disagg --ttft-limit 10000 --compile
  --quantize-linear-action DISABLED --log-level info
)
```

在项目根目录的 Bash 中执行。`DEVICE` 必须是当前项目已注册、能够容纳所选模型的设备画像；可将参数替换成自己的合法寻优配置。`TEST_DEVICE` 默认容量不保证能容纳示例模型，不应为对照而修改它的内存容量。`pipefail` 使日志管道保留寻优命令的失败状态。

```bash
python -m cli.inference.throughput_optimizer "${common_args[@]}" 2>&1 | tee "$LOG_DIR/template-enabled.log"
```

默认启用模板。命中已注册模板的请求使用模板重放，未命中请求回退原生仿真；查看日志确认实际路由。

```bash
python -m cli.inference.throughput_optimizer "${common_args[@]}" --no-template-graph 2>&1 | tee "$LOG_DIR/template-disabled.log"
```

禁用注册模板及工作负载复用，走原生 Meta 路径。两轮使用相同设备、模型和请求参数，均保留 `--compile`，只改变模板开关。开关控制模板优化，不控制编译；对照同时包含工作负载复用的影响，不能将全部耗时差直接归因于模板重放。

### 4.3 使用 text_generate 做单点对照

`text_generate` 在 `--compile` 下默认发现已注册模板；命中时使用模板，未覆盖时回退原生仿真。
它与寻优共享模板目录、路由日志和 `--no-template-graph` 开关，不需要指定模板文件名。

```bash
: "${DEVICE:?请先设置已注册且容量足够的设备画像名称}"
common_args=(
  --model-id zai-org/GLM-5.3 --device "$DEVICE" --num-devices 32
  --ep-size 32 --moe-dp-size 1 --dcp-size 1 --compile
  --quantize-linear-action W8A8_DYNAMIC --quantize-attention-action INT8
  --performance-model analytic --log-level info
)
prefill_args=(--tp-size 8 --num-queries 37 --query-length 16379 --context-length 0)
decode_args=(--tp-size 16 --num-queries 23 --query-length 1 --context-length 16892 --decode)
```

以上使用原生 78 层 GLM-5.3。Prefill 输入为 16379；Decode 的代表总长度为 16893，对应输入 16380、输出 1024、MTP=0。
验证外推时，须在相同配置桶内按 `(K=ceil(B/DP), S)` 排除训练等价点；仅更换全局 B 不构成独立验证。

```bash
python -m cli.inference.text_generate "${common_args[@]}" "${prefill_args[@]}"
python -m cli.inference.text_generate "${common_args[@]}" "${prefill_args[@]}" --no-template-graph
```

第一条验证 Prefill 模板路径，应看到 `[template HIT]`；第二条明确走原生路径。核对预测执行时间、吞吐和内存结果，比较进程耗时评估取图加速。

```bash
python -m cli.inference.text_generate "${common_args[@]}" "${decode_args[@]}"
python -m cli.inference.text_generate "${common_args[@]}" "${decode_args[@]}" --no-template-graph
```

以相同方式验证 Decode。两轮始终保留 `--compile`，只改变模板开关。需要逐算子核对时，两轮分别增加 `--chrome-trace <不同的输出文件>`。
自定义融合插件、模板身份未包含的量化或模块布局等配置会回退原生路径，原因见 `[template FALLBACK]`。

## 5. 常见问题

| 情况 | 处理方式 |
|---|---|
| 计划目录已经存在 | 继续旧任务用 `run --resume`；创建新计划使用新目录 |
| 报告为 `partial` | 查看 `targets`、`errors` 和 `stop`，区分未执行、拒绝、失配与流程错误 |
| 预算耗尽 | 明确增加累计预算后使用 `--resume` 继续 |
| 输入参数非法 | 命令直接报错，按错误信息修正参数；不会将错误输入转为模板回退 |
| 模板损坏或实例化内部错误 | 直接报告错误；核对模板文件或重新构建，不按普通未命中处理 |
| 模板拒绝目标配置 | 查看原因，核对模型/量化/并行/编译参数；扩展样本后重新构建和验证 |
| 模型、依赖或代码版本变化导致恢复失败 | 创建新运行目录重新制作，不修改旧来源记录绕过检查 |
| 采集失败、超时或记录仍为 running | 查看 `logs/` 并确认真实进程是否结束，排除原因后开启修正轮次 |
| 已生成文件但寻优未命中 | 检查注册状态、运行环境及模型/层数/compile 是否一致，重新加载 worker |
| 不希望覆盖本地注册 | 执行 `run` 时加 `--no-register`；已覆盖的旧文件可从报告所列备份恢复 |

实际覆盖以独立比较报告为准；支持指定模型并不代表该模型所有参数组合均能命中。

Prefill 的全局请求数不同时，若官方 DP 切分后的本地批量相同，可使用已有批量代表点的公式。
查询长度变化时，仍须匹配结构分支并具备可推导的 shape 公式；模板不以已采集的最大查询长度作为统一上限。
Decode 按本地批量分别构建固定结构桶，在桶内推导不同上下文长度的 shape，不跨本地批量拟合。
