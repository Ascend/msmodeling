# RFC: Throughput Optimizer 适配 Dflash / DSpark 支持

## 1. Overview

Status (状态): Draft  
Author(s) (作者): linjingjie  
Created (创建日期): 2026-08-04  
Updated (更新日期): 2026-08-12（CLI 迁移为 `--speculative-method` / `--num-speculative-tokens` / `--acceptance-length`；Prefill 挂 draft；G2/G3 已落地）  
Related Issue/PR (相关 Issue/PR):  
Related Docs (相关文档):

- [Dflash 统一建模方案](./rfc_dflash_support_zh.md)（DFlash backbone / Draft 构图真源）
- TensorCast 层：`tensor_cast/layers/dflash.py`、`tensor_cast/layers/dspark.py`

**范围**：在**严格不改变**现有 `throughput_optimizer` 寻优能力与默认行为的前提下，把已落地的统一 **Dflash** 与 **DSpark** 建模接入吞吐寻优入口。  
**入口**：`cli/inference/throughput_optimizer.py`  
**核心服务**：`serving_cast/parallel_runner.py`、`serving_cast/service/*_throughput_optimizer.py`  
**建模真源**：

- Dflash：`tensor_cast/layers/dflash.py` / `dflash_qwen3.py`
- DSpark：复用 DFlash backbone + `tensor_cast/layers/dspark.py`（Markov / Confidence 串行头）

本 RFC **只覆盖寻优接线与折算**；层内算子账本以统一建模实现为准。DSpark 专属设计见 **§9**。

### 硬性门禁（全文最高优先级）

| # | 门禁 | 要求 |
|---|------|------|
| **G1** | **现有功能零回归** | 未启用 Dflash / DSpark 时，`throughput_optimizer` 的 CLI 默认、参数解析、TP/EP/MOE-DP/MTP 搜索、聚合/分离/PD 配比、折算、summary 语义与改前**比特级行为一致**（允许日志措辞微调，不允许结果/组合数/默认值漂移）。 |
| **G2** | **三方严格互斥** | `--speculative-method` / MTP（候选中任一 `>0`）**两两互斥，至多启用其一**。冲突必须在进入寻优主循环前 **fail-fast**，禁止静默覆盖、禁止「A 构图 + B 折算」混用。 |
| **G3** | **从属参数依赖主开关** | `--num-speculative-tokens` / `--acceptance-length` / `--num-draft-layers` / `--draft-model-config-path` 须先有 `--speculative-method`；`--dspark-markov-*` 须 `--speculative-method dspark`。未传主开关却传从属参数 → **fail-fast**。**禁止**仅凭 block 隐式启用。 |

未满足 G1/G2/G3 的实现视为 **本 RFC 不合格**，不得合入。

---

## 1.1 Summary

`text_generate` 与 `throughput_optimizer` 均已支持 Dflash / DSpark：主开关、G2 三方互斥、G3 从属/共享参数依赖、构图（`DflashWrapper` / `DsparkWrapper`）、Decode fold（DSpark → Dflash → MTP）与标签透传。本文以**当前实现**为真源，固化门禁与语义，防止回归。

本提案要求（按优先级）：

1. **G1 现有功能零回归（最高）**：未传 `--speculative-method` 时，寻优行为与改前**完全一致**。新增代码必须以显式 flag 门控。
2. **G2 三方严格互斥（最高）**：`--speculative-method {dflash,dspark}`、MTP（候选中任一 `>0`）不得两两共存；冲突必须报错退出。
3. **G3 参数依赖（最高）**：从属参数仅允许在对应主开关后出现；共享 draft 参数要求 `--speculative-method`。
4. **显式开启**：
   - Dflash：`--speculative-method dflash`；Decode `query_len = dflash_block_size`；折算 `/(accept+1)`，`accept` clamp 到 **`block_size-1`**。
   - DSpark：`--speculative-method dspark`；Decode `query_len = dspark_block_size`；折算同形，但 `accept` clamp 到 **`block_size`**（`sample_from_anchor=True`）。详见 §9。
   - Prefill/Decode **均挂并执行 draft**；`acceptance_length` **只折算 Decode**（TTFT 不折算，但 Prefill 延迟含 draft 算力）。
5. **本阶段不改寻优算法骨架**：不改 concurrency 搜索策略、不改 SLO 判定公式骨架、不把 draft block/acceptance/markov 强制扩进 TP×EP×MTP 笛卡尔积。

---

## 1.2 Motivation

- 用户需要在 **TPOT / 吞吐** 口径下评估「开 Dflash / DSpark」相对基线 / MTP 的收益，而不仅是单点 `text_generate`。
- DSpark = DFlash 并行 backbone + 块内串行 Markov/Confidence；寻优侧需保证 **构图（满 block）与折算（accept 上界不同）** 一致，并让 Markov 算力进入 Decode 延迟账本。
- 折算、CLI、互斥与构图接线已落地；文档需与实现一致，避免把 Prefill 误写成「仅 target / 不挂 draft」，或把共享 draft 参数误写成「仅 `--speculative-method dflash`」。

---

## 1.3 Goals / Non-Goals

### Goals

- **保证 G1**：未开 `--speculative-method` 时现有 `throughput_optimizer` 功能与结果不受影响。
- **保证 G2**：`--speculative-method` / MTP 两两互斥，冲突 fail-fast。
- **保证 G3**：从属参数依赖对应主开关；共享 draft 参数依赖 `--speculative-method`。
- CLI：`--speculative-method` 为主开关；其后方可覆盖 block / layers / acceptance / draft config；DSpark 另可覆盖 markov-rank / markov-head。
- `UserInputConfig` → `ConfigResolver.update_dflash_config` / `update_dspark_config` → `DflashWrapper` / `DsparkWrapper`。
- Decode 前向 `query_len = block`；Prefill/Decode **均挂并执行 draft**（`DflashWrapper`/`DsparkWrapper`），TTFT 账本含 draft 算力；`acceptance_length` **只折算 Decode**。
- Agg / Disagg / PD-ratio 的 **Decode 延迟折算**在启用 draft 时走对应分支；未启用时仍仅 MTP/基线。
- 结果展示可辨识 DFlash / DSpark；**未开时标签字符串与改前一致**。

### Non-Goals

- **禁止**为接入 Dflash/DSpark 而修改未开路径上的默认参数、搜索组合生成、警告阈值、PD 模式语义、MTP 折算公式。
- **禁止**未传主开关却接受从属参数（含「block≥2 隐式启用」）。
- 不在本阶段把 `block_size` / `acceptance_length` / `markov_*` 纳入与 TP×EP×MTP 同级的**强制**笛卡尔积搜索（可选后续 RFC；且仍须满足 G2）。
- 不建模真实 acceptance 控制流、KV crop、DynamicCache；**Confidence 不驱动动态 verify 深度**（仅计入 TensorCast 算力）。
- 不修改 Dflash/DSpark 层内算子实现（`context_kv_proj` / Markov 数值正确性等）。
- 不为「`--compile` + draft」增加特殊建议文案。

---

## 2. 现状（As-Is，以代码为准）

| 组件 | 状态 |
|------|------|
| `cli/inference/throughput_optimizer.py` | 已落地 `--speculative-method`、G2 三方互斥、G3 从属/共享 draft 参数依赖、block/acceptance 解析与 clamp |
| `cli/inference/text_generate.py` | 已有 `--speculative-method` + 从属/共享参数与互斥（**无** `--*-acceptance-length`；acceptance 仅吞吐 CLI） |
| `UserInputConfig` / `ConfigResolver` | 已有 `dflash*` / `dspark*` 与 `update_*_config`；optimizer / text_generate 均已接线 |
| `OptimizerData` | 已有 `dflash_*` / `dspark_*`；未开主开关时保持 None/0（G1） |
| `BaseThroughputOptimizer._fold_decode_latency_ms` | 已实现 DSpark → Dflash → MTP 单一分支 |
| `_resolve_forward_shape` | Decode 优先 `dspark_block` / `dflash_block`；Prefill 仍用有效 prompt 长度 |
| `format_parallel_label` | 可带 `DSpark=...` / `DFlash=...` / MTP；未开时与改前一致 |
| `parallel_runner` | Prefill/Decode 均按主开关填充 `OptimizerData` / UserInput 并构图 draft（`is_prefill` 只影响 DCP） |
| Dflash / DSpark 构图 | tensor_cast 已落地；本 RFC 不改层内实现 |

---

## 3. 方案设计

> **DSpark**：CLI / 折算上界 / 标签等与 DFlash 有差异的部分统一见 **§9**。本节以 Dflash 为主叙述；凡写「与 MTP 互斥」处，实现上均须扩展为 **Dflash / DSpark / MTP 三方互斥（G2）**。

### 3.0 规范性要求（Normative）

以下条款为 **MUST**（必须满足）：

1. **MUST** 以显式布尔条件门控一切 Dflash 专用逻辑：  
   `dflash_enabled = (args.speculative_method == "dflash")`（**仅** `--speculative-method dflash` flag，不以 `block_size` 隐式开启）。  
   当 `dflash_enabled == False` 时，执行路径 **MUST NOT** 写入 `model.dflash_config`，**MUST NOT** 将 `OptimizerData.dflash_block_size` 设为 `>=2`，**MUST NOT** 改变 MTP/`query_len`/折算分支选择。
2. **MUST** 在 CLI 解析完成、进入 `run_multi_device_loop` / worker 构图之前完成：  
   (a) **G3** 从属参数依赖检查；  
   (b) **G2** MTP 互斥检查。  
   冲突时 **MUST** 以非 0 退出并给出明确错误信息。
3. **MUST** 将下列情形均视为「MTP 已启用」（任一成立即与 `--speculative-method dflash` 冲突）：  
   - `args.num_mtp_tokens > 0`；  
   - `args.num_mtp_token_sizes`（或等价候选列表）中存在任一 `> 0` 的值。  
   仅当 MTP 候选为空或全部为 `0` 时，才允许 `--speculative-method dflash`。
4. **MUST NOT** 在 `--speculative-method dflash` 启用时仍向 `UserInputConfig.num_mtp_tokens` 写入 `>0`，**MUST NOT** 同时设置 `mtp_config` 与 `dflash_config`。
5. **MUST NOT** 为迁就 Dflash 而修改：`--tp-sizes`/`--ep-sizes`/`--moe-dp-sizes` 默认推断、`max-search-combinations` 默认值、concurrency 搜索策略默认值、量化默认值、PD 模式开关语义。
6. **MUST** 保证未开 `--speculative-method dflash` 时：`resolve_parallel_search_candidates` 的输入输出与改前一致；`format_parallel_label` 在无 Dflash 字段时与改前字符串一致。
7. **MUST（G3）**：Dflash-only 从属缺 `--speculative-method`、共享 draft 从属缺 `--speculative-method`、DSpark 从属缺 `--speculative-method dspark` 时，均不得以「用户显式传入」形式出现（见 3.2.2）；违者 fail-fast。

### 3.1 兼容性硬约束（G1：不改变现有寻优功能）

```text
dflash_enabled = (args.speculative_method == "dflash")   # G3：禁止用 block_size 隐式开启
mtp_active     = (num_mtp_tokens > 0) OR any(c > 0 for c in mtp_candidates)

IF speculative_method is None AND any_method_dependent_explicit:
    FAIL-FAST   # G3：从属参数依赖 --speculative-method

IF dflash_enabled AND mtp_active:
    FAIL-FAST   # G2：互斥，禁止继续

IF NOT dflash_enabled:
    # G1：与今日完全一致 —— 禁止进入任何 Dflash 专用分支
    不设置 model.dflash_config
    OptimizerData.dflash_block_size 保持 None/0（折算条件不成立）
    decode query_len = num_mtp_tokens + 1（MTP 关则为 1）
    折算：仅 MTP 公式或无投机（与改前相同）
    搜索组合 = TP × EP × MOE-DP × MTP（与改前相同）
ELSE:
    # 仅 Dflash 分支；MTP 必须已为 0 / 空候选
    num_mtp_tokens := 0；mtp 搜索候选 := [0] 或空且不产生 >0
    构图 + 折算按 3.3–3.4
```

**G1 验收（强制）**：

- 全量既有 `tests/regression/cli/test_throughput_optimizer.py` 在**不增加 Dflash 参数**时必须通过。  
- serving_cast optimizer 相关回归在未开 Dflash 时必须通过。  
- 对比改前：相同基线命令的搜索组合数、最优行关键数值字段（吞吐/TPOT/TTFT/并行标签中的非 Dflash 部分）不得无故变化。

**G2 / G3 验收（强制）**：见 3.2 互斥矩阵、3.2.2 从属参数规则与第 5 节用例。

### 3.2 CLI 设计与互斥 / 参数依赖

在 `throughput_optimizer` 的 Model 参数组增加：

| 参数 | 类型 | 默认 | 角色 | 说明 |
|------|------|------|------|------|
| `--speculative-method` | choice | None | **主开关** | `dflash` / `dspark`；显式传入才启用 |
| `--num-speculative-tokens` | int | 0 | **从属** | 需 `--speculative-method`；`n>=1` 时 `block_size=n+1`；`0`=builtin/config |
| `--acceptance-length` | float | 5.0 | **从属（仅吞吐）** | 需 `--speculative-method`；dflash clamp 到 `B-1`，dspark clamp 到 `B`；不改构图 |
| `--num-draft-layers` | int | 0 | **共享从属** | `--speculative-method` 后可用；0=用 config |
| `--draft-model-config-path` | str | None | **共享从属** | `--speculative-method` 后可用 |

**既有 MTP CLI 保持不变**（不得改名、不得改默认）：

| 参数 | 说明 |
|------|------|
| `--num-mtp-tokens` | 行为与今日一致；与 `--speculative-method` 互斥 |
| `--mtp-acceptance-rate` | 行为与今日一致；仅 MTP 路径消费 |

#### 3.2.1 与 MTP 互斥矩阵（G2）

| `--speculative-method dflash`？ | MTP 启用？（见 3.0.3） | 结果 |
|--------------|------------------------|------|
| 否 | 否 | 基线寻优（无投机）——与改前一致 |
| 否 | 是 | **仅 MTP**——与改前一致 |
| 是 | 否 | **仅 Dflash** |
| 是 | 是 | **非法：立即 error 退出** |

#### 3.2.2 从属参数依赖（G3，MUST）

定义参数集合（与 `validate_draft_spec_cli_args` 一致）：

```text
METHOD_DEPENDENT = {   # validate_draft_spec_cli_args._METHOD_DEPENDENT_OPTIONS
  --num-speculative-tokens,
  --acceptance-length,   # 仅 throughput_optimizer；text_generate 无此参数
  --num-draft-layers,
  --draft-model-config-path,
}
DSPARK_ONLY = {
  --dspark-markov-rank,
  --dspark-markov-head,
}
```

**判定「用户显式传入」**：检查 `sys.argv` 是否包含对应 option 字符串（因 `--acceptance-length` 默认 5.0，**不能**仅用「!=5.0」判断）。

**规则**：

```text
IF speculative_method is None AND any_method_dependent_explicit:
    parser.error("--num-speculative-tokens / --acceptance-length / "
                 "--num-draft-layers / --draft-model-config-path require --speculative-method")

IF speculative_method != "dspark" AND any_dspark_only_explicit:
    parser.error("--dspark-markov-rank / --dspark-markov-head require --speculative-method dspark")
```

| 命令 | 结果 |
|------|------|
| （无 draft 相关参数） | 合法，基线 |
| `--speculative-method dflash` | 合法，启用（block 用 builtin） |
| `--speculative-method dflash --num-speculative-tokens 15` | 合法 |
| `--speculative-method dspark --num-draft-layers 4` | 合法（共享从属） |
| `--num-speculative-tokens 15`（无 `--speculative-method`） | **非法（G3）** |
| `--num-draft-layers 4`（无 `--speculative-method`） | **非法（G3）** |
| `--acceptance-length 3`（无 `--speculative-method`） | **非法（G3）** |
| `--draft-model-config-path x.json`（无主开关） | **非法（G3）** |

> 说明：`text_generate` 与 `throughput_optimizer` **一致**——必须以显式 `--speculative-method` 启用；**禁止**仅凭 `block_size >= 2` 隐式启用。差异仅在于 acceptance 折算参数只存在于吞吐 CLI。

非法示例（G2）：

```text
--speculative-method dflash --num-mtp-tokens 2
--speculative-method dflash --num-mtp-tokens 0 2          # 候选中含 >0
```

合法示例：

```text
# 仅 MTP（现有功能）
--num-mtp-tokens 2 --mtp-acceptance-rate 0.9 0.6

# 仅 Dflash
--speculative-method dflash
--speculative-method dflash --num-speculative-tokens 15 --acceptance-length 5

# 显式 MTP=0 可与 --speculative-method dflash 并存（0 视为未启用 MTP）
--speculative-method dflash --num-mtp-tokens 0
```

校验顺序（`arg_parse` / `main`，**MUST**）：

1. 解析参数。  
2. **G3**：Dflash 从属缺 `--speculative-method`、共享 draft 从属缺 `--speculative-method`、DSpark 从属缺 `--speculative-method dspark` → error。  
3. 规范化 MTP 候选（保持现有 `_normalize_mtp_token_values` 语义）。  
4. 判定 `dflash_enabled` / `dspark_enabled` / `mtp_active`。  
5. **G2**：三方至多其一 → 否则 error（在 `run_multi_device_loop` 之前）。  
6. 若 `--speculative-method dflash` 且 `block_size==0` → 从 builtin/path 解析 `block_size`。  
7. `acceptance_length` clamp 到 `block_size-1`。  
8. 若 `dflash_enabled`：强制 `num_mtp_tokens=0`，且 **MUST NOT** 再启用 `update_mtp_config(>0)`。

**搜索策略（本阶段，满足 G1）**：

- Dflash 参数为**单点配置**，**MUST NOT** 改变 `resolve_parallel_search_candidates` 对 TP/EP/MOE-DP/MTP 的既有计算。  
- `--speculative-method dflash` 开启时 MTP 侧 **MUST** 为未启用；搜索组合与「MTP=0 的今日行为」一致的 TP×EP×MOE-DP。  
- **MUST NOT** 因引入 Dflash 而提高未开 `--speculative-method dflash` 时的 `max-search-combinations` 触发频率。

### 3.3 数据流

```text
throughput_optimizer CLI (--speculative-method dflash ...)
  → UserInputConfig.from_args (dflash*)
  → ConfigResolver.update_dflash_config
  → maybe_enable_dflash → DflashWrapper（既有）
  → parallel_runner.OptimizerData.dflash_block_size / dflash_acceptance_length
  → Decode forward: query_len = block   # draft + verify 满图
  → Prefill forward: target + draft（query_len=有效 prompt；acceptance 不折算）
  → latency_ms_decode_folded = record.latency_ms / (accept + 1)
  → TPOT / throughput 汇总（原有公式，输入改为折算后延迟）
```

### 3.4 Decode / Prefill 语义

| 阶段 | 行为 |
|------|------|
| Prefill / TTFT | **target + draft**（`DflashWrapper` Prefill 路径）；`query_len`=有效 prompt；`dflash_acceptance_length` **不参与折算**（TTFT 延迟仍含 draft 算力） |
| Decode / TPOT（仅 `dflash_enabled`） | 一次仿真步 = draft(block) + target verify(block)；`query_len = dflash_block_size`；折算 `/(accept+1)` |
| Decode / TPOT（仅 MTP，未开 Dflash） | **与改前完全一致**：`query_len = num_mtp_tokens + 1`；折算 `sum(rates[:n])+1` |
| Decode / TPOT（均未开） | **与改前完全一致**：`query_len = 1`；无投机折算 |

折算选择（**互斥后的单一分支**，禁止双公式叠加；优先级 DSpark > Dflash > MTP）：

```text
IF dspark_enabled:          # 此时 dflash/mtp 必为 False（G2 已保证）
    accept = clamp(dspark_acceptance_length, 0, block_size)   # 上界 B（sample_from_anchor）
    latency /= (accept + 1)
ELIF dflash_enabled:        # 此时 mtp 必为 False
    accept = clamp(dflash_acceptance_length, 0, block_size - 1)  # 上界 B-1
    latency /= (accept + 1)
ELIF num_mtp_tokens > 0:    # 现有 MTP 路径，禁止改动公式
    latency /= (sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1)
ELSE:
    latency 不变
```

`_resolve_forward_shape` 伪代码（未开 draft 的分支 **MUST** 保持原表达式）：

```python
if is_decode:
    dspark_block = optimizer_data.dspark_block_size or 0
    dflash_block = optimizer_data.dflash_block_size or 0
    if dspark_block >= 2:
        resolved_query_len = query_len or dspark_block
    elif dflash_block >= 2:
        resolved_query_len = query_len or dflash_block
    else:
        # 现有逻辑，不得改写为其它默认值
        resolved_query_len = query_len or (self.num_mtp_tokens + 1)
    ...
```

### 3.5 已落地文件（验收清单）

| 文件 | 职责 | G1/G2 注意 |
|------|------|-----------|
| `cli/inference/throughput_optimizer.py` | `--speculative-method` CLI；G2/G3；MTP 互斥 | 未开 Dflash 时不得改 MTP 解析/默认 |
| `serving_cast/parallel_runner.py` | 仅当 `dflash_enabled` 填充 `OptimizerData.dflash_*` 与 UserInput | Prefill/Decode 均挂；未开时字段保持 None/0 |
| `serving_cast/service/base_throughput_optimizer.py` | `_resolve_forward_shape` / `_fold_decode_latency_ms` Dflash 分支 | `else` 保持原 MTP/`+1` 逻辑一字不差 |
| `serving_cast/service/disagg_throughput_optimizer.py` | Decode 折算对齐 3.4 单一分支 | 未开 Dflash 时必须仍走原 MTP 折算 |
| `serving_cast/service/utils.py` | `format_parallel_label` 可选追加 DFlash | 未开 Dflash 时返回值与改前相同 |

**明确不改（G1）**：TP/EP/MOE-DP 候选生成函数本体、concurrency 搜索、`max-search-combinations` 默认值、PD ratio 组合算法、量化默认值、MTP acceptance 长度校验逻辑（除非与 Dflash 互斥检查相邻增加，不得削弱原校验）。

### 3.6 Agg / Disagg / PD-ratio

| 模式 | Prefill 任务 | Decode 任务 |
|------|--------------|-------------|
| 聚合 | target + draft（无 acceptance 折算） | Dflash 满图 + 折算 |
| 分离 | 同上 | 同上 |
| PD 配比 | Prefill 实例同样挂 Dflash（当前实现不按阶段拆掉 draft） | Decode 实例开 Dflash + 折算 |

开启方式：同一 CLI 全局 `--speculative-method dflash`；`parallel_runner` Prefill/Decode **均**填充 `dflash_*` 并构图 `DflashWrapper`（`is_prefill` 只影响 DCP=1）。Prefill 返回 primary=target logits，draft 子图仍入账（防 `--compile` DCE）。

### 3.7 展示与可观测性

- `format_parallel_label`：例如 `... | DFlash=16/acc=5`；DSpark 见 §9.7（`DSpark=.../markov=...`）。  
- chrome-trace 文件名：在既有 `mtp{n}` 规则旁增加 `dflash{block}` / `dspark{block}`（仅开启时）。  
- 日志：打印 block / draft layers / acceptance（复用 `UserInputConfig` 已有 print）。

---

## 4. 分阶段落地

### Phase 0（本 RFC 必达）

1. CLI + **G3 从属参数依赖** + **G2 互斥 fail-fast** + block 解析（含 **§9 DSpark**）  
2. `parallel_runner` 仅在对应主开关时灌 `OptimizerData` / `UserInput`  
3. Decode `query_len = block`（gated）  
4. Disagg 折算对齐 3.4 / §9.4 单一分支  
5. **G1 回归**：未开 `--speculative-method` 全绿且与改前基线一致  
6. **G2/G3 测试**：互斥与从属参数非法组合全部非 0 退出  
7. Dflash 冒烟：`--speculative-method dflash` + agg decode + tpot（无 MTP）  
8. DSpark 冒烟：见 §9.9（`--speculative-method dspark` + fold clamp 到 B）

### Phase 1（可选，另开变更）

- `--num-speculative-tokens` 多候选搜索，纳入组合计数与 warning  
- skill / 文档示例命令更新  

### Phase 2（明确不做）

- acceptance 分布采样、KV 回滚仿真  
- **任何形式的 Dflash+MTP 同时开启或混合折算**（永久违反 G2）  

---

## 5. 测试设计

### 5.1 G1：现有功能零回归（未开 Dflash）

| 用例 | 验证 |
|------|------|
| 默认 CLI | 不传 `--speculative-method` 时，解析结果中 `dflash_enabled=False`；MTP/并行默认与改前一致 |
| 搜索组合数 | 同命令下 TP×EP×MOE-DP×MTP 组合数与改前相同 |
| MTP-only | `--num-mtp-tokens 2` 等现有用例：构图/折算/summary 行为不变 |
| 基线无投机 | 无 MTP 无 Dflash：`query_len=1`，无折算 |
| 标签 | `format_parallel_label` 不含 DFlash 后缀，与改前相同 |
| 回归套件 | `tests/regression/cli/test_throughput_optimizer.py` 全过 |

### 5.2 G2 / G3：互斥与参数依赖

| 用例 | 验证 |
|------|------|
| `--speculative-method dflash` + `--num-mtp-tokens 2` | 非 0 退出；错误含 mutually exclusive / 互斥 |
| `--speculative-method dflash` + MTP 多候选含 `>0` | 同上（进入寻优前失败） |
| `--num-speculative-tokens 15`（无 `--speculative-method`） | **G3** 非 0 退出；错误含 require `--speculative-method` |
| `--acceptance-length 3`（无 `--speculative-method`） | **G3** 失败（即使默认本就是 5，显式传入也须拒绝） |
| `--num-draft-layers 4`（无 `--speculative-method`） | **G3** 失败 |
| `--speculative-method dspark --num-draft-layers 4` | **合法**（共享从属） |
| `--draft-model-config-path x`（无 draft 主开关） | **G3** 失败 |
| 互斥/依赖失败时 | **MUST NOT** 写出部分结果 / **MUST NOT** 启动 worker 构图 |

### 5.3 Dflash 功能（仅在 G2 通过后）

| 用例 | 验证 |
|------|------|
| OptimizerData 透传 | worker 内 `dflash_block_size` / `acceptance` 正确 |
| Decode shape | `query_len==block`；trace 含 draft `attention`×N |
| 折算 | `folded = raw/(accept+1)`；accept clamp；**不**再叠加 MTP 分母 |
| Prefill | 挂并执行 draft；acceptance **不**折算 TTFT；primary 仍为 target logits |
| Disagg decode | 折算与 base 3.4 一致 |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 误改默认搜索 / MTP（违反 G1） | 全部 Dflash 逻辑 `if dflash_enabled` 门控；G1 专项回归为合入阻断条件 |
| 静默混用 MTP 构图 + Dflash 折算（违反 G2） | CLI 与 worker 入口双检互斥；禁止只改折算不检构图 |
| 无 `--speculative-method` 却吃掉 block/acceptance（违反 G3） | argv 判定显式传入；单测覆盖「仅从属参数」 |
| 只折算未改 `query_len` | Phase 0 将 shape 与折算绑在同一 `dflash_block_size>=2` |
| Disagg 与 Agg 折算分叉 | 共用 3.4 单一分支 helper |
| 显存变大导致 batch 上界下降 | 仅 Dflash 启用时的预期结果；**不得**因此改全局 reserved memory 默认 |
| `CopyLayerWrapper` 与 draft 层 | 构图侧已处理；寻优侧无需再改 |

---

## 7. 示例命令

### 7.1 基线（行为不变）

PD混部场景

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --tpot-limits 50
```

PD分离场景 Prefill模式

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --disagg \
    --ttft-limits 2000
```

PD分离场景 Decode 模式

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --num-devices 8 \
    --input-length 3500 \
    --output-length 1500 \
    --compile \
    --quantize-linear-action W8A8_DYNAMIC \
    --quantize-attention-action DISABLED \
    --disagg \
    --tpot-limits 50
```

### 7.2 启用 Dflash

```bash
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 16 \
  --input-length 3500 --output-length 128 \
  --tpot-limits 50 --compile \
  --speculative-method dflash --num-speculative-tokens 15 --acceptance-length 5 \
  --quantize-linear-action W4A8_DYNAMIC
```

### 7.3 非法组合（必须失败）

```bash
# G2：与 MTP 互斥
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --input-length 3500 --output-length 128 \
  --speculative-method dflash --num-mtp-tokens 2

# G3：从属参数缺少 --speculative-method
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --input-length 3500 --output-length 128 \
  --num-speculative-tokens 15

python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --input-length 3500 --output-length 128 \
  --acceptance-length 5
```

### 7.4 仅 MTP（G1：现有功能，行为不变）

```bash
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --input-length 3500 --output-length 128 \
  --tpot-limits 50 \
  --num-mtp-tokens 2 --mtp-acceptance-rate 0.9 0.6
```

---

## 8. Drawbacks

1. 本阶段不搜 `block_size`，最优 block 需用户多次手动跑。  
2. `acceptance_length` 为外生标量，吞吐对 accept 敏感；需在报告中标明假设。  
3. Dflash / DSpark decode 比单 token decode 更重，相同 SLO 下最优 concurrency 可能变小——这是特性结果而非回归。  
4. DSpark 串行 Markov 增加 Decode 延迟；gated/rnn 比 vanilla 更重——标签应标明 `markov=` 便于对比。

---

## 9. DSpark 特性设计（寻优接线）

> 算法真源：[Dflash 统一建模方案](./rfc_dflash_support_zh.md)；实现见 `tensor_cast/layers/dspark.py`（G5：独立于 `dflash.py`）。  
> 本节只规定 **throughput_optimizer 如何启用、构图口径与折算**；不改 Markov 数值语义。

### 9.1 DSpark 相对 DFlash（寻优视角）

```text
DSpark = DFlash 并行 backbone（一次）
       + 块内串行 MarkovHead / ConfidenceHead / sample（N = block_size）
```

| 维度 | DFlash | DSpark |
|------|--------|--------|
| Backbone / Decoder | Qwen3 DFlash | **复用** |
| 额外模块 | 无 | `MarkovHead`（vanilla/gated/rnn）+ 可选 `ConfidenceHead` |
| `sample_from_anchor` | `False` | **`True`** |
| Decode `query_len` | `block_size` | **`block_size`（同）** |
| 有效投机长度上界 | `block_size - 1` | **`block_size`** |
| acceptance clamp | `[0, B-1]` | **`[0, B]`** |
| 折算公式形 | `latency / (accept + 1)` | **同形，分母上界不同** |
| Confidence | — | **只计入算力**；本阶段 **不**动态截断 verify |
| 寻优主开关 | `--speculative-method dflash` | **`--speculative-method dspark`** |

要点：

1. **构图**：Decode 一次仿真步 = 满 block draft（含串行头）+ target verify；`query_len = dspark_block_size`。  
2. **折算**：外生标量 `dspark_acceptance_length`；**MUST** clamp 到 `block_size`（不是 `B-1`）。  
3. **Markov 算力**：由 TensorCast `DsparkWrapper._sample_sequential` 入账；寻优侧无需单独加折算项。  
4. **Prefill / TTFT**：与 DFlash 相同——**挂并执行 draft**；acceptance **不参与折算**（TTFT 延迟仍含 draft 算力）。

### 9.2 硬门禁在 DSpark 上的落地

```text
dspark_enabled = (args.speculative_method == "dspark")
dflash_enabled = (args.speculative_method == "dflash")
mtp_active     = (num_mtp_tokens > 0) OR any(c > 0 for c in mtp_candidates)

IF (NOT dspark_enabled) AND any_dspark_dependent_explicit:
    FAIL-FAST   # G3

IF (NOT dflash_enabled) AND (NOT dspark_enabled) AND any_shared_draft_dependent_explicit:
    FAIL-FAST   # G3：--num-draft-layers / --draft-model-config-path

IF count_true(dspark_enabled, dflash_enabled, mtp_active) > 1:
    FAIL-FAST   # G2：三方至多其一

IF NOT dspark_enabled AND NOT dflash_enabled:
    # G1：与今日完全一致
    ...
ELIF dspark_enabled:
    num_mtp_tokens := 0；禁止写入 dflash_config
    填充 OptimizerData.dspark_*；dflash_* := None
    构图 maybe_enable_dspark → DsparkWrapper
    Decode query_len = dspark_block；fold 按 §9.4
ELIF dflash_enabled:
    # 既有 §3 Dflash 分支
    ...
```

**MUST NOT**：

- 同时设置 `dspark_config` 与 `dflash_config` / `mtp_config(>0)`。  
- 用 `OptimizerData.dflash_*` 承载 DSpark（字段分离，避免 accept 上界误用 `B-1`）。  
- 因 Confidence 输出改写 `dspark_acceptance_length`（本阶段禁止）。

### 9.3 CLI 设计

在 `throughput_optimizer` Model 参数组增加（对齐 `text_generate`，并补齐寻优必需的 acceptance）：

| 参数 | 类型 | 默认 | 角色 | 说明 |
|------|------|------|------|------|
| `--speculative-method` | choice | None | **主开关** | 取值 `dspark` 启用 DSpark（与 dflash/MTP 互斥） |
| `--num-speculative-tokens` | int | 0 | **从属** | ≥2 覆盖 block；0=builtin/config |
| `--acceptance-length` | float | 5.0 | **从属** | 折算标量；clamp 到 `block_size` |
| `--dspark-markov-rank` | int | 256 | **从属** | `0`=不建 MarkovHead；默认 256 |
| `--dspark-markov-head` | choice | `vanilla` | **从属** | `vanilla` / `gated` / `rnn` |
| `--num-draft-layers` | int | 0 | **共享从属** | 需 `--speculative-method dspark` 或 `--speculative-method dflash` |
| `--draft-model-config-path` | str | None | **共享从属** | 需 `--speculative-method dspark` 或 `--speculative-method dflash` |

DSpark 从属集合：

```text
DSPARK_DEPENDENT = {
  --num-speculative-tokens,
  --acceptance-length,
  --dspark-markov-rank,
  --dspark-markov-head,
}
SHARED_DRAFT_DEPENDENT = {
  --num-draft-layers,
  --draft-model-config-path,
}
```

判定「显式传入」与 §3.2.2 相同（看 `sys.argv` / `SUPPRESS`；**不能**仅用「!= 默认值」判断 acceptance / markov-rank）。

#### 9.3.1 互斥矩阵（G2 扩展）

| `--speculative-method dspark` | `--speculative-method dflash` | MTP 启用？ | 结果 |
|------------|------------|------------|------|
| 否 | 否 | 否 | 基线——与改前一致 |
| 否 | 否 | 是 | 仅 MTP——与改前一致 |
| 否 | 是 | 否 | 仅 Dflash |
| 是 | 否 | 否 | **仅 DSpark** |
| 其它任一「两者以上为真」 | | | **非法：立即 error** |

非法示例：

```text
--speculative-method dspark --speculative-method dflash
--speculative-method dspark --num-mtp-tokens 2
--speculative-method dspark --speculative-method dflash --num-mtp-tokens 0   # 仍与 --speculative-method dflash 冲突
--num-speculative-tokens 15                 # 无 --speculative-method dspark → G3
--speculative-method dflash --dspark-markov-rank 128        # Markov 挂在错误 method → G3
```

合法示例：

```text
--speculative-method dspark
--speculative-method dspark --num-speculative-tokens 15 --acceptance-length 8
--speculative-method dspark --dspark-markov-rank 256 --dspark-markov-head gated
--speculative-method dspark --num-draft-layers 2 --draft-model-config-path ./draft.json
--speculative-method dspark --num-mtp-tokens 0            # 0 视为未启用 MTP
```

校验顺序（相对 §3.2 扩展，**MUST**）：

1. 解析参数。  
2. **G3**：Dflash 从属 / DSpark 从属 / 共享 draft 从属依赖检查。  
3. 规范化 MTP 候选。  
4. 判定 `dspark_enabled` / `dflash_enabled` / `mtp_active`。  
5. **G2**：至多其一。  
6. 若 `--speculative-method dspark` 且 `block_size==0` → 从 builtin/path 解析。  
7. `dspark_acceptance_length` clamp 到 `block_size`（**不是** `B-1`）。  
8. 若 `dspark_enabled`：强制 `num_mtp_tokens=0`；`dflash=False`；只走 `update_dspark_config`。

**搜索策略**：与 Dflash 相同——DSpark 参数为**单点配置**，**MUST NOT** 改变 TP×EP×MOE-DP×MTP 笛卡尔积生成。

### 9.4 Decode / Prefill 与折算

| 阶段 | 行为 |
|------|------|
| Prefill / TTFT | **target + draft**（`DsparkWrapper` 继承 Prefill 路径；Markov/Confidence 仅 Decode propose）；`dspark_acceptance_length` **不参与折算** |
| Decode / TPOT（`--speculative-method dspark`） | 一次仿真步 = draft(block，含 N 步 Markov/Confidence) + target verify(block)；`query_len = dspark_block_size` |
| 折算 | `latency_ms / (accept + 1)`，`accept ∈ [0, block_size]` |

```text
# 与实现 BaseThroughputOptimizer._fold_decode_latency_ms 对齐
IF dspark_block_size >= 2:
    accept = clamp(dspark_acceptance_length ?? 5.0, 0, dspark_block_size)
    return latency_ms / (accept + 1)
```

与 DFlash 的**唯一折算差异**：上界是 **`B` 不是 `B-1`**。  
原因：`sample_from_anchor=True` 时 draft 提案长度为 `block_size`，平均接受长度可到满块。

**禁止**：

- DSpark 路径再叠加 MTP `sum(rates)`。  
- 用 Confidence 动态改写本步 `accept`（Phase 0 / 本 RFC）。  
- 把 Prefill 误写成「不挂 draft」（当前实现 Prefill **会**挂并执行 draft backbone）。

### 9.5 数据流

```text
throughput_optimizer CLI (--speculative-method dspark ...)
  → G3/G2 校验
  → UserInputConfig.from_args (dspark*)
  → ConfigResolver.update_dspark_config
  → maybe_enable_dspark → DsparkWrapper（DFlash backbone + heads）
  → parallel_runner.OptimizerData.dspark_block_size / dspark_acceptance_length / dspark_markov_rank
       （同时 dflash_* = None）
  → Decode forward: query_len = block   # draft(含串行头) + verify
  → Prefill forward: target + draft（query_len=有效 prompt；acceptance 不折算）
  → latency_ms_decode_folded = raw / (accept + 1)   # accept ≤ B
  → TPOT / throughput 汇总（原有公式）
```

### 9.6 Agg / Disagg / PD-ratio

| 模式 | Prefill 任务 | Decode 任务 |
|------|--------------|-------------|
| 聚合 | target + draft（无 acceptance 折算） | DSpark 满图 + 折算 |
| 分离 | 同上 | 同上 |
| PD 配比 | Prefill 实例同样挂 DSpark（当前实现不按阶段拆掉 draft） | Decode 实例开 DSpark + 折算 |

开启方式：同一 CLI 全局 `--speculative-method dspark`；`parallel_runner` Prefill/Decode **均**填充 `dspark_*` 并构图 `DsparkWrapper`（`is_prefill` 只影响 DCP=1）。

### 9.7 展示与可观测性

- `format_parallel_label`：例如 `... | DSpark=16/acc=8/markov=256`。  
  - 有 DSpark 字段时 **不要**再追加 `DFlash=`。  
  - 未开 DSpark/Dflash 时字符串与改前一致（G1）。  
- chrome-trace 文件名：在既有 `mtp{n}` / `dflash{block}` 旁增加 `dspark{block}`（仅开启时）。  
- 日志：打印 block / draft layers / acceptance / markov_rank / markov_head（复用 `UserInputConfig` print）。

### 9.8 已落地文件（验收清单）

| 文件 | 职责 | 状态提示 |
|------|------|----------|
| `cli/inference/throughput_optimizer.py` | `--speculative-method`（含 dspark）CLI；G2/G3；与 MTP 互斥 | **已落地** |
| `serving_cast/parallel_runner.py` | Prefill/Decode 均在 `dspark_enabled` 时灌 `OptimizerData.dspark_*` | **已落地** |
| `serving_cast/service/base_throughput_optimizer.py` | fold / shape：DSpark 优先于 Dflash | 已实现 |
| `serving_cast/service/disagg_throughput_optimizer.py` | Decode 折算走 `_fold_decode_latency_ms` | 已对齐 |
| `serving_cast/service/utils.py` | `format_parallel_label` 追加 `DSpark=...` | 已实现 |
| `tensor_cast/layers/dspark.py` | Markov/Confidence / Wrapper | 建模真源；本 RFC 不改 |

**明确不改（G1）**：与 §3.5 相同——TP/EP/MOE-DP 候选生成、concurrency 搜索、PD ratio 算法、MTP 公式本体。

### 9.9 分阶段落地（DSpark）

#### Phase 0（本 RFC 必达）

1. `throughput_optimizer` CLI：`--speculative-method dspark` + 从属参数 + 共享 draft 参数依赖  
2. G2 三方互斥 + G3 从属依赖 fail-fast  
3. `parallel_runner` 仅在 `--speculative-method dspark` 时填充 `dspark_*`；强制清空 `dflash_*` / MTP  
4. Decode `query_len = block`；fold clamp 到 **B**  
5. 标签 `DSpark=B/acc=.../markov=...`  
6. G1：未开任何 draft 时全绿且与改前一致  
7. 冒烟：`--speculative-method dspark` + agg decode + tpot（无 MTP/Dflash）；`vanilla` Markov

#### Phase 1（可选）

- `--dspark-markov-head gated|rnn` 在寻优冒烟中覆盖（算力差异可见）  
- `--num-speculative-tokens` 多候选搜索（另开变更，计入组合 warning）  
- skill / 用户文档示例更新  

#### Phase 2（明确不做）

- Confidence 驱动的动态 verify 长度 / 调度  
- acceptance 分布采样、KV 回滚  
- **任何** DSpark+Dflash / DSpark+MTP 同时开启或混合折算  

### 9.10 测试设计（DSpark）

#### G1 回归（未开 DSpark/Dflash）

与 §5.1 相同；额外确认：不传 `--speculative-method` 时 `dspark_enabled=False`，标签无 `DSpark=`。

#### G2 / G3

| 用例 | 验证 |
|------|------|
| `--speculative-method dspark` + `--speculative-method dflash` | 非 0；互斥 |
| `--speculative-method dspark` + `--num-mtp-tokens 2` | 非 0；互斥 |
| `--num-speculative-tokens 15`（无 `--speculative-method`） | G3 失败 |
| `--acceptance-length 3`（无 `--speculative-method`） | G3 失败（看 argv） |
| `--dspark-markov-rank 128`（无 `--speculative-method dspark`） | G3 失败 |
| `--num-draft-layers 4`（无 draft 主开关） | G3 失败 |
| 失败时 | MUST NOT 启动 worker / 写出部分结果 |

#### DSpark 功能

| 用例 | 验证 |
|------|------|
| OptimizerData | `dspark_block_size` / `acceptance` / `markov_rank` 正确；`dflash_*` 为 None |
| Decode shape | `query_len == block`；trace 含 draft attention×N + Markov 串行算子 |
| 折算 | `folded = raw/(accept+1)`；`accept` 可到 `block`；**不**用 `B-1` |
| Prefill | 挂并执行 draft；acceptance **不**折算 TTFT |
| Disagg decode | 与 base fold helper 一致 |
| 标签 | 含 `DSpark=`；不含 `DFlash=` / `MTP=` |

### 9.11 示例命令

```bash
# 仅 DSpark（合法）
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 16 \
  --input-length 3500 --output-length 128 \
  --tpot-limits 50 --compile \
  --speculative-method dspark --num-speculative-tokens 15 --acceptance-length 8 \
  --dspark-markov-rank 256 --dspark-markov-head vanilla \
  --quantize-linear-action W4A8_DYNAMIC

# gated Markov（合法；算力更重）
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --input-length 3500 --output-length 128 \
  --tpot-limits 50 \
  --speculative-method dspark --num-speculative-tokens 7 --dspark-markov-head gated

# 非法：与 Dflash 互斥
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --speculative-method dspark --speculative-method dflash

# 非法：与 MTP 互斥
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --speculative-method dspark --num-mtp-tokens 2

# 非法：G3 缺主开关
python -m cli.inference.throughput_optimizer <model> \
  --device <DEVICE> --num-devices 8 \
  --num-speculative-tokens 15
```

### 9.12 风险与缓解（DSpark 增量）

| 风险 | 缓解 |
|------|------|
| 误用 DFlash 的 `B-1` clamp | `OptimizerData.dspark_*` 独立字段；fold 分支优先读 dspark |
| 与 Dflash 标签/字段串线 | G2 互斥 + runner 互斥清空对方字段 |
| Markov 头未入账却按高 accept 折算 | Phase 0 冒烟检查 trace 含 embedding/mm（或 addmm）×N |
| Confidence 被误当作动态调度 | Non-Goals + 单测：accept 仅为 CLI 标量 |
| gated/rnn 默认开启导致对比不公 | 默认 `vanilla`；文档标明 head 类型 |

---

## Appendix

### Glossary

| 术语 | 含义 |
|------|------|
| **G1** | 现有 `throughput_optimizer` 功能零回归：未开 `--speculative-method` 时行为与改前一致 |
| **G2** | `--speculative-method` / MTP **两两互斥**：冲突 fail-fast，禁止混用构图/折算 |
| **G3** | 从属参数仅在对应主开关后合法；共享 draft 参数需 `--speculative-method`；禁止 block_size 隐式启用 |
| Dflash 启用条件 | **仅** `--speculative-method=dflash`（`dflash_enabled = (args.speculative_method == "dflash")`） |
| DSpark 启用条件 | **仅** `--speculative-method dspark`（`dspark_enabled = (args.speculative_method == "dspark")`） |
| Dflash 从属参数 | `--num-speculative-tokens` / `--acceptance-length`（及仅 Dflash 的覆盖项） |
| DSpark 从属参数 | `--num-speculative-tokens` / `--acceptance-length` / `--dspark-markov-rank` / `--dspark-markov-head` |
| 共享 draft 从属参数 | `--num-draft-layers` / `--draft-model-config-path`（需 `--speculative-method`） |
| MTP 启用条件 | `num_mtp_tokens > 0` 或 MTP 搜索候选中存在 `>0` |
| `dflash_block_size` | 含 anchor 的 block 长；Decode `query_len`；accept 上界 **B-1** |
| `dflash_acceptance_length` | 折算用平均接受长度；仅 `--speculative-method dflash` 路径；clamp 到 `B-1` |
| `dspark_block_size` | 含 anchor 的 block 长；Decode `query_len`；accept 上界 **B**（`sample_from_anchor`） |
| `dspark_acceptance_length` | 折算用平均接受长度；仅 `--speculative-method dspark` 路径；clamp 到 `B` |
| `dspark_markov_rank` / `markov_head` | Markov 头容量与类型；影响 Decode 算力账本，不改变折算公式形 |
| 现有寻优功能不变 | 未传 `--speculative-method` 时，CLI 默认、MTP、搜索、折算、PD 模式与今日一致 |

### 合入检查清单（PR）

- [ ] 未传 `--speculative-method` 时，既有 CLI / serving 回归全绿  
- [ ] 仅 MTP 命令结果与改前一致（抽样对比）  
- [ ] `--speculative-method dflash` + 任意 MTP`>0` 候选 → 非 0 退出  
- [ ] `--speculative-method dspark` + `--speculative-method dflash` / MTP`>0` → 非 0 退出  
- [ ] 无主开关但传从属参数 → 非 0 退出（G3）  
- [ ] `--speculative-method` 路径下 `num_mtp_tokens==0` 且未创建对方 draft config  
- [ ] DSpark fold：`accept` clamp 到 `block_size`（不是 `B-1`）  
- [ ] 未开 draft 时 `format_parallel_label` / 搜索组合数无漂移  

### 与统一建模 RFC 的关系

- **构图 / KV / 算子账本**：DFlash 以统一 Dflash 建模为准；DSpark 以 `dspark.py` + 算法文档 §2.8 为准。  
- **本文件**：只规定 `throughput_optimizer` 如何在 **G1（零回归）**、**G2（三方互斥）**、**G3（参数依赖主开关）** 约束下接线与折算；DSpark 专章见 **§9**。
