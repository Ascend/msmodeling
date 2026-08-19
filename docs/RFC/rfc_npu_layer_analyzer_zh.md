# NPU Layer Analyzer 设计文档

状态 (Status): Draft
作者 (Authors): @yuyinkai
创建日期 (Created): 2026-08-01
更新日期 (Updated): 2026-08-18

---

## 1. 概述

### 1.1 简介

NPU Layer Analyzer是一套NPU前向推理算子分析与层结构对比工具集，从 Chrome Trace Event JSON / kernel_details CSV 出发，完成 Forward 切分 → 层提取 → 子结构标注 → **双工具对比** 全流程。本 RFC 描述该工具集的整体架构、核心算法、模块设计与输出规范。

工具集由 5 个独立脚本组成，构成一条完整的数据流水线：

| 工具 | 输入                 | 输出                      | 作用 |
|------|--------------------|-------------------------|------|
| `trace_json_to_csv.py` | trace JSON         | kernel_details CSV      | Chrome Trace Event → 标准 CSV |
| `npu_layer_analyzer.py` | kernel_details CSV | forward_XXX_layerN.csv   | Forward 切分 + 层提取 + Stage 标注 |
| `layer_analyzer.py` | kernel_details CSV | `_layered.csv`/`_layerN.csv` | 全局标注 + 层提取 + Stage 标注 |
| `layer_compare.py` | 两个 layer CSV       | compare_result.xlsx     | 按 Stage 对比时间/算子/Shape |
| `npu_layer_compare.py` | CSV + JSON/CSV     | 完整输出目录                  | 统一入口，一键跑完全流程 |

核心价值：

- **双工具交叉验证**：`npu_layer_analyzer`（NPU 侧 trace，含 Forward 切分）与 `layer_analyzer`（框架侧 trace，含全局标注）独立提取层结构，再通过 `layer_compare` 按 Stage 对齐，定位两侧算子/时间差异
- **Attention 锚点层边界**：以 Attention（而非 RMSNorm/上下采样）作为一层起始，避免 NPU 未融合 RMSNorm 散开导致的层错位
- **2 段 Stage 标注**：每层固定拆为 `Attention → FFN`（Dense 层）或 `Attention → MOE`（MoE 层）两段，与 Transformer DecoderLayer 语义对齐
- **未融合 RMSNorm 识别**：以 `aten.rsqrt.default` 为锚点向前向后扩展，将散开的 8 个基础算子识别为一个 RMSNorm
- **MoE 自动检测 + 代表层选取**：自动识别 MoE 层并额外选取一个 MoE 代表层，与 Dense 代表层分别导出
- **通信算子排除**：对比时自动排除 `all_reduce` / `all_gather` / `hcom_*` 等通信算子，保证算法时间纯净

### 1.2 动机

在 NPU 推理性能分析中，存在以下痛点：

| 痛点 | 说明 |
|------|------|
| Trace 文件庞大难读 | 一次 forward 的 kernel_details.csv 可达数万行，人工定位层边界耗时且易错 |
| NPU 侧 RMSNorm 未融合 | NPU 上 RMSNorm 常被拆成 8 个基础算子（view/add/pow/mean/rsqrt/mul...），无法用单一正则识别 |
| 层边界锚点不统一 | 以 RMSNorm 为层边界时，未融合 RMSNorm 会让一层散开成多段；以 Attention 为锚点更稳定 |
| 两侧 trace 差异难定位 | NPU 侧 trace 与框架侧 trace 算子粒度不同，需逐 Stage 对齐才能发现"少了哪个算子"或"时间差在哪段" |
| Prefill / Decode 混杂 | 一次采集包含多次 forward（prefill + 多次 decode），需自动切分才能单独分析 |
| MoE 层与 Dense 层混排 | MoE 模型一层 Dense 一层 MoE 交替，分析时需分别选代表层 |
| 通信算子干扰时间统计 | All-Reduce 等通信算子占用大量时间，但与算法本身无关，需在对比时排除 |
| 输出格式分散 | 各工具输出 CSV 散落多个目录，需统一为 xlsx 多 Sheet 便于查看 |

### 1.3 目标

- 提供 5 个可独立使用的脚本，覆盖 JSON→CSV、Forward 切分、层提取、对比全流程
- 通过 `npu_layer_compare.py` 统一入口一键跑完全流程，输出 3 个 xlsx（npu_out / layer_out / compare_result）
- 层边界以 Attention 为锚点，稳定应对未融合 RMSNorm 场景
- 每层标注 2 段 Stage（Attention / FFN 或 MOE），由 MoE 自动检测结果（内部 `is_moe` 标志）决定第二段类型，与 Transformer 语义对齐
- 自动识别未融合 RMSNorm 序列（rsqrt 锚点）
- 自动检测 MoE 层并分别选取 Dense / MoE 代表层
- 对比时自动排除通信算子，保证 Stage 时间纯净
- 支持 task-id 定位特定 Forward segment

## 环境依赖

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Python | ≥ 3.10 | 使用 dataclass(slots=True)、新 match 语法 |
| openpyxl | ≥ 3.0 | xlsx 生成（layer_compare.py） |

安装：

```bash
pip install "openpyxl>=3.0"
```

**非目标**：

- 不做算子级性能建模（仅做 Stage 级时间汇总对比）
- 不做跨 forward 的趋势分析（每次只导出代表层）
- 不做训练场景的层分析（仅针对推理 prefill/decode）
- 不做分布式多机通信拓扑分析（通信算子仅做排除）
- 不替代 tensor_cast 仿真框架（仅作为仿真 vs 实测对比的外部工具）

---

## 2. 项目架构分析

### 2.1 工具集概览

```markdown
npu_layer_analyzer/
├── README.md                    # 项目文档
├── trace_json_to_csv.py         # JSON → CSV 转换
├── layer_common.py              # 共享工具模块（正则/子结构标注/Stage 提取/CSV 写出）
├── layer_analyzer.py            # 全局标注 + 层提取（Attention 锚点）
├── npu_layer_analyzer.py        # Forward 切分 + 层提取（含 task-id 定位）
├── layer_compare.py             # 双工具层对比 → xlsx（2 或 4 Sheet）
├── npu_layer_compare.py         # 统一入口（一键全流程）
├── rfc_npu_layer_analyzer_zh.md # 本文档（中文）
├── rfc_npu_layer_analyzer_en.md # 本文档（英文）
├── samples/
│   ├── kernel_details.csv       # 示例：NPU 侧 trace CSV
│   └── qwen1.json               # 示例：框架侧 Chrome Trace JSON
└── output/                      # 示例输出
    ├── npu_out.xlsx
    ├── layer_out.xlsx
    └── compare_result.xlsx
```

### 2.2 数据流

整体数据流如下，`npu_layer_compare.py` 串联所有步骤：

```markdown
┌──────────────┐                ┌──────────────────────┐
│ kernel_      │  npu_layer_    │ npu_out.xlsx         │
│ details.csv  │  analyzer.py   │  ├── summary         │
│ (NPU 侧)     │ ─────────────→ │  ├── forward_XXX     │
└──────────────┘                │  └── forward_XXX_layerN│
                                └──────────────────────┘
                                          │
                                          │ (layer CSV)
                                          ▼
┌──────────────┐  trace_json_   ┌──────────────────────┐
│ qwen1.json   │  to_csv.py     │ layer_out.xlsx       │
│ (框架侧)     │ ─────────────→ │  ├── *_kernel_details│
└──────────────┘       │        │  ├── *_layered       │
                       │ layer_  │  └── *_layerN       │
                       ▼ analyzer│ └──────────────────────┘
                  ┌──────────────┐         │
                  │ *_layerN.csv │         │
                  └──────────────┘         │
                                          ▼
                                ┌──────────────────────┐
                                │ compare_result.xlsx  │
                                │  ├── 总比较          │
                                │  └── 算子明细        │
                                └──────────────────────┘
```

两个 layer CSV 分别来自：

- **文件 A**：`npu_layer_analyzer.py` 的 `forward_XXX_layerN.csv`（NPU 侧，经过 Forward 切分，`N` 为层号）
- **文件 B**：`layer_analyzer.py` 的 `*_layerN.csv`（框架侧，全局标注后提取，`N` 为层号）

### 2.3 核心算法

#### 2.3.1 Forward 切分（npu_layer_analyzer.py）

Forward 切分将一次采集的 CSV 切分为多个 forward segment，核心逻辑：

1. **主 stream 选择**：统计每个 stream 的 embedding/attention/rows 数，按 `(attention, embedding, not_na, rows)` 评分选主 stream（attention 第一优先级，因为 attention 最能代表计算主流，embedding 可能在非计算 stream 上）
2. **Gap 阈值自动计算**：取所有正 gap，按降序排列找最大 `high/low` 比值（≥5 且 high≥1000us），否则 fallback 到 `max(1000, median*20, p95*3)`
3. **Embedding 锚点切分**：

   - 有多个 embedding：相邻 embedding 之间为一个 segment（`embedding-to-embedding`）
   - 最后一个 embedding：向后扩展到第一个大 gap（`embedding-to-gap`）
4. **未覆盖行补切**：剩余主 stream 行按 gap 阈值补切（`gap` method）
5. **Segment 分类与校验**：
   - 含 embedding → `prefill`（embedding 检查在 `output_rows` 上进行，即所有 stream 的算子，避免 embedding 不在主 stream 时误判为 decode）
   - gap method 且 attention 数匹配 → `decode`
   - 否则 → `unknown`
   - 校验：attention 数是否在 `expected ± tolerance`，内部 gap 是否超阈值，边界是否可信
6. **层提取**：`extract_layer_rows` 仅保留主 Stream ID 的算子，过滤非主 stream 的算子（如通信算子），保证层 CSV 时间纯净

#### 2.3.2 层边界：Attention 锚点

层边界以 **Attention** 为起始锚点（而非 RMSNorm / 上下采样），原因：

- **多 Attention 模型**：取第一对相邻 Attention 之间的算子作为一层
- **单 Attention 模型**：从 Attention 到其后第一个 SwiGlu/MLP

Attention 识别规则（以下模式均识别为 Attention）：

```markdown
- `attention` / `infer_attention` / `mla` / `ring_mla` / `grouped_attention`
- `recurrent` / `attn_chunk_gated`（Recurrent 模型）
- `delta_rule` / `gated_delta`（Recurrent Attention）
```

#### 2.3.3 Stage 标注（2 段）

每个层提取 CSV 包含 `Stage` 列，标注 2 个阶段（Dense 层与 MoE 层第二段不同，由 `is_moe` 参数控制）：

| Stage | 含义 | 关键算子 |
|-------|------|----------|
| `Attention` | 第一个 Attention 及其后续 RmsNorm 后剩余算子 | FusedInferAttentionScore / delta_rule |
| `FFN` | Dense 层第二段：AddRmsNormBias → RmsNorm（含两端） | AddRmsNormBias / 未融合 RMSNorm 序列 / MatMul → SwiGlu → MatMul |
| `MOE` | MoE 层第二段：AddRmsNormBias → RmsNorm（含两端） | AddRmsNormBias / GroupedMatmul / DispatchFFNCombine / MoeGatingTopK |

定位逻辑（在 `extract_substructure` 中）：

1. 找层内第一个 Attention → `Attention` stage（Attention Stage = RmsNorm 后剩余算子 + ATT → AddRmsNormBias 前）
2. Attention 之后从第一个 AddRmsNormBias 开始，到下一个 RmsNorm（含两端）→ 第二段 Stage
3. Dense 层第二段标为 `FFN`，MoE 层第二段标为 `MOE`，由 `is_moe` 参数控制

#### 2.3.4 未融合 RMSNorm 识别

当 RMSNorm 未融合为单个算子时，以下算子序列会被识别为一个 RMSNorm：

```markdown
aten.view.default → aten.add.Tensor → prims.convert_element_type.default →
aten.pow.Tensor_Scalar → aten.mean.dim → aten.add.Tensor →
aten.rsqrt.default → aten.mul.Tensor → aten.mul.Tensor
```

识别策略（`mark_unfused_rmsnorm`）：

1. 遍历所有行，找到 `Name` 包含 `rsqrt` 的行作为锚点
2. 向前扩展（最多 15 行）：连续命中 `UNFUSED_NORM_OPS` 集合则扩展起始位置
3. 向后扩展（最多 10 行）：连续命中则扩展结束位置
4. 将 `[start, end]` 范围内所有行的 `Marker` 标记为 `NORM`

这样 `refine_sub_blocks` 和 `extract_substructure` 就能通过 `Marker` 列识别它们为 RMSNorm。

#### 2.3.5 MoE 检测与代表层选取

**MoE 检测**：同时检查 `main_rows`（主 stream）和 `all_rows`（所有 stream，非主流算子按时间重叠归属到对应层），通过正则 `\b(?:moe|router|expert|topk|gating|gate_proj|shared_expert|GroupedMatmul|DispatchFFNCombine|MoeGatingTopK)\b` 匹配，命中则该层标记为 MoE 层。检查 `all_rows` 是因为 MoE 相关算子（如 `GroupedMatmul`、`DispatchFFNCombine`、`MoeGatingTopK`）可能不在主 stream 上。

**代表层选取**（`pick_representative_layer`）：

- 优先从含 Attention 的 Dense 层中选，取前 1/3 位置（避免首尾边界层）
- 若无含 ATT 的 Dense 层，从所有 Dense 层中取前 1/3
- MoE 层取中间位置
- `layer_analyzer.py` 支持 `--layer-index` 指定 Dense 层号；`npu_layer_analyzer.py` 自动选取

#### 2.3.6 通信算子排除

对比时自动排除以下通信算子（不计入 Duration 累加）：

正则：`all_reduce|all_gather|allgather|reduce_scatter|reduceScatter|allReduce|hcom_`

在 `layer_compare.py` 的 `split_by_stages` 中过滤，保证 Stage 时间纯净。

### 2.4 输出结构

#### 2.4.1 npu_layer_analyzer 输出

```markdown
forward_segments/
├── summary.csv                        # Forward 汇总（每行一个 forward）
├── forward_000.csv                     # Forward 0 全部算子
├── forward_000_layer1.csv              # Dense 代表层（纯 Dense 模型，N 为层号）
├── forward_000_layer5_dense.csv        # Dense 代表层（MoE 模型，N 为层号）
├── forward_000_layer16_moe.csv         # MoE 代表层（仅 MoE 模型，N 为层号）
└── ...
```

文件命名规则（`N` 为层号，如 `layer1` / `layer32`）：

- 纯 Dense 模型：`forward_XXX_layerN.csv`
- Dense + MoE 模型 Dense 代表层：`forward_XXX_layerN_dense.csv`
- Dense + MoE 模型 MoE 代表层：`forward_XXX_layerN_moe.csv`

`summary.csv` 字段：`forward_index`, `segment_kind`（prefill/decode/unknown）, `is_valid`, `method`, `main_stream`, `start/end_time_us`, `duration_us`, `attention_count_main`, `embedding_count_main`, `max_internal_gap_us`, `boundary_gap_us`, `split_reason`, `output_file`, `layer_dense_file`, `layer_moe_file` 等。

#### 2.4.2 layer_analyzer 输出

```markdown
<base_stem>_layered.csv                # 全局标注（所有行 + Layer/Marker/Structure/Is_Key 列）
<base_stem>_layerN.csv                 # Dense 代表层 + Stage 标注（纯 Dense 模型，N 为层号）
<base_stem>_layerN_dense.csv           # Dense 代表层（Dense + MoE 模型，N 为层号）
<base_stem>_layerN_moe.csv             # MoE 代表层（仅 MoE 模型，N 为层号）
```

#### 2.4.3 统一入口输出（npu_layer_compare.py）

```markdown
<output_dir>/
├── npu_out.xlsx                # npu_layer_analyzer 输出（多 Sheet）
│   ├── summary                 #   Forward 切分汇总
│   ├── forward_XXX             #   指定 Forward 全部算子
│   └── forward_XXX_layerN      #   层提取 + Stage 标注（N 为层号）
├── layer_out.xlsx              # layer_analyzer 输出（多 Sheet，来自 JSON 转换）
│   ├── <stem>_kernel_details   #   JSON 转换后的 CSV
│   ├── <stem>_layered          #   全局标注（所有行）
│   └── <stem>_layerN           #   层提取 + Stage 标注（N 为层号）
└── compare_result.xlsx         # 对比结果（Sheet 数随模型类型变化）
    ├── Dense_总比较            #   Dense 层按 Stage 汇总时间对比（Dense+MoE 模型）
    ├── Dense_算子明细          #   Dense 层逐算子并排对比（Dense+MoE 模型）
    ├── MoE_总比较              #   MoE 层按 Stage 汇总时间对比（Dense+MoE 模型）
    └── MoE_算子明细            #   MoE 层逐算子并排对比（Dense+MoE 模型）
```

`compare_result.xlsx` Sheet 数量：

- **Dense + MoE 模型**：4 Sheet（`Dense_总比较` / `Dense_算子明细` / `MoE_总比较` / `MoE_算子明细`）
- **单一类型模型（纯 Dense 或纯 MoE）**：2 Sheet（`总比较` / `算子明细`）

统一入口会将中间 CSV 目录合并为 xlsx（每个 CSV 一个 Sheet），并删除原 CSV 目录，保证输出整洁。

---

## 3. 方案设计

### 3.1 模块设计

#### 3.1.1 文件清单

```markdown
npu_layer_analyzer/
├── trace_json_to_csv.py         ← JSON → CSV 转换（去重 + 字段映射）
├── layer_common.py              ← 共享工具模块（正则 / refine_sub_blocks / extract_substructure / 未融合 RMSNorm / 代表层选取 / write_layer_csv）
├── npu_layer_analyzer.py        ← Forward 切分 + 层提取 + 子结构标注
├── layer_analyzer.py            ← 全局标注 + 层提取 + 子结构标注
├── layer_compare.py             ← 双工具层对比 → xlsx（2 或 4 Sheet）
└── npu_layer_compare.py         ← 统一入口（子进程调度 + CSV→xlsx 合并）
```

`layer_common.py` 是 `npu_layer_analyzer.py` 与 `layer_analyzer.py` 共享的工具模块，包含：

- **正则常量**：Attention / Embedding / NORM / MLP / MoE / 通信算子等匹配正则
- **`refine_sub_blocks`**：子结构状态机，为层内每行标注 `Structure` 列（仅内部打印摘要使用）
- **`extract_substructure`**：2 段 Stage 提取，由 `is_moe` 参数控制第二段为 `FFN` 或 `MOE`
- **未融合 RMSNorm 识别**：以 `aten.rsqrt.default` 为锚点向前向后扩展
- **`pick_representative_layer`**：代表层选取（Dense 优先含 ATT 的前 1/3，MoE 取中间）
- **`write_layer_csv`**：层 CSV 写出，排除 `Structure` 列

#### 3.1.2 核心设计：双工具独立 + Stage 对齐对比

工具集采用 **双工具独立提取 + Stage 对齐对比** 的架构：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **双工具独立 + Stage 对齐** | 两侧独立提取，互不污染；Stage 对齐后差异一目了然；通信算子可独立排除 | 两侧 Stage 划分需一致，否则对齐失败 | ✅ 采用 |
| 单工具统一提取 | 无对齐问题 | 无法交叉验证；NPU 侧未融合 RMSNorm 会污染框架侧标注 | ❌ 不采用 |

两个工具通过共享模块 `layer_common.py` 复用相同的 Stage 标注逻辑（`extract_substructure`）和子结构标注逻辑（`refine_sub_blocks`），保证两侧 Stage 划分一致。

#### 3.1.3 子结构标注状态机

`refine_sub_blocks` 使用状态机为层内每行标注 `Structure` 列：

| Structure 标签 | 含义 | 触发条件 |
|------|------|------|
| `1a_Norm`, `1b_Norm`... | RMSNorm（多个用字母区分） | 遇到 NORM，且在 Attention 之前或之后但非 pre-MLP |
| `2_Attention` | Attention 算子 | 遇到 ATT |
| `3_Linear(post-att)` | Attention 后投影层 | Attention 之后，非 NORM/MLP |
| `4_Norm(pre-MLP)` | MLP 前 RMSNorm | NORM 且位置 ≥ `pre_mlp_norm_pos` |
| `5_MLP` | MLP 子结构 | MLP 且在 Attention 之后 |

`Is_Key` 列标记关键边界算子（NORM / ATT / MLP）为 `★`，便于 Excel 筛选。

#### 3.1.4 Forward 切分策略

`npu_layer_analyzer.py` 的 Forward 切分支持多种边界方法：

| Method | 说明 | 适用场景 |
|------|------|------|
| `embedding-to-embedding` | 相邻 embedding 之间 | 多次 prefill 连续采集 |
| `embedding-to-gap` | embedding 到第一个大 gap | 最后一次 prefill + 后续 decode |
| `gap` | 纯 gap 阈值切分 | 无 embedding 锚点的 decode 段 |

Gap 阈值自动计算策略（`auto_gap_threshold`）：

1. 收集所有正 gap
2. 降序排列，找相邻 `high/low` 比值最大且 `high≥1000us` 的对
3. 若最大比值 ≥ 5，取 `(high+low)/2` 作为阈值
4. 否则 fallback 到 `max(1000, median*20, p95*3)`

### 3.2 关键数据结构

#### 3.2.1 KernelRow（npu_layer_analyzer.py）

```python
@dataclass(frozen=True)
class KernelRow:
    raw: dict[str, str]          # 原始 CSV 行
    line_no: int                 # 行号
    source_index: int            # 源索引（用于去重）
    stream_id: str
    task_id: str
    name: str
    op_type: str
    start_us: float
    duration_us: float
    end_us: float
    is_attention: bool           # 预计算的 Attention 标记
    is_embedding: bool           # 预计算的 Embedding 标记
    full_name: str
```

#### 3.2.2 Segment（npu_layer_analyzer.py）

```python
@dataclass
class Segment:
    index: int                   # forward 索引
    method: str                  # 切分方法（embedding-to-embedding/gap 等）
    main_stream: str
    start_us: float
    end_us: float
    main_rows: list[KernelRow]   # 主 stream 行
    output_rows: list[KernelRow] # 输出行（含相关 stream）
    split_reason: str
    boundary_gap_us: float
    boundary_gap_before_task: str
    boundary_gap_after_task: str
    segment_kind: str            # prefill / decode / unknown
    is_valid: bool
    validity_reason: str
```

#### 3.2.3 层提取输出列

层提取 CSV 的列顺序（priority 列在前）：

| 列 | 说明 |
|------|------|
| `Layer` | 层号 |
| `Stage` | 2 段 Stage 标注（Attention / FFN 或 MOE） |
| `Is_Key` | `★` 标记关键边界算子 |
| `Stream ID` / `Task ID` / `Name` / `Type` | 原始字段 |
| `Start Time(us)` / `Duration(us)` | 时间字段 |
| `Input Shapes` / `Output Shapes` | Shape 字段 |
| `Full Name` | 完整算子名 |
| `Marker` | 全局标记（EMBED/ATT/NORM/MLP/MATMUL/LINEAR/COMM/SAMPLE） |

注：`Structure` 列仅用于内部打印摘要（`refine_sub_blocks` 状态机标注），`write_layer_csv` 写 CSV 时排除该列，最终 CSV 不含 `Structure`。

### 3.3 CLI 设计

#### 3.3.1 npu_layer_compare.py（统一入口）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv` | - | 输入 CSV（给 npu_layer_analyzer） |
| `--json` | - | 输入 JSON 或 CSV（给 layer_analyzer） |
| `-o` / `--output-dir` | `compare_test` | 输出目录 |
| `--task-id` | - | 算子 Task ID，定位特定 Forward |
| `--npu-only` | - | 只跑 npu_layer_analyzer |
| `--layer-only` | - | 只跑 layer_analyzer |
| `--no-compare` | - | 不跑对比 |
| `--layer-index` | - | 指定 Dense 层号（传给 layer_analyzer） |

#### 3.3.2 npu_layer_analyzer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `kernel_details.csv` | 输入 CSV |
| `--output-dir` | `forward_segments` | 输出目录 |
| `--task-id` | - | 算子 Task ID，定位 Forward |
| `--attention-tolerance` | 0 | Attention 数允许偏差 |
| `--attention-pattern` | `attention` | Attention 匹配正则 |
| `--embedding-pattern` | `embed` | Embedding 匹配正则 |
| `--main-stream` | 自动 | 主 stream ID |
| `--gap-us` | 自动 | 手动 gap 阈值（us） |
| `--no-layer-export` | - | 不导出层分析 CSV（只切 forward） |

> 导出策略内部固定为 `first-valid-by-kind`（导出每种类型第一个有效 forward）；指定 `--task-id` 时自动切换到 `task-id` 策略。`expected_attention` 固定为 0（禁用检查），`forward_kind` 固定为 `all`（导出所有类型），`layer_index` 固定为自动选取。

#### 3.3.3 layer_analyzer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | - | 输入 CSV（必填） |
| `--output` / `-o` | 自动 | 输出路径前缀 |
| `--delimiter` | `attention` | 层边界锚点（attention / norm） |
| `--layer-index` | - | 指定 Dense 层号 |
| `--attention-pattern` / `--norm-pattern` | - | 自定义匹配正则 |

> 全局标注和层提取固定启用（无 `--no-global` / `--no-layer` 开关）；HTML 输出已移除。

#### 3.3.4 layer_compare.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-a` | - | 文件 A（层 CSV，来自 npu_layer_analyzer） |
| `-b` | - | 文件 B（层 CSV，来自 layer_analyzer） |
| `-o` | `compare_result.xlsx` | 输出 xlsx 路径 |

#### 3.3.5 trace_json_to_csv.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | - | 输入 trace JSON（必填） |
| `--output` / `-o` | 自动 | 输出 CSV 路径 |

---

## 4. 测试设计

### 4.1 单元测试

| 测试项 | 验证点 |
|------|------|
| `trace_json_to_csv` 基本转换 | JSON → CSV 字段映射正确，去重生效 |
| `choose_main_stream` | 按 `(attention, embedding, not_na, rows)` 评分自动选主 stream，attention 优先 |
| `auto_gap_threshold` | 大 gap 比值 ≥5 时正确识别阈值；否则 fallback |
| `build_segments` embedding-to-embedding | 相邻 embedding 之间正确切分 |
| `build_segments` embedding-to-gap | 最后 embedding 到大 gap 正确切分 |
| `build_segments` gap 补切 | 未覆盖行按 gap 阈值补切 |
| `classify_and_validate_segment` prefill | 含 embedding → `prefill` |
| `classify_and_validate_segment` decode | gap method + attention 匹配 → `decode` |
| `mark_unfused_rmsnorm` | rsqrt 锚点向前向后扩展，8 个基础算子标记为 NORM |
| `refine_sub_blocks` 状态机 | 5 个 Structure 标签正确切换 |
| `extract_substructure` 2 段 Stage | Attention/FFN（Dense）或 Attention/MOE（MoE）正确定位，由 `is_moe` 控制 |
| `detect_moe_in_segment` | 同时检查 main_rows 和 all_rows，MoE 算子（含 GroupedMatmul/DispatchFFNCombine/MoeGatingTopK）命中的层正确标记 |
| `pick_representative_layer` | 优先选含 ATT 的 Dense 层前 1/3 |
| `split_by_stages` 通信排除 | all_reduce/hcom 算子被排除 |
| `compare` 总比较 + 算子明细 | 单一类型 2 Sheet / Dense+MoE 4 Sheet 数据正确，小计行正确 |

### 4.2 集成测试

| 测试项 | 测试内容 | 验证点 |
|------|------|------|
| 端到端全流程 | `npu_layer_compare.py --csv ... --json ...` | 3 个 xlsx 生成，无 crash |
| task-id 定位 | `--task-id 41500` | 正确定位到含该 task 的 forward segment |
| MoE 模型 | MoE 模型 trace | 额外生成 `_layerN_moe.csv`（N 为层号） |
| 单 Attention 模型 | 仅 1 个 Attention 的层 | 从 Attention 到第一个 SwiGlu/MLP 正确提取 |
| 未融合 RMSNorm | NPU 侧未融合 RMSNorm | 8 个基础算子识别为一个 RMSNorm |
| 通信算子排除 | 含 all_reduce 的层 | 对比时通信时间不计入 |

### 4.3 实测验证案例

以下是使用 `samples/` 目录示例数据的验证结果：

| 场景 | 命令 | 验证点 |
|---|---|---|
| **基本全流程** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json` | 生成 npu_out.xlsx / layer_out.xlsx / compare_result.xlsx |
| **task-id 定位** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json --task-id 41500` | 定位到含 task 41500 的 forward |
| **指定输出目录** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json -o my_output` | 输出到 my_output 目录 |
| **只跑 NPU 侧** | `python npu_layer_compare.py --csv samples/kernel_details.csv --npu-only` | 只生成 npu_out.xlsx |
| **只跑框架侧** | `python npu_layer_compare.py --json samples/qwen1.json --layer-only` | 只生成 layer_out.xlsx |
| **指定层号** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json --layer-index 5` | 提取第 5 层 |

---

## 5. 缺点和风险

| 风险 | 影响 | 应对措施 |
|------|------|------|
| Attention 正则不匹配 | 新模型 Attention 算子名不在正则中，层提取失败 | 提供 `--attention-pattern` 参数自定义正则；正则覆盖多种变体（mla/ring_mla/delta_rule 等） |
| 未融合 RMSNorm 扩展越界 | rsqrt 锚点向前向后扩展可能误纳相邻非 NORM 算子 | 限制向前 15 行、向后 10 行；扩展时检查 `UNFUSED_NORM_OPS` 集合 |
| Gap 阈值自动计算偏差 | 采集噪声导致 gap 分布异常，阈值偏大/偏小 | 多策略 fallback（比值法 → median/p95）；支持 `--gap-us` 手动指定 |
| 两侧 Stage 不对齐 | NPU 侧与框架侧 Stage 划分不一致，对比失败 | 两侧通过 `layer_common.py` 共享 `extract_substructure` 逻辑，保证 Stage 划分一致 |
| MoE 检测误判 | `gate_proj` 命中但实际非 MoE 模型；新增 `GroupedMatmul`/`DispatchFFNCombine`/`MoeGatingTopK` 扩大命中面 | 正则限定 `\b` 边界；同时检查 `main_rows` 和 `all_rows` 提高召回；MoE 检测影响代表层选取及第二段 Stage 标注（FFN/MOE），误判会导致 Stage 标签错位 |
| 单 Attention 层提取 | 仅 1 个 Attention 时取到第一个 SwiGlu/MLP，可能跨层 | 文档标注此场景；`--layer-index` 可指定层号 |
| 通信算子正则遗漏 | 新通信算子名未覆盖 | 正则覆盖常见变体（all_reduce/all_gather/hcom_/reduceScatter） |
| xlsx Sheet 名截断 | Excel 限制 Sheet 名 31 字符 | `_sheet_name_from_csv` 截断到 31 字符 |

---

## 6. 现有技术

| 项目/工具 | 做法 | 借鉴与差异 |
|------|------|-----------|
| Chrome Trace Event 格式 | 标准性能 trace 格式（torch.profiler 导出） | `trace_json_to_csv.py` 解析 `ph=X` + `cat=analytic` 事件，去重后转 CSV |
| NPU profiling 工具 | 导出 kernel_details.csv（含 aicore/aiv 详细指标） | 本工具仅使用 `Name/Type/Start Time/Duration/Shapes` 等基础字段 |
| torch.profiler layer_table | PyTorch 内置层级别 profiling | 本工具不依赖 torch，纯 CSV/JSON 离线分析；支持 NPU 侧 trace |
| MindStudio Insight | NPU 官方 profiling 可视化工具 | 本工具聚焦层结构对比，不替代可视化；输出 xlsx 便于 Excel 二次分析 |

---

## 附录

### 附录 A：参考资料

- [Chrome Trace Event 格式规范](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
- [torch.profiler 文档](https://pytorch.org/docs/stable/profiler.html)
- [openpyxl 文档](https://openpyxl.readthedocs.io/) — xlsx 生成依赖

### 附录 B：术语表

| 术语 | 说明 |
|------|------|
| Forward Segment | 一次前向推理的算子集合，由 embedding 锚点或 gap 切分 |
| Layer | Transformer DecoderLayer，以 Attention 为起始锚点 |
| Stage | 层内 2 段子结构：Attention / FFN（Dense 层）或 MOE（MoE 层） |
| RSN | Residual + RMSNorm（Residual Side Norm） |
| Structure | 层内子结构标签（1a_Norm / 2_Attention / 3_Linear(post-att) / 4_Norm(pre-MLP) / 5_MLP），仅内部打印摘要使用，CSV 输出不含该列 |
| Marker | 全局算子类型标记（EMBED/ATT/NORM/MLP/MATMUL/LINEAR/COMM/SAMPLE） |
| 未融合 RMSNorm | NPU 上 RMSNorm 被拆成 8 个基础算子（view/add/pow/mean/rsqrt/mul） |
| MoE | Mixture of Experts，混合专家层 |
| Dense | 非 MoE 的标准 Transformer 层 |
| Prefill | 首次前向，含 embedding，序列长度 > 1 |
| Decode | 增量前向，无 embedding，序列长度 = 1 |
| Gap Threshold | Forward 切分的间隔阈值（us），大于此值视为 forward 边界 |

### 附录 C：文档更新历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-08-01 | 初始版本：NPU Layer Analyzer 工具集设计文档 | @yuyinkai |
| v1.1 | 2026-08-05 | Stage 标注由 4 段（Attention/RSN/上下采样/RSN）改为 2 段（Attention/FFN 或 MOE），由 `is_moe` 控制第二段；主 stream 评分改为 attention 优先 `(attention, embedding, not_na, rows)`；`--forward-kind` 默认值由 `prefill` 改为 `all` 并新增到两个工具参数表；`--expected-attention` 默认值由 64 改为 0（≤0 禁用检查）；文件命名加入层号 `N`（`forward_XXX_layerN.csv` / `forward_XXX_layerN_dense.csv` / `forward_XXX_layerN_moe.csv`）；embedding 检查改为在 `output_rows`（所有 stream）上进行，避免误判 decode；`extract_layer_rows` 仅保留主 Stream 算子；MoE 检测改为同时检查 `main_rows` 和 `all_rows`，正则新增 `GroupedMatmul`/`DispatchFFNCombine`/`MoeGatingTopK`；CSV 输出移除 `Structure` 列（仅内部打印摘要使用）；`compare_result.xlsx` 在 Dense+MoE 模型下输出 4 Sheet（`Dense_总比较`/`Dense_算子明细`/`MoE_总比较`/`MoE_算子明细`），单一类型仍为 2 Sheet；新增 `layer_common.py` 共享工具模块 | @yuyinkai |
| v1.2 | 2026-08-18 | CLI 精简：`npu_layer_analyzer.py` 移除 `--expected-attention`/`--forward-kind`/`--layer-index`/`--export-policy`（内部固定为禁用检查/导出所有类型/自动选取层号/first-valid-by-kind 策略），新增 `-i` 短选项；`layer_analyzer.py` 移除 `--no-html`/`--no-global`/`--no-layer`（全局标注和层提取固定启用，HTML 输出移除）；`npu_layer_compare.py` 移除 `--expected-attention`/`--forward-kind`（`--layer-index` 仅传给 layer_analyzer）；`write_annotated_html` 及相关 HTML 代码删除 | @yuyinkai |
