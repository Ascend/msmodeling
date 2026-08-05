# NPU Forward Inspector

NPU 前向推理算子分析与层结构对比工具集。从 Chrome Trace Event JSON / kernel_details CSV 出发，
完成 **Forward 切分 → 层提取 → 子结构标注 → 双工具对比** 全流程。

## 功能概览

| 工具 | 输入 | 输出 | 作用 |
|------|------|------|------|
| `trace_json_to_csv.py` | trace JSON | kernel_details CSV | Chrome Trace Event → 标准 CSV |
| `npu_layer_analyzer.py` | kernel_details CSV | forward_XXX_layerN.csv | Forward 切分 + 层提取 + Stage 标注 |
| `layer_analyzer.py` | kernel_details CSV | `*_layered.csv` / `*_layerN.csv` | 全局标注 + 层提取 + Stage 标注 |
| `layer_compare.py` | 两个 layer CSV | compare_result.xlsx | 按 Stage 对比时间/算子/Shape |
| `npu_layer_compare.py` | CSV + JSON/CSV | 完整输出目录 | **统一入口**，一键跑完全流程 |
| `layer_common.py` | - | - | 共享工具：正则、子结构标注、层提取（被两侧复用） |

## 工程结构

```text
npu_layer_analyzer/
├── README.md                    # 本文档
├── trace_json_to_csv.py         # JSON → CSV 转换
├── layer_common.py              # 共享工具（正则、子结构标注、层提取）
├── layer_analyzer.py            # 全局标注 + 层提取（Attention 锚点）
├── npu_layer_analyzer.py        # Forward 切分 + 层提取（含 task-id 定位）
├── layer_compare.py             # 双工具层对比 → xlsx（2-4 Sheet）
├── npu_layer_compare.py         # 统一入口（一键全流程）
└── samples/
    ├── kernel_details.csv       # 示例：NPU 侧 trace CSV
    └── qwen1.json               # 示例：框架侧 Chrome Trace JSON
```

## 依赖

```text
Python >= 3.10
openpyxl          # layer_compare.py 生成 xlsx
```

安装：

```bash
pip install openpyxl
```

## 快速开始

### 一键全流程（推荐）

`--csv` 是 NPU 侧 trace CSV（如 kernel_details.csv），`--json` 是框架侧 trace JSON（如 qwen1.json），
两者是不同来源的 trace，对比结果体现两侧差异。

```bash
# CSV(NPU侧) + JSON(框架侧) → 全部结果
python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json

# 指定 task-id 定位特定 Forward segment
python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json --task-id 41500

# 指定输出目录
python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json -o my_output
```

输出结构（全部 xlsx，无 CSV 残留）：

```text
<output_dir>/
├── npu_out.xlsx                # npu_layer_analyzer 输出（多 Sheet）
│   ├── summary                 #   Forward 切分汇总
│   ├── forward_XXX             #   指定 Forward 全部算子
│   └── forward_XXX_layerN      #   层提取 + Stage 标注（含层号）
├── layer_out.xlsx              # layer_analyzer 输出（多 Sheet，来自 JSON 转换）
│   ├── qwen1_kernel_details    #   JSON 转换后的 CSV
│   ├── qwen1_kernel_details_layered  #   全局标注（所有行）
│   └── qwen1_kernel_details_layerN   #   层提取 + Stage 标注（含层号）
└── compare_result.xlsx         # 对比结果
    ├── Dense_总比较             #   Dense 层按 Stage 汇总（仅 Dense+MoE 模型）
    ├── Dense_算子明细           #   Dense 层逐算子并排对比
    ├── MoE_总比较               #   MoE 层按 Stage 汇总
    └── MoE_算子明细             #   MoE 层逐算子并排对比
```

> 纯 Dense 或纯 MoE 模型只有 2 个 Sheet（总比较 + 算子明细），无 Dense_/MoE_ 前缀。

### 单独使用各工具

#### 1. JSON → CSV

```bash
python trace_json_to_csv.py -i trace.json -o kernel_details.csv
```

#### 2. npu_layer_analyzer（Forward 切分 + 层提取）

```bash
# 自动选第一个有效 Forward
python npu_layer_analyzer.py -i kernel_details.csv

# 按 task-id 定位
python npu_layer_analyzer.py -i kernel_details.csv --task-id 41500

# 指定层号
python npu_layer_analyzer.py -i kernel_details.csv --layer-index 21
```

#### 3. layer_analyzer（全局标注 + 层提取）

```bash
# 全局标注 + 层提取（默认行为）
python layer_analyzer.py -i kernel_details.csv --delimiter attention

# 只提取代表层（不做全局标注和 HTML）
python layer_analyzer.py -i kernel_details.csv --no-global --no-html
```

#### 4. layer_compare（层对比）

```bash
python layer_compare.py -a forward_003_layer.csv -b kernel_details_layer.csv -o compare.xlsx
```

## 核心算法

### 主 Stream 选择

按 `(attention 数, embedding 数, 非 N/A, 总行数)` 评分选主 stream：

1. **attention 数最多**的 stream 优先（attention 最能代表计算主流）
2. attention 数相同时，**embedding 数最多**的优先
3. 仍相同时，非 N/A stream 优先，最后按总行数

> 层提取只保留主 Stream 算子，过滤其他 Stream ID 的通信/调度算子。

### 层边界：Attention 锚点

层边界以 **Attention** 为起始锚点（非 RMSNorm / 上下采样）：

- **多 Attention 模型**：取第一对相邻 Attention 之间的算子作为一层
- **单 Attention 模型**：从 Attention 到其后第一个 SwiGlu/MLP

### Stage 标注（2 段）

每个层提取 CSV 包含 `Stage` 列，标注 2 个阶段：

| Stage | 含义 | 关键算子 |
|-------|------|----------|
| `Attention` | Attention 阶段（含 pre-ATT NORM 残余 + ATT → AddRmsNormBias 前） | FusedInferAttentionScore / multihead_latent_attention |
| `FFN` | Dense 层的 FFN 阶段（AddRmsNormBias → MLP → RmsNorm） | AddRmsNormBias / MatMul → SwiGlu → MatMul / RmsNorm |
| `MOE` | MoE 层的 MOE 阶段（AddRmsNormBias → MoE → RmsNorm） | AddRmsNormBias / GroupedMatmul / DispatchFFNCombine / RmsNorm |

> Dense 层第二段为 `FFN`，MoE 层第二段为 `MOE`。层类型由 MoE 检测自动判断。

### Attention 识别规则

以下模式均识别为 Attention：

- `attention` / `infer_attention` / `mla` / `ring_mla` / `grouped_attention`
- `recurrent` / `attn_chunk_gated`（Recurrent 模型）
- `delta_rule` / `gated_delta`（Recurrent Attention）

### 未融合 RMSNorm 识别

当 RMSNorm 未融合为单个算子时，以下算子序列会被识别为一个 RMSNorm：

```text
aten.view.default → aten.add.Tensor → prims.convert_element_type.default →
aten.pow.Tensor_Scalar → aten.mean.dim → aten.add.Tensor →
aten.rsqrt.default → aten.mul.Tensor → aten.mul.Tensor
```

以 `rsqrt` 为锚点，向前向后扩展关联算子。

### 通信算子排除

对比时自动排除以下通信算子（不计入 Duration 累加）：

- `all_reduce` / `all_gather` / `allgather` / `reduce_scatter` / `hcom_*`

## 对比结果说明

### Sheet1: 总比较

| 列 | 说明 |
|----|------|
| Stage | 阶段名（Attention / FFN 或 MOE / TOTAL） |
| 文件A_Duration(us) | npu_layer_analyzer 该 Stage 累加时间（排除通信） |
| 文件B_Duration(us) | layer_analyzer 该 Stage 累加时间（排除通信） |
| Diff(us) | A - B |
| Diff(%) | 差异百分比 |

### Sheet2: 算子明细

逐算子并排对比，按 Stage 分组，每段末尾有小计行。

## 命令参数速查

### npu_layer_compare.py（统一入口）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv` | - | 输入 CSV（给 npu_layer_analyzer） |
| `--json` | - | 输入 JSON 或 CSV（给 layer_analyzer） |
| `-o` | `compare_test` | 输出目录 |
| `--task-id` | - | 算子 Task ID，定位特定 Forward |
| `--expected-attention` | 0 | 预期每 Forward 的 Attention 数（<=0 禁用检查） |
| `--forward-kind` | `all` | 导出哪种 forward 类型（all/prefill/decode） |
| `--npu-only` | - | 只跑 npu_layer_analyzer |
| `--layer-only` | - | 只跑 layer_analyzer |
| `--no-compare` | - | 不跑对比 |
| `--layer-index` | - | 指定 Dense 层号 |

### npu_layer_analyzer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | kernel_details.csv | 输入 CSV |
| `--output-dir` | forward_segments | 输出目录 |
| `--task-id` | - | 算子 Task ID，定位 Forward |
| `--expected-attention` | 0 | 预期每 Forward 的 Attention 数（<=0 禁用） |
| `--forward-kind` | `all` | 导出哪种 forward 类型（all/prefill/decode） |
| `--layer-index` | - | 指定 Dense 层号 |
| `--export-policy` | first-valid-by-kind | 导出策略 |

### layer_analyzer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | - | 输入 CSV |
| `--output` / `-o` | 自动 | 输出路径前缀 |
| `--delimiter` | attention | 层边界锚点（attention / norm） |
| `--layer-index` | - | 指定 Dense 层号 |
| `--no-html` | - | 不输出 HTML |
| `--no-global` | - | 不输出全局标注 |
| `--no-layer` | - | 不输出层提取 |
