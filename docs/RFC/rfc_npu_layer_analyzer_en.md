# NPU Layer Analyzer Design Document

Status: Draft
Authors: @yuyinkai
Created: 2026-08-01
Updated: 2026-08-05

---

## 1. Overview

### 1.1 Introduction

NPU Layer Analyzer is a **toolset for NPU forward inference operator analysis and layer structure comparison**. Starting from Chrome Trace Event JSON / kernel_details CSV, it completes the full pipeline: **Forward segmentation → layer extraction → substructure annotation → dual-tool comparison**. This RFC describes the overall architecture, core algorithms, module design, and output specification of the toolset.

The toolset consists of 5 independent scripts that form a complete data pipeline:

| Tool | Input | Output | Purpose |
|------|------|------|------|
| `trace_json_to_csv.py` | trace JSON | kernel_details CSV | Chrome Trace Event → standard CSV |
| `npu_layer_analyzer.py` | kernel_details CSV | forward_XXX_layerN.csv | Forward segmentation + layer extraction + Stage annotation |
| `layer_analyzer.py` | kernel_details CSV | *_layered.csv / *_layerN.csv | Global annotation + layer extraction + Stage annotation |
| `layer_compare.py` | two layer CSVs | compare_result.xlsx | Stage-by-Stage comparison of time/operators/shapes |
| `npu_layer_compare.py` | CSV + JSON/CSV | complete output directory | **Unified entry**, one-click full pipeline |

Core values:

- **Dual-tool cross-validation**: `npu_layer_analyzer` (NPU-side trace, with Forward segmentation) and `layer_analyzer` (framework-side trace, with global annotation) independently extract layer structures, then align by Stage via `layer_compare` to locate operator/time differences between the two sides
- **Attention anchor layer boundary**: Uses Attention (not RMSNorm/upsampling) as the layer start anchor, avoiding layer misalignment caused by unfused NPU RMSNorm scattering
- **2-stage annotation**: Each layer is split into 2 fixed stages: `Attention → FFN` (Dense layers) or `Attention → MOE` (MoE layers), aligned with Transformer DecoderLayer semantics. Attention Stage = operators after the previous RmsNorm + (ATT → before AddRmsNormBias); FFN/MOE Stage = AddRmsNormBias → RmsNorm (inclusive). The second stage name is controlled by the `is_moe` parameter.
- **Unfused RMSNorm recognition**: Uses `aten.rsqrt.default` as an anchor to expand forward/backward, recognizing 8 scattered base operators as a single RMSNorm
- **MoE auto-detection + representative layer selection**: Automatically identifies MoE layers and additionally selects a MoE representative layer, exported separately from the Dense representative layer
- **Communication operator exclusion**: Automatically excludes `all_reduce` / `all_gather` / `hcom_*` communication operators during comparison, ensuring pure algorithm time

### 1.2 Motivation

In NPU inference performance analysis, the following pain points exist:

| Pain Point | Description |
|------|------|
| Trace files are huge and hard to read | A single forward's kernel_details.csv can have tens of thousands of rows; manually locating layer boundaries is time-consuming and error-prone |
| NPU-side RMSNorm is unfused | On NPU, RMSNorm is often split into 8 base operators (view/add/pow/mean/rsqrt/mul...), which cannot be identified by a single regex |
| Inconsistent layer boundary anchors | When using RMSNorm as layer boundary, unfused RMSNorm scatters one layer into multiple segments; Attention is a more stable anchor |
| Hard to locate trace differences between two sides | NPU-side and framework-side traces have different operator granularities; Stage-by-Stage alignment is needed to find "missing operators" or "time gaps" |
| Prefill/Decode mixed | One capture contains multiple forwards (prefill + multiple decodes); automatic segmentation is needed to analyze them separately |
| MoE and Dense layers interleaved | MoE models alternate Dense and MoE layers; representative layers need to be selected separately during analysis |
| Communication operators interfere with time stats | All-Reduce and other communication operators consume significant time but are unrelated to the algorithm itself; they must be excluded during comparison |
| Scattered output formats | Each tool outputs CSVs scattered across multiple directories; needs unification into multi-sheet xlsx for easy viewing |

### 1.3 Goals

- Provide 5 independently usable scripts covering the full pipeline: JSON→CSV, Forward segmentation, layer extraction, comparison
- Unified entry via `npu_layer_compare.py` for one-click full pipeline, outputting 3 xlsx files (npu_out / layer_out / compare_result)
- Layer boundary uses Attention as anchor, stable for unfused RMSNorm scenarios
- Each layer annotated with 2 stages (Attention / FFN for Dense layers, Attention / MOE for MoE layers), aligned with Transformer semantics
- Auto-recognize unfused RMSNorm sequences (rsqrt anchor)
- Auto-detect MoE layers and select Dense/MoE representative layers separately
- Auto-exclude communication operators during comparison, ensuring pure Stage time
- Support task-id positioning for specific Forward segments

**Non-goals**:

- No operator-level performance modeling (only Stage-level time aggregation comparison)
- No cross-forward trend analysis (only representative layers exported each time)
- No training scenario layer analysis (only for inference prefill/decode)
- No distributed multi-node communication topology analysis (communication operators only excluded)
- Not a replacement for the tensor_cast simulation framework (only an external tool for simulation vs. measurement comparison)

---

## 2. Project Architecture Analysis

### 2.1 Toolset Overview

```text
npu_layer_analyzer/
├── README.md                    # Project documentation
├── trace_json_to_csv.py         # JSON → CSV conversion
├── layer_common.py              # Shared utilities (regexes, substructure annotation, Stage extraction, CSV writing)
├── layer_analyzer.py            # Global annotation + layer extraction (Attention anchor)
├── npu_layer_analyzer.py        # Forward segmentation + layer extraction (with task-id positioning)
├── layer_compare.py             # Dual-tool layer comparison → xlsx (2 Sheets)
├── npu_layer_compare.py         # Unified entry (one-click full pipeline)
├── rfc_npu_layer_analyzer_zh.md # This document (Chinese)
├── rfc_npu_layer_analyzer_en.md # This document (English)
├── samples/
│   ├── kernel_details.csv       # Sample: NPU-side trace CSV
│   └── qwen1.json               # Sample: framework-side Chrome Trace JSON
└── output/                      # Sample output
    ├── npu_out.xlsx
    ├── layer_out.xlsx
    └── compare_result.xlsx
```

### 2.2 Data Flow

The overall data flow is as follows, with `npu_layer_compare.py` chaining all steps:

```
┌──────────────┐                ┌──────────────────────┐
│ kernel_      │  npu_layer_    │ npu_out.xlsx         │
│ details.csv  │  analyzer.py   │  ├── summary         │
│ (NPU side)   │ ─────────────→ │  ├── forward_XXX     │
└──────────────┘                │  └── forward_XXX_layerN│
                                └──────────────────────┘
                                          │
                                          │ (layer CSV)
                                          ▼
┌──────────────┐  trace_json_   ┌──────────────────────┐
│ qwen1.json   │  to_csv.py     │ layer_out.xlsx       │
│ (framework   │ ─────────────→ │  ├── *_kernel_details│
│  side)       │       │        │  ├── *_layered       │
└──────────────┘       │ layer_  │  └── *_layerN       │
                       │ analyzer│ └──────────────────────┘
                       ▼ py      │          │
                  ┌──────────────┐         │
                  │ *_layerN.csv │         │ (layer CSV)
                  └──────────────┘         │
                                          ▼
                                ┌──────────────────────┐
                                │ compare_result.xlsx  │
                                │  ├── 总比较           │
                                │  └── 算子明细         │
                                └──────────────────────┘
```

The two layer CSVs come from:
- **File A**: `npu_layer_analyzer.py`'s `forward_XXX_layerN.csv` (NPU side, after Forward segmentation)
- **File B**: `layer_analyzer.py`'s `*_layerN.csv` (framework side, after global annotation and extraction)

### 2.3 Core Algorithms

#### 2.3.1 Forward Segmentation (npu_layer_analyzer.py)

Forward segmentation splits one capture's CSV into multiple forward segments. Core logic:

1. **Main stream selection**: Counts embedding/attention/rows per stream, scores by `(attention, embedding, not_na, rows)` to select main stream. Attention is the first priority because it best represents the compute main stream; embedding may run on non-compute streams and is therefore demoted to second priority.
2. **Gap threshold auto-calculation**: Collects all positive gaps, sorts descending to find max `high/low` ratio (≥5 and high≥1000us), otherwise falls back to `max(1000, median*20, p95*3)`
3. **Embedding anchor segmentation**:
   - Multiple embeddings: segment between adjacent embeddings (`embedding-to-embedding`)
   - Last embedding: extends forward to first large gap (`embedding-to-gap`)
4. **Uncovered row supplementation**: Remaining main stream rows are segmented by gap threshold (`gap` method)
5. **Segment classification and validation**:
   - Contains embedding → `prefill`. Embedding count is checked against `output_rows` (all streams), not just `main_rows`, because embedding may run on a non-main stream and would otherwise be misclassified as `decode`.
   - Gap method and attention count matches → `decode`
   - Otherwise → `unknown`
   - Validation: attention count within `expected ± tolerance` (disabled when `expected ≤ 0`), internal gap within threshold, boundary trusted

#### 2.3.2 Layer Boundary: Attention Anchor

Layer boundary uses **Attention** as the start anchor (not RMSNorm/upsampling), because:

- **Multi-Attention models**: Takes operators between the first pair of adjacent Attentions as one layer
- **Single-Attention models**: From Attention to the first SwiGlu/MLP after it

Attention recognition rules (all of the following patterns are recognized as Attention):
- `attention` / `infer_attention` / `mla` / `ring_mla` / `grouped_attention`
- `recurrent` / `attn_chunk_gated` (Recurrent models)
- `delta_rule` / `gated_delta` (Recurrent Attention)

`extract_layer_rows` keeps only main-Stream ID operators when building the per-layer row set, filtering out non-main-stream operators (e.g. communication operators) so they do not interfere with layer boundary splitting and Stage annotation. Non-main-stream MoE operators are still captured separately by `detect_moe_in_segment` (see 2.3.5).

#### 2.3.3 Stage Annotation (2 Stages)

Each layer extraction CSV contains a `Stage` column annotating 2 stages. The second stage name depends on whether the layer is MoE (controlled by the `is_moe` parameter):

| Stage | Meaning | Key Operators |
|-------|------|----------|
| `Attention` | Operators after the previous RmsNorm + (ATT → before AddRmsNormBias) | FusedInferAttentionScore / delta_rule |
| `FFN` (Dense layers) / `MOE` (MoE layers) | AddRmsNormBias → RmsNorm (inclusive) | AddRmsNormBias → MatMul → SwiGlu → MatMul → RmsNorm (Dense) or GroupedMatmul / DispatchFFNCombine / MoeGatingTopK (MoE) |

Positioning logic (in `extract_substructure`):
1. Layer boundary: from one Attention to just before the next Attention (single-Attention models extend to end)
2. Find the first ATT, the first NORM (`AddRmsNormBias`, recorded as `first_norm`) and the second NORM (next layer's pre-ATT `RmsNorm`, recorded as `second_norm`) on the main stream
3. **Attention Stage** = operators after `second_norm` (belonging to the tail of the previous layer's residual, kept continuous with the current layer) + (ATT → just before `first_norm`). `AddRmsNormBias` belongs to the second stage, not Attention.
4. **FFN/MOE Stage** = `first_norm` → `second_norm` (inclusive). When `is_moe=True` the stage is named `MOE`, otherwise `FFN`.
5. Result rows are reordered so the Attention Stage is continuous (tail of previous layer placed before ATT), keeping the layer non-segmented.
6. Fall-back: if no NORM is found the whole layer is `Attention`; if only one NORM is found, ATT → before NORM is `Attention` and NORM → end is `FFN`/`MOE`.

#### 2.3.4 Unfused RMSNorm Recognition

When RMSNorm is not fused into a single operator, the following operator sequence is recognized as one RMSNorm:

```
aten.view.default → aten.add.Tensor → prims.convert_element_type.default →
aten.pow.Tensor_Scalar → aten.mean.dim → aten.add.Tensor →
aten.rsqrt.default → aten.mul.Tensor → aten.mul.Tensor
```

Recognition strategy (`mark_unfused_rmsnorm`):
1. Iterate all rows, find rows with `Name` containing `rsqrt` as anchor
2. Expand backward (up to 15 rows): if consecutive rows hit the `UNFUSED_NORM_OPS` set, extend start position
3. Expand forward (up to 10 rows): if consecutive rows hit, extend end position
4. Mark all rows in `[start, end]` range with `Marker = NORM`

This allows `refine_sub_blocks` and `extract_substructure` to recognize them as RMSNorm via the `Marker` column.

#### 2.3.5 MoE Detection and Representative Layer Selection

**MoE detection** (`detect_moe_in_segment`): Uses the `MOE_RE` regex (matching `moe` / `router` / `expert` / `GroupedMatmul` / `DispatchFFNCombine` / `MoeGatingTopK` etc.) to identify MoE layers. Detection checks **both `main_rows` and all rows** (`all_rows`); non-main-stream operators (e.g. `GroupedMatmul` running on another stream) are assigned to a layer by time overlap with the main-stream layer time range, so MoE layers whose expert operators are not on the main stream are still detected.

**Representative layer selection** (`pick_representative_layer`):
- Prioritize Dense layers containing Attention, take the top 1/3 position (avoiding boundary layers at start/end)
- If no Dense layer with ATT, take top 1/3 of all Dense layers
- For MoE layers, take the middle position
- Supports `--layer-index` to specify Dense layer number

#### 2.3.6 Communication Operator Exclusion

During comparison, the following communication operators are automatically excluded (not counted in Duration accumulation):

Regex: `all_reduce|all_gather|allgather|reduce_scatter|reduceScatter|allReduce|hcom_`

Filtered in `layer_compare.py`'s `split_by_stages`, ensuring pure Stage time.

### 2.4 Output Structure

#### 2.4.1 npu_layer_analyzer Output

```
forward_segments/
├── summary.csv                        # Forward summary (one row per forward)
├── forward_000.csv                     # Forward 0 all operators
├── forward_000_layerN.csv              # Dense representative layer + Stage annotation (pure Dense model, N = layer number, e.g. layer1, layer32)
├── forward_000_layerN_dense.csv        # Dense representative layer (MoE models)
├── forward_000_layerN_moe.csv          # MoE representative layer (MoE models only)
└── ...
```

Filename rule: when the model has both Dense and MoE layers, both `_dense` and `_moe` suffixes are used; for a pure Dense model the unsuffixed `forward_XXX_layerN.csv` is emitted. `N` is the actual layer number selected by `pick_representative_layer` (e.g. `forward_000_layer5_dense.csv`).

`summary.csv` fields: `forward_index`, `segment_kind` (prefill/decode/unknown), `is_valid`, `method`, `main_stream`, `start/end_time_us`, `duration_us`, `attention_count_main`, `embedding_count_main`, `max_internal_gap_us`, `boundary_gap_us`, `split_reason`, `output_file`, `layer_dense_file`, `layer_moe_file`, etc.

#### 2.4.2 layer_analyzer Output

```
<base_stem>_layered.csv                # Global annotation (all rows + Layer/Marker/Structure/Is_Key columns)
<base_stem>_layered.html               # Global annotation HTML (with color highlighting)
<base_stem>_layerN.csv                 # Dense representative layer + Stage annotation (pure Dense model, N = layer number)
<base_stem>_layerN_dense.csv           # Dense representative layer (MoE models)
<base_stem>_layerN_moe.csv             # MoE representative layer (MoE models only)
```

#### 2.4.3 Unified Entry Output (npu_layer_compare.py)

```
<output_dir>/
├── npu_out.xlsx                # npu_layer_analyzer output (multi-sheet)
│   ├── summary                 #   Forward segmentation summary
│   ├── forward_XXX             #   Specified Forward all operators
│   └── forward_XXX_layerN      #   Layer extraction + Stage annotation (N = layer number)
├── layer_out.xlsx              # layer_analyzer output (multi-sheet, from JSON conversion)
│   ├── <stem>_kernel_details   #   JSON-converted CSV
│   ├── <stem>_layered          #   Global annotation (all rows)
│   └── <stem>_layerN           #   Layer extraction + Stage annotation
└── compare_result.xlsx         # Comparison result (Sheet names kept in Chinese)
    ├── Dense+MoE model: up to 4 Sheets
    │   ├── Dense_总比较        #   Dense layer Stage-by-stage time comparison
    │   ├── Dense_算子明细      #   Dense layer operator-by-operator side-by-side comparison
    │   ├── MoE_总比较          #   MoE layer Stage-by-stage time comparison
    │   └── MoE_算子明细        #   MoE layer operator-by-operator side-by-side comparison
    └── Single type (pure Dense or pure MoE): 2 Sheets
        ├── 总比较              #   Stage-by-stage time comparison
        └── 算子明细            #   Operator-by-operator side-by-side comparison
```

The unified entry merges intermediate CSV directories into xlsx (one sheet per CSV) and deletes the original CSV directories, ensuring clean output. `compare_result.xlsx` is produced by `layer_compare.py` (2 Sheets per layer type) and, for Dense+MoE models, merged by `npu_layer_compare.py` with `Dense_` / `MoE_` prefixes (Chinese sheet names `总比较` / `算子明细` are preserved).

---

## 3. Solution Design

### 3.1 Module Design

#### 3.1.1 File List

```text
npu_layer_analyzer/
├── trace_json_to_csv.py         ← JSON → CSV conversion (dedup + field mapping)
├── layer_common.py              ← Shared utilities (regexes, refine_sub_blocks, extract_substructure, unfused RMSNorm, pick_representative_layer, write_layer_csv)
├── npu_layer_analyzer.py        ← Forward segmentation + layer extraction + substructure annotation
├── layer_analyzer.py            ← Global annotation + layer extraction + substructure annotation + HTML output
├── layer_compare.py             ← Dual-tool layer comparison → xlsx (2 Sheets)
└── npu_layer_compare.py         ← Unified entry (subprocess scheduling + CSV→xlsx merge)
```

#### 3.1.2 Core Design: Dual-tool Independent + Stage-aligned Comparison

The toolset adopts a **dual-tool independent extraction + Stage-aligned comparison** architecture:

| Approach | Pros | Cons | Choice |
|------|------|------|------|
| **Dual-tool independent + Stage alignment** | Both sides extract independently without pollution; differences are clear after Stage alignment; communication operators can be excluded independently | Both sides must have consistent Stage划分, otherwise alignment fails | ✅ Adopted |
| Single-tool unified extraction | No alignment issue | Cannot cross-validate; NPU-side unfused RMSNorm pollutes framework-side annotation | ❌ Not adopted |

Both tools share the same Stage annotation logic (`extract_substructure`) and substructure annotation logic (`refine_sub_blocks`), ensuring consistent Stage划分 on both sides. These shared utilities (regexes, `refine_sub_blocks`, `extract_substructure`, unfused RMSNorm recognition, `pick_representative_layer`, `write_layer_csv`) are extracted into the `layer_common.py` module and imported by both `npu_layer_analyzer.py` and `layer_analyzer.py`, avoiding duplicated maintenance.

#### 3.1.3 Substructure Annotation State Machine

`refine_sub_blocks` uses a state machine to annotate each row's `Structure` column:

| Structure Label | Meaning | Trigger Condition |
|------|------|------|
| `1a_Norm`, `1b_Norm`... | RMSNorm (multiple distinguished by letters) | NORM encountered, before Attention or after but not pre-MLP |
| `2_Attention` | Attention operator | ATT encountered |
| `3_Linear(post-att)` | Post-Attention projection layer | After Attention, not NORM/MLP |
| `4_Norm(pre-MLP)` | Pre-MLP RMSNorm | NORM and position ≥ `pre_mlp_norm_pos` |
| `5_MLP` | MLP substructure | MLP and after Attention |

The `Is_Key` column marks key boundary operators (NORM / ATT / MLP) with `★` for easy Excel filtering.

> Note: The `Structure` column is an **internal annotation** used by `refine_sub_blocks` and the print summary (`print_structure_summary`). It is **excluded from the layer CSV output** by `write_layer_csv` (only `Layer` / `Stage` / `Is_Key` and original fields are written), so the CSV stays focused on the 2-stage `Stage` annotation.

#### 3.1.4 Forward Segmentation Strategy

`npu_layer_analyzer.py`'s Forward segmentation supports multiple boundary methods:

| Method | Description | Use Case |
|------|------|------|
| `embedding-to-embedding` | Between adjacent embeddings | Multiple prefills captured consecutively |
| `embedding-to-gap` | Embedding to first large gap | Last prefill + subsequent decode |
| `gap` | Pure gap threshold segmentation | Decode segments without embedding anchors |

Gap threshold auto-calculation strategy (`auto_gap_threshold`):
1. Collect all positive gaps
2. Sort descending, find adjacent `high/low` pair with max ratio and `high≥1000us`
3. If max ratio ≥ 5, use `(high+low)/2` as threshold
4. Otherwise fall back to `max(1000, median*20, p95*3)`

### 3.2 Key Data Structures

#### 3.2.1 KernelRow (npu_layer_analyzer.py)

```python
@dataclass(frozen=True)
class KernelRow:
    raw: dict[str, str]          # Original CSV row
    line_no: int                 # Line number
    source_index: int            # Source index (for dedup)
    stream_id: str
    task_id: str
    name: str
    op_type: str
    start_us: float
    duration_us: float
    end_us: float
    is_attention: bool           # Pre-computed Attention flag
    is_embedding: bool           # Pre-computed Embedding flag
    full_name: str
```

#### 3.2.2 Segment (npu_layer_analyzer.py)

```python
@dataclass
class Segment:
    index: int                   # Forward index
    method: str                  # Segmentation method (embedding-to-embedding/gap, etc.)
    main_stream: str
    start_us: float
    end_us: float
    main_rows: list[KernelRow]   # Main stream rows
    output_rows: list[KernelRow] # Output rows (including related streams)
    split_reason: str
    boundary_gap_us: float
    boundary_gap_before_task: str
    boundary_gap_after_task: str
    segment_kind: str            # prefill / decode / unknown
    is_valid: bool
    validity_reason: str
```

#### 3.2.3 Layer Extraction Output Columns

Layer extraction CSV column order (priority columns first):

| Column | Description |
|------|------|
| `Layer` | Layer number |
| `Stage` | 2-stage annotation (Attention / FFN for Dense, Attention / MOE for MoE) |
| `Is_Key` | `★` marks key boundary operators |
| `Stream ID` / `Task ID` / `Name` / `Type` | Original fields |
| `Start Time(us)` / `Duration(us)` | Time fields |
| `Input Shapes` / `Output Shapes` | Shape fields |
| `Full Name` | Full operator name |
| `Marker` | Global marker (EMBED/ATT/NORM/MLP/MATMUL/LINEAR/COMM/SAMPLE) |

> The `Structure` substructure label (1a_Norm / 2_Attention / ...) is computed internally by `refine_sub_blocks` for the print summary, but is intentionally excluded from the CSV output by `write_layer_csv`.

### 3.3 CLI Design

#### 3.3.1 npu_layer_compare.py (Unified Entry)

| Parameter | Default | Description |
|------|--------|------|
| `--csv` | - | Input CSV (for npu_layer_analyzer) |
| `--json` | - | Input JSON or CSV (for layer_analyzer) |
| `-o` / `--output-dir` | `compare_test` | Output directory |
| `--task-id` | - | Operator Task ID, locates specific Forward |
| `--expected-attention` | 0 | Expected Attention count per Forward (`<=0` disables the check) |
| `--forward-kind` | `all` | Which forward type to export: `prefill` / `decode` / `all` (default exports all forward types, not just prefill) |
| `--npu-only` | - | Only run npu_layer_analyzer |
| `--layer-only` | - | Only run layer_analyzer |
| `--no-compare` | - | Skip comparison |
| `--layer-index` | - | Specify Dense layer number |

#### 3.3.2 npu_layer_analyzer.py

| Parameter | Default | Description |
|------|--------|------|
| `--input` | `kernel_details.csv` | Input CSV |
| `--output-dir` | `forward_segments` | Output directory |
| `--task-id` | - | Operator Task ID, locates Forward |
| `--expected-attention` | 0 | Expected Attention count per Forward (`<=0` disables the check) |
| `--attention-tolerance` | 0 | Attention count tolerance |
| `--attention-pattern` | `attention` | Attention match regex |
| `--embedding-pattern` | `embed` | Embedding match regex |
| `--main-stream` | auto | Main stream ID |
| `--gap-us` | auto | Manual gap threshold (us) |
| `--export-policy` | `first-valid-by-kind` | Export policy (first-valid-by-kind/all/indexes/task-id) |
| `--forward-kind` | `all` | Which forward type to export: `prefill` / `decode` / `all` (default exports all forward types, not just prefill) |
| `--layer-index` | - | Specify Dense layer number |
| `--no-layer-export` | - | Skip layer analysis CSV (only segment forward) |

#### 3.3.3 layer_analyzer.py

| Parameter | Default | Description |
|------|--------|------|
| `--input` / `-i` | - | Input CSV (required) |
| `--output` / `-o` | auto | Output path prefix |
| `--delimiter` | `attention` | Layer boundary anchor (attention / norm) |
| `--layer-index` | - | Specify Dense layer number |
| `--no-html` | - | Skip HTML output |
| `--no-global` | - | Skip global annotation |
| `--no-layer` | - | Skip layer extraction |
| `--attention-pattern` / `--norm-pattern` | - | Custom match regex |

#### 3.3.4 layer_compare.py

| Parameter | Default | Description |
|------|--------|------|
| `-a` | - | File A (layer CSV, from npu_layer_analyzer) |
| `-b` | - | File B (layer CSV, from layer_analyzer) |
| `-o` | `compare_result.xlsx` | Output xlsx path |

#### 3.3.5 trace_json_to_csv.py

| Parameter | Default | Description |
|------|--------|------|
| `--input` / `-i` | - | Input trace JSON (required) |
| `--output` / `-o` | auto | Output CSV path |

---

## 4. Test Design

### 4.1 Unit Tests

| Test Item | Verification Point |
|------|------|
| `trace_json_to_csv` basic conversion | JSON → CSV field mapping correct, dedup works |
| `choose_main_stream` | Auto-selects stream by `(attention, embedding, not_na, rows)`, attention first |
| `auto_gap_threshold` | Correctly identifies threshold when gap ratio ≥5; otherwise falls back |
| `build_segments` embedding-to-embedding | Correctly segments between adjacent embeddings |
| `build_segments` embedding-to-gap | Correctly segments last embedding to large gap |
| `build_segments` gap supplementation | Uncovered rows segmented by gap threshold |
| `classify_and_validate_segment` prefill | Contains embedding (checked against `output_rows`) → `prefill` |
| `classify_and_validate_segment` decode | Gap method + attention match → `decode` |
| `mark_unfused_rmsnorm` | rsqrt anchor expands forward/backward, 8 base operators marked as NORM |
| `refine_sub_blocks` state machine | 5 Structure labels switch correctly |
| `extract_substructure` 2 stages | Attention / FFN (or MOE) correctly positioned; `is_moe` switches second stage name |
| `detect_moe_in_segment` | Checks both `main_rows` and `all_rows`; layers with MoE operator hits (incl. non-main-stream) correctly marked |
| `pick_representative_layer` | Prioritizes Dense layers with ATT in top 1/3 |
| `extract_layer_rows` main-stream filter | Non-main-stream operators filtered out before Stage annotation |
| `write_layer_csv` Structure exclusion | `Structure` column present internally but excluded from CSV output |
| `split_by_stages` communication exclusion | all_reduce/hcom operators excluded |
| `compare` summary + operator detail | 总比较 / 算子明细 data correct, subtotal rows correct |

### 4.2 Integration Tests

| Test Item | Test Content | Verification Point |
|------|------|------|
| End-to-end full pipeline | `npu_layer_compare.py --csv ... --json ...` | 3 xlsx files generated, no crash |
| task-id positioning | `--task-id 41500` | Correctly locates forward segment containing this task |
| MoE model | MoE model trace | Additional `_layerN_moe.csv` generated; `compare_result.xlsx` contains 4 Sheets (Dense_总比较 / Dense_算子明细 / MoE_总比较 / MoE_算子明细) |
| Pure Dense model | Dense-only model trace | Single `_layerN.csv` generated; `compare_result.xlsx` contains 2 Sheets (总比较 / 算子明细) |
| Single-Attention model | Layer with only 1 Attention | Correctly extracts from Attention to end |
| Unfused RMSNorm | NPU-side unfused RMSNorm | 8 base operators recognized as one RMSNorm |
| Communication operator exclusion | Layer with all_reduce | Communication time excluded during comparison |

### 4.3 Validation Cases

The following are validation results using sample data from the `samples/` directory:

| Scenario | Command | Verification Point |
|---|---|---|
| **Basic full pipeline** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json` | Generates npu_out.xlsx / layer_out.xlsx / compare_result.xlsx |
| **task-id positioning** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json --task-id 41500` | Locates forward containing task 41500 |
| **Custom output dir** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json -o my_output` | Outputs to my_output directory |
| **NPU side only** | `python npu_layer_compare.py --csv samples/kernel_details.csv --npu-only` | Only generates npu_out.xlsx |
| **Framework side only** | `python npu_layer_compare.py --json samples/qwen1.json --layer-only` | Only generates layer_out.xlsx |
| **Specify layer number** | `python npu_layer_compare.py --csv samples/kernel_details.csv --json samples/qwen1.json --layer-index 5` | Extracts layer 5 |

---

## 5. Drawbacks and Risks

| Risk | Impact | Mitigation |
|------|------|------|
| Attention regex mismatch | New model's Attention operator name not in regex, layer extraction fails | Provide `--attention-pattern` parameter for custom regex; regex covers multiple variants (mla/ring_mla/delta_rule, etc.) |
| Unfused RMSNorm expansion overflow | rsqrt anchor may incorrectly include adjacent non-NORM operators when expanding | Limit forward to 15 rows, backward to 10 rows; check `UNFUSED_NORM_OPS` set during expansion |
| Gap threshold auto-calculation bias | Capture noise causes abnormal gap distribution, threshold too large/small | Multi-strategy fallback (ratio method → median/p95); supports manual `--gap-us` |
| Stage misalignment between sides | NPU-side and framework-side Stage划分 inconsistent, comparison fails | Both sides share `extract_substructure` logic, ensuring consistent Stage划分 |
| MoE detection false positive | `MOE_RE` matches an operator name (e.g. `GroupedMatmul`) but the model is not actually MoE | MoE detection checks both `main_rows` and `all_rows` and only affects representative layer selection / second-stage naming, not Stage annotation correctness |
| Single-Attention layer extraction | With only 1 Attention, extracts to end, may cross layers | Document this scenario; `--layer-index` can specify layer number |
| Communication operator regex gaps | New communication operator names not covered | Regex covers common variants (all_reduce/all_gather/hcom_/reduceScatter) |
| xlsx Sheet name truncation | Excel limits Sheet names to 31 chars | `_sheet_name_from_csv` truncates to 31 chars |

---

## 6. Prior Art

| Project/Tool | Approach | Reference and Differences |
|------|------|-----------|
| Chrome Trace Event format | Standard performance trace format (exported by torch.profiler) | `trace_json_to_csv.py` parses `ph=X` + `cat=analytic` events, deduplicates and converts to CSV |
| NPU profiling tools | Export kernel_details.csv (with detailed aicore/aiv metrics) | This tool only uses basic fields like `Name/Type/Start Time/Duration/Shapes` |
| torch.profiler layer_table | PyTorch built-in layer-level profiling | This tool does not depend on torch, pure CSV/JSON offline analysis; supports NPU-side traces |
| MindStudio Insight | NPU official profiling visualization tool | This tool focuses on layer structure comparison, not replacing visualization; outputs xlsx for Excel secondary analysis |

---

## Appendix

### Appendix A: References

- [Chrome Trace Event Format Spec](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
- [torch.profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [openpyxl Documentation](https://openpyxl.readthedocs.io/) — xlsx generation dependency
- Project README: [README.md](./README.md)

### Appendix B: Glossary

| Term | Description |
|------|------|
| Forward Segment | Operator collection of one forward inference, segmented by embedding anchor or gap |
| Layer | Transformer DecoderLayer, with Attention as start anchor |
| Stage | 2-stage substructure within a layer: Attention / FFN (Dense) or Attention / MOE (MoE) |
| RSN | Residual + RMSNorm (Residual Side Norm) |
| Structure | Internal substructure label within layer (1a_Norm / 2_Attention / 3_Linear(post-att) / 4_Norm(pre-MLP) / 5_MLP); used by `refine_sub_blocks` and the print summary, excluded from layer CSV output |
| Marker | Global operator type marker (EMBED/ATT/NORM/MLP/MATMUL/LINEAR/COMM/SAMPLE) |
| Unfused RMSNorm | RMSNorm split into 8 base operators on NPU (view/add/pow/mean/rsqrt/mul) |
| MoE | Mixture of Experts, mixed expert layer |
| Dense | Standard Transformer layer, not MoE |
| Prefill | First forward, contains embedding, sequence length > 1 |
| Decode | Incremental forward, no embedding, sequence length = 1 |
| Gap Threshold | Interval threshold (us) for Forward segmentation; values above this are considered forward boundaries |

### Appendix C: Document History

| Version | Date | Changes | Author |
|------|------|---------|------|
| v1.0 | 2026-08-01 | Initial version: NPU Layer Analyzer toolset design document | @yuyinkai |
| v1.1 | 2026-08-05 | Stage annotation 4→2 (Attention / FFN or MOE, controlled by `is_moe`); main stream selection changed to attention-first `(attention, embedding, not_na, rows)`; `--forward-kind` default changed to `all` (export all forward types); `--expected-attention` default changed to 0 (check disabled); layer CSV filename now includes layer number (`_layerN` / `_layerN_dense` / `_layerN_moe`); `embedding_main` now checks `output_rows` to avoid misclassifying decode; `extract_layer_rows` keeps only main-Stream operators; MoE detection now checks both `main_rows` and `all_rows` (non-main-stream ops assigned by time overlap), `MOE_RE` extended to GroupedMatmul / DispatchFFNCombine / MoeGatingTopK; `Structure` column removed from CSV output (internal only); `compare_result.xlsx` supports up to 4 Sheets with Chinese names (Dense_总比较 / Dense_算子明细 / MoE_总比较 / MoE_算子明细); shared `layer_common.py` module added; `--forward-kind` parameter added to `npu_layer_compare.py` and `npu_layer_analyzer.py` parameter tables | @yuyinkai |
