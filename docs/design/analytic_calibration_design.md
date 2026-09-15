# Design Document: Analytic Calibration for Analytic Performance Model / Analytic Performance Model 实测校准

## Revision History (修订记录)

| Date (日期) | Version (修订版本) | Change Description (修改描述) | Author (作者) | RFC Document (RFC文档) |
| --- | --- | --- | --- | --- |
| 2026-09-10 | 1.0 | Initial SQLite analytic calibration design (完成 analytic calibration 设计) | MindStudio-Modeling | N/A |

---

## 1. Background (背景描述)

TensorCast 的 `AnalyticPerformanceModel` 根据算子复杂度、设备峰值能力、内存带宽和通信拓扑估算算子时延。该模型具有输入覆盖范围广、无需逐 shape 实测的优点，但对 NPU kernel 的 launch overhead、实际 MMA 利用率、量化/融合开销和通信实现差异只能进行近似。

Analytic Calibration 的目标是复用仓库已有的实测单算子性能数据，将 profiling CSV 中的实测时延转换为 TensorCast semantic operation 的校准曲线，并在运行时对 raw analytic 结果进行有边界的修正。该能力不替换 analytic 模型，也不依赖 runtime 读取 `op_mapping.yaml`、原始 CSV 或 audit 文件。

本设计覆盖两条闭环：

```text
运行时：CLI → UserInputConfig → ModelRunner → raw analytic → semantic signature → SQLite lookup → calibrated result

离线：profiling CSV → family builder → SQLite profile + audit YAML → CLI 参数加载
```

当前设计的核心目标是：

- 用一个 SQLite artifact 统一保存 MM、GMM、Communication 和 Attention 校准数据；
- 使用 TensorCast operation semantic signature 进行 runtime 匹配，不把物理 kernel 名称带入 runtime 推断；
- 对已测 shape 使用 exact/curve/interpolation 结果，对未覆盖或越界 shape 安全回退 raw analytic；
- 保留 raw analytic 的统计信息，支持 bound、source、confidence 和 fallback 原因诊断；
- 通过独立 audit YAML 保存 CSV 来源、hash、拒绝行、曲线压缩误差和 holdout 误差。

## 2. Design (方案设计)

### 2.1 Architecture overview (总体架构)

```text
                         Offline build path
  MM/GMM/HCCL/Attention CSV
             │
             ├── mm_build.py
             ├── gmm_build.py
             ├── comm_build.py
             └── attention_build.py
             │
             ├── profile validation
             ├── compressed family documents
             ├── indexed Attention points
             └── audit YAML
             │
             ▼
       SQLite profile
             │
             │ --analytic-calibration-profile
             ▼
  CLI → UserInputConfig → ModelRunner
                              │
                              ▼
                    AnalyticPerformanceModel
                              │ raw result
                              ▼
              CalibratedAnalyticPerformanceModel
                              │
          build_calibration_signature(OpInvokeInfo, raw result)
                              │
                              ▼
               ProfileCalibrationDataSource.lookup
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      family rule matching              Attention SQLite index
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                 CalibrationRule.apply(raw result)
                              │
                              ▼
             calibrated result + diagnostic statistics
```

### 2.2 Runtime call chain (运行时调用链)

#### 2.2.1 CLI and configuration (CLI 与配置)

`text_generate` 和 `throughput_optimizer` 暴露以下参数：

| 参数 | 作用 |
| --- | --- |
| `--analytic-calibration-profile <FILE>` | 指定 SQLite calibration profile |
| `--analytic-calibration-stack <STACK>` | 显式指定 profile 对应的软件栈；单 stack profile 可自动选择 |

参数通过 `UserInputConfig.from_args()` 进入 `ModelRunner.create_performance_models()`。`analytic` 始终运行 raw analytic；`calibrated` 使用 SQLite profile 包装 analytic，且必须显式指定 `--analytic-calibration-profile`；`profiling` 使用实测数据库。三个模型可同时选择并在输出表格中分别显示 `analytic total/avg`、`calibrated total/avg` 和 `profiling total/avg`。

#### 2.2.2 Model wrapper (模型包装)

`CalibratedAnalyticPerformanceModel` 对每个 `OpInvokeInfo` 执行：

1. 调用 `AnalyticPerformanceModel.process_op()` 得到 raw `PerformanceModel.Result`；
2. 通过 `build_calibration_signature()` 提取 TensorCast semantic operation、dtype 和 shape/features；
3. 从 `ProfileCalibrationDataSource` 查找最佳规则；
4. 命中时调用规则生成 calibrated result；
5. 未命中时返回原始时延，并在 statistics 中写入 `source=ANALYTIC_RAW` 与 `fallback_reason`。

校准结果保留 raw result 的 bound 和资源统计；校准只修改 execution time 及相关校准诊断字段。

#### 2.2.3 Profile loading and isolation (Profile 加载与隔离)

runtime 只接受 SQLite 文件。加载时校验：

- 文件存在且可打开；
- SQLite `schema_version == 3`；
- profile device 与当前 `DeviceProfile.name` 一致；
- profile software stack 与显式或自动选择的 stack 一致。

device/stack 不匹配时 profile 被禁用，MM/GMM/Communication/Attention 均回退 raw analytic，不能只屏蔽部分 family。

### 2.3 Semantic signature (语义签名)

签名构造只依赖 `OpInvokeInfo`、raw analytic result 和当前 device，不读取 `op_mapping.yaml`。签名包含：

- `tc_op`：TensorCast operation overload 名称；
- `op_family`：`matmul`、`attention`、`communication` 或 `other`；
- `device`、`dtype`；
- family-specific features。

当前 family-specific features：

| Family | 关键特征 |
| --- | --- |
| Dense/BMM MM | `m/k/n`、`mm_kind`、`quantization`、`compute_dtype`、`min_kn/max_kn/mn/aspect` |
| Quantized linear | `m/k/n`、`weight_quant`、`w8a8/w4a8/fp8/mxfp4`、activation dtype |
| GMM | `m_total/k/n/gemm_n`、`num_experts`、`gmm_variant`、dtype、distribution、weight orientation |
| MLA projection | projection role、`m/k/n`、dense/batch/transpose-batch、quantization |
| Dense/Sparse Attention | query/KV token、head、head-dim、phase、KV length、block size、layout、sparse features |
| Communication | collective、group size、topology tier、`message_bytes`、`analytic_message_bytes` |

### 2.4 Offline build chain (离线构建调用链)

每个 builder 读取一种实测数据并生成或追加一个 SQLite profile。追加构建必须使用相同 device/software stack。

```text
database directory
  ├── op_mapping.yaml (仅离线推断 device，runtime 不读取)
  ├── MatMul*.csv / QuantBatchMatmulV3.csv
  ├── GroupedMatmul*.csv
  ├── HCCL CSV
  └── FusedInferAttentionScore.csv / SparseFlashAttention.csv
```

#### MM builder

`mm_build.py` 将 CSV 的 `Average Duration(us)` 作为 standalone measured latency，按设备 MMA 峰值计算：

```text
ideal_mma_latency = 2 × M × K × N / device_mma_peak
utilization = ideal_mma_latency / measured_latency
```

曲线 identity 为：

```text
kind + quantization + compute_dtype + K + N
```

支持 dense、batch、transpose-batch 和 weight-quant 语义。profile 中保存 M 曲线、压缩 knot、domain、regions 和 confidence。

#### GMM builder

`gmm_build.py` 当前接入 INT8 activation + INT8 weight 的：

- `GroupedMatmul` plain variant；
- `GroupedMatmulSwigluQuant` SwiGLU quant variant。

曲线 identity 为 `num_experts + K + N + gemm_N`。GMM 使用 `calibration_mode=total_latency`，测得时延中包含 grouped matmul、SwiGLU、量化和 routing 等 kernel 级开销。

#### Communication builder

`comm_build.py` 当前接入：

- `all_reduce`；
- `all_gather`；
- `reduce_scatter`；
- `all_to_all`。

曲线 identity 为：

```text
collective + dtype + group_size + topology_tier
```

每条曲线保存有序的 `[message_bytes, latency_us]` 点。`reduce_scatter` 和 `all_to_all` 使用显式字节语义转换，确保 TensorCast signature 与 HCCL CSV 的 message bucket 一致。

#### Attention builder

`attention_build.py` 当前接入：

- `FusedInferAttentionScore`：普通 Attention、量化 Attention 和 dense MLA semantic ops；
- `SparseFlashAttention`：sparse MLA semantic ops。

Attention 采用 feature-mask + ordered feature values 索引，不对 feature 做近似插值。相同 semantic key 的重复样本在写入 SQLite 时以 latency median 聚合，并保存 `sample_count`。

后续 Attention builder 可以通过 `--base-profile` 追加新 family；已有 FIA/SFA 模型和点必须保留。

### 2.5 Calibration rule and interpolation behavior (规则与插值行为)

#### MM/GMM

匹配顺序为：

1. region 规则优先按 family/dtype/quantization/op semantic identity 和 shape selector 匹配；
2. shape-aware 曲线要求 M 落在测量范围内，并在 log-log 空间对 M 做幂律插值；
3. K/N（GMM 还包括固定 `gemm_n` 和 exact `num_experts`）在 log-shape 空间做三角插值；
4. 三角插值不可用时，仅对距离不超过 `_MAX_SHAPE_FALLBACK_DISTANCE=2.0` 的局部候选使用 KNN 加权或最近邻；
5. 无合法候选、M 越界、shape 距离超限或校准异常时回退 raw analytic。

插值结果会标记：

- `confidence=measured`：M 命中实际曲线 knot 的 exact 曲线点；
- `confidence=interpolated`：三角或 M 的 log-log 插值；
- `confidence=extrapolated`：KNN/nearest-neighbor 近邻估计。

#### Communication

只在同一 curve identity 内进行 message-byte 分段线性插值；曲线范围外不外推，回退 raw analytic。

#### Attention

只做 exact semantic feature lookup，不做跨 feature-mask、跨 phase、跨 dtype 或跨 kernel family 的插值；没有 exact point 时回退 raw analytic。

### 2.6 SQLite data model (SQLite 数据模型)

SQLite profile 的核心表：

| 表 | 内容 |
| --- | --- |
| `metadata` | schema version、profile id、device、software stack，以及按 family 保存的 `calibration_sources` 溯源 metadata |
| `family_document` | MM/GMM/Communication 压缩后的 JSON family document |
| `attention_model` | Attention model id、kernel type、TensorCast ops、dtype |
| `attention_schema` | feature names 和 specificity |
| `attention_point` | ordered feature values、median latency、sample count |

Audit YAML 不作为 runtime 输入。它记录输入文件 hash、接受/拒绝行、曲线数量、压缩误差和 holdout 误差，供离线复核。
MM、GMM、Communication 和 Attention builder 分别更新 `calibration_sources.mm/gmm/communication/attention`，
追加构建不得覆盖其他 family 的数据库与 audit 溯源。

### 2.7 Impact and boundaries (影响范围与边界)

影响范围：

- analytic performance model 的 execution time；
- text generation、throughput optimizer 和 Chrome trace 中的 operator latency；
- op bound 诊断中保留 raw/calibrated 来源和匹配信息。

不改变：

- profiling performance model 的 CSV lookup 流程；
- `op_mapping.yaml` 的格式和 runtime profiling 路径；
- TensorCast operation 的计算语义；
- 未命中校准规则时的 raw analytic 数值。

当前限制：

- profile 必须是单一 device/software stack 的 SQLite artifact；
- Attention 当前只支持 exact lookup；
- GMM 当前限定 INT8 activation/weight；
- profile 之外的 dtype、kernel family 或 shape 不会被自动编造校准值；
- runtime 不支持 YAML calibration profile。

## 3. Usage Instructions (使用说明)

### 3.1 Build a profile (构建 profile)

先用 MM builder 创建基础 profile：

```powershell
uv run python -m tensor_cast.performance_model.calibration.mm_build `
  <profiling-database> `
  --target dense=MatMulV2,MatMulV3,MatMulCommon `
  --target weight_quant:w8a8=QuantBatchMatmulV3 `
  --output <profile.sqlite> `
  --audit-output <mm_audit.yaml>
```

再使用 `--base-profile <profile.sqlite>` 依次追加 GMM、Communication 和 Attention：

```powershell
uv run python -m tensor_cast.performance_model.calibration.gmm_build <profiling-database> `
  --base-profile <profile.sqlite> --output <profile.sqlite> --audit-output <gmm_audit.yaml>

uv run python -m tensor_cast.performance_model.calibration.comm_build <hccl-directory> `
  --base-profile <profile.sqlite> --output <profile.sqlite> --audit-output <comm_audit.yaml>

uv run python -m tensor_cast.performance_model.calibration.attention_build <profiling-database> `
  --device <device> --software-stack <stack> --target fia=FusedInferAttentionScore `
  --target sfa=SparseFlashAttention --base-profile <profile.sqlite> `
  --output <profile.sqlite> --audit-output <attention_audit.yaml>
```

每次追加都必须保持 device 和 software stack 一致。输入 CSV 缺失、为 Git LFS pointer 或没有合法样本时，builder 默认失败；MM 可通过 `--allow-missing-kernels` 显式允许部分候选构建。

### 3.2 Run with calibration (运行时调用)

```powershell
uv run python -m cli.inference.text_generate \
  --performance-model calibrated \
  --analytic-calibration-profile <profile.sqlite> \
  ...
```

也可以对 throughput optimizer 使用相同参数：

```powershell
uv run python -m cli.inference.throughput_optimizer \
  --performance-model calibrated \
  --analytic-calibration-profile <profile.sqlite> \
  ...
```

### 3.3 Runtime diagnostics (运行时诊断)

每个算子的 statistics 会包含：

- `source`: `ANALYTIC_RAW` 或 `ANALYTIC_CALIBRATED`；
- `calibration.profile_id`；
- `rule_id` / `match_rule`；
- `confidence`；
- `match_level`；
- `fallback_reason`；
- 插值时的 `source_curves` 或 Attention `sample_count`。

这些字段可用于 Chrome trace、模型性能报告和回归分析。

## 4. Test Design (测试设计)

### 4.1 Unit tests (单元测试)

- `test_calibrated_analytic.py`
  - raw fallback 和 rule apply；
  - MM/GMM semantic signature；
  - M、K/N、KNN、nearest-neighbor 行为；
  - Communication 字节归一化和曲线边界；
  - `calibrated` 模型必须显式加载 profile，`analytic` 保持 raw 路径；
  - 多模型输出列使用 `analytic total/avg`、`calibrated total/avg` 和 `profiling total/avg`。
- `test_sqlite_calibration_profile.py`
  - Attention 重复 key median 聚合；
  - SQLite indexed lookup；
  - 非 SQLite 文件拒绝；
  - 错误 device/stack 不命中。

### 4.2 Builder tests (构建器测试)

- `test_mm_calibration_build.py`
  - MM 曲线拟合、dtype 分区、缺失/非法 CSV 行；
  - audit 和 source metadata。
- `test_comm_calibration_build.py`
  - collective identity 分曲线；
  - duplicate message bucket 拒绝。
- `test_attention_calibration_build.py`
  - FIA 对普通 Attention/MLA semantic ops 的声明；
  - FIA/SFA 通过 base profile 增量追加且不丢失已有模型。

### 4.3 CLI and integration tests (CLI 与集成测试)

- `tests/regression/cli/test_spec_cli.py` 验证 text_generate 参数解析；
- `tests/regression/cli/test_throughput_optimizer.py` 验证 throughput optimizer 参数解析；
- ModelRunner 集成测试验证 profile 配置只改变 analytic route，不改变 profiling route。

### 4.4 Required validation (提交前验证)

提交前至少执行：

```powershell
uv run pytest tests/benchmark/ops/perf_database/test_calibrated_analytic.py `
  tests/benchmark/ops/perf_database/test_sqlite_calibration_profile.py `
  tests/benchmark/ops/perf_database/test_attention_calibration_build.py `
  tests/benchmark/ops/perf_database/test_mm_calibration_build.py `
  tests/benchmark/ops/perf_database/test_comm_calibration_build.py -q

uv run pytest tests/regression/cli/test_spec_cli.py tests/regression/cli/test_throughput_optimizer.py -q
```

还应检查：

- profile 使用当前 SQLite schema；
- 所有 family 的 device/stack 一致；
- audit 中的输入 hash 与构建输入一致；
- Attention 增量构建后已有 model id 仍存在；
- 越界查询全部安全回退 raw analytic；
- runtime 不读取 CSV、audit 或 `op_mapping.yaml`。

---
