# Shape 网格生成

`generate_shape_grid.py` 会针对目标 HuggingFace 模型自动运行多组 throughput optimizer，捕获性能数据库实际收到的
Kernel 查询，并把可 replay 的 Shape 追加到数据库 CSV。用户显式指定但查询未命中的 `--ops` 会自动使用通用理论规则
生成，结果统一供 microbench 实测回填。也可以通过 `--optimizer-args-file` 直接传入实际执行的寻优参数，
按真实 workload 生成定向 Shape。

```powershell
python tools/perf_data_collection/generate_shape_grid.py `
  --database-path <性能数据库目录> `
  --target-models <HuggingFace模型ID> `
  --rows 1000
```

公开参数只有：

- `--database-path`：必选，目标数据库目录；
- `--rows`：每个 CSV 本轮最多新增行数，默认 1000；
- `--target-models`：可传一个或多个 HF 模型 ID，触发内部采样扫描；
- `--optimizer-args-file`：可选，实际寻优参数 workload spec（YAML/JSON），与 `--target-models` 至少提供一个，可同时提供；
- `--ops`：可选，指定最终需要扩展的 op replay 支持算子；
- `--seed`：可选，控制候选稳定顺序，默认 0；
- `--report-path`：可选，机器可读 JSON 生成报告输出路径；缺省写入自动查询缓存目录。

已有行、重复行和拒绝行不占 `--rows`。以相同参数再次运行会继续补后续候选，而不是重新采样同一批行。

`--ops` 的优先级高于 workload 来源。例如查询到 `A/B/C/D`，而用户传入 `--ops C D E`，最终只生成 `C/D/E`：
`C/D` 使用查询网格，未查询到的 `E` 使用通用理论网格。未传 `--ops` 时，以实际查询到的 replay 算子作为生成集合。
理论配置明确标记为 `skip` 的算子不会兜底生成。数据库不区分 Shape 来源。

工具内部会自动运行单卡长度/batch 基线、TP/EP/MoE-DP/DCP/MTP 分轴扫描、少量并行交叉边界、长序列
chunked-prefill，以及数据库支持时的 compile+SP+DFC。W8A8 动态量化之外还会覆盖 BF16 基线和代表性的 INT8 KV
cache；这些策略不增加新的公开参数。

## 实际寻优参数模式（--optimizer-args-file）

将实际执行的 `throughput_optimizer` 命令等价转换为 YAML/JSON workload spec，字段名与 CLI 的 snake_case
语义一一对应，省略键与省略 CLI flag 行为一致；未知或不支持字段 fail closed（`dp_sizes`、speculative
三件套、multimodal、PD-ratio 等当前不可用字段会给出定向错误）。`pp_sizes` 与 `pp_layer_partitions`
同样受支持：`pp_sizes` 缺省为仅 PP=1（旧行为），空列表为 2 的幂默认范围；PP>1 要求
`num_mtp_tokens` 包含 0（即不允许推测性 MTP），显式 `pp_layer_partitions` 时每个请求的 PP>1 都需要等长分区且分区
求和等于模型 `num_hidden_layers`。

```yaml
workloads:
  - name: prefill-20k
    model: zai-org/GLM-5.1
    device: ATLAS_800_A3_752T_128G_DIE
    num_devices: 32
    input_length: 20000
    output_length: 1024
    disagg: true
    ttft_limit: 10000
    max_batched_tokens: 20000
    batch_range: [1, 32]
    tp_sizes: [1, 2, 4, 8, 16]
    ep_sizes: [4, 8, 16, 32]
    moe_dp_sizes: [1]
    num_mtp_tokens: [0]
    quantize_linear_action: W8A8_DYNAMIC
    quantize_attention_action: DISABLED
    reserved_memory_gb: 10
    compile: true
    compilation_config: [enable_sequence_parallel, enable_dispatch_ffn_combine]
    enable_shared_expert_tp: true
  - name: decode-20k
    model: zai-org/GLM-5.1
    device: ATLAS_800_A3_752T_128G_DIE
    num_devices: 32
    input_length: 20000
    output_length: 1024
    disagg: true
    tpot_limit: 70
    max_batched_tokens: 80000
    batch_range: [1, 32]
    tp_sizes: [1, 2, 4, 8, 16]
    ep_sizes: [4, 8, 16, 32]
    moe_dp_sizes: [1]
    dcp_sizes: [1, 2, 4, 8, 16]
    num_mtp_tokens: [0, 2]
    mtp_acceptance_rates: [0.9, 0.6]
    quantize_linear_action: W8A8_DYNAMIC
    quantize_attention_action: DISABLED
    reserved_memory_gb: 10
    compile: true
    compilation_config: [enable_sequence_parallel, enable_dispatch_ffn_combine]
    enable_shared_expert_tp: true
```

```powershell
python tools/perf_data_collection/generate_shape_grid.py `
  --database-path <性能数据库目录> `
  --optimizer-args-file workload-spec.yaml `
  --rows 1000
```

该模式下的语义约定：

- spec 的 `device` 必须与目标数据库设备一致，否则报错；
- 并行候选先经 optimizer 同款解析，再展开为单并行配置的独立子进程，与内部模式共享 checkpoint 缓存；
- **optimizer-args exact demand 不受 `--rows` 截断**：spec workload 的 exact 行永远全部写入；组合模式下，
  target-model workload 的 exact 行仍与 Coverage 插值和 constraint fallback 候选共享 `--rows` 预算；
- 每条 exact demand 都有确定状态（已有有效行 / 生成新行 / 重复 / 投影拒绝 / replay 校验拒绝 / 算子不支持），
  禁止静默遗漏；
- spec 内容以稳定 digest 计入缓存键与报告，同一 spec 复跑可复现、可审计；
- spec 的 `device` 必须是注册 DeviceProfile。

## 生成报告与 replay preflight

所有新增行写入 CSV 前会执行纯 Python replay preflight：加载对应 `op_replay/*_run.py` 并 stub 掉 NPU
张量构建，真实执行该 Kernel 的 `build_case` 契约校验 Shape/dtype/format/辅助张量约束；失败行不写入并记录原因。

生成结束后输出机器可读 JSON 报告（写入查询缓存目录，或 `--report-path` 指定路径），包含 workload 摘要与
spec digest、demand → 状态台账、每算子 preflight 结果、生成前后 CSV 行数与待 microbench 回填行数。

生成结束后再运行 microbench：

```powershell
python tools/perf_data_collection/start_microbench.py `
  --database-path <性能数据库目录> `
  --update-mode missing-only
```

Shape 网格生成不执行 NPU 实测，新增行的耗时为 0。A3 replay、coverage、插值误差和 text_generate B2B 仍需单独验收。

详细设计见 [查询驱动的 Shape 网格生成设计](../../docs/design/query_driven_shape_grid_generation.md)。
