# 参数推荐规则

本文档记录第一版启发式规则。规则面向首次使用用户，优先保守、稳定、可解释。

## 通用规则

- 默认 benchmark：`ais_bench`。
- 核心约束：`DP * TP * PP == world_size`。
- 首次寻优优先推荐 `DP * TP * PP == world_size`，即尽量用满卡。
- TP 从 `world_size` 的因子中选择，同时要求能整除 `num_attention_heads`。
- 单机场景首次寻优默认 PP 为 1；多机场景或模型很大时再考虑 PP。
- 根据模型 config、单卡显存、TP、dtype 和业务 token 长度估算 KV cache 容量。

## MindIE 字段

首次寻优推荐字段：

- `max_batch_size`：根据 KV cache 容量估算。
- `max_prefill_batch_size`：`max_batch_size` 的比例；TTFT 优先时更低，吞吐优先时更高。
- `max_prefill_token`：覆盖首次寻优业务负载的 prefill token 压力。
- `max_queue_deloy_mircroseconds`：TTFT 优先时更低，吞吐优先时更高。
- `support_select_batch`：TTFT 优先时关闭，吞吐或均衡场景开启。
- `prefill_time_ms_per_req`、`decode_time_ms_per_req`：作为可搜索调度参数。
- `max_preempt_count`：按 `max_batch_size` 的比例生成，首次寻优保持较低范围。
- `prefill_policy_type`、`decode_policy_type`：使用 optimizer 已支持的枚举候选 `[0, 1, 3]`。
- `tp`、`dp`：作为并行参数；`dp` 默认由 `world_size / tp` 派生。
- `moe_ep`、`moe_tp`：仅在模型为 MoE 或 config 中存在专家数信息时推荐，`moe_tp` 由 `moe_ep` 派生。
- `CONCURRENCY`、`REQUESTRATE`：`ais_bench` 压测负载参数。

MindIE 第一版默认 PP 为 1，除非用户确认自己的配置里有 PP 字段。因此脚本输出的约束说明统一写作 `DP * TP * PP == world_size`；在未显式配置 PP 的场景下，TOML 约束默认等价为 `$dp * $tp == $NPU_COUNT`。

## vLLM 字段

首次寻优推荐字段：

- `MAX_MODEL_LEN`：固定为 `input_len_max + output_len_max`，并受模型最大上下文限制。
- `MAX_NUM_SEQS`：根据 KV cache 容量估算。
- `MAX_NUM_BATCHED_TOKENS`：与输入长度和 `MAX_NUM_SEQS` 联动。
- `TENSOR_PARALLEL_SIZE`：选择能整除 attention heads 的 world_size 因子。
- `PIPELINE_PARALLEL_SIZE`：首次单机场景固定为 1；多机或大模型再考虑增加。
- `DATA_PARALLEL_SIZE`：派生固定值；只有用户当前 vLLM 启动模式支持时才接入命令。
- `GPU_MEMORY_UTILIZATION`：保守范围 `0.85-0.92`，降低首次启动 OOM 风险。
- `BLOCK_SIZE`：KV cache block size 候选，建议用本机 `vllm serve --help` 确认可用枚举。
- `ENABLE_PREFIX_CACHING`：presence flag，首次推荐使用 vLLM 默认值，不纳入搜索，也不显式传参。
- `ENABLE_CHUNKED_PREFILL`：presence flag，首次推荐使用 vLLM 默认值，不纳入搜索，也不显式传参。
- `COMPILATION_CONFIG`：字符串枚举，默认空值；确认环境支持后可加入 cudagraph 编译配置候选。
- `CONCURRENCY`、`REQUESTRATE`：`ais_bench` 压测负载参数。

presence flag 字段需要谨慎处理。vLLM 已有模型感知默认值的 flag 默认不写入 `vllm.command.others`；需要强制启用的 discovery flag 直接写字面量；需要搜索的字符串枚举才使用 `$COMPILATION_CONFIG` 这类占位符。

## vLLM Help Discovery 字段

当 `discovery.enabled = true` 且提供 `vllm_help_text` 或 `vllm_help_text_path` 时，脚本会额外扫描 help 输出。当前支持的可选追加规则：

- help 包含 `--max-num-partial-prefills` 且业务有长 prefill 时，追加 `MAX_NUM_PARTIAL_PREFILLS`。
- help 包含 `--long-prefill-token-threshold` 且业务有长 prefill 时，追加 `LONG_PREFILL_TOKEN_THRESHOLD`。
- help 包含 `--disable-chunked-mm-input` 且 `model.is_multimodal = true` 时，追加固定 flag `DISABLE_CHUNKED_MM_INPUT`。
- help 包含 `--enable-expert-parallel` 且模型为 MoE 时，追加固定 flag `ENABLE_EXPERT_PARALLEL`。

追加字段会标记 `source = "vllm --help"` 和 `optional = true`，并在 `result.discovery.added_parameters` 中列出。

## Benchmark 影响

benchmark 类型会影响 benchmark 侧 target field 和性能指标解释，但不应该覆盖基于硬件和模型推导出的服务侧参数范围。

第一版中，用户未指定 benchmark 时始终输出 `ais_bench` 字段。

## 启动命令发现

用户提供的启动命令是可运行骨架，不是完整搜索空间。对 vLLM，help 输出可能发现模型相关的可选参数。除核心首次寻优字段外，发现到的额外参数默认保持固定或作为可选候选。

## 输出交接规则

脚本输出同时服务三类消费方：

- 用户阅读：看 `recommendations` 和 `toml_snippet`。
- agent 总结：看 `assumptions`、`constraints`、`discovery`。
- 配置 skill 应用：看 `config_skill_handoff`。

`config_skill_handoff.apply_commands` 只生成当前 `optix-config` CLI 能表达的命令，并使用 `--option=value` 形式承载以 `--` 开头的参数值。`ais_bench` 相关字段保留在 `target_fields` 和 `notes` 中，不写入 `toml_snippet` 的 target-field 块，因为当前 Settings loader 对 `ais_bench` target field 的处理存在兼容风险。
