---
name: optix-param-recommend
description: 当首次使用 msmodeling optix 的用户需要根据硬件、模型、负载和优化目标推荐 MindIE/vLLM 寻优参数、搜索范围、benchmark 侧字段或 config.toml 片段时使用。
---

# 参数范围推荐

## 目标

为首次使用 `msmodeling optix` 的用户推荐保守、可解释的 MindIE / vLLM 寻优参数和初始搜索范围。用户不清楚应该调哪些参数、参数范围如何设置、benchmark 参数如何配置时，使用本 skill。

## 必须遵循的流程

缺少必需信息时，不要输出最终参数范围。

1. 收集必需上下文：
   - 推理框架：`mindie` 或 `vllm`
   - 硬件信息：`single_card_mem_gb`、`world_size`、`num_per_nodes`、`num_nodes`
   - 模型信息：本地 `config.json` 路径，或显式模型结构字段
   - 业务负载：平均/最大输入 token，平均/最大输出 token
   - 优化目标：`throughput`、`ttft`、`tpot` 或 `balanced`
2. 将已收集信息写入一个 JSON context 文件。
3. 执行 `.agents/skills/optix-param-recommend/scripts/recommend_params.py --context <context.json>`。
4. 如果脚本返回 `status: need_more_info`，只询问返回的 `next_question`，不要给最终范围。
5. 如果脚本返回 `status: ok`，再总结推荐结果并附上 TOML 片段。

如果用户没有指定 benchmark，MindIE 和 vLLM 都默认使用 `ais_bench`。

## Context 格式

最小 JSON 结构如下：

```json
{
  "engine": "vllm",
  "hardware": {
    "single_card_mem_gb": 64,
    "world_size": 8,
    "num_per_nodes": 8,
    "num_nodes": 1
  },
  "model": {
    "config_path": "/path/to/model/config.json"
  },
  "workload": {
    "input_len_avg": 1024,
    "input_len_max": 4096,
    "output_len_avg": 256,
    "output_len_max": 512
  },
  "target": "balanced",
  "discovery": {
    "enabled": false
  }
}
```

完整输入说明见 `references/input-schema.md`。

## 推荐规则

核心并行约束是：

```text
DP * TP * PP == world_size
```

首次使用时，优先推荐在稳定可行的前提下用满卡：

```text
DP * TP * PP == world_size
```

脚本会根据模型 config、单卡显存、TP、dtype 和业务负载估算模型权重与 KV cache 容量，然后推荐：

- MindIE：`max_batch_size`、`max_prefill_batch_size`、`max_prefill_token`、排队/调度策略参数、`tp`、`dp`、MoE 专家并行参数、`ais_bench` 压测参数。
- vLLM：`MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS`、`MAX_MODEL_LEN`、TP/PP/DP 参数、`GPU_MEMORY_UTILIZATION`、`BLOCK_SIZE`、前缀缓存、分块 prefill、编译配置、`ais_bench` 压测参数。

`ENABLE_PREFIX_CACHING` 和 `ENABLE_CHUNKED_PREFILL` 默认交给 vLLM 自身的模型感知默认值，不作为搜索维度，也不写入启动命令。只有用户明确要求覆盖默认行为时，才把对应 flag 放进 `vllm.command.others`。

详细启发式规则见 `references/parameter-rules.md`。

## 启动命令发现

如果用户提供 vLLM 启动命令，把它当作可运行命令骨架，不要当作完整搜索空间。可以执行只读帮助命令：

```bash
vllm serve --help
vllm bench serve --help
```

第一版只把发现到的额外参数作为可选候选。对 MindIE，优先读取 `config.json`，不要依赖命令行帮助。

如需启用可选发现功能，先执行只读 help 命令，将输出写入 context：

```json
{
  "discovery": {
    "enabled": true,
    "vllm_help_text_path": "/tmp/vllm-serve-help.txt"
  }
}
```

脚本会根据模型和负载决定是否追加 `MAX_NUM_PARTIAL_PREFILLS`、`LONG_PREFILL_TOKEN_THRESHOLD`、`DISABLE_CHUNKED_MM_INPUT`、`ENABLE_EXPERT_PARALLEL` 等候选。`result.discovery.added_parameters` 会列出本轮由 help 发现追加的字段。

vLLM 的 presence flag 参数分两类处理：`ENABLE_PREFIX_CACHING` 和 `ENABLE_CHUNKED_PREFILL` 使用 vLLM 默认值，不写入命令；`COMPILATION_CONFIG` 这类需要搜索的字符串枚举才用 `$COMPILATION_CONFIG` 占位符接入 `vllm.command.others`。

## 输出要求

脚本返回 `status: ok` 时，输出：

- 输入摘要和假设。
- 如仍有假设，明确标注。
- 推荐参数表：参数名、范围、默认值、是否参与搜索、推荐理由。
- `toml_snippet` 中的 `config.toml` 片段。
- `config_skill_handoff`：给 `optix-config` skill 使用的机器可读交接对象。
- `DP * TP * PP == world_size` 约束解释。
- `next_command` 中的下一步命令。

不要静默应用 TOML 片段。修改现有 `config.toml` 前必须征得用户同意。

## 与配置 Skill 衔接

`config_skill_handoff` 是给 `optix-config` 的稳定输出契约，包含：

- `consumer_skill`：固定为 `optix-config`。
- `target_fields`：规范化后的推荐字段，保留 `section`、`name`、`dtype`、`min`、`max`、`value`、`dtype_param`、`search` 和推荐理由。
- `vllm_command_others`：vLLM 需要放入 `[vllm.command].others` 的片段；可搜索参数使用 `$VAR` 占位符，需要显式覆盖默认值的 presence flag 才使用字面量。
- `apply_commands`：可交给 config skill 审阅执行的 `auto_config.py` 命令清单。
- `notes`：当前 config skill CLI 无法直接覆盖的部分，例如 `ais_bench` section。

交接原则：

- `toml_snippet` 用于人工审阅和精确复制。
- `config_skill_handoff.target_fields` 用于后续做自动应用。
- `config_skill_handoff.apply_commands` 用于当前已有 config skill 的命令式衔接。
- vLLM 的 `MAX_NUM_BATCHED_TOKENS` 和 `MAX_NUM_SEQS` 已在工具的 `VllmCommand` 中内置占位符，不需要再追加到 `others`。
- `ENABLE_PREFIX_CACHING`、`ENABLE_CHUNKED_PREFILL` 这类 vLLM 可自行决定默认值的 flag 不写入 target_field，也不写入 command；`ENABLE_EXPERT_PARALLEL` 这类由 discovery 明确识别的 MoE flag 才写入 command 字面量。


## 快速命令

打印 context 模板：

```bash
# 打印 context 模板
python .agents/skills/optix-param-recommend/scripts/recommend_params.py --print-template

# 执行推荐
python .agents/skills/optix-param-recommend/scripts/recommend_params.py --context /path/to/context.json
```
