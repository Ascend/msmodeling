# 参数推荐输入 Schema

收集首次寻优信息时使用本参考文档。

## 必需字段

```json
{
  "engine": "mindie | vllm",
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
  "target": "throughput | ttft | tpot | balanced"
}
```

`benchmark_policy` 是可选字段。缺省时使用 `ais_bench`。

`discovery` 是可选字段。缺省时不做 `--help` 参数发现。

## 没有 config.json 时的模型字段

如果用户无法提供模型 `config.json`，需要收集：

- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `torch_dtype`
- `max_position_embeddings`
- `vocab_size`，如用户知道

可选字段：

- `model_weight_gb`
- `num_parameters_billion`
- `model_type`
- `is_moe`
- `num_experts`
- `num_local_experts`
- `n_routed_experts`
- `is_multimodal`
- `is_quantized`

## 可选 discovery 字段

```json
{
  "discovery": {
    "enabled": true,
    "vllm_help_text_path": "/tmp/vllm-serve-help.txt",
    "vllm_help_text": "也可以直接放入 vllm serve --help 的输出"
  }
}
```

使用建议：

- `enabled` 为 `false` 或缺省时，只输出基础推荐参数。
- `vllm_help_text_path` 适合 agent 先执行 `vllm serve --help > /tmp/vllm-serve-help.txt` 后再调用脚本。
- `vllm_help_text` 适合测试或用户直接粘贴 help 输出。
- 当前 discovery 只分析 vLLM help；MindIE 仍优先读取 config。
- discovery 会结合 `model.is_multimodal`、`model.is_moe` 和业务输入长度决定是否追加参数。

## 问题顺序

每轮只问一个主题：

1. 硬件信息。
2. 模型路径或模型结构。
3. 业务 token 长度。
4. 优化目标。
5. 可选：启动命令或 MindIE config 路径。
6. 可选：如果用户同意执行本地只读命令，收集 `vllm serve --help` 输出用于 discovery。

如果 `recommend_params.py` 返回 `need_more_info`，优先询问返回的 `next_question`。
