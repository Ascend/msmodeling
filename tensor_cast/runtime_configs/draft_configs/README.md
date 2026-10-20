# Draft Runtime Configs 使用说明

## 1 简介

本目录存放 TensorCast 运行时配置中的通用 draft 配置 JSON。Dflash 与通用 DSpark 共用同一套 draft 骨架；默认配置文件为 [`dflash_draft_builtin.json`](./dflash_draft_builtin.json)。模型已适配原生 DSpark draft 时，默认使用原生结构，不加载这份通用配置。

| 项目 | 路径 / 参数 |
| --- | --- |
| 内置默认 | [`dflash_draft_builtin.json`](./dflash_draft_builtin.json) |
| 外部覆盖 | `--draft-model-config-path` → JSON 文件，或指向包含 `config.json` 的目录 |

## 2 快速用法

在 `text_generate` 或 `throughput_optimizer` 中启用投机解码时，需先设置 `--speculative-method`，再按需传入 draft 相关从属参数：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --num-queries 8 \
  --query-length 8 \
  --context-length 4500 \
  --decode \
  --device TEST_DEVICE \
  --speculative-method dflash \
  --num-speculative-tokens 7 \
  --num-draft-layers 6 \
  --compile
```

说明：

- 必须显式传入 `--speculative-method`；仅设置 `--num-speculative-tokens` 等从属参数不会启用。
- `--query-length` 应不小于 draft block 长度（`block_size = --num-speculative-tokens + 1`；未指定时使用内置 config 默认值 `8`）。
- 自定义 draft 结构时，通过 `--draft-model-config-path` 指向外部 JSON；完整字段说明见下文。

## 3 覆盖优先级

DSpark 未指定 `--num-draft-layers` 和 `--draft-model-config-path` 时：

- 已适配原生 draft 的模型使用模型声明的层数、辅助隐藏状态来源和窗口大小。例如 DeepSeek V4.1-Flash 默认使用原生 3 层 draft，而非通用 Qwen3 draft 骨架。
- 没有原生 draft 适配的模型使用通用配置，默认 6 层。其他模型的 MTP 层数声明不会被当作原生 DSpark 配置。
- 显式指定的 CLI 参数或外部 draft 配置继续优先生效，并接受对应构建器的结构校验；DeepSeek V4.1 原生路径目前仍要求 draft 层数与原生声明一致。

以下配置文件规则适用于通用 draft 路径，以及用户显式提供的外部 draft 配置：

加载 draft 配置时，各字段的最终取值按下面顺序确定；**后一步会覆盖前一步**：

1. **基底**：先读取内置 [`dflash_draft_builtin.json`](./dflash_draft_builtin.json)，或由 `--draft-model-config-path` 指定的 JSON。
2. **CLI 覆盖**：若传入 `--num-speculative-tokens`（`n ≥ 1`）或 `--num-draft-layers`（`> 0`），对应地改写 `block_size`（`block_size = n + 1`）或 `num_hidden_layers`。
3. **主模型对齐（始终生效）**：`hidden_size`、`vocab_size`、`max_position_embeddings` 在运行时强制与 target 主模型一致，JSON 中的同名值会被忽略。

因此，自定义 JSON 通常只需关心 draft 结构字段；主模型维度不必写入，也不应依赖 profile 里的旧值。

## 4 常用字段

完整字段定义请参考 [`dflash_draft_builtin.json`](./dflash_draft_builtin.json)。下表列出自定义 profile 时最常修改的字段：

| 字段 | 内置默认 | 说明 | CLI 可覆盖 |
| --- | ---: | --- | --- |
| `block_size` | `8` | draft block 长度（含 anchor），与 `--query-length` 对齐 | 是，`--num-speculative-tokens`（`n ≥ 1` → `block_size = n + 1`） |
| `num_hidden_layers` | `6` | draft 层数 | 是，`--num-draft-layers`（`> 0`） |
| `layer_types` | 6× `sliding_attention` | 逐层注意力类型：`full_attention` / `sliding_attention`；长度需等于层数 | 否（层数被 CLI 改写时自动同步长度） |
| `dflash_config.target_layer_ids` | `[1,12,24,35,47,58]` | 从 target 抽取 aux hidden 的层号；须非空。`--num-draft-layers` 只改 draft 层数并同步 `layer_types`，**不会**拓充本列表。若 `max(id) >= target.num_hidden_layers`，运行时按主模型层数等间隔重采样；`--num-draft-layers` 超过主模型层数则报错 | 否 |
| `mask_token_id` | `163838` | draft noise 中的 MASK 占位 token；也可写在 `dflash_config.mask_token_id` | 否 |
| `model_type` | `"qwen3"` | draft 模型族；内置栈请保持 `"qwen3"` | 否 |
| `num_attention_heads` | `64` | draft 注意力头数 | 否 |
| `num_key_value_heads` | `8` | GQA 的 KV 头数 | 否 |
| `head_dim` | `128` | 每头维度 | 否 |
| `intermediate_size` | `18432` | draft MLP 中间层宽度 | 否 |
| `hidden_act` | `"silu"` | MLP 激活函数 | 否 |
| `attention_bias` | `false` | QKV / out-proj 是否带 bias | 否 |
| `rms_norm_eps` | `1e-5` | RMSNorm 的 epsilon | 否 |
| `rope_theta` | `50000.0` | RoPE 基频 | 否 |
| `rope_scaling` | yarn 字典 | RoPE 缩放配置 | 否 |
| `sliding_window` | *（可省略）* | sliding attention 层窗口大小；含 sliding 层且未设时默认 `2048` | 否（自动推导） |
| `hidden_size` | — | 隐层宽度 | 否（运行时由主模型对齐） |
| `vocab_size` | — | 词表大小 | 否（运行时由主模型对齐） |
| `max_position_embeddings` | — | 位置 / RoPE 长度预算 | 否（运行时由主模型对齐） |

自定义 JSON 不必包含 `hidden_size`、`vocab_size`、`max_position_embeddings`，也不应依赖 profile 中的旧值。

## 5 自定义 profile

1. 复制 [`dflash_draft_builtin.json`](./dflash_draft_builtin.json)（或按上表结构字段编写最小 JSON）。
2. 按目标模型调整 `block_size`、`num_hidden_layers`、`layer_types`、`dflash_config.target_layer_ids`。
3. 运行时传入 `--speculative-method dflash` 或 `dspark`，并通过 `--draft-model-config-path` 指定 JSON 路径。

```bash
python -m cli.inference.text_generate /data/models/Qwen3-32B \
  --num-queries 4 \
  --query-length 8 \
  --context-length 2048 \
  --decode \
  --device TEST_DEVICE \
  --speculative-method dflash \
  --draft-model-config-path /path/to/my_dflash_draft/config.json \
  --compile
```

## 6 约束与注意

- `--speculative-method` 与 MTP（`--num-mtp-tokens`）互斥；`dflash` / `dspark` 通过 `--speculative-method` 单选。
- DSpark 复用同一套 draft JSON 骨架；Markov 相关参数通过 `--dspark-markov-rank` / `--dspark-markov-head` 配置，不写在该 JSON 中。
- `--num-draft-layers` 不得大于主模型 `num_hidden_layers`。内置 `target_layer_ids`（按 64 层 Qwen3 标定）在主模型更浅时会按层数等间隔重采样，而不是直接报错。
- draft 自有 Linear 不参与 `--quantize-linear-action` 量化。
- `--acceptance-length` 仅用于 `throughput_optimizer` 的 Decode 吞吐折算，不参与 `text_generate` 构图。
