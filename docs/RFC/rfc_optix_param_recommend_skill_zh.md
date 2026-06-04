# RFC: 参数范围推荐器（param-recommend Skill）

## 元数据

| 项目 | 内容 |
|:-----|:--------|
| **状态** | 实施中 |
| **作者** | wendellX |
| **创建日期** | 2026-05-21 |
| **最后更新** | 2026-05-30 |
| **相关链接** | [Skill 文档](../../.agents/skills/optix-param-recommend/SKILL.md) |

---

## 1. 概述

### 1.1 背景与问题

首次使用 `ms_serviceparam_optimizer` 的用户面临以下痛点：

| 痛点 | 说明 |
|------|------|
| 参数繁多 | MindIE / vLLM 涉及数十个服务化参数，用户不清楚应该调哪些 |
| 范围难定 | 参数范围设置过大搜索效率低，过小可能错过最优解 |
| 上下文缺失 | 不知道硬件约束（显存、并行度）和业务负载（输入/输出 token 分布）如何影响参数选择 |
| benchmark 配置不熟悉 | AISBench、vllm_benchmark 的参数配置需要参考文档 |

### 1.2 目标

本提案引入一个 Claude Code skill——`param-recommend`，用于通过渐进式对话收集用户场景信息，推荐保守、可解释的寻优参数和初始搜索范围。

**核心目标**：

- 通过渐进式提问收集必需上下文（硬件、模型、负载、优化目标）
- 基于启发式规则推荐参数范围，不依赖机器学习模型
- 输出可读的推荐结果和 `config.toml` TOML 片段
- 与 `optix-config` skill 衔接，支持自动配置

**非目标**：

- 不自动修改 `config.toml`（需用户确认）
- 不做参数组合的效果预测
- 不支持仿真模式的参数推荐

### 1.3 核心价值

- **渐进式引导**：首轮只问最关键的几个问题，避免信息过载
- **保守推荐**：首次使用推荐稳定可行的参数组合，降低风险
- **可解释性**：每个推荐参数都附带推荐理由
- **自动化衔接**：通过 `config_skill_handoff` 与配置 skill 无缝对接

---

## 2. 详细设计

### 2.1 架构概览

```text
用户输入（场景描述）
       │
       ▼
┌─────────────────────────────────┐
│   param-recommend Skill          │
│   .agents/skills/                │
│   param-recommend/SKILL.md        │
└───────────────┬─────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   recommend_params.py 脚本       │
│   ├─ 加载模型 config.json        │
│   ├─ 估算模型权重和 KV cache     │
│   ├─ 选择并行度配置              │
│   └─ 生成推荐参数表              │
└───────────────┬─────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   JSON 输出               │
    │   ├─ status: ok /         │
    │   │   need_more_info      │
    │   ├─ recommendations[]    │
    │   ├─ toml_snippet         │
    │   └─ config_skill_handoff │
    └──────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   AI Agent 格式化输出     │
    │   + 传给 config skill     │
    └──────────────────────────┘
```

### 2.2 核心并行约束

参数推荐必须满足核心并行约束：

```
DP * TP * PP == world_size
```

首次使用时，优先推荐在稳定可行的前提下用满卡：

```
DP * TP * PP == world_size
```

### 2.3 Context JSON 输入格式

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

### 2.4 推荐规则

#### 2.4.1 模型权重估算

基于 `config.json` 字段估算模型权重大小：

| 字段 | 说明 |
|------|------|
| `hidden_size` | 隐藏层维度 |
| `num_hidden_layers` | 层数 |
| `num_attention_heads` | 注意力头数 |
| `intermediate_size` | FFN 中间层维度（MoE） |
| `num_experts` | 专家数量（MoE） |

dtype 字节映射：

| dtype | 字节数 |
|-------|--------|
| int4 / nf4 / fp4 | 0.5 |
| int8 / uint8 | 1 |
| bfloat16 / float16 | 2 |
| float32 | 4 |

#### 2.4.2 KV Cache 容量估算

```
KV_cache_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
```

根据单卡显存和模型权重估算最大并发 token 数。

#### 2.4.3 并行度选择策略

1. 从大到小遍历可行的 TP 值（而非从小到大）
2. 优先选最大的可行 TP，释放更多 KV cache 空间
3. 满足约束 `DP * TP * PP == world_size` 时优先用满卡

### 2.5 输出结构

#### 状态：need_more_info

```json
{
  "status": "need_more_info",
  "next_question": "请提供模型 config.json 路径，或显式提供 hidden_size、num_hidden_layers 等字段"
}
```

#### 状态：ok

```json
{
  "status": "ok",
  "recommendations": [
    {
      "section": "vllm",
      "name": "MAX_NUM_SEQS",
      "min": 64,
      "max": 256,
      "value": 128,
      "dtype": "int",
      "search": true,
      "reason": "基于最大并发 token 数和 BLOCK_SIZE 估算"
    }
  ],
  "toml_snippet": "...",
  "config_skill_handoff": {
    "consumer_skill": "optix-config",
    "target_fields": [...],
    "vllm_command_others": [...],
    "apply_commands": [...],
    "notes": [...]
  },
  "next_command": "python scripts/recommend_params.py --context /path/to/context.json"
}
```

### 2.6 特殊参数处理

| 参数 | 处理策略 |
|------|----------|
| `ENABLE_PREFIX_CACHING` | 使用 vLLM 默认值，不作为搜索维度 |
| `ENABLE_CHUNKED_PREFILL` | 使用 vLLM 默认值，不作为搜索维度 |
| `COMPILATION_CONFIG` | 使用 `$COMPILATION_CONFIG` 占位符接入 others |
| `ais_bench` CONCURRENCY/REQUESTRATE | 仅保留在 JSON handoff 中，不生成 TOML |

### claude 嵌套 config 处理

部分模型（如 Qwen3.5、Qwen3-VL、Kimi）将配置嵌套在 `text_config` 下：

```python
config = {
    "hidden_size": 4096,
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 28
    }
}
```

脚本会自动检查并合并 `text_config` 字段。

---

## 3. 使用说明

### 3.1 调用方式

在 Claude Code 对话中直接使用：

```text
/param-recommend
```

### 3.2 典型使用流程

**用户**：我要用 vLLM 在 8 卡 A100 上跑 Qwen3-8B 推理，输入平均 1K，最大 4K，输出平均 256，最大 512。优化吞吐优先。

**第一轮（助手）**：收集信息不足，询问：

- 确认模型 config.json 路径
- 确认单卡显存大小（默认 80GB）

**用户**：config.json 在 /workspace/models/qwen3-8b/config.json，单卡 80GB。

**第二轮（助手）**：信息已足够，运行参数推荐脚本进行分析，准备推荐结果。

**第三轮（助手）**：输出推荐结果，包含：
- 输入摘要和假设
- 推荐参数表
- TOML 片段
- 下一步命令

### 3.3 使用约束

1. **核心并行约束必须满足**：`DP * TP * PP == world_size`
2. **不要静默应用 TOML 片段**：修改 `config.toml` 前必须征得用户同意
3. **量化模型支持**：4-bit 量化类型正确识别，不高估权重大小

---

## 4. 参考资料

- [《Skill 文档》](../../.agents/skills/optix-param-recommend/SKILL.md)
- [《输入格式说明》](../../.agents/skills/optix-param-recommend/references/input-schema.md)
- [《参数推荐规则》](../../.agents/skills/optix-param-recommend/references/parameter-rules.md)
- [《服务化自动寻优工具文档》](../../docs/zh/serviceparam_optimizer_instruct.md)
