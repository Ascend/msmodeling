---
name: optix-config
description: Automates the configuration of msmodeling optix config.toml for parameter optimization. Use when modifying optimizer parameters, setting up VLLM/MindIE target fields, configuring benchmark tools (AISBench/vllm_benchmark), or preparing config.toml for service parameter optimization.
---

# msmodeling optix 寻优工具配置管理

## 前置条件

使用本 skill 前，请确保：

1. **已完成 optix 安装**：在 msmodeling 仓库根目录执行 `uv sync`（自动创建 `.venv` 并安装 msmodeling，无需 `pip install -e .`）。不要在该 venv 里装 vllm。
2. **部署栈默认在系统环境**：机器上应已部署 vLLM、MindIE 和测评工具。只有 PATH 特殊时才配 `OPTIX_DEPLOY_PATH` 或 `config.toml` 的 `[deploy] path_prefix`，`OPTIX_DEPLOY_PATH` 优先。
3. **config.toml 文件已存在**（位于 `optix/config.toml`，也可通过 `-c` 参数指定其他路径）
4. **了解寻优参数类型**（见下文参数类型说明）

## 快速开始

### 完整配置流程示例

**MindIE / VLLM + AISBench 配置**:
```bash
# 1. 应用场景模板
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario standard --engine mindie

# 2. 配置 AISBench
python ./.agents/skills/optix-config/scripts/auto_config.py --set-ais-bench \
    --models /path/to/models.yaml \
    --datasets /path/to/datasets.yaml \
    --mode perf \
    --ais-num-prompts 3000
```

**VLLM + vllm_benchmark 配置**:
```bash
# 1. 应用场景模板
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario standard --engine vllm

# 2. 配置 VLLM 服务
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-command \
    --model /data/models/deepseek-v3 \
    --served-name deepseek-v3

# 3. 配置 vllm_benchmark
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-benchmark \
    --model /data/models/deepseek-v3 \
    --served-name deepseek-v3 \
    --dataset-name random \
    --vllm-num-prompts 500
```

## 场景模板配置

### 使用方式

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario <场景> --engine <引擎> [选项]
```

### Supported Scenarios

| Scenario | Description | n_particles | iters | Features |
|---------|-------------|-------------|-------|----------|
| `quick-test` | Quick test | 5 | 3 | Small range, fast verification |
| `standard` | Standard optimization | 10 | 5 | Balance depth and breadth, 10×5 iterations |
| `deep-optimize` | Deep optimization | 30 | 20 | Large range, fine-grained search; enable `pso_top_k=3-5` |
| `ttft-priority` | TTFT priority | 15 | 10 | High ttft_penalty, low tpot_penalty |
| `tpot-priority` | TPOT priority | 15 | 10 | High tpot_penalty, low ttft_penalty |
| `throughput` | Throughput priority | 20 | 10 | Latency penalty set to 0 |

### Time Budget Auto Calculation

```bash
# Calculate optimal parameters based on available time
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario deep-optimize --time-budget 8h
```

**Time Estimation Notes**:
- Each seed runs the service twice (warmup + formal test)
- Total time ≈ n_particles × iters × 2 × single test time

### Deep Optimization (pso_top_k)

For the `deep-optimize` scenario, it is recommended to enable the fine-tuning seed parameter `pso_top_k`. This parameter controls how many top-performing candidates are selected for fine-grained re-exploration during the search, helping to converge toward better optima.

**Recommended values: 3–5**

| Value | Behavior |
|-------|----------|
| `3` | Tighter selection, faster convergence, slightly higher risk of missing global optimum |
| `4` | Balanced (recommended default) |
| `5` | Broader exploration, better chance of finding global optimum, slower convergence |

**Configuration (in config.toml)**:
```toml
[data_storage]
pso_top_k = 4
```

> **Note**: This parameter is independent of the scenario presets and must be manually added to `config.toml` after applying `--scenario deep-optimize`.

## Parameter Configuration

### Add Search Parameter (with optimization range)

> **Important**: The `--value` parameter is required, used to specify the default value for the parameter.
>
> **Enum Parameter Default Rule**: If the user does not specify `--value`, the script automatically selects the first **non-empty value** as the default (instead of an empty string).

```bash
# Integer parameter
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name MAX_BATCH_SIZE \
    --min 10 --max 400 --dtype int --value 100

# Float parameter
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name GPU_MEMORY_UTILIZATION \
    --min 0.8 --max 0.95 --dtype float --value 0.9

# Enum parameter
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name TENSOR_PARALLEL_SIZE \
    --dtype enum --enum-values "[1,2,4,8,16]" --value 4

# String enum parameter (containing JSON values, script auto adds single-quote escaping)
# User input format: --enum-values '["", "--config {\"key\": \"value\"}"]'
# Script auto generates: dtype_param = ["", "--config '{\"key\": \"value\"}'"]
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name COMPILATION_CONFIG \
    --dtype enum \
    --enum-values '["", "--compilation-config {\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}"]' \
    --cli-arg=""

# Ratio parameter (relative to another parameter)
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name MAX_PREFILL_RATIO \
    --min 0.1 --max 0.7 --dtype ratio --dtype-param max_batch_size --value 0.3

# Factories parameter (dp = 16 / tp)
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param \
    --engine vllm --param-name DP \
    --dtype factories \
    --factories-config '{"target_name":"TENSOR_PARALLEL_SIZE","product":16,"dtype":"int"}' --value 4
```

### Add Fixed Parameter (not participating in optimization)

```bash
# Fixed integer
python ./.agents/skills/optix-config/scripts/auto_config.py --add-fixed-param \
    --engine vllm --param-name MAX_MODEL_LEN \
    --value 16384 --dtype int

# Fixed string
python ./.agents/skills/optix-config/scripts/auto_config.py --add-fixed-param \
    --engine vllm --param-name COMPILATION_CONFIG \
    --value "" --dtype str

# Fixed boolean
python ./.agents/skills/optix-config/scripts/auto_config.py --add-fixed-param \
    --engine vllm --param-name ENABLE_PREFIX_CACHING \
    --value true --dtype bool
```

### Parameter Type Description

| dtype | Meaning | Required CLI Args | Example CLI |
|-------|---------|-------------------|-------------|
| `int` | Integer parameter | --min, --max | `min=10, max=400` |
| `float` | Float parameter | --min, --max | `min=0.8, max=0.95` |
| `bool` | Boolean parameter | --value | `value=true/false` |
| `str` | String parameter | --value | `value="string"` |
| `enum` | Enum parameter | --enum-values | `enum-values="[1,2,4,8]"` |
| `ratio` | Ratio parameter | --dtype-param | `dtype-param=target_param` |
| `factories` | Factories parameter | --factories-config | `factories-config='{"target_name":"TP","product":16}'` |
| `times` | Times parameter | --dtype-param | `dtype-param='{"target_name":"TP","product":16}'` |

> **注意**: CLI 参数名与 config.toml 中的键名不完全一致。`--enum-values` 和 `--factories-config` 在生成的 config.toml 中对应的键名均为 **`dtype_param`**。例如:
> - `--dtype enum --enum-values "[1,2,4]"` → `dtype = "enum"` + `dtype_param = [1, 2, 4]`
> - `--dtype factories --factories-config '{...}'` → `dtype = "factories"` + `dtype_param = {...}`

## 服务配置

### VLLM 命令配置

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-command \
    --model /path/to/model \
    --served-name my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --others "--trust-remote-code --enable-expert-parallel"
```

配置字段：
- `model`: 模型路径
- `served_model_name`: 服务模型名
- `host`: 服务主机
- `port`: 服务端口号
- `others`: 其他启动参数（支持 `$VAR` 变量引用）

## 测评工具配置

### AISBench 配置

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --set-ais-bench \
    --models "/path/to/models.yaml" \
    --datasets "/path/to/datasets.yaml" \
    --mode perf \
    --ais-num-prompts 3000
```

### vllm_benchmark 配置

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-benchmark \
    --model /path/to/model \
    --served-name my-model \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset-name random \
    --vllm-num-prompts 500 \
    --others "--input-len 128 --output-len 256"
```

## 高级用法

### 预览模式（不实际修改）

所有命令都支持 `--dry-run` 预览：

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario standard --engine vllm --dry-run
```

### 配置文件路径指定

```bash
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario standard \
    --config-path /custom/path/config.toml
```

### 组合使用示例

```bash
# 完整配置一条命令（使用 && 串联）
python ./.agents/skills/optix-config/scripts/auto_config.py --scenario standard --engine vllm && \
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-command --model /data/model --served-name model && \
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param --engine vllm --param-name TP --dtype enum --enum-values "[1,2,4,8]" --value 4 && \
python ./.agents/skills/optix-config/scripts/auto_config.py --add-search-param --engine vllm --param-name DP --dtype factories --factories-config '{"target_name":"TP","product":8}' --value 2 && \
python ./.agents/skills/optix-config/scripts/auto_config.py --set-vllm-benchmark --model /data/model --dataset-name random
```

## 配置验证

修改完成后，验证配置是否正确：

```bash
# 检查 TOML 语法
python -c "import tomllib; tomllib.load(open('optix/config.toml', 'rb'))"

# 查看帮助确认工具可用
msmodeling optix --help
```

## 常见问题

**Q: 如何删除已添加的参数？**
- 手动编辑 config.toml，删除对应的 `[[engine.target_field]]` 块

**Q: 如何修改已有参数的范围？**
- 重新执行 `--add-search-param` 命令，会自动更新同名参数

**Q: 参数未生效？**
- 检查参数名是否正确（区分大小写）
- 确认 `config_position` 设置正确（通常为 `"env"`）
- 验证 TOML 语法无错误

**Q: 如何查看当前所有配置？**
- 直接查看 config.toml 文件
- 或使用 `cat config.toml | grep -A 10 "target_field"`
