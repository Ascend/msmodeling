# PSO 分支流程

> 状态：骨架。本分支由 `optix-param-recommend`（推荐）+ `optix-config`（写入）合并而来，
> 并吸收 agent 模式沉淀的经验资产（presets / experience_injector / known_patterns）。

## 目标

为 optix 内置 PSO 寻优（`PSOOptimizer`，基于 pyswarms）提供知识增强的初始配置：
推荐哪些参数、范围怎么定、benchmark 怎么配，写入 config.toml 后启动。

与旧 param-recommend 的核心差别：**推荐依据不再是孤立启发式，而是共享知识层**——
experience_report（Roofline/KV 预算/TP 安全范围）+ vendor preset 官方值 + known_patterns，
自身启发式仅兜底。

## 流程

### 1. 前置（已由 SKILL.md「共享前置流程」完成）

- 共享骨架：0a benchmark 确认 → config 就绪复检 → collect_context → experience_report.md
  生成并阅读，见 `../SKILL.md`「共享前置流程」，本文件不重复。PSO / config 模式复检配置用
  `config_writer.py --check-only`（`config_preflight.py` 是 agent 模式门禁，勿跑）
- 用户只想"要推荐不要执行"时走**最小前置**（不启动 optix）：0a 确认 benchmark + 收集
  硬件/模型/负载信息 → collect_context 生成 context → experience_injector + recommend_params，
  到第 2 步输出即止；跳过 preflight 完整回填与 config_writer 写入。

### 2. 生成推荐

```bash
python <SKILL_ROOT>/scripts/recommend_params.py --context <context.json>
```

context.json 兼容旧 `optix-param-recommend` 的输入格式（engine / hardware / model /
workload / target / discovery），也可直接消费 collect_context 的 context.json（实现期打通）。

输出必须包含：

- 输入摘要与假设（未确认的假设明确标注）
- 推荐参数表：参数名、范围、默认值、是否参与搜索、**推荐理由（引用 experience_report 或 preset 出处）**
- `toml_snippet`：config.toml 片段（人工审阅用）
- `config_skill_handoff` → 更名语义为 `assistant_handoff`：`consumer_skill` 填 `optix-assistant`（旧值为 `optix-config`，本 skill 为唯一消费者）
- `apply_commands`：给 `config_writer.py` 的命令清单（`--target-field` 写搜索参数/范围/固定参数）
- `DP * TP * PP == world_size` 约束解释

> **推荐校准规则**：
> - vendor preset 命中的参数：默认值取官方推荐值，范围围绕官方值展开（±1档）
> - experience_report 给出安全边界的参数：范围不得越过硬约束（KV 预算、显存上限）
> - `ENABLE_PREFIX_CACHING` / `ENABLE_CHUNKED_PREFILL` 等交给引擎默认值，不作为搜索维度
> - benchmark 必须询问用户确认；无偏好时默认 `ais_bench`（选择准则与配置见 `benchmark.md`）

### 3. 应用写入（用户确认后）

```bash
# 应用推荐值与搜索范围（来自 apply_commands，逐条执行或串联；config_writer 薄层写入）
python <SKILL_ROOT>/scripts/config_writer.py --config <绝对路径> \
  --target-field engine=<engine> name=<NAME> config_position=<Z> dtype=<D> value=<V> min=<MIN> max=<MAX>
# vllm 服务命令（host/port/model/served-name/others）缺失时先配：
python <SKILL_ROOT>/scripts/config_writer.py --config <绝对路径> \
  --set vllm.command.model=<model> --set vllm.command.served_model_name=<name> \
  --set vllm.command.host=127.0.0.1 --set vllm.command.port=8000 \
  --set-others vllm="--trust-remote-code"
# benchmark 配置（vllm_benchmark / ais_bench）
python <SKILL_ROOT>/scripts/config_writer.py --config <绝对路径> \
  --set-others vllm_benchmark="--random-input-len 2048 --random-output-len 512"
python <SKILL_ROOT>/scripts/config_writer.py --config <绝对路径> \
  --set ais_bench.command.models=<短名> --set ais_bench.command.mode=perf \
  --append-others ais_bench="--datasets <短名> --num-prompts 3000"
```

**PSO 计算预算（`n_particles`/`iters`）**：不是推荐产出，optix 启动直接读 config.toml 现值
（仓库默认 `n_particles=8, iters=4`）。需要调整时**先问用户**愿花多少墙钟/要多彻底，按
"总时长 ≈ n_particles × iters × 2 × 单次测试时长"估算后，
`config_writer.py --config <绝对路径> --set n_particles=<N> --set iters=<N>` 写入——禁止静默套档位。

- 所有命令支持 `--dry-run` 预览；修改现有 config.toml 前必须征得用户同意

### 4. 启动与收尾

```bash
# 命中 preset 场景先应用 env
source <run-dir>/launch_env.sh
msmodeling optix -e <engine> -b <benchmark> -c <config.toml>
# 轮次结束后
source <run-dir>/unset_env.sh
```

- 未配置 `optimizer_strategy="agent"` 时 optix 自动走内部 PSO，消费 config.toml 搜索空间
- 寻优完成后的结果解读、瓶颈分析引导用户使用 `throughput-optimizer-explainer`

## 与 agent 分支的衔接

本分支的推荐结果（assistant_handoff）可被 agent 分支的 `collect_context.py
--recommend-result` 消费，作为 agent 模式 Round 1 的初始搜索空间——用户可先 PSO 快速摸底，
再切 agent 模式深挖。
