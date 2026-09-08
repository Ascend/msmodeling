# Benchmark 选择与配置指南

> 本文件是 benchmark（`vllm_benchmark` / `ais_bench`）的唯一流程指导，两种寻优模式共用。
> 此前默认值与配置指引散落在 PSO 分支（默认 `ais_bench`）与 agent 分支（默认 `vllm_benchmark`）
> 且互相矛盾，本文件统一。

## 1. 选择准则（两种模式强制，先询问再配置）

**核心规则（参数推荐与 agent 模式同样适用）**：
1. **必须主动询问用户使用哪个压测工具**（`vllm_benchmark` / `ais_bench`），禁止猜值、
   禁止静默套默认——询问是流程的一步，不是可选项。
2. 用户无偏好时，**默认 `ais_bench`**（昇腾生态实际使用最频繁的工具）。
3. 以下场景做引导建议（作为询问话术，不替代询问）：

| 场景 | 建议 benchmark | 原因 |
|---|---|---|
| 用户明确要 vllm bench serve 原生压测 | `vllm_benchmark` | 原生 `vllm bench serve` 命令，字段直接映射 |
| 引擎 = MindIE | `ais_bench` | `vllm_benchmark` 绑定 vLLM，MindIE 无 vllm bench |
| 需要自定义压测负载（并发模型/真实请求序列） | `ais_bench` | ais_bench 以 Python 配置（`models`/`datasets`）定义负载形态 |
| 其余场景 / 无偏好 | `ais_bench`（默认） | 实际使用最频繁 |

4. benchmark 影响 benchmark 侧 target field 与性能指标解释，**不覆盖**基于硬件/模型推导的服务侧参数范围。

## 2. vllm_benchmark（用户选定时）

### 必填项（preflight 检查，缺一不可）

| 字段 | 说明 | 写入方式 |
|---|---|---|
| `model` / `served_model_name` | 与被测服务一致 | `config_preflight --set-model` / `--set-served-name` |
| `host` / `port` | 服务地址 | `config_writer.py --config <config.toml> --set vllm_benchmark.command.host=127.0.0.1 --set vllm_benchmark.command.port=8000` |
| **输入输出长度** | vllm bench 默认 random dataset，**必须显式给 `--random-input-len`（建议同时 `--random-output-len`）**，或配 `dataset_name`/`dataset_path` | `config_writer.py --config <config.toml> --set-others vllm_benchmark="--random-input-len <N> --random-output-len <N>"`，或 `config_preflight --set-bench <path>` |

### 推荐配置命令

```bash
python <SKILL_ROOT>/scripts/config_writer.py --config <config.toml> \
  --set vllm_benchmark.command.model=<model> \
  --set vllm_benchmark.command.served_model_name=<name> \
  --set vllm_benchmark.command.host=127.0.0.1 --set vllm_benchmark.command.port=8000 \
  --set vllm_benchmark.command.dataset_name=random --set vllm_benchmark.command.num_prompts=500 \
  --set-others vllm_benchmark="--random-input-len 128 --random-output-len 256"
```

`--set-others` 整体覆盖 `[vllm_benchmark.command].others`，支持 `$VAR` 占位符（与 `[vllm.command].others` 规则一致）。

## 3. ais_bench（MindIE 引擎 / 自定义负载）

> **配置形态与放置规则（本文唯一流程指导，实测验证）**：
> 1. AISBench 的 `models`/`datasets` 配置是 **Python 文件**（mmengine 风格，定义 `models = [...]` / `datasets = [...]` 列表），**不是 YAML**。
> 2. CLI 的 `--models/--datasets` **只接受包内可被唯一解析的短名**（`match_cfg_file` 递归搜索，匹配 0 个或 >1 个都报 `UTILS-MATCH-001`），**不接受绝对路径**。**具体匹配规则随 ais_bench 版本而变**——当前实测按配置文件 **basename** 全树 fnmatch（带 `子目录/` 前缀或绝对路径均 0 命中）；authority 是**运行环境 `ais_bench --search` 实测**，不是本文。搜索目录为 `<args.config_dir>/models` 与 **`<ais_bench 包根>/benchmark/configs/models`**（注意是 `benchmark/configs/`，不是仓根的 `configs/`）。editable 安装下包根即 `python -c "import ais_bench; import os; print(os.path.dirname(ais_bench.__file__))"`。写入 config 后由 `config_preflight --check-only` 的 ais_bench `--search` 门禁统一校验（错误在门禁期拦截，不拖到 optix 启动）。
> 3. CLI **强制要求显式 `--datasets`**（缺省报 `TMAN-CMD-001: You must specify --datasets`）；optix 的命令只带 `--models/--mode/--work-dir`，因此 `--datasets` 必须经 `[ais_bench.command].others = "--datasets <短名>"` 传递。
> 4. 启动前必做预检：`ais_bench --models <短名> --datasets <短名> --search`（毫秒级，打印解析路径表格）。

### 必填项（preflight 检查）

| 字段 | 说明 | 写入方式 |
|---|---|---|
| `models` | ais_bench 模型清单 **Python 配置**（`models = [dict(...)]`） | 文件放入 `<pkg>/benchmark/configs/models/<子目录>/`；`config_writer.py --config <config.toml> --set ais_bench.command.models=<短名>`（preflight 检查 `[ais_bench.command].models`） |
| `datasets` | 数据集 **Python 配置**（`datasets = [dict(...)]`） | 文件放入 `<pkg>/benchmark/configs/datasets/<子目录>/`；经 `--append-others ais_bench="--datasets <短名>"` 传递（config_writer 已支持 token 级并入 others） |
| `mode` | `perf` / `concurrency` / `throughput` | `--set ais_bench.command.mode=<mode>` |
| 压测规模 | `--num-prompts <N>` | 经 `--append-others ais_bench="--num-prompts <N>"` 传递（同上） |

### 推荐配置命令

```bash
# 1) 编写 models / datasets 配置（基于 AISBench 内置模板，Python）：
#    models:  <pkg>/benchmark/configs/models/optix_custom/<name>.py
#      参考 api_examples/demo_infer_vllm_api_perf.py + models/vllm_api/vllm_api_stream_chat.py，
#      VLLMCustomAPIChatStream 配置 host_ip/host_port/max_out_len/batch_size/generation_kwargs
#    datasets: <pkg>/benchmark/configs/datasets/optix_custom/<name>.py
#      参考 datasets/synthetic/synthetic_gen_string.py（RequestCount / StringConfig 控制规模与输入输出长度）
# 2) 预检（必须能解析出 --models 与 --datasets 两行）
ais_bench --models <短名> --datasets <短名> --search
# 3) 写入 config.toml（datasets/num-prompts 自动并入 others）
python <SKILL_ROOT>/scripts/config_writer.py --config <config.toml> \
  --set ais_bench.command.models=<短名> --set ais_bench.command.mode=perf \
  --append-others ais_bench="--datasets <短名> --num-prompts 3000"
```

> **兼容性注意**：ais_bench 相关字段保留在 `config_skill_handoff.target_fields` 与 `notes` 中，
> **不写入 `toml_snippet` 的 target-field 块**——当前 Settings loader 对 ais_bench target field
> 的处理存在兼容风险（recommend_params 输出与 `config_writer.py` 均遵循此约定：ais_bench 段不产生
> target-field 写命令，CONCURRENCY/REQUESTRATE 等 scheduler env 保留在 handoff `target_fields`）。
>
> **optix schema 注意**：`AisBenchCommandConfig` 只有 `models/mode/work_dir/others`
> 四字段。`datasets`/`num_prompts` 必须以 `--datasets X --num-prompts N` 形式放进 `others`，
> 写成 `[ais_bench.command].datasets` 独立键会被 optix 忽略。

## 4. 两种模式下的落地

| 环节 | vllm_benchmark | ais_bench |
|---|---|---|
| preflight 检查 | `--benchmark vllm_benchmark`；检查 `[vllm_benchmark.command].*` 与输入输出 | `--benchmark ais_bench`；检查 `[ais_bench.command].models` |
| 回填 | `config_preflight --set-bench <path>` → `dataset_path`；其余走 config_writer | `config_preflight --set-bench <path>` → `models`；其余走 config_writer |
| 启动 | `msmodeling optix -e <engine> -b vllm_benchmark -c <config.toml>` | `msmodeling optix -e <engine> -b ais_bench -c <config.toml>` |

## 5. 常见坑

- vllm_benchmark 不配输入输出长度 → preflight 报"未配置输入输出"，**不要用 benchmark 的 tqdm/默认值绕过**，必须显式给出；
- ais_bench 的 models 配置缺 `[ais_bench.command].models` 指向 → preflight 报缺失，先 `config_writer.py --set ais_bench.command.models=<短名>` 再复检；
- ais_bench 的 `--models/--datasets` 传**绝对路径** → 必报 `UTILS-MATCH-001 match config file failed`（只按包内短名匹配），配置需放入 `<pkg>/benchmark/configs/models|datasets/` 下；
- ais_bench 的 `--models/--datasets` 写成 `子目录/文件名`（含目录前缀）→ **同样 `UTILS-MATCH-001`**（匹配规则随版本而变，当前实测按 basename，不含目录）；短名 = config 文件名去 `.py`、**不带目录**，新建 model/dataset 配置须保证 basename 在整棵 `models|datasets` 树下唯一；一律以 `ais_bench --search` 唯一解析为准；
- ais_bench 只配 models 不配 datasets → 启动时 `TMAN-CMD-001 You must specify --datasets`；经 `[ais_bench.command].others = "--datasets <短名>"` 传递；
- ais_bench 的 `--search` 是零副作用预检命令，启动前必跑一次（拦下上述两类配置故障）；
- 两个 benchmark 的 host/port 都指**被测服务**，不是本机调试端口。

## 6. 引导话术与建议值表（agent 引导用户时统一口径，禁止临场发挥）

### 6.1 询问话术（两种模式通用）

```
请确认压测工具（默认 ais_bench，昇腾生态最常用）：
1. ais_bench — 默认；MindIE 引擎 / 自定义压测负载（models/datasets Python 配置）
2. vllm_benchmark — 仅当你需要 vllm bench serve 原生压测时选择
请输入 1 或 2（直接回车 = ais_bench）：
```

用户完全没概念时，按 **6.3 典型场景模板**给出一套完整建议值，逐项确认后再写入。

### 6.2 建议值表（均为"建议值，必须用户确认"，禁止静默填入）

| 项 | ais_bench | vllm_benchmark |
|---|---|---|
| mode | `perf`（备选 concurrency / throughput） | —（无 mode 字段） |
| 压测规模 | `--num-prompts` 建议 3000 | `--num-prompts` 建议 500 |
| 输入长度 | datasets 配置 `StringConfig.Input` 定义 | 对话场景 `--random-input-len 1024`；长文本 4096；需业务确认 |
| 输出长度 | datasets 配置 `StringConfig.Output` 定义 | 对话场景 `--random-output-len 256`；长文本 512；需业务确认 |
| host/port | —（无 host/port，由 models 配置 `host_ip/host_port` 定义负载） | 默认 `127.0.0.1:8000`，**必须与 `[vllm.command]` 一致** |
| models/datasets | Python 配置放 `<pkg>/benchmark/configs/{models,datasets}/`；`--models <短名> --datasets <短名>`（经 others 传） | `--dataset-name random` 或 `--dataset-path <path>` |

### 6.3 典型场景模板（用户无概念时按此引导，标注"按典型场景建议，请确认"）

| 场景 | benchmark | 输入/输出 | 压测规模 | SLO 参考（秒） |
|---|---|---|---|---|
| 对话服务（Qwen 72B / 8 卡） | ais_bench | —（datasets 配置定义） | 3000 | TTFT ≤ 3.0 / TPOT ≤ 0.1 |
| 长文本生成 | vllm_benchmark | 4096 / 512 | 500 | TTFT ≤ 5.0 / TPOT ≤ 0.15 |
| 批量推理（离线） | ais_bench | —（datasets 配置定义） | 5000 | 可放宽，无实时约束 |

> SLO 建议仅作引导起点：TTFT/TPOT 必须用户按业务确认（单位秒，`--ttft-slo`/`--tpot-slo`）。

### 6.4 引导完成流程（回到 SKILL.md 共享前置链）

**0. 前置——先问用户、禁止自找**：以下项**必须先询问用户**，agent 不得自行文件系统搜索/推断/自动填充：
- 模型权重路径 / 服务模型名 → 先问；用户不知道且同意后才 `config_preflight --discover` 查找
- ais_bench `models`/`datasets` **短名** → 先问；用户不知道且同意后才 `ais_bench --search` 探测
- 使用卡数 → 先问；用户不知道且同意后才探测可见卡
- **输入/输出长度** → 必问（禁止静默用占位默认），见步骤 2

1. 0a 询问 benchmark（上文话术）→ 无偏好默认 ais_bench
2. 收集必填项：每项给出 6.2 建议值 → 用户确认 → 记下。**输入/输出长度必问**：
   - vllm_benchmark → 确认后 `config_writer.py --config <config.toml> --set-others vllm_benchmark="--random-input-len <N> --random-output-len <N>"` 写入 `[vllm_benchmark.command].others`
   - ais_bench → 用户确认 datasets 配置的 Input/Output 长度
   - 长度同时传给 `collect_context --input-len <N> --output-len <N>` 供 Roofline/KV 数值推导
3. `config_writer.py` 写入（`--set` + `--set-others`/`--append-others`，命令见 §2/§3）
4. `config_preflight.py --check-only` 复检 → 退出 0 → 生效配置确认表 → 用户确认 → 进入寻优
5. models/datasets Python 配置用户没有时：指向 AISBench 官方文档（models 配置说明）+
   内置模板 `configs/api_examples/demo_infer_vllm_api_perf.py`（models）与
   `configs/models/vllm_api/vllm_api_stream_chat.py`、`datasets/synthetic/synthetic_gen_string.py`，
   说明"需要一份模型任务配置（Python），可先按内置模板创建并放入 `<pkg>/benchmark/configs/{models,datasets}/`"，
   再经 `config_writer.py --set ais_bench.command.models=<短名> --append-others ais_bench="--datasets <短名>"` 写入 + `--search` 预检。
