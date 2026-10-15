# Environment Preparation（环境准备）

本文件写 agent 模式在「配置就绪」和「Round 0（初始化与自检）」两个阶段的具体操作步骤；
整体阶段编排见 `./agent.md`。
只覆盖单机本地场景。optix（msmodeling）是唯一支持的执行后端——它不可用时没有替代方案，
直接停止。

**先读哪里**：整体步骤链（先做什么后做什么）只看 `../SKILL.md` 的「共享前置流程」——
0a benchmark 询问、config.toml 发现与门禁（`config_preflight.py --discover` / `--check-only`）、
必须先问用户的清单（卡数 / 模型路径与服务模型名 / ais_bench 短名 / 输入输出长度，
禁止猜值）、写入分工、配置确认表。这些内容只在 SKILL.md 维护，本文件不抄写、不另起版本，
有出入以 SKILL.md 为准。本文件只写 agent 模式每一步的执行细节：命令字段明细见
`config_preflight.py --help` 与 `../references/benchmark.md`，optix CLI 与
config.toml 字段参考见 `../references/optix-reference.md`，benchmark 工具检查与安装见
下文「benchmark 工具可执行性与隔离安装」。

> 下文中「阻塞项」= 不通过就不许往下走，必须先解决再继续。

## 流程

1. **optix 可用性检查与安装（阻塞项）**：
   - 先检查 uv 虚拟环境中 `msmodeling optix` 是否可用（如 `.venv/bin/msmodeling`）。
   - 缺失时安装：克隆 msmodeling 仓库后 `uv sync`（自动建 `.venv`、可编辑安装含
     `msmodeling optix` CLI）；或用发布 whl 装进 uv/venv 虚拟环境
     （`pip install msmodeling-<version>-py3-none-any.whl`，whl 链接见
     <https://gitcode.com/Ascend/msmodeling/releases>）。
   - **严禁装进系统 Python**：msmodeling 会带上 `torch`、`transformers` 等包，
     会冲掉系统部署栈里 vLLM/MindIE 已验证的同名依赖。vLLM、MindIE、测评工具
     始终用系统环境，msmodeling/optix 只用虚拟环境。
   - 装完验证：`msmodeling --help` 无报错即可。
   - 无法安装（无网络且无离线包等）时报告原因并停止，不得绕过。

### benchmark 工具可执行性与隔离安装（阻塞项）

1. **可执行性检查**：如果当前这台机器就是跑 trial 的机器，先在一个不带 msmodeling
   venv 的 shell 里确认上一步选定的 benchmark 工具能跑——
   `command -v ais_bench` / `vllm bench serve --help`。注意
   `config_preflight.py` 的 ais_bench `--search` 和包定位检查只是"只告警不拦截"
   （因为写 config.toml 的机器可能不是跑 benchmark 的机器），不能替代本检查。
2. **缺失时当场安装**（无人值守/headless 同样适用，禁止因此停下提问；仅安装不可行——
   无网络且无离线包等——才停止报告）。安装规则：测评工具装**独立 venv**，经
   `OPTIX_DEPLOY_PATH=<venv 根>`（启动 optix 的 shell 里 export，或写入 `launch_env.sh`）
   或 config.toml `[deploy] path_prefix` 暴露给 optix——optix 会把 `<prefix>/bin` **前插**
   到子进程 PATH 并按工具逐个解析（`deploy_env.py`），独立 venv 不含 vllm/torch，不会遮蔽
   系统部署栈。AISBench 标准安装（PyPI 包 `ais_bench_benchmark`，华为云镜像已同步）：

   ```bash
   uv venv --python 3.10 /opt/aisbench-venv    # AISBench 钉 numpy<2，Python 3.10 是已验证组合
   uv pip install --python /opt/aisbench-venv/bin/python setuptools ais_bench_benchmark==3.1.20260630
   #  setuptools 必装：uv venv 不带它，ais_bench 运行时 import pkg_resources 会直接失败
   #  版本钉死保证可复现；升级版本前先实测 ais_bench --help 与 --search 再放行
   /opt/aisbench-venv/bin/ais_bench --help     # 验证可执行
   ```

   装完在本轮寻优全程保持 `OPTIX_DEPLOY_PATH=/opt/aisbench-venv` 生效。
3. **两个禁止**：① 禁止装进 msmodeling 的 `.venv`（optix 拒绝解析落在 msmodeling venv
   内的命令，且会冲掉 optix 自身依赖）；② 禁止直接装进系统 Python（ais_bench 的依赖
   可能与系统部署栈里 vLLM/MindIE 已验证的同名包冲突）。

2. **发现并确认 `config.toml`**：`config_preflight.py --discover` 列出候选，
   找不到或多个时必须向用户确认，禁止猜测。
3. **收集环境与 agent 专属配置**：
   读取已确认的 config.toml，收集模型路径、served model name、硬件、benchmark 负载、
   目标与预算。缺项按入口约定询问；本步骤先收集，完成案例读取与方法核对后，在步骤 5
   集中写入并预检。用户已明确给出的值沿用，不重复确认。

   **agent 模式特有的额外要求**（全自动/交互式都必须做）：

   - 寻优目标：`tpot_slo` / `ttft_slo`（单位秒）必填且 > 0 → 必须问用户目标值
     （即使有默认值也要确认，禁止静默填）→ `config_preflight --ttft-slo <x> --tpot-slo <y>`；
     用户无概念时按 `../references/benchmark.md` §6.3 典型场景模板给建议值，
     标注"按典型场景建议，请确认"。
   - 轮数 / trial / 时间预算 → 与用户确认，在步骤 5 写入 optix 实际读取的 config.toml：
     `config_preflight --set-max-rounds <N> --set-max-trials <N> --set-time-limit-minutes <N>`
     （写入 `[agent_optimizer]`，是 optix 运行终止的真实依据；
     `time_limit_minutes` ≤ 1440（24h））。
   - agent 模式 → 确认是否开启 → `config_preflight --set-agent-mode`
     （自动对齐 run_id + 清空配置中的 run_dir 字段 + 写入 `use_request_rate_calibration=false`）。
   - **把 benchmark 负载参数写进配置（全自动/headless 模式强制）**：user_task 已给全
     benchmark 参数（输入/输出长度、请求数 + 并发）时无需询问，但必须完成三件事——
     ① 步骤 6 的 `collect_context` 带 `--input-len/--output-len/--num-requests` +
     并发旗标（固定 `--concurrency`，可调 `--concurrency-min/--concurrency-max`），
     把它们登记进 `context.json` 的 `benchmark_spec`；② 按 `references/benchmark.md` §3
     核对/重写 ais_bench models/datasets 配置，以及 ais_bench 包里的
     `synthetic_config.py`（负载参数以这个文件为准）（**禁止静默复用环境现存配置**）；
     ③ 并发按形态写入 CONCURRENCY target_field（`config_writer.py --target-field
     engine=<engine> name=CONCURRENCY config_position=env dtype=int ...`）——固定并发
     钉死 `value=<并发> min=<并发> max=<并发>`；并发范围写 `value=<下限> min=<下限>
     max=<上限>` 使其成为搜索维度（并发数以 CONCURRENCY target_field 为准，optix 每
     trial 据此改写 models 配置 batch_size），随后 `config_preflight --check-only
     --context <context.json>` 的 benchmark 一致性门禁必须通过。②③ 随步骤 5 完成。
   - **只认一个 config.toml**：`--check-only` / `--set-*` 回填、optix 启动，必须全部
     使用同一个**显式绝对路径**（`--config <绝对路径>`），禁止某一步自己再去发现
     config.toml——避免 preflight 校验的文件与 optix 实际消费的文件不一致；指定卡号
     （可选）经 `ASCEND_RT_VISIBLE_DEVICES`（可写 shell env / `OPTIX_DEPLOY_PATH` 相关）
     与 `--world-size` 交叉校验。
     **注意**：agent 模式**不**跑 `apply_launch_config.py`（那是 PSO 的 presets 流程）。
   - **场景冻结清单（寻优前必做）**：开跑前与用户显式声明本次寻优的**冻结项 /
     可搜项**。业务给定值（并发 C、输入/输出长度）是
     **验收条件，不是调参对象**——写入 config.toml 钉死（`target_field` 单值枚举或 `min=max=value`）；其中的 target_field 在首次建立搜索空间时保留为常量定义，不作为可调维度。
     用户未显式给值的参数默认属搜索维度（如约束 `tp×dp=world_size`），不得代用户钉死。只有清单里的可搜项才进候选生成（执行侧约束见 `./data-analysis.md`「候选生成契约」）。
     **诊断性 C 扫描 ≠ 寻优**：冻结 C 时仍可临时降 C 跑少量诊断点做并发梯度测量，
     但只用于机制定位 / 瓶颈分型（如边界 regime 识别），结果**不进入验收判定、
     不推翻冻结口径**（不产生"更优验收配置"主张）；若要把诊断发现转为寻优维度
     （如解冻 C），必须先回到本清单重新声明口径再开跑。

   > **run-id 自动化**：run-id 是内部标识（复用已有或自动生成 `run-<时间戳>`），
   > **用户无需提供**。config.toml 模板的 `run_id = "default"` 是占位（无实义）——
   > 首次 `--set-agent-mode` 会自动生成真实 run-id 写入并取代它；已有真实 run-id
   > （run-xxx）再 `--set-agent-mode` 会沿用（同一实验续跑）。agent 捕获
   > config_preflight 输出的 `run-id:`，后续 `collect_context --run-id <run-id>`
   > 保持一致；optix 才能在同一目录找到 `candidates.round-N.json`。

   > **边界**：完整性问题（缺模型路径/SLO）由本步骤引导用户；
   > Round 0 案例收集走 `knowledge/case-knowledge`（见步骤 4），**不要**跑
   > `apply_launch_config.py` / 读 `presets/ascend_vllm_presets.json`（PSO 专用）。
4. **收集 case-knowledge 官方案例（强制）**：
   **先完整阅读** `../knowledge/case-knowledge/SKILL.md`，再按该 skill 搜案例。
   Agent 模式跳过 vendor preset 注入；Round 0 的官方案例**只**走这里，**禁止**读
   `presets/ascend_vllm_presets.json`、**禁止**跑 `apply_launch_config.py` /
   `source launch_env.sh`（那是 PSO 分支）。
   本步骤只提取配置建议，交步骤 5 核对后写入；此时尚无 context.json，检索使用步骤 3
   已确认的模型、硬件与目标，不依赖搜索空间。

   ```bash
   # 工作目录以 optix-assistant skill 根为准；--model 与 --goal 二选一
   python <SKILL_ROOT>/knowledge/case-knowledge/scripts/search_cases.py --model <前缀>
   # 未绑模型、按主目标搜时：
   # python <SKILL_ROOT>/knowledge/case-knowledge/scripts/search_cases.py --goal <ttft|tpot|throughput>
   ```

   - 按 case-knowledge SKILL「多命中与无命中」使用过滤和分页；每批只打开 stdout
     返回的最多 5 个文件，继续读取时保持条件并使用下一 offset。读取前不凭文件名推断拓扑或收益。
   - 有 `## 启动脚本`：作为 `config.toml` 基础配置的参考。提取参数、固定配置及其案例依据，不限定为 TOML 已有字段。
     拓扑核算及不匹配处理按 case Skill；仅忽略不适用的并行取值，其他参数逐项核对。不整套照搬、不硬拼两条快照。
   - 有 `kind: method_card_positive`：不当启动命令；对照「改动 + 结果数字」，
     规则见 case-knowledge SKILL。
   - 无可采用案例时继续步骤 5，由目标路由与方法卡确定基础配置。

5. **方法核对与集中写入配置（阻塞项）**：
   基于已确认环境，按 `modeling-knowledge` 的 goals → method-index → relationships → methods
   路由读命中卡，核对案例适用性、组合条件、起点和取值范围；首轮无实测时走无基线分支，
   不凭模型名推断瓶颈。此时可直接读取模型配置和已确认环境，无需提前生成 context。

   - 汇总环境配置、运行节、拟采用的案例参数、固定项及未采用项的原因。用户确认值与
     冻结项优先；案例建议不自动覆盖。写入前沿用已确认信息，仅缺项或约束变更需要确认。
   - 用 `config_writer.py --config <config.toml绝对路径> --set ...` 写基础配置，
     `--target-field ...` 登记参数，`--append-others` 或结构化路径写固定 flags。
     范围须有方法卡、配置或实测依据；只有单个数值参考时用 `min=max=value`，不自行扩展档位。
   - 采用投机方法时，用 `python <SKILL_ROOT>/scripts/config_writer.py --config <config.toml绝对路径> --set '<engine>.command.others.speculative-config={"method":"<method>","model":"<草稿路径>"}'`
     登记实际需要的 method/model；无需草稿模型时省略 model。可调子键通过 `target_field`
     登记；投机 eager 使用 `speculative_enforce_eager`。首次 TOML 基线所需配置必须完整。
   - 随后用 `config_preflight.py --config <config.toml绝对路径> --engine <engine> --benchmark <benchmark>`
     携带步骤 3 已确认的 SLO、预算及 `--set-agent-mode` 旗标写入运行配置，捕获 run-id。
     这里的集中写入由多个脚本调用完成；每个调用失败均须修正后继续。**禁止手写 TOML**。
   - 写入后执行 `python <SKILL_ROOT>/scripts/config_preflight.py --check-only --config <config.toml绝对路径> --engine <engine> --benchmark <benchmark>`。
     校验模型、benchmark、`optimizer_strategy="agent"`、`use_request_rate_calibration=false`
     和 run-id；通过后才采集 context。基础参数写入 TOML，Round 0 不建立搜索空间。

6. **采集上下文**：配置就绪后记录本次硬件、模型、负载及预算。

   ```bash
   python <SKILL_ROOT>/scripts/collect_context.py \
     --run-dir <run-dir> --config-toml <config.toml绝对路径> \
     --engine vllm --benchmark <benchmark> --model-name <服务模型名> --model-path <权重路径> \
     --world-size <总卡数> --input-len <输入长度> --output-len <输出长度> \
     --num-requests <请求数> --concurrency <固定并发>
   ```

   `<benchmark>` 使用入口已确认的 `ais_bench` 或 `vllm_benchmark`；引擎旗标使用当前引擎。
   并发可调时，用 `--concurrency-min/--concurrency-max` 替代 `--concurrency`。
   `context.json` 记录环境事实、负载、约束和配置来源，不生成 `search_space`；空间由 agent 在首次优化时建立。

7. **配置参数与负载复检**：完成上下文采集后执行：

   ```bash
   python <SKILL_ROOT>/scripts/check_search_space.py --run-dir <run-dir>
   python <SKILL_ROOT>/scripts/config_preflight.py --check-only --config <config.toml绝对路径> --engine <engine> --benchmark <benchmark> --context <run-dir>/context.json
   ```

   参数类型、默认值、并行资源、投机配置及 benchmark 一致性均通过后继续。

8. **进入 Round 1 优化**：按 `./data-analysis.md`「每轮搜索空间」建立空间，再生成首轮候选。
   每轮 1 个候选，首轮基于步骤 7 复检通过的基础配置建立基线。

## 输出要求

输出已确认配置、run 目录、`context.json`（无搜索空间）和 `checklist_report.md`；随后进入 `data-analysis.md` 准备空间和首轮候选。
