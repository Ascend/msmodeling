# scripts/ 脚本归属矩阵

> 本目录采用**平铺结构**（不分子目录），原因：
> ① 多个脚本依赖同目录导入（`from optix_assistant_common import ...`），拆目录即断链；
> ② SKILL.md / references/*.md 中所有命令路径均写死 `<SKILL_ROOT>/scripts/<脚本名>`；
> ③ 脚本升级归属（专属→公共）只改标注，不挪文件。
>
> **脚本归属以此表为单一事实源**；SKILL.md 与分支文档只表达流程，不重复维护清单。
> 归属变更流程：改本表 → 改脚本 docstring 头标注 → 同步受影响文档，三处一致才算完成。

## 归属定义

- `shared-precheck` — 共享前置：PSO / agent 两种寻优模式执行（config 模式只走 config_writer，见 SKILL.md"共享前置流程"）
- `config-write` — config.toml 写入执行器：三种模式都可能调用
- `pso` — 仅 PSO 分支使用（references/pso-mode.md）
- `agent` — 仅 agent 分支使用（workflows/ 下流程文件）
- `lib` — 共享库，不单独执行
- `tooling` — 工具链脚本，寻优运行时不执行

## 脚本清单

| 脚本 | 归属 | 用途 | 消费文档 |
|---|---|---|---|
| `config_preflight.py` | agent | config.toml 发现 / **agent 模式配置就绪门禁**（strategy=agent、run-id 对齐、calibration=false）+ `[agent_optimizer]` 回填 | SKILL.md、workflows/agent.md |
| `collect_context.py` | shared-precheck | run 目录环境与负载事实采集（含 `--input-len/--output-len/--num-requests` + 并发旗标 `--concurrency` 固定 / `--concurrency-min/--concurrency-max` 可调 登记 `benchmark_spec`） | SKILL.md、workflows/agent.md |
| `check_search_space.py` | shared-precheck | 直接检查 TOML 基础参数和资源可行性，不要求 context 已有搜索空间 | SKILL.md、workflows/agent.md |
| `apply_launch_config.py` | pso | PSO 模式 preset 命中后应用 env/additional_config（launch_env.sh + unset_env.sh）；agent 模式不用 | pso-mode.md |
| `optix_assistant_common.py` | lib | JSON 读写、模型族识别、target_fields→search_space 转换等共享函数 | 被 4 个脚本 import |
| `estimation.py` | lib | 估算物理量单一宿主（dtype 字节 / per-token KV / 权重总量 / KV 预算 / TP 整除候选） | 被 recommend_params.py 等估算消费者 import |
| `config_writer.py` | config-write | config.toml 写入执行器（引擎命令/benchmark/搜索参数/固定参数，parse→mutate→serialize→check 薄层） | SKILL.md、pso-mode.md、workflows/agent.md |
| `recommend_params.py` | pso | 基于模型、硬件和负载的 PSO 参数与范围推荐 | pso-mode.md |
| `test_recommend_params.py` | pso | recommend_params 单测 | 开发用 |
| `test_estimation.py` | lib | estimation.py 单测（锁估算物理量数值契约，防回退私有公式） | 开发用 |
| `summarize_history.py` | agent | 轮次摘要追加 summary.md | workflows/agent.md |
| `validate_candidates.py` | agent | 优化阶段的空间完整性、TOML 一致性、候选装配、分析与引用检查、routing_review 路由判定检查；检查已成功轮次的 context 归档 | workflows/data-analysis.md「候选展示与确认」 |
| `generate_pd_config.py` | agent | 按固定设备边界生成独立 PD 轮次配置 | references/pd.md |
| `summarize_pd_search.py` | agent | PD 覆盖记录与独立复测证据汇总 | references/pd.md |
| `export_best_config.py` | agent | 收敛后导出最优配置/serve 命令/handoff | workflows/agent.md |
| `case_base_extract.py` | agent | ① 案例 md → `case_base_raw.json`（Round 0 案例底座装配，只读案例库） | workflows/environment-preparation.md、references/case-base-automation.md |
| `audit_base_coverage.py` | agent | ④ 底座覆盖率审计（`missing`/`range_gap`，带退出码语义）；③ 复用其渲染函数避免漂移 | 同上 |
| `apply_case_base.py` | agent | ③ 结构化底座写入器（缺口 → `config_writer` 调用序列，默认干跑，写后断言 env 为 str） | 同上 |
| `case_base_rules.json` | agent | ② 案例底座分类 / 携带通道 / 区间策略规则表（数据文件，判据外部化，改判据不改代码） | 同上 |
| `run_casebase_pipeline.sh` | agent | 一条命令复跑 ①→④→③→④（Round 0 案例底座自动装配入口） | 同上 |

## tools（presets 目录下的工具链，非运行时）

| 脚本 | 归属 | 用途 |
|---|---|---|
| `presets/scripts/harvest_vllm_ascend_presets.py` | tooling | 从 docs.vllm.com.cn 收割官方配置 → draft JSON |
| `presets/scripts/curate_draft.py` | tooling | draft 人工/子 agent 校准 |
| `presets/scripts/validate_presets_schema.py` | tooling | preset JSON schema 校验（提交门禁） |


## 方法卡推荐规则

Round 0 采集事实并检查 TOML，不生成搜索空间。首次优化时 agent 读取 TOML 和方法卡，建立 `context.search_space`；后续自主维护并记录变更，候选 `params` 只写取值。`validate_candidates.py` 校验定义、TOML 一致性和候选，不改写输入。每轮进程退出后将 context.json 复制为 context.round-N.json，保留工作文件供下一轮更新。`export_best_config.py` 从最佳轮次的 context 归档取得历史定义和配置来源；归档缺失或损坏时拒绝导出。
