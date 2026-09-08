# scripts/ 脚本归属矩阵

> 本目录采用**平铺结构**（不分子目录），原因：
> ① 多个脚本依赖同目录导入（`from common import ...`），拆目录即断链；
> ② SKILL.md / references/*.md 中所有命令路径均写死 `<SKILL_ROOT>/scripts/<脚本名>`；
> ③ 脚本升级归属（专属→公共）只改标注，不挪文件。
>
> **脚本归属以此表为单一事实源**；SKILL.md 与分支文档只表达流程，不重复维护清单。
> 归属变更流程：改本表 → 改脚本 docstring 头标注 → 同步受影响文档，三处一致才算完成。

## 归属定义

- `shared-precheck` — 共享前置：PSO / agent 两种寻优模式执行（config 模式只走 config_writer，见 SKILL.md"共享前置流程"）
- `config-write` — config.toml 写入执行器：三种模式都可能调用
- `pso` — 仅 PSO 分支使用（references/pso-mode.md）
- `agent` — 仅 agent 分支使用（workflows/agent.md + references/agent-mode.md）
- `lib` — 共享库，不单独执行
- `tooling` — 工具链脚本，寻优运行时不执行

## 脚本清单

| 脚本 | 归属 | 用途 | 消费文档 |
|---|---|---|---|
| `config_preflight.py` | agent | config.toml 发现 / **agent 模式配置就绪门禁**（strategy=agent、run-id 对齐、calibration=false）+ `[agent_optimizer]` 回填 | SKILL.md、workflows/agent.md |
| `collect_context.py` | shared-precheck | run 目录上下文采集与搜索空间装配 | SKILL.md、workflows/agent.md |
| `experience_injector.py` | shared-precheck | 经验注入报告（Roofline/KV 预算/硬约束/种子经验） | SKILL.md、pso-mode.md、workflows/agent.md |
| `check_search_space.py` | shared-precheck | 搜索空间自检 checklist | SKILL.md、workflows/agent.md |
| `apply_launch_config.py` | shared-precheck | preset 命中后应用 env/additional_config（launch_env.sh + unset_env.sh） | SKILL.md、workflows/agent.md |
| `common.py` | lib | JSON 读写、known_patterns 匹配、target_fields→search_space 转换等共享函数 | 被 4 个脚本 import |
| `estimation.py` | lib | 估算物理量单一宿主（dtype 字节 / per-token KV / 权重总量 / KV 预算 / TP 整除候选）——experience_injector 与 recommend_params 曾双实现，公式收口于此 | 被 experience_injector.py、recommend_params.py import |
| `config_writer.py` | config-write | config.toml 写入执行器（引擎命令/benchmark/搜索参数/固定参数，parse→mutate→serialize→check 薄层） | SKILL.md、pso-mode.md、workflows/agent.md |
| `recommend_params.py` | pso | 知识增强的参数与范围推荐（消费 experience_report + vendor preset） | pso-mode.md |
| `test_recommend_params.py` | pso | recommend_params 单测 | 开发用 |
| `test_estimation.py` | lib | estimation.py 单测（锁估算物理量数值契约，防回退私有公式） | 开发用 |
| `summarize_history.py` | agent | 轮次摘要追加 summary.md | workflows/agent.md |
| `export_best_config.py` | agent | 收敛后导出最优配置/serve 命令/handoff | workflows/agent.md、agent-mode.md |

## tools（presets 目录下的工具链，非运行时）

| 脚本 | 归属 | 用途 |
|---|---|---|
| `presets/scripts/harvest_vllm_ascend_presets.py` | tooling | 从 docs.vllm.com.cn 收割官方配置 → draft JSON |
| `presets/scripts/curate_draft.py` | tooling | draft 人工/子 agent 校准 |
| `presets/scripts/validate_presets_schema.py` | tooling | preset JSON schema 校验（提交门禁） |
