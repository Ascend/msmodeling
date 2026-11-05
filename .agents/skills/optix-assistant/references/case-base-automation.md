# 案例底座自动化（Round 0 案例参数 → config.toml 底座）

**什么时候用**：`workflows/environment-preparation.md` 步骤 5（方法核对与集中写入配置）。
案例收集（步骤 4）完成后，先用本流程把案例里**可自动采用**的条目一次性写进新 run 的底座，
再只对残余项（`needs_review`）逐条裁决。**不要**再对着案例手工逐条 `--set`。

**一句话契约**：把「案例/方法卡里的参数 → 当前 run 的 `config.toml` 底座」这一步做成
可审计、可干跑、可复检的流水线；每条改动都能回溯到 `source_case + line_no + reason`。

## 四段流水线与一键入口

```
① case_base_extract.py   案例 md → case_base_raw.json        （只读案例知识库）
② case_base_rules.json   分类 / 携带通道 / 区间策略的规则表  （改判据改这里，不改代码）
③ apply_case_base.py     缺口 → config_writer 调用序列 → 写入（默认干跑）
④ audit_base_coverage.py 复检覆盖率：missing/range_gap，带退出码语义
```

推荐用一条命令复跑全流程（① → ④ → ③ → ④）；脚本自行以**仓根为 cwd** 调 `config_writer`
（否则 `import optix` 会命中别的 checkout），并允许整体搬移（所有工具路径相对脚本自身解析）。

```bash
SRC=<基准/冻结 config.toml 绝对路径> \
NEW=<新底座 config.toml 绝对路径> \
bash <SKILL_ROOT>/scripts/run_casebase_pipeline.sh                # 干跑：只出清单
SRC=... NEW=... APPLY=1 bash <SKILL_ROOT>/scripts/run_casebase_pipeline.sh   # 写入
```

可选变量：`CASE=<案例 md>`（缺省复用上次产物记录的案例）、`AUDIT_DIR=<产物目录>`
（缺省 `<repo>/case_base_out`）、`ROOT=<optix 仓根>`、`PY=<选定环境的 python>`（默认 `python3`）、`CT=<容器名>`（显式启用 Docker）、`FORCE_COPY=1`（允许覆盖已有目标）、
`ENGINE=<vllm|mindie>`。

产物（`<AUDIT_DIR>/out/`）：`case_base_raw.json`、`base_coverage.json`、
`base_apply_plan.json`（逐条 argv/shell/reason/verify 断言/manual 清单）、
`base_coverage.after.json`。

退出码：`0` 干跑完成或写入+自检通过；`1` 写后自检失败；`2` 参数/路径/冻结契约错误。

> ③ 拒绝原地补写冻结配置：目标 `--config` 等于审计产物记录的配置路径时报错；
> 使用 `--copy-from <冻结配置> --config <新路径>`；干跑仅读取源配置，写入在临时文件中完成并通过校验后发布。
> 已有目标默认拒绝覆盖；确需替换时显式设置 `FORCE_COPY=1`。

## ② 规则表的判据（不抄字面量，脚本现读真源）

| 键 | 作用 |
|---|---|
| `symbol_sources` | 真源 = 文件 + 变量名：`OWNED_FLAGS`（`optix/optimizer/pd_adapter.py`）、`OWNED_ENV` / `TOPOLOGY_FIELDS`（`optix/optimizer/agentic/pd.py`）、`JSON_SUBKEY_MAP` / `BENCHMARK_ONLY_FIELDS`（`optix/serve_plan.py`） |
| `orchestration_flags` | 编排类 flag，永不自动写 |
| `review_flags` | 需裁决类 flag（`--enforce-eager` / `--compilation-config` / `--async-scheduling` / `--additional-config` / `--speculative-config`），永不自动写 |
| `env_categories` | `toolchain`（`LD_LIBRARY_PATH` 等）与 `node_placeholder`（值含 `${`/`xxxx`/`nic_name` 等）判定，均不搬 |
| `adopt_policy` | 携带通道：flag → `--append-others`（**禁止 `--set-others`**，那是整体覆盖会抹掉既有 `--quantization`/`--speculative-config`）；容器子键 → 结构化 `--set`；env → TOML dot 路径 `vllm_pd.env_{role}.{VAR}` |
| `range_policy` | `widen_range`：`new_min=min(案例值,当前min)`、`new_max=max(案例值,当前max)`，`value` 不动；枚举的 `choices` 扩展转人工 |
| `needs_review_handling` | 只输出建议命令，不执行；由 agent 按方法卡与互斥关系逐条裁决并落盘依据 |

`--engine` 必须与该 run 底座里 `others` 的实际宿主段一致（看 `base_coverage.json` 的 `notes`）。
写错段会凭空造出一个不生效的分区。

## 故意「不自动采用」的边界（转人工）

- `needs_review`：`--additional-config`（整容器）、`--additional-config:enable_cpu_binding`、
  `--additional-config:recompute_scheduler_enable`、`--speculative-config:enforce_eager`。
- P/D 取值不同的 flag / 容器子键：`[<engine>.command].others` 是 **P/D 共享通道**，写进去双侧同值
  → 表达不了角色差异，转人工（或改走 role 后缀 `target_field`：整容器作为 str 枚举）。
- 枚举型区间缺口、固定值或用户固定参数的区间冲突：自动扩域会改变约束 → 转人工。
- 单角色 flag 不写共享 `others`；没有安全的角色通道时转人工。
- 框架占用项（命中 `OWNED_FLAGS`，写进 `others` 会 `ValueError`）/ `orchestration_flags`：只出建议。
- 工具链 / 节点占位 env：随部署变化，不搬。

`recompute_scheduler_enable` 不在 `JSON_SUBKEY_MAP` 里，**不能**当搜索维度；唯一可表达通道是
「整个 `--additional-config` 作为 str 枚举 + role 后缀 `target_field`」，或整容器静态下发。

## 实测结论（2026-10-23/24 在 116.204.45.24，非推断）

输入 = 冻结底座 `config.pd-qwen3500n256-20261023.toml`（5 315 B），案例库 = `case-knowledge/cases/official/**`。

| 指标 | 写入前 | 写入后 |
|---|---|---|
| `adopted` | 14 | **24** |
| `missing` | 14 | **4**（全部 `needs_review`） |
| `range_gap` | 2 | **0** |
| env 落地类型 | — | 8/8 均为 `str` ✅ |

实测写入动作 `auto_actions 16 / manual 4`：
6 条 flag 并入 `others`（`--no-enable-prefix-caching`、`--enable-expert-parallel`、`--seed 1024`、
`--max-model-len 16384`、`--trust-remote-code`、`--no-disable-hybrid-kv-cache-manager`）、
8 条 env 落位（P/D 各 4：`VLLM_ENGINE_READY_TIMEOUT_S=30000`、
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`、`VLLM_TORCH_PROFILER_WITH_STACK=0`、
`HCCL_OP_EXPANSION_MODE=AIV`）、2 条区间放宽（`MAX_NUM_BATCHED_TOKENS_decode [256,1024]→[128,1024]`、
`GPU_MEMORY_UTILIZATION_decode [0.85,0.95]→[0.85,0.96]`）。

## 处理过的三个真坑（勿再踩）

1. **`config_writer --check` 在 PD 配置上必然失败**：`--check` 走单机 Settings 校验，会报
   `cluster / vllm_pd: Extra inputs are not permitted`，**与写入内容无关**（原样 `--check-only` 同样报）。
   → 写入统一用 `--print`，验收靠「写后 TOML 回读」+「重跑 ④ 审计」。
2. **`config_writer` 必须用 cwd = 仓根调用**：容器默认 cwd `/workspace` 时 `import optix` 会命中
   *另一个* checkout（`/data1/ssq/msmodeling-x`），API 不一致，在 `check_service_semantics` 直接抛
   `ImportError: cannot import name 'validate_service_parameters'`。
3. **PD 角色 env 表必须保留字符串**：`vllm_pd.env_prefill` / `env_decode` 的 schema 是
   `dict[str, str]`，数字样值（`VLLM_ENGINE_READY_TIMEOUT_S = 30000`）被推断成 int 时，
   PD schema 会**静默拒绝整份配置并退回 localhost**。
   该缺陷已修入仓内 `config_writer.py::_is_str_typed_path`（含 `test_config_writer.py` 两条回归），
   不再需要外挂补丁副本。

## 收尾动作（每个 run 都必须做）

1. 对 `base_apply_plan.json` 的 `manual` 清单逐条给出 `decision + reason + evidence`，
   落盘到该轮 direction 的分析字段。
2. 重跑 `python <SKILL_ROOT>/scripts/config_preflight.py --check-only --config <新底座> --engine <engine> --benchmark <benchmark>`，
   再执行步骤 5 其余写入（MTP 准备、运行配置、SLO/预算）。
3. 只有复检通过才进入步骤 6 采集 context；装配完成后的 `config.toml` 在本 run 内冻结。
