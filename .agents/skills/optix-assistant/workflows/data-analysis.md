# Data Analysis（数据分析与候选生成）

本文件写 agent 模式每轮循环中「读数据源 → 生成候选 → 总结 → 收敛检测」这些分析
工作的具体步骤；整体编排见 `./agent.md`。

候选生成与判优的规则也写在本文件（「候选生成规则」「fitness 公式解读」两节）；
失败分类（Layer 1）判据见 `./execution-validation.md`「失败分类」，
Layer 2 升级规则见 `./agent.md`「失败恢复」。

## 输入

每轮候选生成前必须读取当前 run 目录的：

1. `context.json` — 本轮环境事实、负载、约束、目标和搜索空间；历史输入读取对应 `context.round-N.json`。
2. `summary.md` — 所有过往轮次摘要、趋势、失败分析（可以只读摘要，数值以
   results 为准）。
3. 全部 `candidates.round-N.json`（含 directions）与 `results.round-N.json` —
   结构化 trial 数据，尤其是上一轮的 `failure` 字段（含
   `category/sub_category/suggested_action/offending_params`）。执行由
   `msmodeling optix` agent 模式（`optimizer_strategy="agent"`）完成，原始
   vLLM/benchmark log **不贴进这些 JSON**（由 optix 另行落盘保存）；本阶段只读
   结构化字段，不读原始 log。
4. `context.source.config_toml` 指向的配置文件——参数定义、固定配置与实际命令。
5. 按目标路由读取的方法卡——推荐起点、组合条件与回退动作。

## 本轮分析记录

本轮分析直接写入 `candidates.round-N.json`：三个分析字段写入每个 `directions[]`
对象，路由判定记录写入顶层 `routing_review`；不再生成独立的 `analysis.round-N.md`。
先完成方向分析与路由判定，再实例化关联候选，随同一个 JSON 落盘；这些字段不替代
`summary.md`。

| 作用域 | 必填字段 | 内容 |
|---|---|---|
| `directions[]` | `evidence_analysis` | 读取的证据及发现，包括来源、具体指标、失败或约束事实；区分已验证事实与未验证假设。 |
| `directions[]` | `refs_analysis` | 读取的知识点（goals 路由、relationships 组合校验、方法卡的前提与边界），以及证据如何匹配知识、为何选择或排除方向；经验建议冲突时记录裁决依据。 |
| `directions[]` | `candidates_analysis` | `search_space` 新增、修改、删除项的名称、前后定义及依据；候选取值、默认继承项、校验与回退动作。空间未变更时明确写明。 |
| 顶层 | `routing_review.hypothesis` | 本轮瓶颈假设，一句话陈述；延续上一轮假设时须写明延续关系。 |
| 顶层 | `routing_review.l1_layers_matched` | 该假设在目标路由表（goals/*.md）中命中的机制层名列表；层名须与 `method-index.md` 分层标题一致。 |
| 顶层 | `routing_review.layers_deferred` | 命中但本轮不行动的层及理由；键为命中层名，值为非空理由，无暂缓层时写 `{}`。 |

`directions[]` 的三个字段均为**非空字符串**，正文可用简短段落或 Markdown，不要求
固定标题、表格或子模板。只记录与该方向有关的分析；`evidence_refs` 继续保留真实
来源的文件路径，分析文字中也应说明来源。完整 JSON 示例见「候选格式」。

### routing_review 填写与校验规则

1. `l1_layers_matched` 如实记录当前假设依据 L1 路由表命中的全部机制层；命中
   范围由假设与路由表决定，不因本轮是否行动而增减。
2. 本轮实际探索哪些层由预算、优先级与组合约束决定；已行动层的依据方法卡照常
   写入 `refs_analysis`。
3. 命中但本轮不行动的层，必须写入 `layers_deferred` 并说明理由；未记录理由的
   跳过视为流程遗漏。

`validate_candidates.py` 检查 `directions` 为非空数组、每个方向的三个字段均为
非空字符串、`routing_review` 的完整性与一致性，并保留路径引用校验；不校验正文
的章节格式或分析结论的正确性。未完成分析或校验未通过时，不得执行本轮候选。
针对 `routing_review`：命中的层不在 `layers_deferred` 中时，`refs_analysis` 必须
引用该层目录（如 通信层 → `knowledge/modeling-knowledge/methods/communication/`）
下的方法卡；引用写方法卡路径或卡名均可，`evidence_refs` 中登记的路径同样计入，
且首次引用某层方法卡前须实际读取该卡正文。假设文本包含某层关键词但未列出对应
层时，校验输出告警，须复核路由匹配是否完整。

## 流程

1. 读取 `context.json` 的事实、约束、优化目标及已有的 `search_space`，并核对 `source.config_toml` 的基础字段；首次优化时空间尚不存在。
2. 汇总已完成 trial，区分成功、失败、跳过和 SLO 违约；将内容区分为已验证事实、
   未验证假设和用户明确约束，未验证假设不能当作结论。
3. 选择基线：优先选择硬约束达标且目标值最优的候选；没有达标配置时选择违约程度
   最低或最近成功的配置，并明确原因。
4. **诊断先行**，基于父轮次现象（summary/results）判定瓶颈场景；现象不足以判定时，
   先经 `../knowledge/modeling-diagnosis/`（症状模式 D1-D9 + 规程 P1-P3）补充
   证据再进路由；首轮没有可诊断数据时明确记录“瓶颈未知”，进入目标路由的无基线分支，
   校验批次和容量并建立基线后再诊断，不凭空指定瓶颈。
5. 基于瓶颈和目标值，阅读 `../knowledge/modeling-knowledge` 下知识卡，知识卡筛选
   根据目标值不同按需加载，详细流程参见 `../knowledge/modeling-knowledge/SKILL.md`。
6. 按「每轮搜索空间」建立或更新 `context.search_space`；整理证据发现、知识分析和变更操作，归入三个分析字段。
7. 基于前面获取的知识，生成少量结构化 `directions`：每个方向包含 `direction_id`、`type`、`trial_type`（`optimization` 或 `diagnosis`）、
   `hypothesis`、`baseline_ref`、`evidence_refs`、`constraints`，以及必填的
   `evidence_analysis`、`refs_analysis`、`candidates_analysis`（schema 见
   `../schemas/candidates.schema.json`）；顶层同步填写 `routing_review`（规则见
   「routing_review：路由判定记录」）。方向描述"为什么探索"，随候选文件
   一起落盘；没有证据引用的方向只能标记为 `unverified`。
8. 从方向实例化候选，每轮固定 1 个（理由见「候选生成契约 · 每轮候选数量」）；
   候选包含 `candidate_id`、`direction_id`、`params`、
   `rationale`、`expected_effect` 和 `risk`。候选必须引用至少一个具体历史依据
   或约束；`params` 只写实际值，全部参数须在 `context.search_space` 登记。
   params 的参数命名与容器子键映射规则见
   `../references/optix-reference.md`（写错会生成非法 serve 命令）。落盘后立即执行
   `python3 <SKILL_ROOT>/scripts/validate_candidates.py --run-dir <run-dir> --round N`（分析、引用、空间一致性与候选装配门禁，
   规则见「候选展示与确认」），**exit 0 才可进入步骤 9**；exit 1 补齐分析或修正引用后重跑校验。
9. 过滤越界、违反组合约束或缺少依据的候选。
10. 按 `./execution-validation.md` 执行本轮；确认进程退出、结果写入并完成 context 归档后，再进入总结。
11. 用 `summarize_history.py --round N` 自动追加 `summary.md`（脚本生成，不手工编辑）。评估本轮方向为 `validated` / `rejected` / `inconclusive`；结论及结果引用写入下一轮 `directions[].evidence_analysis`，收尾时写入 `final_report.md`，不回写已执行候选的方向状态。
12. 检查收敛和预算；本轮成功且继续探索时，才返回步骤 1 准备 `candidates.round-N+1.json`。本轮失败按 `./agent.md`「同轮失败与重试计数」处理，保持轮号；扩展范围前按「参数范围扩展」核对配置和实测依据——调整只落在本 run 的 `context.search_space`，不得修改 `config.toml`。

## 候选生成规则

### 生成前必须完成的检查

1. **核对参数定义**：按下节维护完整 `context.search_space`；候选只写值。启动前按当前 context 复检参数；执行期间不修改 context。
2. **核对方法前提**：采用推荐值前读取对应方法卡及组合关系；固定参数不能因方法建议而自动解冻。
3. **读取上一轮 failure 字段**：提取 `offending_params` 与 `suggested_action`，
   作为后续轮次避开失败参数区域的依据。
4. **场景冻结约束**：只有环境准备阶段「场景冻结清单」（见
   `./environment-preparation.md`）声明的**可搜项**才可进候选生成；冻结项
   （业务给定并发 C、输入/输出长度、既定部署项等，已钉死进 config.toml）不得
   成为候选参数或方向的 `target_params`；已登记的 TOML 字段仍须保留在空间中，以常量默认值参与装配。

### 每轮搜索空间

- Round 0 不生成 `search_space`。首次候选准备时，由 agent 读取 TOML 基础定义、方法卡和已确认约束，建立 `search_space.parameters/constants`；后续根据实测决定保留、修改、新增或删除，并在生成候选前写回 context。校验脚本只检查，不自动初始化或补齐空间。新增字段（env 类与 run 类一致）直接写入本空间回填，无需 TOML 对应 target_field，optix 会正式装载进最终执行命令：run 类渲染为引擎 CLI flag（`name` 须为引擎真实参数名并通过校验器的 `--help` 核验），env 类写为服务进程环境变量（`name` 转大写须为合法环境变量名）——不会因 TOML 缺失而丢失，也不会被渲染成非法命令；禁止通过修改 TOML 引入。回填后立即执行 `validate_candidates.py` 且退出 0，才可生成候选。
- 条目提供 `name`、`dtype`、`config_position`、`default`、`min`、`max`；枚举另有 `choices`，`min=0/max=1` 是枚举执行坐标。新增字段支持 `int/float/bool/enum`；字符串和数组用枚举表达。没有范围依据的单个数值可设 `min=max=default`，不编造搜索区间。`constants` 分组只放真正的常量：必须设 `constant` 字段或 `min==max`；只用 `default`+`choices` 表达倾向取值的条目仍是可调参数，放 `parameters` 分组（校验脚本会拒掉 constants 分组里的可变定义）。`config_position=run` 的条目 `name` 必须用引擎真实参数名（与命令行 `--flag` 对应，如 `tensor_parallel_size`），不得用方法卡缩写（如 `pcp`）；`validate_candidates.py` 会对照引擎 `--help` 的合法 flag 列表逐条核验。
- 保留全部 TOML 基础字段，类型、配置位置和派生关系须一致（冲突时以 TOML 为准修正空间）；可调字段的 `min`/`max`/`choices`/`default` 以本空间为准，TOML 登记的初始范围不构成运行期边界，TOML `constant` 项钉死。`config.toml` 在 Round 0 装配完成后冻结，运行期间不得修改；确需变更 R0 级事实（benchmark 工具、负载形态、引擎或模型）时终止当前 run 重新装配。删除空间里的条目不能删除 TOML 已有配置；额外字段可以删除，候选也须移除引用。
- `params` 只保存取值；省略的字段仍会采用空间默认值。新增定义时明确默认值，删除特性时同时核对 TOML 固定配置，不能用省略候选值表示关闭。
- 约束表达式（`context.constraints.expressions`）由 agent 按需自写，Round 0 脚本只采集 world_size 等事实、不再生成表达式。`$VAR` 按名字引用空间条目（大小写不敏感），`$WORLD_SIZE`/`$NPU_COUNT` 取 world_size 事实。只有在读完知识库、确认场景需要约束（如并行度乘积等于卡数）时才编写，且必须理解每个符号的语义再落笔；表达式不是登记空间条目的依据，空间条目只能来自 TOML 基础定义、知识库方法和实测判断。表达式或空间变化后在 `candidates_analysis` 说明依据。
- 在 `candidates_analysis` 说明变更条目、前后定义、依据、本轮取值与默认继承项，并记录本轮第几次执行（首次为 1，重试为 2）；重试使用新的 candidate_id。分析文字不代替结构化定义；无需复制未改动的全部参数。
- 以机制理由让参数保持默认（不搜索）时，论证必须覆盖取值域的**两个方向**（调大与调小、开与关）都无作用路径。只能论证单方向的（如只论证调大无差异、未排除调小的收益）记为未验证假设：补一个 `trial_type: "diagnosis"` 探针实测，或在 `candidates_analysis` 显式标注"未验证：仅论证了 × 方向"。未验证假设不得写成结论。
- 候选落盘后执行 `validate_candidates.py --run-dir <run-dir> --round N`。脚本检查定义、TOML 一致性、投机配置、实际并行取值、候选校验及字段装配，并校验顶层 routing_review；校验不改写候选或 context。启动前再次校验；空间变化须重新分析。进程退出后按执行流程归档 `context.round-N.json`，成功轮的候选、结果和归档不得覆盖。
- 每轮进程退出后才准备下一轮，不预生成未来候选；optix 一次启动只读一次 context。若有未成功的旧轮，按 `./agent.md`「同轮失败与重试计数」修复重试或暂停，门禁禁止带着新空间越过它。历史导出读取最佳结果所在轮的 `context.round-N.json`；归档缺失或损坏时报错，不使用当前 context 替代。

### 知识参考优先级

1. 用户冻结项及配置硬约束。
2. 本 run 实测结果（`results.round-*.json`）。案例或其他来源附带的实测声明不算本 run 实测。
3. 方法卡正文的适用条件与组合约束。
4. 官方案例配置与方法卡推荐起点；采用时在 rationale 说明依据。

案例参数与方法卡约束冲突时以方法卡为准：先按方法卡约束生成候选，把案例写法列为后续轮次的
单变量对照方向，本 run 实测更优后方可保留；裁决过程写入 rationale。

**case-knowledge 官方案例（agent Round 0 / Round 1 基线）**：按
`../knowledge/case-knowledge/SKILL.md` 用 `search_cases.py` 命中的
`## 启动脚本` 作为 `config.toml` 基础配置参考。Round 0 在采集 context 前汇总环境与案例，经方法卡核对（冲突按上述「知识参考优先级」裁决）后集中写入 TOML；后续轮次采用新案例配置时，把案例取值与新增字段按「每轮搜索空间」和 `../references/parameter-rules.md`「运行中新增字段」写回 `context.search_space` 并复检，随后生成候选；禁止通过 `config_writer.py` 修改 TOML 引入。案例不直接覆盖候选。多命中按 case Skill 过滤并分页读取，排序不代表推荐优先级。实测结果（results.round-*.json）优先于官方案例。Agent 跳过 vendor preset 注入，不把 presets JSON 作为自动推荐来源。

**hardware 是参考不是门槛**：官方案例的硬件场景是推荐配置，卡数不匹配
**不阻断逐项参考**。拓扑按 case Skill 核算；不匹配时忽略并行取值，
其余配置经适用性判断后仍可登记到 TOML。

**modeling-knowledge 方法卡**：具体详解下列章节描述。

### modeling-knowledge 方法卡（按目标指标路由加载）

候选生成除上述知识参考外，须按 `../knowledge/modeling-knowledge/SKILL.md` §0
调用链**完整执行**（诊断先行 → 前提探测 → 路由 → 校验 → 读卡，不跳步）：

1. **诊断先行**：基于父轮次现象（summary/results）判定瓶颈场景；现象不足以判定时，
   先经 `../knowledge/modeling-diagnosis/`（症状模式 D1-D9 + 规程 P1-P3）补充
   证据再进路由；首轮没有可诊断数据时明确记录“瓶颈未知”，进入目标路由的无基线分支，
   校验批次和容量并建立基线后再诊断，不凭空指定瓶颈。
2. **前提探测**：按 modeling-knowledge/SKILL.md §2 记录架构 / MoE / 版本 / 平台 /
   测试集特征等过滤值（`knowledge/modeling-knowledge/goals/<目标>.md` 的 L1 与
   `knowledge/modeling-knowledge/method-index.md` 的 L2 过滤依据）。
3. **查路由表**：读 `goals/<目标>.md`（throughput / tpot / ttft，SLO 优先级定主目标）
   L1 路由表（`knowledge/modeling-knowledge/goals/<目标>.md`）→
   L2 统一方法卡索引（`knowledge/modeling-knowledge/method-index.md`）：先按目标指标作用保留直接/条件作用卡和必要依赖，再按适用场景、决定性特征和排除条件过滤。
4. **校验组合**：`../knowledge/modeling-knowledge/relationships.md` 查
   依赖 / 互斥 / 推荐叠加 / 收益稀释。
5. **读方法卡**：直接打开命中的 `methods/<层>/<卡>.md`，读取正文与参数推荐章节。基础配置在 config.toml 登记；本轮额外字段及默认值写入 `context.search_space`，候选只写取值；定义必须有方法卡、运行配置或已验证结果依据。
   evidence_refs、分析字段和 rationale 引用实际读过的卡片或运行结果；引用必须能解析到文件，不能引用目录。
6. **基线与回退**：无基线时，经当前目标路由的无基线分支读取
   `../knowledge/modeling-knowledge/methods/scheduling/request-scheduling.md` 与
   `../knowledge/modeling-knowledge/methods/generic/memory-budget.md`，确认批次与容量可行性。
   特性不可用 / 负收益时恢复上一可行配置，按剩余瓶颈重新路由；多条件同时满足可叠加读卡，
   并执行 relationships 的参数联动检查。没有诊断证据时先建立基线，不推断瓶颈。

方法正文的适用条件与组合约束优先于推荐起点与案例参数；用户冻结项和配置硬约束优先于方法建议。冲突及取舍写入 rationale。

### 候选格式

以下为完整候选示例，不包含参数定义副本。搜索空间和来源须符合所述条件，实际填写必须使用本轮读取的证据。

```json
{
  "round": 1,
  "directions": [
    {
      "direction_id": "d1-baseline",
      "type": "baseline",
      "trial_type": "optimization",
      "status": "proposed",
      "hypothesis": "按 config.toml 的参数定义与 context.json 的约束建立吞吐基线，效果待实测。",
      "baseline_ref": "首轮，无历史实测基线",
      "evidence_refs": [
        "context.json",
        "knowledge/modeling-knowledge/goals/throughput.md",
        "knowledge/modeling-knowledge/method-index.md",
        "knowledge/modeling-knowledge/relationships.md",
        "knowledge/modeling-knowledge/methods/scheduling/request-scheduling.md",
        "knowledge/modeling-knowledge/methods/generic/memory-budget.md"
      ],
      "evidence_analysis": "config.toml 登记 MAX_NUM_BATCHED_TOKENS 可选值 32768/65536，并发固定为 64；依据请求调度方法卡，比较两个 token budget 档位。当前无历史实测，吞吐与瓶颈尚待验证。",
      "refs_analysis": "已读取 knowledge/modeling-knowledge/goals/throughput.md、knowledge/modeling-knowledge/method-index.md、knowledge/modeling-knowledge/relationships.md、knowledge/modeling-knowledge/methods/scheduling/request-scheduling.md 和 knowledge/modeling-knowledge/methods/generic/memory-budget.md，核对目标路由、组合约束与批次参数边界。现有证据不足以确认具体瓶颈，先按经验建议建立基线，暂不叠加其他特性。",
      "candidates_analysis": "本轮第 1 次执行。首次建立 search_space：保留 TOML 基础定义、默认值和固定约束，无额外参数；r1-a 取 MAX_NUM_BATCHED_TOKENS=65536，其余参数继承空间默认值。执行前通过分析字段与引用门禁，再由 optix 校验搜索空间和约束；若 OOM，按同轮失败规则判断是否重试，并评估 32768 档位。",
      "constraints": [
        "并发固定为 64，在搜索空间中保留为常量，不作为可调维度"
      ]
    }
  ],
  "candidates": [
    {
      "candidate_id": "r1-a",
      "direction_id": "d1-baseline",
      "params": {
        "MAX_NUM_BATCHED_TOKENS": 65536
      },
      "rationale": "参考请求调度方法卡建立基线；分析见 d1-baseline 的三个分析字段。",
      "expected_effect": "建立吞吐基线，具体指标待实测。",
      "risk": "较大批次可能增加显存占用；失败时按方向中的回退动作处理。"
    }
  ],
  "routing_review": {
    "hypothesis": "首轮无可诊断实测，瓶颈未知，先按目标路由的无基线分支建立基线。",
    "l1_layers_matched": ["调度层", "通用层"],
    "layers_deferred": {}
  }
}
```

例如启用有依据的 `enforce_eager`：先将下列条目加入 `context.search_space.parameters`（其余条目保留）：

```json
{
  "name": "enforce_eager",
  "dtype": "bool",
  "config_position": "run",
  "default": true,
  "min": 0,
  "max": 1
}
```

候选仍写 `"params": {"enforce_eager": true}`；`candidates_analysis` 说明新增依据、默认值和本轮选择。无依据时不新增。

### 候选校验规则（`optix/optimizer/agentic/validation.py::validate_candidate`）

每个候选执行前强制校验；**有 error 即被拒、不消耗 trial**，拒绝原因写入
`results.round-N.json` 的 `validation` 节（valid/errors/warnings 逐候选列出）。
写候选时按下表自查：

| 规则 | 拒/警 |
|---|---|
| 参数名未登记在 context.search_space | **拒**：`Unknown parameter` |
| 空间缺少必填定义、与 TOML 不一致，或上一成功轮缺少 context 归档 | **拒**：Skill 执行前门禁 |
| `dtype=enum`：值不在 `choices` 内 | **拒** |
| `dtype=int`：值非整数；`int/float/bool/range`：非数值或超出 `[min, max]` | **拒** |
| `dtype=ratio`：非数值或超出 `[min, max]`；`dtype_param` 必须是目标字段名字符串、该字段须存在且有解析值 | **拒** |
| `dtype` 为引擎派生字段（`share`/`ternary_factories`/`ternary_times`/`le_enum`） | **拒**：这些值由引擎从依赖字段计算，候选不得显式设置 |
| 常量/派生命名字段（`constant` 已设或 `min==max`）被显式设为不同值 | **拒**（同值仅 warning） |
| 启用投机但缺少 method/model，或候选并行取值与实际卡数不符 | **拒**：Skill 执行前门禁 |
| `context.constraints.expressions` 表达式不满足（`$VAR` 引用参数；`$WORLD_SIZE`/`$NPU_COUNT` = world_size） | **拒**：`Constraint failed` |
| 组合安全：`GPU_MEMORY_UTILIZATION > 0.95` | **拒**（几乎必 OOM） |
| 组合安全：KV 估算 > 80% 单卡显存；`CONCURRENCY>128` 且 `MAX_NUM_BATCHED_TOKENS > 2×默认值` | 仅 warning（不拦） |
| 空间中 `config_position=run` 的条目（变量与常量都会进最终命令）映射出的 `--flag` 不在当前引擎运行时支持的 CLI 列表里（`vllm serve --help=all` 等动态探测；探测不到二进制时该检查跳过） | **拒**：validate_candidates.py 执行前门禁 |

另注意：缺省值按 schema `default` 补齐后参与校验；`factories`/`times`/`ratio`
候选可以设，但校验侧会按引擎推导规则复算，与推导值冲突即拒。

### 每轮候选数量

**每轮固定 1 个候选**。trial 串行执行，同轮多候选意味着后面的候选在看不到前面
结果的情况下生成，浪费 trial 预算；逐轮"跑一个 → 分析 → 再定下一个"样本效率最高。

- 首轮根据环境准备阶段复检通过的基础配置生成基线候选，建立参照；
- 探索靠跨轮的 directions 规划保证覆盖（见 candidates 协议），不靠同轮堆候选；
- 结果异常、漂移或久未复测时，下一轮重跑基线候选确认稳定性。

### 诊断点候选（`trial_type: "diagnosis"`）

当存在明确**单变量假设**需受控验证（对应
`../knowledge/modeling-diagnosis/procedures.md` P2 单变量对照语义）且目标维度
**在当前搜索空间内未被场景冻结清单钉死**时，可生成 `trial_type: "diagnosis"` 的方向及其候选（候选经 `direction_id` 继承标记）。诊断点是真实
trial（走 optix 候选管道），只是结果解释权被降级。典型用途：失败归因的单变量
对照，以及「每轮搜索空间」中只论证了单方向的冻结假设的反向探针。

触发前提（缺失不得生成）：

1. 单变量假设明确，可写出 `hypothesis` 与判读标准；
2. 目标维度在搜索空间内且未被冻结——被钉死的参数走不了候选通道（optix
   `validate_candidate` 直接拒）；客户端压测类探针（并发梯度等，服务端冻结）
   **不走本通道**——该类探测需独立旁路执行且结果须与优化结果集隔离，当前
   流程未提供该通道，相关探测需求应在轮次外单独评估。

**判优排除（强制）**：`trial_type: "diagnosis"` 候选的结果不得进入——基线选择、fitness
对比与收敛趋势计算、验收判定与 `best_result.json` 候选池；
仅可作为下轮 `directions.evidence_refs` 的证据来源。

`type` 继续描述搜索策略（如 `factor_isolation`），`trial_type` 独立描述结果用途。
新方向必须显式填写 `trial_type`；候选必须用 `direction_id` 关联方向，不自行覆盖分类。
agent 按结果所属轮次和 `candidate_id` 回查同轮候选，再通过 `direction_id` 读取方向的 `trial_type`；
据此单独汇报诊断证据、排除正常判优和收敛判断。
agent 必须结合候选分类（`trial_type` 类型）解释，不能直接将这些原始统计作为排除诊断后的统计。
文件轮号与引擎轮次预算仍按原有规则计数，不承诺诊断自动豁免 `max_rounds`。
最终导出由 `export_best_config.py` 自动回查同轮候选并过滤 diagnosis；关联不完整、
类型非法或结果标记与候选冲突时报错。成功轮候选归档必须保留，不得改写。
诊断会实际消耗运行时间，仍须遵守原有时间预算；不因不计正常轮数而无限追加诊断。

### 参数范围扩展

候选到达配置范围边界或全部失败时，结合实际指标、错误日志和方法卡判断下一步。范围调整只修改 `context.search_space`：扩 `min`/`max`/`choices` 或改 `default`，在 direction 分析中记录修改依据，校验退出 0 后执行；TOML 标记 `constant` 的参数不得解冻。禁止修改 `config.toml` 扩范围。失败重试保持同一轮号，保留用户冻结项。运行中的轮次使用启动时加载的配置。

## 候选展示与确认（执行前强制，两种模式都要）

**分析与引用门禁（先于展示）**：`candidates.round-N.json` 落盘后立即执行
`python3 <SKILL_ROOT>/scripts/validate_candidates.py --run-dir <run-dir> --round N`，校验
`directions` 为非空数组，各方向的 `evidence_analysis` / `refs_analysis` /
`candidates_analysis` 均为非空字符串（不强制正文模板），顶层 `routing_review`
的完整性及命中层与引用卡的对应关系（规则见「本轮分析记录」），以及 evidence_refs /
hypothesis / rationale / risk 和三个分析字段中路径类引用的可解析性：

- skill 相对路径（`knowledge/…`、`workflows/…`、`references/…`、`schemas/…`、
  `scripts/…`、`presets/…`）→ 按 skill 根解析，必须存在且是**文件**
  （只引目录也算违规，如 `knowledge/modeling-knowledge`）；
- run 产物（`results.round-*.json`、`summary.md`、`trial_logs/…` 等）→ 按 run 目录解析；
- 不带路径前缀的自由文本（数据摘要、preset 场景描述等）→ 放行不校验。

exit 0 才可展示与执行；exit 1 必须补齐分析或修正引用后重跑（伪引用或目录引用均被拒绝）。完全不写 refs 的方向按「流程」标 `unverified`，仍须填写三个分析字段。

生成 `candidates.round-N.json` 后、启动执行前，必须用**表格**向用户展示本轮
候选的参数清单（让用户无需翻 JSON），例如：

```
Round 1 候选参数（1 个）
候选ID | MAX_NUM_BATCHED_TOKENS | MAX_NUM_SEQS | tp | dp | cudagraph_mode     | GPU_MEMORY_UTILIZATION
r1-a   | 16384                  | 48           | 4  | 4  | FULL_DECODE_ONLY    | 0.85
```

**表格上方必须先写"每个候选的选择依据"**（rationale + 引用 summary.md/results
的具体数据），方便交互模式下用户快速判断确认/修改/跳过，而不是只看到参数表。

- **交互式模式**：展示后暂停，等用户选择：1.确认执行 2.修改某个候选 3.跳过本轮。
- **全自动模式**：展示后直接流转，不停顿、不再询问确认。

> **候选校验由 optix 执行**：`msmodeling optix` agent 模式读
> `candidates.round-N.json` 时自行调 `validate_candidate`（搜索空间/约束/
> 运行时 flag 校验），**无效候选自动跳过**。Skill 门禁检查当前空间与 TOML 一致性及此前成功轮的 context 归档，并调用同一参数校验函数；生成候选时
> 仍需主动避开 combo safety 风险（KV cache 溢出、并发-批次冲突）。optix 只校验
> 参数与约束，**不校验分析字段与引用**——这两项由 `validate_candidates.py` 门禁负责
> （见上），两者互补、缺一不可。

## 评价规则

先检查硬约束可行性，再比较优化目标；`fitness` 只能作为辅助值。候选必须引用具体
历史结果或约束，不得把预期效果写成实测结果。

- 硬约束判定必须使用约束定义中声明的统计量（`statistic`，缺省 mean）；其他
  统计量只作参考记录。
- 软目标（如 TTFT target）不阻塞判定，违反时单独记录并在总结中提示。
- 单变量归因：一轮内若同时改变多个参数导致性能突变，下一轮应安排因子隔离
  （`factor_isolation`）候选确认增益来源。
- 数据反直觉时（如 batch 增大吞吐反降），先怀疑测量质量与噪声，安排复测或
  因子隔离，不得直接写入结论。
- 如果历史信息不足、数据矛盾或文件不完整，应报告信息缺口，不能自行猜测
  配置值或性能结果；信息缺口伴随"无方向可走"时，优先按「候选生成契约 ·
  modeling-knowledge 方法卡」诊断先行路径补证据，或生成符合触发前提的
  `trial_type: "diagnosis"` 诊断点候选，不得随机扩大搜索。

## fitness 公式解读（判优前必读）

optix 的 fitness 是把各项指标按权重加起来的一个分数
（`optix/optimizer/performance_tunner.py`），越小越优：

```text
fitness = 0.4 * (gen_speed_target / speed)         # 吞吐项，量级通常 2-13
        + 0.2 * exp(ttft_penalty * (TTFT/slo - 1)) # TTFT 惩罚，ttft_penalty=0 时恒为 0.2
        + 0.3 * exp(tpot_penalty * (TPOT/slo - 1)) # TPOT 惩罚，边界附近 ≈0.3
        + 0.1 * exp(success_rate_penalty * (1/success_rate - 1))
```

**何时 fitness 排名不可信**：
- `ttft_penalty=0` 时 TTFT 项是常数，fitness 对 TTFT 完全无感知；
- 吞吐项量级（`0.4*5300/speed`，speed 400-700 时约 2.9-5.3）远大于 SLO 惩罚项时，
  高吞吐但 TPOT 违例的候选会拿到更优 fitness —— 这是"软约束"的固有缺陷；
- 因此**硬约束场景（TPOT/TTFT 有明确上限）必须以 `slo_violated` 为第一排序键**：
  本轮最佳、`export_best_config.py`、汇总报告都优先"达标 + 吞吐最大"，
  纯 fitness 只作兜底（无任何达标候选时）。

## 收敛检测

每轮结束后检查。**收敛与预算的数值以本 run Round 0 冻结的
`config.toml [agent_optimizer]` 为准**；下面的数值只是默认值参考，可能过时。
轮数 / trial / 时间预算必须在 Round 0 装配时通过 `config_preflight`
`--set-max-rounds/--set-max-trials/--set-time-limit-minutes` 写入。Round 0 后
`config.toml`（包括 `[agent_optimizer]`）整体冻结，运行期间不得修改，不要把默认值或运行配置写回本文件。

- **趋势条件**：比较本轮 best_fitness 和全局 best_fitness（SLO 达标优先，见
  「fitness 公式解读」），连续 `convergence_rounds`
  轮改善 < 1%（默认 **6**）。
- **覆盖条件（必须同时满足）**：`optix` 会用 `search_space_coverage` 检查——
  已执行轮次 ≥ `min_rounds`（默认 4）、已完成 trial ≥ `min_trials`（默认 4）。
  agent 模式每轮固定 1 个候选，两个门等价，取较大者生效。
  任一不满足时输出"覆盖率不完整"及原因，**继续下一轮**，不收敛。
- 两个条件都满足才输出"收敛退出"并停止；跑满 `[agent_optimizer].max_rounds`
  （默认 32）自然终止。
- **时间预算**：`[agent_optimizer].time_limit_minutes` 限制每次单轮执行（默认
  1440 分钟，preflight 校验其 ≤1440）。OptiX 在下一 trial 启动前检查期限；
  当前 trial 完成后才停止本轮，同轮重试重新计时。
  Agent 在生成下一轮候选、同轮重试和恢复任务前，读取各轮
  `progress.round-N.json`，将顶层 `cumulative_elapsed_s` 每轮计入一次，
  重新计算全局 trial 累计耗时。该字段包含本轮历次成功、失败及重试的已完成
  trial 耗时；`trials[].elapsed_s` 只记录本次执行的单个 trial，不再另行累加。
  该口径不包含 baseline、Agent 分析及进程间等待时间，不能作为整个任务的墙钟耗时。
  预算耗尽或不足以完成下一次执行时，Agent
  不再启动新轮或重试，并执行 `export_best_config.py` 和生成 `final_report.md`。

> **注意**：collect_context 不控制上述参数——其同名预算旗标仅作信息记录，
> 运行时以 config.toml `[agent_optimizer]` 为准。

## 收尾

收敛或预算耗尽时，配合入口执行 `export_best_config.py` 导出
`best_result.json`（含 `selection_policy`/`source_ref`/`selected_at`）、
`serve_command.*` 和 `optimizer_config_handoff.json`，并提供 final_report.md
所需的分析数据（逐参数探索过程必须引用具体轮次和数据）。
