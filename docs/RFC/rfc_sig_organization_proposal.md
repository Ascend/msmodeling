
# msmodeling 子 SIG 分工与运作规范

> 以 SIG 为维度明确子系统归属与检视责任，消除灰色地带；辅以积分机制约束与激励看护质量。
> 生成日期：2026-07

---

## 一、背景与定位

### 1.1 项目背景

msmodeling（MindStudio Modeling）是一个全系统性能仿真与分析框架，核心包含两大组件：

- **TensorCast**：PyTorch / diffusers 程序性能仿真器，无需物理硬件即可预测模型在目标硬件上的执行性能。
- **ServingCast**：系统级推理服务仿真与吞吐优化。

围绕这两个核心，项目还延伸出 Optix（服务化自动寻优）、CLI 入口、profiling 数据采集工具链（microbench）、Web UI、文档与 Skill 体系等子系统。代码目录横跨 `tensor_cast/`、`serving_cast/`、`optix/`、`cli/`、`tools/`、`web_ui/`、`docs/`、`.agents/skills/` 等，参与人员约 20 人。

随着子系统增多、人员增长，出现了三类问题：

1. **灰色地带**：部分目录无人认领或多 SIG 交叉，PR 提出后找不到明确看护人；
2. **检视拖延**：缺乏 SLA 约束，PR 长期无人检视，阻塞合入；
3. **检视质量参差**：部分检视流于形式，或仅靠 AI 生成套话即通过，未能真正看护质量。

本规范通过划分子 SIG、定义角色与检视流程、引入积分机制来系统性地解决上述问题。

### 1.2 定位说明（重要）

- **子 SIG 是商业项目管理的补充手段，而非替代。** 日常的商业项目目标管理、排期、绩效考核仍以原有体系为准；子 SIG 仅在"代码归属边界"与"检视责任传递"两个维度上做轻量约束。子 SIG 不替代主管的行政决策，也不替代架构 SIG 的技术仲裁。
- **以荣誉驱动为主。** 与开源社区的检视运作一样，本项目的检视以荣誉与责任感为第一驱动力。积分机制是软性激励，目的是让责任可追溯、让贡献被看见，而非引入繁琐的行政考核流程。
- **荣誉为主不等于无约束。** 对长期看护不力、检视敷衍者，设有渐进式退出机制（见第五节）。积分比值异常会作为信号反馈，但本身不直接等同于绩效扣减。

---

## 二、角色与职责

| 角色 | 职责 | 关键约束 |
|------|------|----------|
| **主 Chair** | 子 SIG 负责人，为所属子系统的**质量与发展**负责；主导技术方向、分配检视任务、对子系统质量兜底 | 不可 approve 自己的 PR |
| **备 Reviewer** | chair 缺位或 chair 自身提交 PR 时代行质量看护；须实质性参与检视 | 须独立给出判断，不得附和 chair |
| **Approver** | 架构 SIG 成员轮值，最终合入把关，提供跨 SIG 仲裁视角 | 不可同时为本 SIG 的 chair / 备 |
| **成员** | 参与 SIG 内开发与检视 | 遵循检视质量规范 |

核心原则：

- **chair 是子 SIG 的负责人**，为相关子系统的质量与发展负责。
- **当 chair 在自己的子 SIG 范围内提交 PR 时，由 reviewer 代为行使质量看护责任**——chair 不自审、不自合，由备 Reviewer 接管该 PR 的检视与流程推动。

---

## 三、检视与合入流程（SLA）

一次 PR 在某子 SIG 目录范围内的完整流转如下：

```text
PR 提交 ──► chair/代行 reviewer 检视 ──► @ approver ──► approver 检视并合入
        (24h 内启动)                (责任传递)       (24h 内完成)
```

> **节假日说明**：上述 SLA 以工作日计时，法定节假日及周末不计入 24 小时倒计时。节假日期间检视不计超时。

### 3.1 启动检视（24 小时）

- chair 必须在 PR 提出 **24 小时内启动检视**。
- 若 chair 无法及时处理，须 **@ 真正相关人进行责任传递**，由被 @ 者承接剩余 SLA；责任一旦传递，计时随之切换到承接人。
- **特例**：若 chair 本人是该 PR 作者，则跳过 chair 环节，直接由备 Reviewer 代行看护并承接 24 小时启动 SLA。

### 3.2 完成检视并移交 approver

- chair（或代行 reviewer）完成检视后，须 **@ approver 并将责任传递给 approver**，明确告知"已检视完毕，请合入"。
- 移交时须附带本人在第四节要求的三项书面评价，作为 approver 复核的依据。

### 3.3 approver 检视与合入（24 小时）

- approver 须在收到移交后 **24 小时内完成检视与合入**。
- approver 有权要求 chair / reviewer 补充评价或返工；返工期间计时暂停，返工完成后恢复。

### 3.4 超时计分

- 每经过 **24 小时未完成当前环节**检视，扣减 1 分（见第五节）。
- 计时随责任 @ 传递而转移：被 @ 的人承接后，从承接时刻起重新计 24 小时。

---

## 四、检视质量保障

为避免检视敷衍了事、或简单 AI 检视后即通过，**每次检视须显式写出以下三项**，缺一不可：

1. **对 PR 的个人理解**：用自己的话说明该 PR 做了什么、解决什么问题（一两句即可，禁止复述 diff）。
2. **功能 / 业务层面评价**：是否正确实现预期功能、是否引入业务风险、是否存在更优方案。
3. **编码与代码质量评价**：命名 / 结构 / 可读性、边界与异常处理、性能与资源占用、是否符合既有约定。

### 4.1 反敷衍规则

- **纯 AI 生成的"看起来正确但无实质判断"的检视视为无效**，等同于未检视，按超时扣分。
- 未包含上述三项书面评价的检视视为未完成。
- 三项齐全且判断到位的检视，方可计入加分（见第五节）。
- approver 若发现 chair / reviewer 的评价敷衍，有权驳回并要求重做，重做期间计入原检视人的超时。

> 该规则同样约束 approver：approver 合入前亦须给出自身对三项的评价或明确复核结论，不得仅凭 chair 移交即机械合入。

---

## 五、积分与奖惩机制

### 5.1 计分规则

| 行为 | 分值 | 说明 |
|------|------|------|
| 完成一次规范检视并合入 | **+1** | 须含三项书面评价、符合 SLA、正确移交 approver |
| 每超时 24 小时未完成当前环节 | **−1** | 计时随责任 @ 传递而转移 |
| 检视缺少三项书面评价 / 敷衍 / 纯 AI 套话 | **−1** | 该次检视不计加分 |

### 5.2 比值审视

- **定期（建议每月）审视各人的扣分 / 加分比值**。
- 比值高者说明看护效果差：付出少、拖延多、质量低。
- 比值低且加分高者，说明看护投入足、质量高，作为荣誉贡献公示与表彰。

### 5.3 退出与惩罚机制（渐进式）

对比值持续异常者，按以下梯度处理：

1. **约谈**：单期比值过高，由架构 SIG 或主管约谈提醒。
2. **限制资格**：连续两期比值过高，限制其 chair / reviewer 资格，降级为普通成员。
3. **退出**：长期看护不力且无明显改善，移出对应子 SIG，由架构 SIG 重新分配人选。

> 对长期无检视活动（既无加分也无扣分）的 chair，若其 SIG 内有 PR 被忽视，同样纳入审视——不作为也是一种失职。

### 5.4 与商业考核的关系

积分本身不直接等同于绩效，但比值异常会作为信号反馈给商业项目管理层；荣誉贡献在评优、署名、公开致谢等方面体现。子 SIG 的定位始终是商业项目的补充手段，不喧宾夺主。

---

## 六、纪律与约束

| 纪律 | 说明 |
|------|------|
| 主 chair 不能 approve 自己的 PR | 强制备 Reviewer / Approver 参与 |
| 任何人兼任 ≤ 2 个 SIG | 指实质性角色（chair / 备 / 成员）；approver 为轮值，不计入兼任上限 |
| 跨 SIG 目录改动需双签 | 改动者所属 SIG + 目录所属 SIG 共审 |
| 影响全局的改动需架构 SIG 共审 | CI 门禁脚本、AGENTS.md 架构约束条款、.importlinter.ini 等 |
| 检视责任不可断链 | chair → reviewer → approver 的 @ 传递必须闭环，禁止"挂起不管" |

---

## 七、目录归属汇总（消除灰色地带）

按目录维度索引，便于"拿到一个路径就知道归谁"：

| 目录 | 归属 SIG | 备注 |
|------|---------|------|
| `tensor_cast/transformers/builtin_model/` | 模型适配 | 语言模型 |
| `tensor_cast/layers/` | 模型适配 | 模型层定义 |
| `tensor_cast/compilation/` | 模型适配 | 编译 pass |
| `tensor_cast/adapter/` | 模型适配 | 适配器工具 |
| `tensor_cast/core/` | 模型适配 | model_builder 等 |
| `tensor_cast/config.py`、`model_config.py`、`model_hub.py` | 模型适配 | 配置 |
| `cli/inference/text_generate.py`、`model_adapter.py` | 模型适配 | 文本生成 |
| `tensor_cast/ops/` | 模型适配 | 仿真算子定义；实测算子查询 SIG 可提映射诉求 |
| `tensor_cast/diffusers/` | 视频生成 | 视频模型 |
| `cli/inference/video_generate.py` | 视频生成 | 视频生成入口 |
| `serving_cast/` | ServingCast | 整体；`service/` 下 optimizer 改动需 Throughput SIG 双签 |
| `tensor_cast/device.py`、`device_profiles/` | ServingCast | 设备配置 |
| `tensor_cast/quantize_utils.py` | ServingCast | 精度防护网 |
| `cli/inference/throughput_optimizer.py` | Throughput 寻优 | 寻优入口 |
| `optix/run_throughput_optimizer_cases.py` | Throughput 寻优 | 寻优用例 |
| `optix/`、`contrib/optix/` | Optix | 整体 |
| `tensor_cast/performance_model/`（empirical / analytic 等逻辑、builtin_model/、custom_op/） | 实测算子查询 | 性能模型逻辑 |
| `tensor_cast/performance_model/profiling_database/` | 实测算子工具链 | profiling 数据与 op_mapping.yaml |
| `tools/perf_data_collection/` | 实测算子工具链 | microbench / op_replay / comm_bench |
| `tools/perf_data_analysis/` | 实测算子工具链 | 数据分析工具 |
| `tests/`、`scripts/`、`pre-commit/`、`build.py` | 测试与基础设施 | 测试与构建 |
| `.pre-commit-config.yaml`、`pyproject.toml` | 测试与基础设施 | 构建配置 |
| `docs/`、`.agents/skills/`、`AGENTS.md`、`CLAUDE.md` | 文档与 Skill | 文档与 AI 约束 |
| `CONTRIBUTING.md`、`README.md` | 文档与 Skill | 贡献指南 |
| `web_ui/` | UI | 前端 |

**根目录文件**（如 `tensor_cast/runtime.py`、`parallel_group.py`、`patch_torch.py`、`__init__.py`）：由改动者所属 SIG 负责，跨 SIG 改动需架构 SIG 共审。

---

## 八、子SIG 完整汇总表

> 以下表格为 11 个 SIG 的完整汇总，每行一个 SIG，便于一览全貌。

| # | SIG 名称 | 类型 | Chair | Reviewer | Approver | 负责目录范围 | 边界说明 |
|---|---------|------|----------|------------|----------|-------------|---------|
| 1 | 架构治理 SIG | 仲裁层 | jhon-117 | Horacehxw、yaohan404、lutean | jiangruitao | 无固定业务目录（仲裁层） | 跨 SIG 仲裁 / RFC 评审 / CODEOWNERS 维护 / .importlinter.ini 配置；跨模块改动共审 |
| 2 | 模型适配 SIG | 业务领域 | ChenHuiwen | stormchasingg、jhon-117、weixin_43113933、wangshen001等模型owner | lutean | tensor_cast/transformers/builtin_model/、tensor_cast/layers/、tensor_cast/compilation/、tensor_cast/adapter/、tensor_cast/core/、tensor_cast/config.py、model_config.py、model_hub.py、cli/inference/text_generate.py、model_adapter.py、tensor_cast/ops/ | 新增语言模型适配必须走 10 步标准流程；视频模型走视频生成 SIG；ops/ 为纯仿真算子定义，实测算子查询 SIG 可提映射诉求 |
| 3 | 视频生成 SIG | 业务领域 | minghang_c | jia_ya_nan | yaohan404 | tensor_cast/diffusers/（含 cache_agent/、diffusers_model.py 等）、cli/inference/video_generate.py | 视频模型（diffusers 系列）与语言模型分离；适配流程可参考 10 步但允许裁剪 |
| 4 | ServingCast SIG | 业务领域 | yuyinkai1 | yaohan404 | lutean | serving_cast/（整体）、serving_cast/service/、tensor_cast/device.py、tensor_cast/device_profiles/、tensor_cast/quantize_utils.py、docs/RFC/rfc_precision_protection.md 相关实现 | serving_cast/service/ 下 optimizer 接口设计由本 SIG 主导；寻优调用逻辑修改需 Throughput SIG 共审（双签） |
| 5 | Throughput 寻优 SIG | 业务领域 | jia_ya_nan | yuyinkai1 | jhon-117 | cli/inference/throughput_optimizer.py、optix/run_throughput_optimizer_cases.py | 对 serving_cast/service/ 下 optimizer 文件修改需 ServingCast SIG 共审（双签）；jia_ya_nan 与 yuyinkai1 互为备，需额外 approver |
| 6 | Optix SIG | 业务领域 | tt0cool | h7star | jhon-117 | optix/（config / optimizer / plugins 等）、contrib/optix/ | 3 人全职；optix/run_throughput_optimizer_cases.py 归 Throughput SIG（跨 SIG 改动双签） |
| 7 | 实测算子查询 SIG | 业务领域 | Horacehxw | zhenghaojie | yaohan404 | tensor_cast/performance_model/（empirical / analytic / bound_analyzer 等性能模型逻辑、builtin_model/、custom_op/） | 查询实测数据用于性能估算；profiling_database 数据由工具链 SIG 产出，跨改双签；ops/ 归模型适配 SIG，映射不匹配时向其提诉求 |
| 8 | 实测算子工具链 SIG | 业务领域 | Secluded_Ocean | zhenghaojie | Horacehxw | tools/perf_data_collection/（microbench、op_replay、comm_bench 等）、tools/perf_data_analysis/、tensor_cast/performance_model/profiling_database/（含 op_mapping.yaml） | microbench 工具链与 profiling 数据采集 / 入库；op_mapping 维护；产出数据供查询 SIG 消费 |
| 9 | 测试与基础设施 SIG | 支撑层 | AvadaKedavrua | jhon-117 | lutean | tests/（UT / ST / skill_eval / perf_database）、scripts/（build / ci_gate / nightly / common）、pre-commit/、build.py、.pre-commit-config.yaml、pyproject.toml | CI 门禁脚本（ci_gate）改动需架构 SIG 共审（影响全局） |
| 10 | 文档与 Skill SIG | 支撑层 | wendellX | eveyin1 | yaohan404 | docs/（RFC / design / user_guide / install_guide / perf_database）、.agents/skills/、AGENTS.md、CLAUDE.md、CONTRIBUTING.md、README.md | AGENTS.md 的架构约束条款改动需架构 SIG 共审 |
| 11 | UI SIG | 支撑层 | zwt__ | lutean | jhon-117 | web_ui/（前端代码）、docs/design/web_ui_frontend_design.md、docs/zh/user_guide/msmodeling_web_ui_user_guide.md、docs/en/user_guide/msmodeling_web_ui_user_guide.md | 如 web_ui 目录尚未建立，以 design 文档为锚点，后续代码目录确定后更新 |

### 补充说明

- **根目录文件**（如 tensor_cast/runtime.py、parallel_group.py、patch_torch.py、__init__.py）：由改动者所属 SIG 负责，跨 SIG 改动需架构 SIG 共审。
- **兼任规则**：任何人兼任不超过 2 个 SIG（兼任 7 人：Horacehxw、zhenghaojie、jia_ya_nan、yuyinkai1、yaohan404、jhon-117、lutean）。
- **Approver 轮值**：jhon-117 负责 3 个 SIG、yaohan404 负责 3 个、lutean 负责 3 个、Horacehxw 负责 1 个、jiangruitao 负责架构 SIG 1 个。
- **Secluded_Ocean**：现担任实测算子工具链 SIG 主 Chair（原外部参与者，转正后承担工具链质量责任）。
- **jiangruitao**：主管，无代码背景，仅做最终决策，不参与技术 review。

---

*本规范基于团队专精背景与项目目录结构制定，旨在消除 SIG 间灰色地带、保障检视质量。积分与奖惩机制以荣誉驱动为主，作为商业项目管理的补充手段。如有调整需求请反馈。*
