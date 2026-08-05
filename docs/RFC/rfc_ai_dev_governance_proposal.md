# msmodeling 项目 AI 开发规范升级方案

> **会议决策材料** | 2026-07 | 项目架构治理专项
> 关键词：SDD 规范驱动开发、PR 治理、责任田划分、理解债

---

## 一、背景：项目正在接近"不可维护临界点"

### 1.1 三个不可忽视的事实

| 维度 | 12 个月前 | 当前 | 增长 |
|------|----------|------|------|
| 代码规模 | ~3 万行 | ~20 万行 | **6.7×** |
| 开发者人数 | — | 50+ 人（vibecoding 为主） | — |
| 单 PR 行数 | 几百行 | **2000-8000 行**成常态 | 5-10× |
| DeepSeek v4 适配 | — | **未走 10 步标准流程** | 规范违反 |

## 四、解决方案：自动化 + 责任田划分 双轮驱动

### 4.1 总览：两层防御纵深

```text
┌─────────────────────────────────────────────┐
│  Layer 1：责任田划分（人）— 防"谁都不负责"      │
│  ├─ 11 个子 SIG（含架构 SIG）                  │
│  ├─ 一主一备一 approver                        │
│  └─ __all__ + import-linter 模块契约            │
├─────────────────────────────────────────────┤
│  Layer 2：自动化（机器）— 防"约束被绕过"         │
│  ├─ SDD 规范驱动开发 + Spec-First Review       │
│  ├─ PR 治理（CI 门禁 + AI 辅助检视）             │
│  ├─ AGENTS.md 统一 AI 约束                     │
│  └─ 理解债管理（语义锚定）                       │
└─────────────────────────────────────────────┘
```

### 4.2 Layer 1：责任田划分（SIG 治理）

#### 4.2.1 为什么是 SIG 模式

业界主流大型项目（Linux kernel / Kubernetes / LLVM / Chromium / Rust）普遍采用 **Subsystem Maintainer / SIG** 模式。这是被几千个大型项目验证过的成熟治理结构。

#### 4.2.2 SIG 划分方案

项目已按子系统边界划分为 **11 个子 SIG**（1 个架构仲裁层 + 10 个业务 / 支撑层），每个 SIG 明确负责的目录范围、主 Chair / 备 Reviewer / Approver 人选及边界说明。

完整的 SIG 划分、目录归属、角色分配与运作规范详见独立文档：

→ [rfc_sig_organization_proposal.md](./rfc_sig_organization_proposal.md)

#### 4.2.3 角色职责与检视流程

每个子 SIG 设 **主 Chair**（子系统质量与发展负责人）、**备 Reviewer**（chair 缺位或自提 PR 时代行看护）、**Approver**（架构 SIG 轮值，最终合入把关）三角色，并配套检视 SLA 与积分机制：

- **检视 SLA**：chair 须在 PR 提出 24h 内启动检视（或 @ 相关人传递责任），完成后 @ approver，approver 须 24h 内完成检视与合入；
- **检视质量保障**：每次检视须显式写出对 PR 的理解、功能 / 业务评价、代码质量评价三项，杜绝敷衍与纯 AI 套话；
- **积分奖惩**：规范检视合入 +1，超时 / 敷衍 −1，定期审视扣分加分比值，比值持续异常者渐进式退出。

角色定义、检视流程、积分机制与纪律约束的完整细节详见：

→ [rfc_sig_organization_proposal.md](./rfc_sig_organization_proposal.md)

#### 4.2.4 模块契约：`__all__` + `import-linter`（低优先级）

**这是把"模块边界"从口头规范升级为可执行约束的关键手段。**

**手段一：显式 `__all__` 声明 Public API**

```python
# tensor_cast/ops/matmul.py
__all__ = ["Matmul", "matmul"]  # 外部只能用这两个

class Matmul: ...
def matmul(a, b): ...

def _internal_matmul_impl(...): ...  # 不在 __all__，禁止外部 import
```

**手段二：`import-linter` 定义依赖方向**

```ini
# .importlinter.ini
[importlinter:contract:layers]
name = Layered architecture
type = layers
layers =
    cli                    # 顶层
    serving_cast
    tensor_cast            # 底层
```

CI 跑 `lint-imports`，跨层依赖直接 fail。**这是防架构腐化的硬手段。**

#### 4.2.5 SIG Chair 的核心 KPI

| 指标 | 内容 |
|------|------|
| 代码质量 | 领域内 bug 率、回归率、复杂度趋势 |
| 用例有效性 | 用例重构完成率、Mutation Testing 杀伤率 |
| 检视时效 | PR 启动检视 ≤ 24h，全流程（检视 → 移交 → 合入）≤ 48h |
| 看护质量 | 积分加分 / 扣分比值低（详见 SIG 运作规范第五节） |
| 技术债清偿 | 每季度至少还 1 项 |

> 检视时效与看护质量的度量规则（24h SLA、三项书面评价、积分奖惩与退出机制）统一以 [rfc_sig_organization_proposal.md](./rfc_sig_organization_proposal.md) 为准。

### 4.3 Layer 2：自动化（机器强制）

#### 4.3.1 SDD 规范驱动开发（Spec-Driven Development）

> **这是 2025-2026 业界共识最强的方向，GitHub 官方亲自下场。**
>
> GitHub 2026 开源 spec-kit（⭐ 98.5K）—— 翻转传统流程：**先写规范，再让 AI 按规范实现**。

**为什么 SDD 比 vibe coding 强**：

| 维度 | 传统 AI 编程 | SDD |
|------|------------|-----|
| 出发点 | 自然语言对话 | 结构化规范文档 |
| 复现性 | 同样输入两次结果不同 | 规范变 → 代码变 |
| 可追溯 | 改了不知道为什么 | 每段代码可追溯到规范条款 |
| 团队协作 | 各人各 AI 各风格 | 单一事实来源 |

**关键纪律**：

- **specify 阶段不提技术栈**（避免 AI 偷懒）
- **clarify 阶段不能跳**（消除歧义）
- **plan 阶段产出 4 个文档**：plan.md / research.md / data-model.md / openapi.yaml

#### 4.3.2 Spec-First Review（核心 review 模式）

**核心思想：把 review 拆成两阶段——事前审 spec（小），事后审"代码是否符合 spec"（机械）**

```text
1. 写 spec.md（< 200 行：What / Why / 边界 / 验收标准）→ 人审 spec
2. spec 通过后，AI 按 spec 生成代码 + 测试（可能 5000 行）
3. 人审代码时只问 3 个问题：
   - 代码是否实现了 spec 的所有条目？
   - 代码有没有做 spec 没要求的事？（防过度工程）
   - 测试是否覆盖 spec 的验收标准？
```

**关键认知**：
> **设计必须完善，人工重点参与检视设计，设计应包含完善的测试，后期验收重点就是代码与设计的一致性。**

#### 4.3.3 大 PR 处理：Commit-as-PR + Spec-First

> **业界 2025-2026 在反思的问题**：
> "PR < 800 行"硬规则会被 vibecoder 用"假拆分"绕过（Stacked PR / 机械切行 / 按层切）。

**Stack Overflow 2026 调研**：
> 78% 的 maintainer 认为 "PR 体积规则"在 AI 时代失效。

**Linus Torvalds 2026 公开信**：
> "Reviewing 5 stacked PRs is harder than reviewing 1 big PR."

**结论**：**PR 大小不是问题，评审单元与工作单元的错配才是问题**。

**落地原则**：

- **能拆就拆 Commit-as-PR**（每个 commit 是独立可 revert 的逻辑单元，Linux 内核模式）
- **不好拆就不拆**——大特性整体性强，强拆只是制造噪音
- **拆不动的大 PR → 必须走 Spec-First Review**（用 spec 降 review 复杂度）

#### 4.3.4 PR 治理（仅包含两类）

**严格定义"PR 治理" = CI 门禁自动化 + AI 辅助检视**，不包含人工 review 决策。

**A. CI 门禁自动化**

| 门禁 | 阈值 | 工具 |
|------|------|------|
| Diff Coverage | ≥ 90%（新增/修改代码） | diff-cover |
| 圈复杂度 | ≤ 15 | radon / ruff |
| 认知复杂度 | ≤ 20 | cognitive-complexity |
| 单函数行数 | ≤ 50 | ruff |
| 单文件行数 | ≤ 500 | 自定义 |
| Import 依赖方向 | 必须符合 layers | import-linter |
| DCO 签名 | 必须 git commit -s | GitCode DCO Action |

**B. AI 辅助检视（推荐 Qodo PR-Agent 自托管）**

| 维度 | 选型理由 |
|------|---------|
| 部署 | 自托管 GitCode CI，**零数据出境**（中国合规） |
| 模型 | 接 **DeepSeek V3**（国内可访问 / 中文好 / 价格 1/10） |
| 测试生成能力 | 三者中最强，正好补 vibecoder 测试质量差 |
| 成本 | 开源免费，50+ 人零成本 |
| 中文 prompt | 可定制（嵌入 AGENTS.md 规范） |

**关键警告**：
> **生成代码的 AI 和 review 的 AI 必须是不同模型**，否则会出现"AI 帮 AI 护短"。

#### 4.3.5 AGENTS.md：统一所有 AI 的开发约束

> **参考华为 vllm-ascend 社区**（仓库根目录已有 AGENTS.md + CLAUDE.md + .agents/ + .claude/ + .gemini/）。

**核心目的**：**保证各种 AI 工具（Claude / Cursor / Gemini / Copilot）都能以同样的约束完成开发**。

**AGENTS.md 必须包含的硬约束**：

1. **环境变量集中管理**（如 vllm_ascend/envs.py 模式）
2. **Patch 架构强 review**（禁止直接添加新模型文件，必须走 BaseModelAdapter）
3. **性能强约束**（hot path 禁用 tensor.item()）
4. **命名 / 风格强约束**（无 magic number / 严格 snake_case）
5. **完整 Review Checklist**（5 类：Code Quality / Testing / Documentation / NPU Considerations / Commit & PR）
6. **PR 三段式模板**（What / Why / How tested）
7. **PR 标题前缀分类**（[Feat] / [Bugfix] / [Refactor] / [Doc] / [Test] / [CI]）

#### 4.3.6 理解债管理（语义锚定）

> InfoQ 2026 提出的新概念。针对 vibecoding 特有债务。

**强制为 AI 生成代码添加结构化注释**：

```python
# @ai-prompt: "Generate idempotent user profile sync"
# @ai-model: deepseek-v3-2026-06
# @ai-constraint: Must not mutate shared state
# @ai-test-gap: Missing test for concurrent sync + delete race
def sync_user_profile(ctx, user): ...
```

**四个锚点的作用**：

| 锚点 | 解决什么理解债 | 说明 |
|------|--------------|------|
| `@ai-prompt` | **意图丢失** | 记录"当时想让 AI 干什么"，避免 3 个月后无人知道为何这么实现 |
| `@ai-model` | **复现性 / 复盘** | 出问题时追溯到具体模型版本（不同模型行为差异大） |
| `@ai-constraint` | **边界不清** | 记录"哪些是不能违反的硬约束"，区分硬约束与 AI 自由发挥 |
| `@ai-test-gap` | **测试假阳性** | 主动声明"哪里还没测"，防止 AI 写的测试"形似神不似" |

**实测效果**：首次修改耗时 42 分钟 → 11 分钟；回归缺陷率 31% → 6.2%。

### 4.4 Layer 3：Refactoring 预算（20% 容量）

> **业界共识**：每个团队应该把大约 **20% 的开发容量（人力时间）专门留给还技术债**。
> 来源：Martin Fowler《重构》+ 《Software Engineering at Google》。

**翻译成本团队场景**：50+ 人团队，每周 2000+ 人时总量，**20% = 400+ 人时/周**，相当于 **10+ 个全职开发者专门做重构**。

**落地三件套**：

1. **Tech-debt Backlog**：像产品需求一样管理技术债（GitCode Issues + tech-debt 标签）
2. **每 sprint 强制还 1 项**：硬规则，不允许"功能永远优先"
3. **每季度 1 周"架构整理周"**：全员还债，参考 Google Spring Cleanup / Microsoft Bug Bash

---

## 五、整体架构示意

```text
┌─────────────────────────────────────────────────────────────┐
│                    项目负责人（Tech Lead）                    │
│                       架构仲裁人                              │
├─────────────────────────────────────────────────────────────┤
│  架构 SIG（仲裁层）                                            │
├─────────────────────────────────────────────────────────────┤
│ 10 个领域 SIG：模型适配 / ServingCast / Throughput ...        │
│ 每个 SIG：一主一备一 approver                                  │
│ 模块契约：__all__ + import-linter                              │
├─────────────────────────────────────────────────────────────┤
│  Layer A：SDD + Spec-First Review（事前审设计，事后审合规）     │
│  Layer B：PR 治理（CI 门禁 + AI 辅助检视 Qodo + DeepSeek）    │
│  Layer C：AGENTS.md（统一 AI 约束）                            │
│  Layer D：理解债管理（语义锚定）                                │
├─────────────────────────────────────────────────────────────┤
│  Refactoring 预算 20%（每 sprint 还 1 项 + 季度架构周）        │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、落地路线图

### P0 立即启动（1-2 周内）

1. **成立 11 个 SIG**，明确主 / 备 / approver 名单
2. **PR 模板强制三段式**（What / Why / How tested）
3. **AGENTS.md 升级**（参考 vllm-ascend 442 行级别）
4. **CI 门禁上线**：diff coverage ≥ 90% / 圈复杂度 ≤ 15 / DCO 签名

### P1 一个月内

1. **核心模块走 Spec-First**（定义核心模块清单）
2. **Qodo PR-Agent 自托管部署** + DeepSeek V3 接入
3. **.importlinter.ini 配置** + 模块依赖方向图
4. **Tech-debt Backlog 建立**

### P2 一个季度内

1. **双签机制落地**（主提 PR 备必审）
2. **RFC 闸门硬规则化**（> 200 行或跨 3 目录必须 RFC）
3. **Refactoring 预算 20% 强制执行**
4. **理解债锚定**（AI 代码强制 @ai-prompt / @ai-constraint）

### P3 半年内

1. **Mutation Testing**（核心模块跑 mutmut）
2. **架构腐化季度盘点**（import-linter + 复杂度趋势）
3. **SIG Chair 轮换机制**（1-2 年轮换，防领地化）

---

## 七、关键认知总结

1. **问题不是"AI 写的代码不好"**，而是约束从文档形式转为可执行形式的迁移没完成。

2. **业界正在分化为三类项目**：
   - 完全闭源 / 关 PR（cURL / Ghostty / tldraw）—— 极端保守
   - SDD + 多重门禁（GitHub 官方推荐）—— **主流方向，我们应走这条**
   - 继续放任 vibe coding —— 必腐化

3. **PR 大小不是问题**，评审单元与工作单元的错配才是问题。解法是 Spec-First Review，不是单纯限行数。

4. **责任田划分方向正确**，但要补：
   - 一主一备一 approver（防 bus factor = 1）
   - 主提 PR 备必审（防责任集中）
   - __all__ + import-linter（防边界穿透）

5. **自动化是唯一可持续方案**：CI 门禁 + AI 辅助检视 + AGENTS.md + 理解债锚定，四件套缺一不可。

6. **Refactoring 预算 20%** 是业界共识，规模翻 7 倍后无此预算必腐化。

7. **华为 vllm-ascend 是最佳参照**：拥抱 AI 但强约束（AGENTS.md 442 行 + PR 三段式 + Review Checklist + DCO）。

---

## 八、决策事项

请会议就以下事项表决：

| 序号 | 决策事项 | 推荐 |
|------|---------|------|
| D1 | 是否成立 11 个 SIG + 一主一备一 approver 制 | ✅ |
| D2 | 是否升级 AGENTS.md 到 vllm-ascend 级别 | ✅ |
| D3 | 是否启动 Spec-First Review 流程 | ✅ |
| D4 | 是否部署 Qodo PR-Agent 自托管 + DeepSeek V3 | ✅ |
| D5 | 是否启用 .importlinter.ini 模块契约 | ✅ |
| D6 | 是否强制 Refactoring 预算 20% | ✅ |
| D7 | 是否建立 Tech-debt Backlog + 季度架构周 | ✅ |

---

## 附录 A：参考案例与数据来源

### 业界关闸门事件（2025-2026）

- cURL 关闭漏洞赏金计划
- Ghostty 禁止未审核 AI 代码 + Vouch 系统
- tldraw 关闭所有外部 PR
- NetBSD 禁止 AI 代码
- Linux 7.1 RC4 Linus Torvalds 公开信
- GitHub 平台推出"禁用 PR"功能

### 业界共识数据

- 78% maintainer 认为 PR 体积规则失效（Stack Overflow 2026）
- AI 代码缺陷率 1.7× / 漏洞率 2.74×（CodeRabbit / Veracode 2025）
- 93.6% 项目第 3 次迭代后失控（127 万行熵值分析）
- Refactoring 20% 预算（Martin Fowler + Google SWE）

### 参考开源治理

- 华为 vllm-ascend：AGENTS.md + PR 三段式 + DCO
- Linux kernel：Subsystem Maintainer 树
- Kubernetes：SIG 治理 + SIG-Architecture 仲裁
- GitHub spec-kit：SDD 8 步流程（⭐ 98.5K）

### 工具选型

- Qodo PR-Agent（自托管）+ DeepSeek V3
- import-linter / archunit-py（架构 Fitness Function）
- diff-cover / SonarQube New Code gate
- mutmut（Mutation Testing）

---

*本方案综合业界 2025-2026 公开数据、案例与最佳实践形成。建议会议表决后立即启动 P0 项。*
