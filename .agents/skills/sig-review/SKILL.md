---
name: sig-review
description: msmodeling SIG 化代码检视技能。PR 作者说"请求检视"或"启动合入"即在 PR 评论区评论 /merge，由常驻服务 后台合入管理服务 自动路由 SIG、分配 reviewer、监控 CI/lgtm/approve 和催审；检视者说"检视PR {number}"自动分析 diff 并提交意见，通过时评论 /lgtm；说"分析PR {number}的检视意见"自动拉取 diff_comment 评论并逐条分析合理性、给出修改建议；还支持查看待检视列表、查看状态、转交、完成检视。不手动指派 assignee，SIG 路由由后台服务完成。适用于 cursor/claude code/opencode/codex 等各类 agent。
metadata:
  version: 3.0.0
  source: mergetrack-handoff
---

# SIG PR 代码检视

## 技能概述

本技能是 msmodeling 项目 **SIG 化代码检视工具**，让 PR 作者和检视者通过自然语言与 AI agent 交互完成检视全流程，无需手动操作 GitCode 网页。

msmodeling 按目录划分为 10 个子 SIG，每个 SIG 有明确的 chair（负责人）、reviewer（备审）、approver（评审）。**合入流程的启动（SIG 路由、reviewer 分配、CI/lgtm/approve 标签监控、催审）由后台运行的合入管理工具 MergeTrack 驱动**：只要在 PR 评论区评论 `/merge`，后台合入管理服务即自动接手。本技能不再手动指派 assignee 或打 `sig:XXX` 标签——SIG 路由由后台服务完成，agent 不获取变更文件、不计算 SIG 归属。本技能聚焦三件事：①评论 `/merge` 启动合入；②reviewer 实际检视并提交意见、通过时评论 `/lgtm`；③分析已有检视意见并给出修改建议。

**核心能力：**

| 能力 | 说明 | 谁用 | 怎么触发 |
|------|------|------|---------|
| 启动合入 | 评论 `/merge`，后台合入管理服务 自动路由 SIG、分配 reviewer、监控 CI/lgtm/approve 和催审 | PR 作者或任何成员 | "请求检视" / "启动合入" |
| 查看待检视 | 列出当前分配给自己的所有待检视 PR | 检视者 | "我有哪些待检视PR" |
| 代码检视 | 获取 PR 代码变更，分析问题，提交结构化检视意见 | 检视者 | "检视PR 123" |
| 查看状态 | 快速查看 PR 的审查人、标签、状态 | 任何人 | "PR 123 状态" |
| 转交检视 | 通知其他 reviewer 接手（不改 assignee，assignee 由 后台合入管理服务 管理） | 当前检视者 | "转给 XXX" |
| 完成检视 | 提交检视结论。通过则评论 `/lgtm`（后台合入管理服务 自动通知 approver），有意见则提交行内意见后等作者修改重评 | 检视者 | "完成检视" |
| 分析检视意见 | 拉取 PR 的 diff_comment 评论，逐条分析合理性并给出修改建议 | PR 作者 / 检视者 | "分析PR 123的检视意见" |

**当用户询问"你能做什么"或"这个技能是干什么的"时，按以下方式回答：**

> 这是 msmodeling 项目的 SIG 化代码检视工具。msmodeling 分为 10 个子 SIG，每个 SIG 有明确的负责人和评审人。合入流程的启动由常驻服务 后台合入管理服务 驱动——评论 `/merge` 即自动路由 SIG、分配 reviewer、监控 CI/lgtm/approve 和催审。这个工具能帮你完成检视全流程：
>
> - **你是 PR 作者**：推送代码后说"请求检视"或"启动合入"，我在 PR 评论区评论 `/merge`，后台合入管理服务 自动分配 reviewer 并飞书通知
> - **你是检视者**：说"我有哪些待检视PR"查看任务，说"检视PR 123"开始检视。检视完如果没问题说"完成检视"，我评论 `/lgtm`，后台合入管理服务 自动通知 approver；如果有修改意见，提交行内意见后等作者修改重评
> - **想转给别人**：说"转给 XXX"，我通知对方接手（不改 assignee，由 后台合入管理服务 管理）
> - **想看状态**：说"PR 123 状态"即可
> - **收到检视意见想分析**：说"分析PR 123的检视意见"，自动拉取所有 diff_comment 评论，逐条分析是否合理、该怎么改
>
> 首次使用需要配置 GitCode 令牌（一次性操作），之后全程自然语言交互，不需要操作 GitCode 网页。

## 适用场景

本技能覆盖 SIG 检视全流程（除常驻 watch 外），支持以下工作流：

| 工作流 | 触发者 | 触发词 | 命令 |
|--------|--------|--------|------|
| **启动合入** | PR 作者或任何成员 | "请求检视" / "启动合入" | `gitcode pr comment "/merge"` |
| **查看待检视** | reviewer / chair | "我有哪些待检视PR" | `gitcode pr list` |
| **查看状态** | 任何人 | "PR 123 状态" | `gitcode pr view --json` |
| **代码检视** | reviewer / chair | "检视PR {number}" | `gitcode pr diff` → `gitcode pr comment` |
| **转交检视** | 当前检视者 | "转给 XXX" | 通知目标 reviewer（不改 assignee） |
| **完成检视** | reviewer（非 PR 作者） | "完成检视" | `gitcode pr comment "/lgtm"` |
| **分析检视意见** | PR 作者 / 检视者 | "分析PR {number}的检视意见" | `gitcode pr comments` → 分析 |

**典型流程：**

1. PR 作者推送代码后对 agent 说"请求检视" → 评论 `/merge`，后台合入管理服务 自动路由 SIG、分配 reviewer、飞书通知
2. reviewer 收到飞书通知 → 对 agent 说"我有哪些待检视PR" → 列出任务
3. reviewer 对 agent 说"检视PR {number}" → 自动分析 diff 并提交检视意见 → 有修改意见则等作者修改后重评
4. 作者修改后再次"请求检视" → 重评 `/merge`，后台合入管理服务 重新通知 reviewer → 循环 2-3 直到无问题
5. reviewer 检视通过 → "完成检视" → 评论 `/lgtm`，后台合入管理服务 自动通知 approver
6. （可选）reviewer 忙不过来 → "转给 {reviewer 用户名}" → 通知对方接手

## 前置条件

- **gitcode CLI 已安装并认证**：

```bash
gitcode version
gitcode auth status
```

认证由 gitcode CLI 全局管理，令牌保存在 git credential store，本技能不单独保存 Token、不直接访问 GitCode API。

- 本地有 msmodeling 仓库克隆（用于阅读完整文件、理解上下文，使检视更深入）

## 硬性规则：禁止用 git diff 获取 PR 变更

> **PR 代码变更的唯一来源是 `gitcode pr diff` 返回的 diff。**
>
> **禁止执行的命令（用于获取 PR 变更时）：**
> - `git diff` — 会因本地分支状态、merge、rebase 等因素识别出大量与 PR 无关的变更
> - `git log -p` / `git show <commit>` — 同样会产生不准确的变更集
>
> **原因：** `git diff` 依赖本地工作区状态，经常因分支未更新、merge 残留、rebase 等原因产生大量无关 diff，导致检视准确性下降、token 消耗激增。GitCode API 返回的 diff 是服务端权威数据，精确对应 PR 的实际变更。
>
> **允许且鼓励的做法：**
> - 执行 `gitcode pr diff <PR编号> -R <TARGET_REPO>` 获取 PR 变更
> - 读取本地仓库文件**理解上下文**（如完整函数定义、类结构、导入关系），使检视更深入
> - `git fetch` / `git pull` 同步本地仓库到最新 master，保证上下文准确

## 默认策略

- **PR 变更的唯一来源是 `gitcode pr diff` 返回的 diff，禁止用 `git diff` 获取变更**
- 本地代码仓用于理解上下文：需要看完整函数、类定义、调用关系时可读取本地文件，但变更内容以 diff 为准
- 只关注 PR 新增代码（diff 中 `+` 开头的行），不审查未修改的上下文代码
- 测试代码（`tests/` 目录下）简单检视：只查明显逻辑错误、断言缺失、边界遗漏，跳过风格和架构问题
- 忽略纯格式问题（由 pre-commit 负责）
- 评论措辞委婉，使用"请考虑"、"建议"等表达

## 启动合入流程（请求检视）

当 PR 作者推送代码后说"请求检视"或"启动合入"，agent 在 PR 评论区评论 `/merge` 启动后台合入管理服务。SIG 路由、reviewer 分配、催审等均由后台服务完成，**agent 不需要获取变更文件、不计算 SIG 归属、不展示 SIG 信息**。

> **`/merge` 由用户主动请求触发，agent 不自动发起**。若 agent 所处工作流（如 issue-delivery）需要推进合入，必须**先确保 CI 全绿**再评论 `/merge`——CI 未通过时先走 `msmodeling-ci-recovery` 修复至全绿，然后再评论 `/merge`。

### 命令

```bash
# 1. 前置检查：确认当前 head CI 已全绿（ci-pipeline-passed / docs-ci-pipeline-success label）
gitcode pr view <PR编号> -R <TARGET_REPO> --json

# 2. 在 PR 评论区评论 /merge 启动合入流程
#    /merge 必须单独成行（后台工具按行精确匹配整行 == "/merge"）
gitcode pr comment <PR编号> -R <TARGET_REPO> --body "/merge"
```

> **不执行** `gitcode api PATCH .../pulls/<PR> -f assignee=...` 指派 chair，**不打** `sig:XXX` 标签——assignee 分配和标签由后台合入管理服务统一完成。agent 只做一件事：评论 `/merge`。

### 做的事

1. **前置检查 CI 已通过**（`ci-pipeline-passed`/`docs-ci-pipeline-success` label）；未通过则先走 `msmodeling-ci-recovery` 修复至全绿
2. 在 PR 评论区评论 `/merge`
3. 后台合入管理服务检测到 `/merge` 后自动：路由 SIG、分配 reviewer、飞书通知 reviewer、监控 CI/lgtm/approve、催审

### 幂等

若 PR 评论区已存在 `/merge` 评论且 后台合入管理服务 已启动（PR 已有 reviewer 分配迹象），提示而不重复评论。

### Agent 输出要求

评论 `/merge` 后，向用户报告：

```
PR #123 已评论 /merge，启动合入流程：
- 后台合入管理服务将自动路由 SIG、分配 reviewer 并飞书通知，无需手动指派
```

## 检视流程

> 以下命令在仓库根目录或任意目录执行，`-R <TARGET_REPO>` 指定目标仓库。

### Step 0: 环境准备

**首先检查令牌是否已配置**（运行任意命令即可检测）：

```bash
gitcode auth status 2>&1 | head -1
```

**如果返回正常 JSON**（PR 列表或空列表）→ 令牌已配置，继续后续步骤。

**如果报错"令牌未配置"** → 这是首次使用，agent 必须主动引导用户配置，不要让用户自己摸索：

> 检测到 GitCode 令牌未配置。这是一次性操作，配置后持久生效。
>
> **请在你自己的终端中运行以下命令**（不要在此对话中粘贴令牌，以保护安全）：
>
> ```bash
> cd <msmodeling 仓库根>
> gitcode auth login  # 交互式配置，令牌保存在 git credential
> ```
>
> 令牌获取方式：GitCode → 设置 → 私人令牌 → 生成新令牌（需要 repo 读写权限）。
>
> 配置完成后告诉我，我会继续。

**用户确认配置完成后**，重新执行 `list` 验证，然后继续后续步骤。

> **不要**让用户在对话中直接粘贴令牌。**不要**尝试用 `export GITCODE_TOKEN=xxx` 在 shell 中设置（会在对话历史和 shell 历史中留痕）。

同步本地仓库到最新 master（用于阅读完整文件、理解上下文）：

```bash
git -C <msmodeling 仓库根> fetch origin
git -C <msmodeling 仓库根> checkout master
git -C <msmodeling 仓库根> pull --ff-only origin master
```

> `<msmodeling 仓库根>` 即技能目录上三级（`.agents/skills/sig-review` 的上三级）。
> 若有未提交改动，先 `git stash` 再同步，同步后按需 `git stash pop`。
>
> **注意：** 此处 `git fetch` / `git pull` 仅用于同步本地代码以提供准确的上下文阅读环境，**不是用来获取 PR 变更**。PR 变更只能通过 Step 1 的 `gitcode pr diff` 获取。**禁止执行 `git diff`。**

### Step 1: 获取 PR 信息

获取 PR 完整信息（详情 + 文件 + diff + 已有评论）：

```bash
gitcode pr view <PR编号> -R <TARGET_REPO> --json
gitcode pr diff <PR编号> -R <TARGET_REPO>
gitcode pr comments <PR编号> -R <TARGET_REPO> --json
```

数据来源：

| 命令 | 提供的信息 |
|------|----------|
| `gitcode pr view --json` | PR 标题、描述、作者、head SHA、标签、审查人、变更行数 |
| `gitcode pr diff` | 变更文件列表和每个文件的 diff（`filename`、`status`、`additions`、`deletions`） |
| `gitcode pr comments --json` | 已有检视评论（用于防重复） |

### Step 2: 理解 PR

1. 分析 PR 标题、描述（`body` 字段），理解作者意图和检视重点
2. 审查变更文件列表，识别变更范围
3. 逐文件阅读 diff，理解每处变更的目的
4. 如需更深入的上下文（如完整函数定义、类结构、调用关系），可读取本地仓库中对应文件的完整内容——但变更内容以 diff 为准
5. 测试代码（`tests/` 目录下）简单检视：只查明显逻辑错误、断言缺失、边界遗漏，跳过风格和架构问题

**大 PR 策略（diff_lines > 500）：**

- 聚焦核心业务模块，优先检视接口定义、配置变更
- 逐文件处理，避免一次性加载所有 diff 导致上下文溢出
- 最多检视 5 个最关键的文件

**文档 PR 策略：**

- 设计文档 / RFC：结合网上信息判断设计合理性
- 其他文档：快速浏览，仅提出明显正确性问题
- diff_lines < 20 的文档 PR 可直接跳过

### Step 3: 生成检视意见

**数量控制：**

| 变更行数 | 最大意见数 |
|---------|-----------|
| < 20 行 | 跳过检视，直接通过 |
| 20 - 100 行 | 1 - 2 个 |
| > 100 行 | 最多 5 个 |

**类别与侧重（按优先级排序）：**

| 类别 | 何时使用 | 示例 |
|------|---------|------|
| 逻辑缺陷 | 代码逻辑有 bug，特定输入下会出错 | 空指针未检查、边界条件遗漏、异常未处理 |
| 性能隐患 | 代码可能导致性能问题 | 热路径中不必要的同步、O(n²) 循环、大对象频繁拷贝 |
| 安全风险 | 代码引入安全漏洞 | 硬编码密钥、SQL 注入、未校验输入 |
| 架构设计 | 设计层面的问题，影响可维护性 | 硬编码判断应改为属性驱动、模块耦合过紧 |
| 代码规范 | 命名、接口设计等规范问题（低优先级） | magic number 应提取为常量 |

**什么应该检视（参考 [检视质量标准](./ref/review-checklist.md)）：**

- 清晰的 bug 和安全问题：彻底检查，即使触发场景窄也不要漏
- 每条意见必须具体、可操作，而非对代码库的泛泛担忧
- 如果不确定但潜在影响大（如数据丢失、安全），可以提出但需明确标注不确定性

**什么不应该检视：**

- 纯格式 / 风格问题（由 pre-commit 负责）
- 代码库其他地方可能已存在的功能（你只看到 diff，不是完整代码库）
- 有意的设计选择，除非引入了明确的缺陷
- 无法确定是问题的"感觉不对"——如果能解释清楚触发场景就提，否则不提

### Step 4: 二次检查（提交前必须执行）

**在提交每条检视意见前，快速检查以下 5 点：**

1. **问题确实存在**：确认指出的问题不是误报，能在 diff 中找到具体代码
2. **行号准确**：`--position` 对应的行必须是 diff 中新增或修改的行（`+` 开头），**绝对不要提交在未修改的上下文行上**
3. **建议可行**：代码建议在实际场景中可执行
4. **语句通顺**：评论语句流畅、表达清晰
5. **措辞得体**：使用委婉表达，避免武断措辞

如有问题，直接修改后再提交。此步骤应在几秒内完成。

### Step 5: 提交检视意见

**短内容（不含代码块）直接传递：**

```bash
gitcode pr comment <PR编号> -R <TARGET_REPO> \
  --path "path/to/file.py" \
  --position 42 \
  # category 逻辑缺陷 (in comment body prefix) \
  --body "【逻辑缺陷】缺少最大重试次数限制，可能导致无限重试，建议添加重试次数上限。"
```

**含代码块的多行内容（推荐方式）：**

将评论内容写入临时文件，再用 `--body-file` 提交。临时文件请写入系统临时目录，提交后删除：

```bash
# 1. 获取系统临时目录（跨平台）
TMPDIR=$(python3 -c "import tempfile; print(tempfile.gettempdir())")

# 2. 写入评论内容（agent 手动添加【review】【类别】前缀）
cat > "$TMPDIR/review_123.md" << 'EOF'
缺少最大重试次数限制，可能导致无限重试，建议添加重试次数上限。代码建议：

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        do_something()
        break
    except Exception:
        if attempt == max_retries - 1:
            raise
```
EOF

# 3. 提交评论
gitcode pr comment 123 -R <TARGET_REPO> \
  --path "path/to/file.py" \
  --position 42 \
  # category 逻辑缺陷 (in comment body prefix) \
  --body-file "$TMPDIR/review_123.md"

# 4. 删除临时文件
rm -f "$TMPDIR/review_123.md"
```

> **重要**：agent 在 `--body` / `--body-file` 的评论正文中手动添加 `【review】【类别】` 前缀。

> **后台合入管理服务 自动流转**：提交行内评论（未解决的 diff_comment）后，后台合入管理服务 检测到开放意见会自动将状态切回 author_rework 并飞书通知作者处理。作者修改后重评 `/merge`，后台合入管理服务 重新通知 reviewer。agent 不手动改 assignee。

**撤回评论（如果发现误报）：**

```bash
# comment_id 是提交评论时返回的 comment_id 字段（数字 ID）
gitcode api DELETE "/repos/<TARGET_REPO>/pulls/comments/<comment_id>"
```

### Step 6: 完成检视

检视意见全部提交后，reviewer 根据是否有修改意见选择完成方式：

**通过（无修改意见或意见已解决）→ 提交检视摘要 + 评论 `/lgtm`：**

```bash
# 1. 提交检视摘要（三项评价），写入临时文件
cat > "$TMPDIR/verdict.md" << 'EOF'
1. 个人理解：本 PR 为 attention 层新增了 KV cache 压缩支持，目的是降低长序列场景的显存占用。
2. 功能评价：压缩逻辑正确，与现有 attention 接口兼容。建议补充 L=0 边界场景的测试。
3. 代码质量：命名清晰，异常处理完整。compress_ratio 提取为常量更好。
EOF

# 2. 提交检视摘要
gitcode pr comment <PR编号> -R <TARGET_REPO> --body-file "$TMPDIR/verdict.md"

# 3. 评论 /lgtm，后台合入管理服务 检测后自动通知 approver
gitcode pr comment <PR编号> -R <TARGET_REPO> --body "/lgtm"
```

**有修改意见 → 提交检视摘要但不评论 `/lgtm`：**

```bash
gitcode pr comment <PR编号> -R <TARGET_REPO> --body-file "$TMPDIR/verdict.md"
```

不评论 `/lgtm`，PR 保持待修改状态。作者看到行内意见后修改，修改完成后再次请求检视。

> **`/lgtm` 与 `/approve` 机制**：reviewer 评论 `/lgtm` → 后台合入管理服务 检测后自动通知 approver 并跟踪 `lgtm` 标签；approver 评论 `/approve` → 后台合入管理服务 检测后标记可合入。两个标签齐了 PR 才允许合入。分配 reviewer/approver 由 后台合入管理服务 基于 `/merge` 自动完成，agent 不手动指派 assignee。

> **注意**：评论 `/lgtm` 是 reviewer 的动作，不是 PR 作者的动作。PR 作者只负责请求检视，reviewer 负责检视和评论 `/lgtm`。

**检视摘要必须包含 SIG 规范要求的三项评价：**

根据 SIG 组织规范，每次检视须显式写出以下三项，缺一不可，否则视为未检视：

1. **对 PR 的个人理解**：用自己的话说明该 PR 做了什么、解决什么问题（一两句即可，禁止复述 diff）
2. **功能 / 业务层面评价**：是否正确实现预期功能、是否引入业务风险、是否存在更优方案
3. **编码与代码质量评价**：命名 / 结构 / 可读性、边界与异常处理、性能与资源占用

**如果只需提交阶段性检视结论但不完成检视：**

```bash
gitcode pr comment <PR编号> -R <TARGET_REPO> --body "阶段性意见：..."
```

**如果需要转给其他 reviewer（转交）：**

不改 assignee（由 后台合入管理服务 管理）。通过飞书或 PR 评论通知目标 reviewer 接手即可，对方检视后评论 `/lgtm` 即完成。

## 分析检视意见

当 PR 作者或检视者说"分析PR {number}的检视意见"，agent 自动拉取该 PR 的所有 diff_comment 评论，逐条分析合理性并给出修改建议。

### 命令

```bash
gitcode pr diff <PR编号> -R <TARGET_REPO> && gitcode pr view <PR编号> -R <TARGET_REPO> --json && gitcode pr comments <PR编号> -R <TARGET_REPO> --json
```

从 `gitcode pr comments` 返回的评论列表中筛选行级检视意见。

### 分析流程

1. **拉取评论**：执行 `gitcode pr comments` 获取 PR 评论列表，筛选行级检视意见
2. **去重**：多位检视者可能提出相同问题，按问题实质（而非措辞）去重，合并为独立问题
3. **逐条分析**：对每个独立问题，结合本地代码仓验证：
   - **问题简介**：一句话概括评论指出的具体问题
   - **是否有道理**：✅ 合理 / ⚠️ 部分合理 / ❌ 不合理，并说明判断依据
   - **改法**：涉及哪些文件、具体怎么改（如需改代码，给出关键片段）
4. **汇总输出**：按优先级排序，输出汇总表

### 输出格式

对每个问题输出：

```
### 问题 N：{问题标题}

**位置**：`文件路径:行号`（如评论未关联行号则标注"全局"）

**问题简介**：{一句话描述评论指出的问题}

**是否有道理**：✅ 合理 / ⚠️ 部分合理 / ❌ 不合理

**判断依据**：{结合代码验证后的分析，引用具体代码说明}

**改法**：{涉及哪些文件、怎么改。如需改代码，给出关键片段}
```

最后输出汇总表：

```
| # | 问题 | 重复数 | 合理性 | 优先级 | 涉及文件 |
|---|------|--------|--------|--------|----------|
| 1 | ... | 3 | ✅ | 高 | (file path) |
```

### Agent 行为要求

- **必须结合本地代码验证**：不能仅凭评论内容判断合理性，必须读取相关代码确认问题是否真实存在
- **去重时按问题实质**：不同检视者可能用不同措辞描述同一问题，应合并为一个独立问题，在"重复数"列标注
- **不合理的评论也要说明原因**：如果评论是误报，说明为什么是误报，引用代码证据
- **改法要具体可操作**：指明文件名、函数名、行号，给出修改后的关键代码片段
- **向用户报告时使用自然语言**，不暴露命令名等技术细节

## 评论格式规范

agent 提交评论时应格式化为：

```
【review】【类别标签】评论正文

（可选代码建议）
```

**代码建议规则：**

1. 逻辑缺陷、性能优化、安全风险类问题**必须提供代码建议**
2. 架构设计类问题可以不提供代码建议，但需说明方向
3. 代码建议可以是伪代码或关键片段，不需要完整代码
4. 简单修改直接给出修改后的关键行即可

**措辞要求：**

- 使用"请考虑"、"建议"、"或许可以"等委婉表达
- 避免"必须"、"应该"、"错误"等武断措辞
- 说明优化的好处，而非仅仅指出问题

**示例（含代码建议）：**

```
【review】【逻辑缺陷】缺少最大重试次数限制，可能导致无限重试，建议添加重试次数上限。代码建议：

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        do_something()
        break
    except Exception:
        if attempt == max_retries - 1:
            raise
```
```

**示例（架构问题，无代码建议）：**

```
【review】【架构设计】这里用 model_type 硬编码判断是否走特定分支，建议改为根据 attention layer 的属性（如 compress_ratio）判断，让其他有相同配置的模型也能复用此逻辑。
```

## 防重复机制

Step 1 通过 `gitcode pr comments` 获取的评论列表 包含该 PR 已有的所有评论。生成新意见前必须：

1. **只关注行级检视评论**（inline comment，带 path 和 position 的），这些是防重复的对象
2. 其他类型的评论（通用 PR 评论等）不参与防重复
3. 检查行级评论中的 `path` 和 `position` 字段，避免在同一文件同一行提出类似意见
4. 避免提出相同观点的不同表述

如果已有行级评论已覆盖某个问题，不要重复提交。

## 模式说明

### 自动检视（默认）

触发词："检视PR {number}"、"review PR {number}"

AI 自动完成全流程（获取 diff → 分析 → 提交意见 → 完成检视），**无需用户等待**。提交前 AI 会显示分析结果摘要（发现了几个问题、分别是什么类别），然后立即提交。用户回来后看到结果，可以撤回、补充或完成检视。

**关键原则：检视永远自动完成，不依赖用户响应。** 即使用户说完"检视PR 123"就去做别的事，检视也会完成。

提交后 AI 告知用户：
> 检视完成，已提交 N 条意见。若有修改意见，后台合入管理服务 会自动通知作者处理。
> 如需调整：说"撤回第 2 条"或"补充一条意见"。
> 如需完成：说"完成检视"，我评论 `/lgtm`，后台合入管理服务 自动通知 approver。
> 下次如想自己引导检视方向，可以说"交互检视PR {number}"。

### 交互检视（可选）

触发词："交互检视PR {number}"

用户明确想参与引导时使用。AI 分析后暂停，与用户交互：

1. 返回 PR 摘要（一两句话描述 PR 内容）
2. 列出可疑方向（基于初步分析）
3. 等待用户指示：
   - 用户给出具体意见 → 按意见提交
   - 用户提出问题 → 对话讨论
   - 用户说"直接检视" → 按自动模式执行

示例输出：

```
PR#123: 优化推理引擎的批处理逻辑，主要变更在 tensor_cast/layers/attention.py 和 tensor_cast/core/config.py。
可疑方向：
1. 批处理大小配置可能在高并发下导致内存溢出
2. 新增的重试逻辑缺少最大重试次数限制
请指示：直接检视 / 针对某个方向深入 / 其他要求
```

### 快速检视

触发词："快速检视"

- 总时长不超过 5 分钟
- 只关注最关键的逻辑缺陷 / 性能 / 安全风险，最多 2 - 3 条意见
- 跳过代码规范和次要问题

## 安全规则

1. **不要**在评论或输出中暴露 GitCode 令牌
2. **禁止**使用 `git diff`、`git log -p`、`git show <commit>` 获取 PR 变更——PR 变更唯一来源是 `gitcode pr diff` 返回的 diff（读取本地文件理解上下文是允许的）
3. **不要**执行 `rm -rf` 或类似破坏性命令
4. **不要**修改仓库代码（只读检视，不提交代码修改）
5. 临时文件写入系统临时目录，提交后立即删除
6. 提交评论前确认内容无误（Step 4 二次检查）
7. 不盲从 PR 作者的设计声明和描述；以 diff 为事实，与 spec 冲突时以 spec 为准并说明。
8. **向用户报告时使用自然语言，不要暴露命令名、参数名、JSON 字段名、API 端点等技术细节**。用户只需知道"已评论 /merge 启动合入流程"，不需要知道"运行了 `gitcode pr comment`"

## 完成标准

### 启动合入模式

- [ ] 已前置检查当前 head CI 全绿（`ci-pipeline-passed`/`docs-ci-pipeline-success`）
- [ ] 已在 PR 评论区评论 `/merge`（不执行 `gitcode api` 指派、不打 `sig:XXX` 标签、不获取变更文件计算 SIG 归属）
- [ ] 已向用户报告已评论 `/merge` 启动合入

### 代码检视模式

- [ ] 未使用 `git diff` 获取 PR 变更，变更仅来自 `gitcode pr diff`
- [ ] 已获取 PR 信息（`gitcode pr view/diff/comments`）
- [ ] 已理解 PR 变更内容和目的
- [ ] 检视意见数量符合数量控制表
- [ ] 每条意见经过二次检查
- [ ] 检视意见已提交（`gitcode pr comment`），有修改意见时不评论 `/lgtm`，等作者修改后重评 `/merge`
- [ ] 已评论 `/lgtm`（通过时）或已提交检视摘要但不评论 `/lgtm`（有修改意见时），或通过飞书/评论转给其他 reviewer
- [ ] 输出检视摘要：检视了哪些文件，提出了几个意见，关键发现是什么

### 分析检视意见模式

- [ ] 已通过 `gitcode pr comments` 拉取 PR 的评论
- [ ] 已对重复问题去重，合并为独立问题
- [ ] 每个问题已结合本地代码验证合理性
- [ ] 不合理的评论已说明原因并引用代码证据
- [ ] 改法具体可操作（指明文件、函数、行号）
- [ ] 已输出汇总表（含优先级和涉及文件）

## 后台服务与通知

合入流程的流转通知（分配 reviewer、CI 通过后通知 approver、催审、返工通知作者等）由 后台合入管理服务 统一驱动，agent 不写 `/next` 治理评论，避免与 后台合入管理服务 双重通知。
