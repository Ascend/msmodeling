---
name: sig-review
description: msmodeling SIG 化代码检视技能，覆盖从"请求检视"到"完成移交"的全流程。PR 作者说"请求检视"自动路由 SIG 并指派 chair；检视者说"检视PR {number}"自动分析 diff 并提交意见；说"分析PR {number}的检视意见"自动拉取 diff_comment 评论并逐条分析合理性、给出修改建议；还支持查看待检视列表、查看状态、责任传递、完成移交。适用于 cursor/claude code/opencode/codex 等各类 agent。
metadata:
  version: 1.1.0
  source: sig-workflow
---

# SIG PR 代码检视

## 技能概述

本技能是 msmodeling 项目 **SIG 化代码检视工具**，让 PR 作者和检视者通过自然语言与 AI agent 交互，完成从"请求检视"到"检视完成移交"的全流程，无需手动操作 GitCode 网页。

msmodeling 按目录划分为 10 个子 SIG，每个 SIG 有明确的 chair（负责人）、reviewer（备审）、approver（评审）。本技能自动将 PR 路由到正确的 SIG 并指派检视人员，确保每个 PR 有人及时审查、责任传递不断链。

**核心能力：**

| 能力 | 说明 | 谁用 | 怎么触发 |
|------|------|------|---------|
| 请求检视 | 自动分析 PR 变更文件归属哪个 SIG，指派对应 chair | PR 作者 | "请求检视" |
| 查看待检视 | 列出当前分配给自己的所有待检视 PR | 检视者 | "我有哪些待检视PR" |
| 代码检视 | 获取 PR 代码变更，分析问题，提交结构化检视意见 | 检视者 | "检视PR 123" |
| 查看状态 | 快速查看 PR 的审查人、标签、状态 | 任何人 | "PR 123 状态" |
| 转交检视 | 将检视责任转给其他人 | 当前检视者 | "转给 XXX" |
| 完成移交 | 提交检视结论。通过则移交给 approver，有意见则转回作者修改 | 检视者 | "完成检视" |
| 分析检视意见 | 拉取 PR 的 diff_comment 评论，逐条分析合理性并给出修改建议 | PR 作者 / 检视者 | "分析PR 123的检视意见" |

**当用户询问"你能做什么"或"这个技能是干什么的"时，按以下方式回答：**

> 这是 msmodeling 项目的 SIG 化代码检视工具。msmodeling 分为 10 个子 SIG，每个 SIG 有明确的负责人和评审人。这个工具能帮你完成检视全流程：
>
> - **你是 PR 作者**：推送代码后说"请求检视"，自动分析你的变更文件属于哪个 SIG，指派对应负责人审查
> - **你是检视者**：说"我有哪些待检视PR"查看任务，说"检视PR 123"开始检视。检视完如果没问题说"完成检视，移交给 approver XXX"；如果有修改意见，提交意见后 PR 会自动转回作者，作者改完会再次请求检视
> - **想转给别人**：说"转给 XXX"即可
> - **想看状态**：说"PR 123 状态"即可
> - **收到检视意见想分析**：说"分析PR 123的检视意见"，自动拉取所有 diff_comment 评论，逐条分析是否合理、该怎么改
>
> 首次使用需要配置 GitCode 令牌（一次性操作），之后全程自然语言交互，不需要操作 GitCode 网页。

## 适用场景

本技能覆盖 SIG 检视全流程（除常驻 watch 外），支持以下工作流：

| 工作流 | 触发者 | 触发词 | 命令 |
|--------|--------|--------|------|
| **分配检视** | PR 作者 | "请求检视" | `assign` |
| **查看待检视** | reviewer / chair | "我有哪些待检视PR" | `list` |
| **查看状态** | 任何人 | "PR 123 状态" | `status` |
| **代码检视** | reviewer / chair | "检视PR {number}" | `fetch` → `comment` |
| **责任传递** | 当前 assignee | "转给 XXX" | `handoff` |
| **完成移交** | reviewer（非 PR 作者） | "完成检视" | `complete` |
| **分析检视意见** | PR 作者 / 检视者 | "分析PR {number}的检视意见" | `fetch` → 分析 |

**典型流程：**

1. PR 作者推送代码后对 agent 说"请求检视" → 自动分析文件并指派 chair
2. chair 收到 GitCode 站内信通知 → 对 agent 说"我有哪些待检视PR" → 列出任务
3. chair 对 agent 说"检视PR {number}" → 自动分析 diff 并提交检视意见 → **PR 自动转回作者修改**
4. 作者修改后再次"请求检视" → 重新指派 chair → 循环 2-3 直到无问题
5. chair 检视通过 → "完成检视，移交给 approver {用户名}" → 提交结论 + 指派 approver
6. （可选）chair 忙不过来 → "转给 {reviewer 用户名}" → 责任传递

## 前置条件

- **GitCode 令牌已配置**（首次使用前执行一次，持久生效）：

```bash
# 在 msmodeling 仓库根目录执行
# 方式 1：直接指定令牌（会在 shell 历史留痕）
python3 .agents/skills/sig-review/scripts/review_api.py auth --token <你的GitCode令牌>

# 方式 2：从 stdin 读取（不在 shell 历史留痕，推荐）
echo '<你的GitCode令牌>' | python3 .agents/skills/sig-review/scripts/review_api.py auth --stdin

# 方式 3：交互式输入（隐藏输入）
python3 .agents/skills/sig-review/scripts/review_api.py auth
```

令牌保存在 `~/.config/sig-review/config.json`（权限 600），后续所有命令自动读取，无需设置环境变量。也支持 `GITCODE_TOKEN` 环境变量作为备选。

- Python 3.8+（脚本仅依赖标准库，零外部依赖）
- 本地有 msmodeling 仓库克隆（用于阅读完整文件、理解上下文，使检视更深入）

## 硬性规则：禁止用 git diff 获取 PR 变更

> **PR 代码变更的唯一来源是 `review_api.py fetch` 返回的 `patch` 字段。**
>
> **禁止执行的命令（用于获取 PR 变更时）：**
> - `git diff` — 会因本地分支状态、merge、rebase 等因素识别出大量与 PR 无关的变更
> - `git log -p` / `git show <commit>` — 同样会产生不准确的变更集
>
> **原因：** `git diff` 依赖本地工作区状态，经常因分支未更新、merge 残留、rebase 等原因产生大量无关 diff，导致检视准确性下降、token 消耗激增。GitCode API 返回的 `patch` 是服务端权威数据，精确对应 PR 的实际变更。
>
> **允许且鼓励的做法：**
> - 执行 `python3 review_api.py fetch <PR编号>` 获取 PR 变更（`patch` 字段）
> - 读取本地仓库文件**理解上下文**（如完整函数定义、类结构、导入关系），使检视更深入
> - `git fetch` / `git pull` 同步本地仓库到最新 master，保证上下文准确

## 默认策略

- **PR 变更的唯一来源是 `review_api.py fetch` 返回的 `patch` 字段，禁止用 `git diff` 获取变更**
- 本地代码仓用于理解上下文：需要看完整函数、类定义、调用关系时可读取本地文件，但变更内容以 `patch` 为准
- 只关注 PR 新增代码（patch 中 `+` 开头的行），不审查未修改的上下文代码
- 测试代码（`tests/` 目录下）简单检视：只查明显逻辑错误、断言缺失、边界遗漏，跳过风格和架构问题
- 忽略纯格式问题（由 pre-commit 负责）
- 评论措辞委婉，使用"请考虑"、"建议"等表达

## 分配检视（请求检视）

当 PR 作者推送代码后说"请求检视"，agent 执行 SIG 路由和 assignee 指派。

### 命令

```bash
# 先用 --dry-run 预览分析结果（不实际指派）
python3 review_api.py assign <PR编号> --dry-run

# 确认后实际指派
python3 review_api.py assign <PR编号>
```

### 脚本做的事

1. 获取 PR 变更文件列表
2. 逐文件匹配 SIG 目录归属表（最长前缀匹配 + fallback 兜底）
3. 特例处理：
   - **chair == PR 作者** → 改指派 reviewer
   - **跨 SIG**（文件分属多个 SIG）→ 指派多个 chair，打 `cross-sig` 标签
   - **根目录未匹配文件**（如 `new_root_tool.py`）→ 标记需架构 SIG 共审
4. 指派 chair（GitCode 自动发站内信通知）
5. 打 `sig:XXX` 标签

### 幂等

若 PR 已有 assignee，脚本提示而不重复指派。

### Agent 输出要求

执行 `assign` 后，向用户报告：

```
PR #123 分析结果：
- 变更文件：5 个
- 匹配 SIG：模型适配（chair: ChenHuiwen）
- 已指派：ChenHuiwen
- 已添加标签：sig:模型适配
- ChenHuiwen 将收到 GitCode 站内信通知
```

跨 SIG 时：

```
PR #123 分析结果：
- 变更文件：8 个
- 匹配 SIG：ServingCast（chair: yuyinkai1）、Throughput寻优（chair: jia_ya_nan）
- 跨 SIG 双签，已指派：yuyinkai1, jia_ya_nan
- 已添加标签：sig:ServingCast, sig:Throughput寻优, cross-sig
```

有未匹配文件时，脚本会自动尝试 fallback 归类（按顶层目录推断 SIG）。fallback 匹配的文件追加：

```
ℹ️ 有 1 个文件未在 SIG 归属表中，已自动归类：
  tensor_cast/new_module/foo.py → 模型适配（fallback 推断）
  建议：将 tensor_cast/new_module/ 目录归属到模型适配 SIG
```

仅当文件连 fallback 都无法匹配时（如根目录的全新文件），才追加：

```
⚠️ 有 1 个文件无法归类：totally_new_file.py
根目录文件需架构 SIG 共审
```

### 未匹配文件的处理策略

| 情况 | 处理 | 说明 |
|------|------|------|
| 文件在已知 SIG 的子目录下但未显式登记 | **自动归类**（fallback 推断）+ 建议补录 | 按顶层目录推断 SIG，出争议再调整 |
| 文件不在任何顶层目录下 | 标记需架构 SIG 共审 | 无法自动推断，需人工判断 |

**原则：优先自动化，出争议再找 SIG。** fallback 匹配的文件会正常指派，chair 如认为不属于自己 SIG 可用 `handoff` 转交。

## 检视流程

> 以下命令均在技能的 `scripts/` 目录下执行（即 `.agents/skills/sig-review/scripts/`）。

### Step 0: 环境准备

**首先检查令牌是否已配置**（运行任意命令即可检测）：

```bash
python3 review_api.py list 2>&1 | head -1
```

**如果返回正常 JSON**（PR 列表或空列表）→ 令牌已配置，继续后续步骤。

**如果报错"令牌未配置"** → 这是首次使用，agent 必须主动引导用户配置，不要让用户自己摸索：

> 检测到 GitCode 令牌未配置。这是一次性操作，配置后持久生效。
>
> **请在你自己的终端中运行以下命令**（不要在此对话中粘贴令牌，以保护安全）：
>
> ```bash
> cd <msmodeling 仓库根>
> echo '<你的GitCode令牌>' | python3 .agents/skills/sig-review/scripts/review_api.py auth --stdin
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
> **注意：** 此处 `git fetch` / `git pull` 仅用于同步本地代码以提供准确的上下文阅读环境，**不是用来获取 PR 变更**。PR 变更只能通过 Step 1 的 `review_api.py fetch` 获取。**禁止执行 `git diff`。**

### Step 1: 获取 PR 信息

获取 PR 完整信息（详情 + 文件 + diff + 已有评论）：

```bash
python3 review_api.py fetch <PR编号>
```

返回 JSON 包含：

| 字段 | 说明 |
|------|------|
| `title`, `body`, `author` | PR 标题、描述、作者 |
| `head_sha` | PR 最新提交 SHA（提交评论时自动使用） |
| `labels`, `assignees` | PR 标签、审查人 |
| `diff_lines` | 总变更行数（新增 + 删除） |
| `files` | 变更文件列表，每个含 `filename`、`status`、`additions`、`deletions`、`patch` |
| `existing_comments` | 已有检视评论（用于防重复） |

### Step 2: 理解 PR

1. 分析 PR 标题、描述（`body` 字段），理解作者意图和检视重点
2. 审查变更文件列表，识别变更范围
3. 逐文件阅读 `patch` 字段，理解每处变更的目的
4. 如需更深入的上下文（如完整函数定义、类结构、调用关系），可读取本地仓库中对应文件的完整内容——但变更内容以 `patch` 为准
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
2. **行号准确**：`--line` 对应的行必须是 diff 中新增或修改的行（`+` 开头），**绝对不要提交在未修改的上下文行上**
3. **建议可行**：代码建议在实际场景中可执行
4. **语句通顺**：评论语句流畅、表达清晰
5. **措辞得体**：使用委婉表达，避免武断措辞

如有问题，直接修改后再提交。此步骤应在几秒内完成。

### Step 5: 提交检视意见

**短内容（不含代码块）直接传递：**

```bash
python3 review_api.py comment <PR编号> \
  --file "path/to/file.py" \
  --line 42 \
  --category "逻辑缺陷" \
  --content "缺少最大重试次数限制，可能导致无限重试，建议添加重试次数上限。"
```

**含代码块的多行内容（推荐方式）：**

将评论内容写入临时文件，再用 `--content-file` 提交。临时文件请写入系统临时目录，提交后删除：

```bash
# 1. 获取系统临时目录（跨平台）
TMPDIR=$(python3 -c "import tempfile; print(tempfile.gettempdir())")

# 2. 写入评论内容（注意：content 不需要包含【review】【类别】前缀，脚本会自动添加）
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
python3 review_api.py comment 123 \
  --file "path/to/file.py" \
  --line 42 \
  --category "逻辑缺陷" \
  --content-file "$TMPDIR/review_123.md"

# 4. 删除临时文件
rm -f "$TMPDIR/review_123.md"
```

> **重要**：`--content` / `--content-file` 提供的是评论正文，脚本会自动添加 `【review】【类别】` 前缀。不要在内容中重复包含前缀。

> **自动移交**：每提交一条检视意见，脚本会自动将 PR 责任人移回 PR 作者。作者收到 GitCode 站内信通知"有新的检视意见需要处理"。SLA 24h 计时清零，等作者修改后再次"请求检视"时重新计时。

**撤回评论（如果发现误报）：**

```bash
# comment_id 是提交评论时返回的 comment_id 字段（数字 ID）
python3 review_api.py withdraw <comment_id>
```

### Step 6: 完成检视

检视意见全部提交后，reviewer 根据是否有修改意见选择完成方式：

**通过（无修改意见或意见已解决）→ 指派 approver：**

```bash
python3 review_api.py complete <PR编号> --to <approver用户名> --event approved --body "..."
```

**有修改意见 → 移交回作者修改（不指派 approver）：**

```bash
python3 review_api.py complete <PR编号> --event comment --body "..."
```

> `--event approved` 时 `--to` 必填（指定 approver），approver 收到站内信通知。
> `--event comment` 时不需要 `--to`，PR 自动转回给作者修改，作者收到站内信通知。

> **注意**：`complete` 是 reviewer 的动作，不是 PR 作者的动作。PR 作者只负责 `assign`，reviewer 负责检视和 `complete`。

**检视摘要（--body）必须包含 SIG 规范要求的三项评价：**

根据 SIG 组织规范，每次检视须显式写出以下三项，缺一不可，否则视为未检视：

1. **对 PR 的个人理解**：用自己的话说明该 PR 做了什么、解决什么问题（一两句即可，禁止复述 diff）
2. **功能 / 业务层面评价**：是否正确实现预期功能、是否引入业务风险、是否存在更优方案
3. **编码与代码质量评价**：命名 / 结构 / 可读性、边界与异常处理、性能与资源占用

示例：

```bash
python3 review_api.py complete 123 --to lutean --event approved --body-file "$TMPDIR/verdict.md"
```

其中 `verdict.md` 内容：

```
1. 个人理解：本 PR 为 attention 层新增了 KV cache 压缩支持，目的是降低长序列场景的显存占用。
2. 功能评价：压缩逻辑正确，与现有 attention 接口兼容。建议补充 L=0 边界场景的测试。
3. 代码质量：命名清晰，异常处理完整。compress_ratio 提取为常量更好。
```

**如果只需提交检视结论但不移交 approver**（如中途提交阶段性意见）：

```bash
python3 review_api.py verdict <PR编号> --event comment --body "阶段性意见：..."
```

**如果需要转给其他 reviewer**（责任传递）：

```bash
python3 review_api.py handoff <PR编号> --to <reviewer用户名>
```

`handoff` 移除自己的 assignee 并指派新人，GitCode 自动通知新人。

## 分析检视意见

当 PR 作者或检视者说"分析PR {number}的检视意见"，agent 自动拉取该 PR 的所有 diff_comment 评论，逐条分析合理性并给出修改建议。

### 命令

```bash
python3 review_api.py fetch <PR编号>
```

从返回的 `existing_comments` 中筛选 `comment_type == "diff_comment"` 的评论。

### 分析流程

1. **拉取评论**：执行 `fetch` 获取 PR 完整信息，从 `existing_comments` 中筛选 `diff_comment` 类型
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
| 1 | ... | 3 | ✅ | 高 | review_api.py |
```

### Agent 行为要求

- **必须结合本地代码验证**：不能仅凭评论内容判断合理性，必须读取相关代码确认问题是否真实存在
- **去重时按问题实质**：不同检视者可能用不同措辞描述同一问题，应合并为一个独立问题，在"重复数"列标注
- **不合理的评论也要说明原因**：如果评论是误报，说明为什么是误报，引用代码证据
- **改法要具体可操作**：指明文件名、函数名、行号，给出修改后的关键代码片段
- **向用户报告时使用自然语言**，不暴露命令名、脚本文件名等技术细节

## 评论格式规范

脚本自动将提交内容格式化为：

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

Step 1 获取的 `existing_comments` 包含该 PR 已有的所有评论。生成新意见前必须：

1. **只关注 `comment_type` 为 `diff_comment` 的评论**（行级检视意见），这些是防重复的对象
2. 其他类型的评论（通用 PR 评论等）不参与防重复
3. 检查 `diff_comment` 类型评论中的 `path` 和 `position` 字段，避免在同一文件同一行提出类似意见
4. 避免提出相同观点的不同表述

如果已有 diff_comment 已覆盖某个问题，不要重复提交。

## 模式说明

### 自动检视（默认）

触发词："检视PR {number}"、"review PR {number}"

AI 自动完成全流程（获取 diff → 分析 → 提交意见 → 完成移交），**无需用户等待**。提交前 AI 会显示分析结果摘要（发现了几个问题、分别是什么类别），然后立即提交。用户回来后看到结果，可以撤回、补充或完成移交。

**关键原则：检视永远自动完成，不依赖用户响应。** 即使用户说完"检视PR 123"就去做别的事，检视也会完成。

提交后 AI 告知用户：
> 检视完成，已提交 N 条意见，PR 已转回给作者修改。
> 如需调整：说"撤回第 2 条"或"补充一条意见"。
> 如需完成：说"完成检视，移交给 approver XXX"。
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
2. **禁止**使用 `git diff`、`git log -p`、`git show <commit>` 获取 PR 变更——PR 变更唯一来源是 `fetch` 返回的 `patch` 字段（读取本地文件理解上下文是允许的）
3. **不要**执行 `rm -rf` 或类似破坏性命令
4. **不要**修改仓库代码（只读检视，不提交代码修改）
5. 临时文件写入系统临时目录，提交后立即删除
6. 提交评论前确认内容无误（Step 4 二次检查）
7. **向用户报告时使用自然语言，不要暴露命令名、脚本文件名、参数名、JSON 字段名、API 端点等技术细节**。用户只需知道"已分析文件并指派了 chair"，不需要知道"运行了 `review_api.py assign --dry-run`"

## 完成标准

### 分配检视模式

- [ ] 已执行 `assign --dry-run` 预览分析结果
- [ ] 已确认路由结果合理（SIG 匹配正确、assignee 正确）
- [ ] 已执行 `assign` 指派 chair 并打标签
- [ ] 已向用户报告分析结果和指派状态

### 代码检视模式

- [ ] 未使用 `git diff` 获取 PR 变更，变更仅来自 `fetch` 返回的 `patch`
- [ ] 已获取 PR 信息（`fetch`）
- [ ] 已理解 PR 变更内容和目的
- [ ] 检视意见数量符合数量控制表
- [ ] 每条意见经过二次检查
- [ ] 检视意见已提交（`comment`），每条意见提交后 PR 自动转回作者
- [ ] 已用 `complete` 完成（approved 指派 approver，comment 移交回作者），或用 `handoff` 转给其他 reviewer
- [ ] 输出检视摘要：检视了哪些文件，提出了几个意见，关键发现是什么

### 分析检视意见模式

- [ ] 已通过 `fetch` 拉取 PR 的 diff_comment 评论
- [ ] 已对重复问题去重，合并为独立问题
- [ ] 每个问题已结合本地代码验证合理性
- [ ] 不合理的评论已说明原因并引用代码证据
- [ ] 改法具体可操作（指明文件、函数、行号）
- [ ] 已输出汇总表（含优先级和涉及文件）
