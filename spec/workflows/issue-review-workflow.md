# Issue 评审工作流

## 适用范围

评审单个 Issue，或评审当前登录用户负责的开放 Issue。

## 步骤

1. 使用 `gitcode auth status` 确认当前用户，不显示 Token。
2. 使用 `gitcode issue list --state open --assignee <user> --json` 获取候选列表，本地过滤掉已打 `approved`/`need-detail-desc`/`pending`/`wontfix` 的 Issue，剩余的即为待评审。`triaged` 由后台自动打上，不代表评审状态，不用于过滤。
3. 读取 Issue、评论、关联 PR、标签和里程碑。
4. 检查 `origin/master` 和当前代码，验证问题是否仍存在或需求是否已实现。
5. 给出结论：接受、拒绝、需补充或阻塞。
6. 输出证据、缺失信息、验收标准、实现建议、风险和下一步。
7. 用户确认后使用 CLI 提交评审评论。
8. 使用 `gitcode issue label --add <结论label>` 追加结论 label。若仓库中不存在该 label，先通过 `gitcode api repos/<owner>/<repo>/labels?name=<label>&color=<color>` 创建。流转时用 `--remove` 删除旧结论 label；类型 label（如 `bug`）不动。
9. 使用 `gitcode issue label --list` 验证结论 label 已生效。**label 未生效时不得认为评审完成**——防重复评审依赖 label，无 label 则后续会话会重复评审。
10. 未经授权不修改 assignee、里程碑和状态。评审结论 label 属于评审流程自身输出，允许打。

## 结论要求

拒绝不能只写“不做”；需补充不能只列问题；接受不能跳过风险。每种结论都必须提供完整理由和可执行建议。

## 评审状态标签

评审结论通过 label 机器可读，防止重复评审。查询候选列表后，本地排除已有结论 label 的 Issue。`triaged` 由后台自动打上，不代表评审状态。

| 结论 | Label | 说明 |
|------|-------|------|
| 接受 | `approved` | 评审通过，可进入开发 |
| 需补充 | `need-detail-desc` | 评审需补充，等待作者完善描述 |
| 24h 等待 | `wait-feedback` | 超时默许等待窗口，与 `need-detail-desc` 同时打 |
| 阻塞 | `pending` | 暂时无法推进 |
| 拒绝 | `wontfix` | 不做处理 |

防重复评审：Issue 已有任意结论 label 时，新会话不重复评审，直接读评论获取详情。**评审完成后必须验证 label 已生效，否则防重复机制失效。**

label 追加与删除：用 `gitcode issue label --add <label>` 追加结论 label，用 `--remove <label>` 删除流转中的旧结论 label。类型 label（如 `bug`）始终保留。若仓库缺少某 label，`--add` 会失败，需先通过 API 创建。

## 需补充的超时默许

评审结论为"需补充"时，Issue 作者拥有首次补充权。若作者在 24 小时内无响应，按以下规则处理：

1. **计时起点**：从"需补充"评审评论发布时刻开始计时。评审时用 `--add` 追加 `need-detail-desc` + `wait-feedback` label。
2. **先行开发**：assignee 可在 24 小时窗口内先行启动开发。Issue 评审不是开发的硬门禁，PR 评审才是。
3. **超时声明前置**：评审评论中直接声明"若 Issue 作者 24 小时内无补充响应，上述补充建议将自动生效"。声明在发布评审时一次性完成，无需 24 小时后发起额外评论；新会话可直接基于评审评论中的补充建议启动开发。
4. **状态标记**：24 小时后无作者响应，Issue 视为 `verified (timeout)`，与经重新评审通过的 `verified (reviewed)` 区分，便于后期质量回溯。此状态由评审评论中的前置声明隐式生效，无需额外操作。label 层面，新会话开发时用 `--remove wait-feedback` + `--add approved` 表示超时默许生效。
5. **作者回归**：作者可在 PR 阶段提出异议并调整 scope；PR 评审是最终质量门禁。
