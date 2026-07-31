# AI Native 使用案例

以下提示词基于 MindStudio-Modeling 的真实模块和贡献方式，可直接改编号或参数后使用。

## 1. 模糊问题转 Issue

```text
TensorCast 在 Qwen3.6 MoE 的 decode 阶段吞吐结果明显偏高。我只有模型配置和一组实测结果。
请先询问缺失信息，核对模型适配、算子映射和性能模型代码，生成完整 Issue 草稿。
提交到 Ascend/msmodeling 前让我确认。
```

预期：AI 补齐环境、配置、复现、期望/实际结果、影响范围、证据、验收标准和潜在代码位置；确认后使用
`gitcode issue create`。

## 2. 评审自己负责的 Issue

```text
查询 Ascend/msmodeling 中负责人是我的所有开放 Issue。
逐个判断为接受、拒绝或需补充信息，给出代码证据、风险、建议和下一步；先展示评论草稿。
```

预期：通过 CLI 识别当前用户并过滤 assignee；每个结论完整且可操作，确认后评论。

## 3. Issue 到 Ready PR

```text
实现 Issue 123，使用 guided 模式。先做需求分析和实现设计，得到我确认后再开发；
完成受影响测试、pre-commit、构建、Draft PR 和 openLiBing CI 闭环，每个阶段在 Issue/PR 留记录。
```

预期：创建非 `master` 分支，按领域选择模型适配、op mapping、ServingCast 或 OptiX Skills；CI 通过或给出
明确阻塞后才建议 Ready。

## 4. 限定范围的自动交付

```text
对 Issue 123 使用 autonomous 模式，范围仅限 tensor_cast 的 op mapping 和对应测试、文档。
可以创建分支、commit、push、Draft PR 和处理 CI；公共接口变化、扩大范围、强推、审批、合并时必须停下。
```

预期：AI 先记录授权边界，普通阶段连续执行，越界时暂停。

## 5. 从个人 Fork 向主仓交付

```text
以 Ascend/msmodeling 的 Issue 123 为正式目标，从 upstream/master 建分支，push 到当前 origin Fork，
再通过 GitCode CLI 向 Ascend/msmodeling 创建 Draft PR。Fork 内部 PR 只作调测，不作为 CI 证据。
```

预期：AI 动态识别 source repository，显式确认 canonical target；最终在主仓 PR 上完成 openLiBing。

## 6. PR 多角色检视与行内评论

```text
检视 PR 456。按 SIG 路由执行架构、正确性、性能、测试、安全和文档角色检查，
先展示 findings；我确认后用 GitCode CLI 提交到准确的新文件行号，并给出风险等级和合入建议。
```

预期：`gitcode pr diff` 是权威 diff；行内位置是新版本实际行号；多角色分析不冒充多个独立 reviewer。

## 7. openLiBing CI 闭环

```text
分析 PR 456 最新 openLiBing 失败。使用 gitcode-pipeline-analyzer 定位日志和根因，
给出修复方案让我确认，然后本地复现、修复、验证、推送并监控下一轮，直到通过或明确阻塞。
```

预期：每轮记录任务链接、失败签名、根因、修复 commit 和验证证据；相同失败无新证据时停止。

## 8. 处理检视意见

```text
分析 PR 456 的未解决意见。逐条判断是否成立、是否已过期或重复，提出修改或解释；
确认后实施、验证，并通过 CLI 回复对应意见。
```

预期：合理意见修改代码并给证据；不合理意见礼貌解释；不擅自将需要权限的线程标记为已解决。

## 9. 暂停后恢复

```text
继续 Issue 123 的 run-id issue-123-20260723-a1b2。先核验 Issue/PR 评论、分支和 head SHA，
告诉我上次完成到哪里，再从第一个未完成门禁继续。
```

预期：不重复创建 Issue、PR 或评论，不假设本地和远端仍保持原状态。
