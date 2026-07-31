---
name: gitcode-pipeline-analyzer
description: |
  Analyze GitCode PR pipelines end-to-end: parse PR pipeline comments, fetch
  failed task logs from openLiBing, inspect stage/job duration metrics, and
  build batch CI reports.

  TRIGGER when: user asks to inspect GitCode PR pipeline status, parse PR comment
  pipeline tables, fetch failed task logs, analyze openLiBing job logs, inspect
  CI duration metrics, build recent PR pipeline reports, or phrases like
  "看PR流水线", "扒流水线日志", "分析失败任务", "获取UT日志", "openlibing日志",
  "统计耗时", "CI报表".
metadata:
  version: 1.0.0
  source: gitcode-cli/skills@6f0d8b0-adapted
---

# GitCode Pipeline Analyzer

用于从 GitCode PR 评论区提取流水线状态，并从 openLiBing 抓取失败任务日志、质检详情和 stage/job 级耗时，支持单 PR 分析和最近 N 个 PR 的批量报表。

旧 skill `gitcode-pipeline-log` 仍保留为兼容别名；新增使用优先选择当前 skill。

## 适用范围

- **前置检查**：先查 PR 评论区是否存在流水线评论。openLiBing 流水线在 PR 评论 `compile` 后触发，执行状态和结果以机器人生成的表格形式发布到 PR 评论区，同时打 `ci-pipeline-running`/`ci-pipeline-failed`/`ci-pipeline-passed` label。**若评论区无流水线评论，说明该仓库未配置 PR 流水线，直接报告"该仓未配置 PR 流水线"并结束，不视为错误。**
- 流水线任务详情链接指向 `openlibing.com`
- 重点定位 `UT_*`、构建任务、`CodeCheck`、`SCA` 的失败原因
- 支持按 `stage`、`job` 和任务前缀做耗时统计
- 支持最近 N 个 PR 的汇总报表与失败明细导出

## 快速用法

先看 PR 评论区：

```bash
gitcode pr comments <pr_number> -R <owner/repo>
```

优先使用脚本：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo Ascend/triton-ascend `
  --pr 1510
```

查看 stage/job 耗时：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo cann/ops-math `
  --pr 2029 `
  --durations `
  --summary-only
```

查看最近 N 个 PR 的细分类耗时报表：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo cann/ops-math `
  --latest 10 `
  --durations `
  --report-format table
```

导出最近 N 个 PR 的细分类耗时报表为 JSON：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo cann/ops-math `
  --latest 10 `
  --durations `
  --report-format json
```

查看最近 N 个 PR 的失败任务详细分析：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo cann/ops-math `
  --latest 10 `
  --failure-details `
  --report-format table
```

将报表直接写入文件：

```powershell
python ".agents\skills\gitcode-pipeline-analyzer\scripts\fetch_pipeline_logs.py" `
  --repo cann/ops-math `
  --latest 10 `
  --durations `
  --report-format table `
  --output "$env:TEMP\ops-math-ci-report.md"
```

## 工作流程

1. 运行 `gitcode pr comments`，提取最新评论中的流水线总链接和任务表。
2. 默认优先关注最新评论里的最新一轮流水线。
3. 若传入 `--latest-failed-run`，则优先回看最近一次存在 `failed` 任务的流水线。
4. 若评论区任务链接里没有 `jobRunId/stepRunId`，则优先用 pipeline detail 回填任务元数据。
5. 默认分析选中流水线里所有 `failed` 任务；没有 `failed` 时再看 `running` 任务。
6. 若需要日志，从任务详情链接中提取：
   `projectId`、`pipelineId`、`pipelineRunId`、`jobRunId`、`stepRunId`
7. 调用 openLiBing 接口：

```text
GET  https://www.openlibing.com/gateway/openlibing-cicd/project/pipeline/pipeline-run/detail
POST https://www.openlibing.com/gateway/openlibing-cicd/project/pipeline/exec-log
```

8. 按任务类型走不同接口：
   - `UT_*` / 构建：`/gateway/openlibing-cicd/project/pipeline/exec-log`
   - `CodeCheck`：`/gateway/openlibing-codecheck/ci-portal/v1/codecheck/event/task/issues/report`
     和 `/gateway/openlibing-codecheck/ci-portal/v1/event/codecheck/task`
   - `SCA`：`/gateway/openlibing-sca/open/person/scan/id`
     和 `/gateway/openlibing-sca/open/scan/scanIssue/query`
9. 对日志类任务分页抓日志，优先定位：
   - 第一条直接失败的测试
   - `short test summary info`
   - 与当前改动最相关的首个失败项
10. 若传入 `--durations`，输出：
   - 流水线总时长 / 实际执行时长
   - 每个 `stage` 的总时长 / 实际时长 / 任务数
   - 按 `Compile`、`UT`、`StaticCheck`、`Quality/Test`、`Other` 聚合的平均耗时
   - Top 慢任务
11. 若传入 `--latest N --durations`，批量输出最近 N 个 PR 的：
   - 单 PR 耗时总表
   - stage 横向均值
   - 任务类型横向均值
12. 若传入 `--latest N --failure-details`，批量输出最近 N 个 PR 的：
   - 失败任务列表
   - 失败摘要 / 触发关键字
   - `CodeCheck` / `SCA` 的结构化摘要
13. 若传入 `--report-format table`，使用 markdown 表格输出，更适合日报/周报或粘贴到 MR/IM
14. 若传入 `--report-format json`，输出结构化 JSON，便于外部脚本或二次加工
15. 若传入 `--output <path>`，将结果写入目标文件
16. 输出前对日志和链接做基础脱敏，避免直接回显 token、签名串、cookie、口令类字段。

## 输出要求

- 先总结任务状态：哪些阶段通过、哪些失败、哪些运行中
- 若启用 `--durations`，优先给出 stage/job 耗时总览
- 再给失败任务日志摘要
- 若是 `CodeCheck`，输出规则名、文件、行号、问题说明
- 若是 `SCA`，输出命中文件、相似度、来源项目、风险说明
- 区分“直接相关失败”和“连带失败”
- 给出脱敏后的 openLiBing 任务链接，方便回看原始页面

## 注意事项

- `GET /project/pipeline/exec-log` 会返回“方法不支持”，必须用 `POST`
- `GET /project/pipeline/pipeline-run/detail` 可直接拿到 `stages[].jobs[]` 的时间字段
- openLiBing 前端页是 SPA，直接打开详情页拿不到日志正文，要走接口
- 当前 GitCode 机器人评论常见的是“任务名称/状态/日志/下载链接”表格，不再稳定包含 stage 列
- 若评论区没有可推断的 `projectId`，耗时统计和任务详情回填可能不可用
- `compile` 评论是 openLiBing 流水线的触发条件，不是 push 后自动触发
- 如果同一个 PR 有多轮 `compile`，优先分析最新评论里的最新流水线
- `ci-pipeline-running` = 流水线执行中；`ci-pipeline-passed` = CI 通过，自动化循环结束；`ci-pipeline-failed` = CI 失败，需诊断并修复后重新触发
- 若最新一轮已通过但你要追历史失败，可用 `--latest-failed-run`
- 若 PR 被关闭又重开，先确认当前有效 PR 编号再抓日志
