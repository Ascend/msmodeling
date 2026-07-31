# Issue 创建工作流

## 状态

```text
intake -> clarifying -> analyzed -> drafted -> confirmed -> created
```

## 步骤

1. 确认 canonical repository、operation target 和 Issue 类型；正式 Issue 使用 `Ascend/msmodeling`，
   Fork 临时调测必须显式选择 Fork。
2. 提取已知事实、期望结果、影响范围和限制。
3. 只询问会改变 Issue 内容或可实施性的缺失信息。
4. 使用 CLI 搜索开放和关闭 Issue；服务端搜索不可靠时，拉取列表后本地过滤。
5. 分析代码、测试、文档和历史，区分事实、推断和待确认项。
6. 根据 Bug、Feature 或 RFC 模板形成草稿。
7. 检查真实标签；不得发明不存在的标签。
8. 对显式 operation target 执行 `gitcode issue create -R <target> --dry-run --json`。
9. 向用户展示最终标题、正文和元数据。
10. 用户确认后创建，并回读验证编号、标题、状态和正文。

## 完成标准

- 不存在未说明的关键假设；
- 验收标准可测试；
- 已执行重复检查；
- 用户确认最终草稿；
- CLI 创建和回读成功。
