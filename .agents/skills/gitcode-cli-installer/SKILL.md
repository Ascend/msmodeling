---
name: gitcode-cli-installer
description: 安装和认证 gitcode CLI（@gitcode-cli/cli），msmodeling AI Native 工作流所有远端操作的唯一入口。一次性安装后持久生效。
metadata:
  version: 1.0.0
  source: issue-299
---

# gitcode CLI 安装器

## 适用场景

- 用户首次配置 gitcode CLI（安装 + 认证）。
- 用户升级 gitcode CLI（`npm update -g @gitcode-cli/cli`）。
- gitcode CLI 版本过低需要重新安装。
- AI agent 检测到 CLI 缺失或未认证时引导用户完成安装。

> 本 Skill 只管安装和认证。CLI 的功能使用由各业务 Skill（sig-review、msmodeling-issue-draft 等）承担。

## 前置条件

- Node.js ≥ 18（npm 随附）。
- GitCode 账号（用于创建个人令牌）。

## 工作流程

### 1. 检查现有安装

```bash
gitcode version 2>/dev/null || echo "NOT_INSTALLED"
```

已安装且版本 ≥ 0.8.0 时跳到步骤 4（认证检查）。

### 2. 配置 npm scoped registry

npm 默认源或华为云镜像可能不含 `@gitcode-cli` scope 或版本滞后。为 `@gitcode-cli` scope 单独指定官方源，不影响其他包的镜像配置：

```bash
npm config set @gitcode-cli:registry https://registry.npmjs.org
```

> 该配置写入用户全局 `~/.npmrc`，只影响 `@gitcode-cli` scope 包，与其他包的镜像源共存。

### 3. 安装

```bash
npm install -g @gitcode-cli/cli
```

安装后验证版本：

```bash
gitcode version
```

> **Windows 注意**：PowerShell 中 `gc` 是 `Get-Content` 的内置别名。使用 `gitcode` 全称，不要用 `gc`。

### 4. 认证

引导用户执行交互式认证。**AI 不得在对话中接触 token 明文**：

> 请在你自己的终端中运行以下命令（不要在此对话中粘贴令牌）：
>
> ```bash
> gitcode auth login
> ```
>
> 令牌获取方式：GitCode 网站 → 设置 → 私人令牌 → 生成新令牌（需 **repo** 读写权限）。

用户确认完成后验证：

```bash
gitcode auth status
```

预期输出包含 `✓ Logged in`。

### 5. 飞书集成（可选）

如需飞书消息通知（Issue/PR assignee 通知等），检查并安装 lark-cli：

```bash
gitcode lark doctor
```

- 输出提示已就绪 → 跳过。
- 提示未安装 → 引导用户执行 `gitcode lark install` 安装 lark-cli，再用 `lark-cli config init` 和 `lark-cli auth login` 完成 OAuth 登录，最后用 `gitcode lark auth status` 验证。

### 6. 升级（已有安装）

```bash
npm update -g @gitcode-cli/cli
gitcode version
```

## 安全规则

遵循 [spec/foundations/gitcode-cli-contract.md](../../../spec/foundations/gitcode-cli-contract.md) §6：

- **禁止**执行 `gitcode auth token`。
- **禁止**执行 `gitcode auth status --show-token`。
- **禁止**读取认证配置文件（`~/.config/gc/` 等）。
- **禁止**打印 `GC_TOKEN` 或 `GITCODE_TOKEN` 环境变量。
- **禁止**要求用户把令牌粘贴到对话中。
- 令牌由用户在**自己的终端**中通过 `gitcode auth login` 交互式输入，保存在本地配置文件 `~/.config/gc/auth.json`（权限 600）。

## 完成标准

- `gitcode version` 输出 ≥ 0.8.0。
- `gitcode auth status` 显示 `✓ Logged in`。
- npm `@gitcode-cli:registry` 已配置为官方源（或已安装且可正常 `npm update -g`）。
- Windows 用户知道使用 `gitcode` 全称而非 `gc`。
- 飞书集成状态已检查（已就绪或已记录跳过原因）。
