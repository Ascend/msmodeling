# RFC: 真机轻量化寻优部署器（optix-deploy Skill）

## 元数据

| 项目 | 内容 |
|:-----|:--------|
| **状态** | 实施中 |
| **作者** | wendellX |
| **创建日期** | 2026-06-01 |
| **最后更新** | 2026-06-01 |
| **相关链接** | [Skill 文档](../../.agents/skills/optix-deploy/SKILL.md) |

---

## 1. 概述

### 1.1 背景与问题

`ms_serviceparam_optimizer` 是基于 PSO 粒子寻优算法的服务化参数自动寻优工具，支持 MindIE 和 VLLM 推理框架。首次使用该工具的用户面临以下痛点：

| 痛点 | 说明                                                     |
|------|--------------------------------------------------------|
| 安装复杂 | 需要先装主包 `ms_service_profiler`，再装 optimizer 子包，依赖顺序容易搞错。 |
| 目录混淆 | 仓库根目录 vs `ms_serviceparam_optimizer/` 子目录，容易在错误位置执行安装。 |
| 环境缺失 | 主包包含 C++ 扩展，缺少 CMake/MSVC 会导致编译失败。                     |
| 验证缺失 | 安装后不知道如何验证工具是否可用。                                      |

### 1.2 目标

本提案引入一个 Claude Code skill——`optix-deploy`，用于自动化安装 msmodeling optix 真机轻量化模式，并验证 CLI 可用性。

**核心目标**：

- 明确安装顺序：先主包，再 optimizer 子包
- 自动识别当前目录并选择正确安装命令
- 检查编译工具依赖
- 验证 CLI 可用性
- 支持卸载

**非目标**：

- 不支持仿真模式（需要 profiling 数据和额外依赖）
- 不自动配置 `config.toml`
- 不执行实际寻优

### 1.3 核心价值

- **顺序明确**：清晰的依赖安装顺序，避免因缺少主包导致安装失败。
- **目录自适配**：自动识别仓库根目录 vs 子目录，选择对应命令。
- **预检查**：安装前检查编译工具，提前发现潜在问题。
- **验证闭环**：安装后验证 CLI，确保工具可用。

---

## 2. 详细设计

### 2.1 架构概览

```text
用户输入（安装/部署请求）
         │
         ▼
┌──────────────────────────────────────┐
│   optix-deploy                      │
│   Skill                             │
│   .agents/skills/                   │
│   optix-deploy/                     │
│   SKILL.md                          │
└──────────────┬───────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │   安装流程                  │
   │   1. 检查仓库是否存在       │
   │   2. 判断当前目录           │
   │   3. 安装 msmodeling       │
   │   4. 验证 CLI               │
   └────────────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │   输出结果                  │
   │   - 安装成功/失败           │
   │   - 失败原因和修复建议       │
   │   - 下一步建议              │
   └────────────────────────────┘
```

### 2.2 仓库结构

```text
msmodeling/                     ← 仓库根目录
└── experimental/
    ├── pyproject.toml          ← optix 安装入口
    └── optix/                 ← 寻优工具源码
```

### 2.3 安装流程

#### 步骤 1：检查仓库是否存在

```bash
ls ms_service_profiler pyproject.toml
```

若不存在，直接克隆：

```bash
git clone https://gitcode.com/Ascend/msserviceprofiler.git
cd msserviceprofiler
```

#### 步骤 2：判断当前目录

根据当前目录选择安装命令：

| 当前目录 | 安装命令 |
|----------|----------|
| 仓库根目录 | `cd experimental && pip install -e .` |

#### 步骤 2：安装

```bash
cd experimental
pip install -e .
```

#### 步骤 6：验证

```bash
msmodeling optix --help
```

### 2.4 卸载流程

```bash
pip uninstall optix
```

卸载前说明会移除 Python 包，征得用户同意后执行。

### 2.5 与其他 Skill 的衔接

| Skill | 衔接时机 |
|-------|----------|
| `optix-config` | 配置 `config.toml` |

### 2.6 输出结果

#### 安装成功

```text
✓ 主包 ms_service_profiler 已安装
✓ 寻优工具 ms_serviceparam_optimizer[real] 已安装
✓ CLI 验证通过：msmodeling optix --help

下一步建议：
1. 运行 /optix-config 配置 config.toml
2. 执行 msmodeling optix 开始寻优
```

#### 安装失败

```text
✗ 安装失败，原因：缺少 CMake
修复建议：
  非 Windows: 安装 CMake 和基础编译工具（如 GCC/CC）
  Windows: 当前工具不支持 Windows
```

---

## 3. 使用说明

### 3.1 调用方式

在 Claude Code 对话中直接使用：

```text
/optix-deploy
```

或通过自然语言触发：

- "安装寻优工具"
- "部署 optix 参数寻优"

### 3.2 典型使用流程

**用户**：我要安装参数寻优工具。

**助手**：

```text
好的，我来帮你安装 msmodeling optix。

首先检查仓库是否存在...
```

**助手**：（执行安装流程，检查目录，安装依赖，验证 CLI）

**助手**：

```text
✓ 安装成功！

验证结果：
- optix：已安装
- CLI 命令：msmodeling optix --help ✓

下一步建议：
1. 运行 /optix-config 配置 config.toml
2. 执行 msmodeling optix 开始寻优
```

### 3.3 使用约束

1. **涉及安装/卸载操作需先说明影响并征得用户同意。**
2. **不要只给命令，要实际执行并验证结果。**
3. **目标运行环境不支持 Windows。**
4. **不要将运行、配置、结果解读内容塞进本 skill。**

---

## 4. 参考资料

- [《Skill 文档》](../../.agents/skills/optix-deploy/SKILL.md)
