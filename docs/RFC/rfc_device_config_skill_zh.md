# RFC: 设备画像自然语言导入器（device_config Skill）

## 元数据

| 项目 | 内容 |
|:-----|:--------|
| **状态** | 实施中 |
| **作者** | wendellX |
| **创建日期** | 2026-05-27 |
| **最后更新** | 2026-05-28 |
| **相关链接** | [Skill 文档](../../.agents/skills/device_config/SKILL.md) |

---

## 1. 概述

### 1.1 背景与问题

TensorCast 项目需要为各种加速器、GPU、NPU 维护 `DeviceProfile` 硬件能力描述。当前社区成员需要手动阅读代码、理解 `DeviceProfile` 数据结构后，才能添加新硬件支持。这一过程存在以下痛点：

| 痛点 | 说明 |
|------|------|
| 门槛高 | 需要理解 `mma_ops`、`gp_ops`、`CommGrid`、`InterconnectTopology` 等专用术语 |
| 易出错 | 单位换算（GiB/TB/s/TFLOPS）和 dtype 映射（FP16→torch.half）容易搞错 |
| 路径混乱 | 不知该写 `tensor_cast/device.py` 还是新建 `tensor_cast/device_profiles/*.py` |
| 难验证 | 没有工具能验证生成结果是否与预期一致 |

### 1.2 目标

本提案引入一个 Claude Code skill——`device_config`，用于引导用户通过自然语言描述硬件信息，自动生成符合 TensorCast 规范的 `DeviceProfile` 代码，并输出可直接执行的推理命令。

**核心目标**：

- 通过渐进式提问降低硬件建模门槛，不需要用户一开始就了解内部结构
- 保证生成结果可导入、可执行、可校准
- 以 `tensor_cast/device.py` 为默认写入目标
- 输出真实可运行的 `--device <PROFILE_NAME>` 命令

**非目标**：

- 不自动更新 live skill 本身（skill 文档与 repo 文档解耦）
- 不做性能回归检测
- 不做自动基线更新

### 1.3 核心价值

- **渐进式引导**：像对话一样逐步收集信息，首轮只问最容易回答的几件事
- **术语友好**：每个专有概念都配中文解释和可直接照抄的回答示例
- **默认值兜底**：缺失信息时给出明确选项，而不是报错或留空
- **结构化输出**：生成结果可直接导入 `tensor_cast/device.py`，并附验证命令

---

## 2. 详细设计

### 2.1 架构概览

```text
用户输入（自然语言硬件描述）
       │
       ▼
  ┌─────────────────────────────────┐
  │   device_config Skill (v0.4.0)  │
  │   .agents/skills/               │
  │   device_config/SKILL.md        │
  └───────────────┬─────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   内部事实表 (每轮维护)     │
    │   confirmed / ambiguous /  │
    │   missing / needs_calibration │
    └────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   生成 DeviceProfile 代码   │
    │   写入 tensor_cast/device.py │
    └────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   验证 + 输出 CLI 命令      │
    │   --device <PROFILE_NAME>   │
    └────────────────────────────┘
```

### 2.2 核心数据结构

#### DeviceProfile 参数对照表

| 参数 | 含义 | 典型来源 |
|------|------|----------|
| `name` | Profile 唯一名称，大写风格如 `ATLAS_800_A3_752T_128G_DIE` | 用户指定 |
| `vendor` | 硬件厂商如 HUAWEI / NVIDIA / AMD | 用户指定 |
| `granularity` | 单卡 / 单 die / 单 chiplet | 用户选择或追问 |
| `memory_size_bytes` | 每 profile 单元的 HBM 容量（bytes） | 用户提供 GiB → 转为 bytes |
| `memory_bandwidth_bytes_ps` | 每 profile 单元的显存带宽（bytes/s） | 用户提供 TB/s → 转为 bytes/s |
| `mma_ops` | 矩阵/张量算力峰值，按 dtype 区分 | 用户提供 TFLOPS → 转为 ops/s |
| `gp_ops` | 通用/vector 算力峰值，按 dtype 区分 | 用户提供或留空/临时估值 |
| `comm_grid` | 通信拓扑：[grid 形状, topologies] | 用户描述多卡/多 die 互联 |
| `compute_efficiency` | 计算效率，默认 0.7 | 用户确认或待校准 |
| `memory_efficiency` | 内存效率，默认 0.6 | 用户确认或待校准 |
| `static_cost` | 调度开销（mma/gp/comm） | 默认值或待校准 |

#### mma_ops / gp_ops dtype 映射

| 用户资料写法 | 对应字段 |
|------------|----------|
| FP16 TFLOPS / Tensor Core 算力 | `mma_ops[torch.half]` |
| BF16 TFLOPS / AI Core 算力 | `mma_ops[torch.bfloat16]` |
| INT8 TOPS | `mma_ops[torch.int8]` |
| FP32 矩阵/张量 TFLOPS | `mma_ops[torch.float32]` |
| FP8 算力 / MXFP8 | `mma_ops[DTYPE_FP8]` |
| FP32 向量算力（GP） | `gp_ops[torch.float32]` |
| BF16 向量算力（GP） | `gp_ops[torch.bfloat16]` |

#### 单位换算规则

| 来源单位 | 目标 | 换算 |
|---------|------|------|
| `T`/`TFLOPS`/`TOPS` | ops/s | `* 1e12` |
| `P`/`PFLOPS`/`POPS` | ops/s | `* 1e15` |
| `GB/s`（带宽，十进制） | bytes/s | `* 1e9` |
| `TB/s`（带宽，十进制） | bytes/s | `* 1e12` |
| `GiB`（容量） | bytes | `* (1024**3)` |
| `TiB` | bytes | `* (1024**4)` |
| `us` / 微秒延迟 | seconds | `* 1e-6` |
| `ns` / 纳秒延迟 | seconds | `* 1e-9` |
| GB 级容量 | bytes | 默认按 GiB 处理；需列入 `needs_calibration` |

### 2.3 工作流程

#### 新手引导模式（默认）

按阶段逐步推进，每轮最多问 2-3 个问题，每个问题都提供"不知道/先跳过/用默认值"的出口：

```text
第一轮（助手）：
  → 复述已理解的信息
  → 提出下一批问题

第二轮（用户）：回复部分问题

第三轮（助手）：展示将写入的字段、默认值和 needs_calibration，确认后再写入

第四轮（助手）：写入 → 验证 → 输出 CLI 命令
```

#### 各阶段信息收集优先级

| 阶段 | 收集项 | 重要性 |
|------|--------|--------|
| 1 | 厂商 / 型号 / profile 名称 | 必须 |
| 2 | 粒度（单卡 / 单 die） | 必须 |
| 3 | 显存容量和带宽 | 必须 |
| 4 | mma_ops（FP16/BF16/INT8 等峰值） | 核心 |
| 5 | gp_ops（vector/general 峰值） | 次要，可留空 |
| 6 | 通信拓扑（多卡互联） | 进阶 |
| 7 | 效率默认值和 static cost | 可待校准 |

### 2.4 CommGrid 拓扑建模

```text
用户描述 → CommGrid 转换规则：

1. grid 形状：覆盖的 profile 单元数量和排列
   - 8 卡机器：[128, 8]（最外层=卡间互联，最内层=NIC 互联）
   - 1 卡单 die：[2]（至少 2 个元素）

2. topologies：从慢到快的层级
   - start_dim=0：覆盖整个 grid 的最慢层（卡间 CLOS）
   - start_dim=1：更快子层（NIC 互联 FULL_MESH）
   - 最后一层：最快内层

3. 互联类型：
   - CLOS：多对多交换，如 InfiniBand / 多级交换
   - FULL_MESH：全互联，如 NVLink / 直连
```

### 2.5 gp_ops 处理策略

`gp_ops` 很多硬件不会单独给，提供以下选项：

1. **留空**：`gp_ops={}` — 影响 softmax、norm、激活等逐元素算子的估算准确性
2. **临时估值**：按 `mma_ops` 的某个比例估算（如 `gp_ops[BF16] ≈ mma_ops[BF16] / 10`），必须标为 `needs_calibration`
3. **留空但加入校准清单**：告知用户缺少 `gp_ops` 会影响哪些算子

### 2.6 写入目标策略

**默认**：`tensor_cast/device.py`（直接追加到现有厂商类）

**例外**（才写入 `tensor_cast/device_profiles/`）：

- 用户明确要求临时 profile / 自定义 profile 文件
- 用户要求隔离实验

### 2.7 写入前检查清单

- [ ] `profile_name` 在 `DeviceProfile.all_device_profiles` 中唯一
- [ ] `DTYPE_FP4` / `DTYPE_FP8` 只在实际用到时导入
- [ ] `CommGrid` 的每个维度都至少为 2，且 `topologies` 数量与 `grid.ndim` 一致
- [ ] 所有写入数值都能追溯到用户输入 / 用户确认的估值 / 明确的兜底默认值
- [ ] 互联带宽方向已确认（单向 vs 双向）
- [ ] 准备好最终 `--device <PROFILE_NAME>` 命令

### 2.8 替代方案分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 纯文档（无 skill） | 无需定制开发 | 用户仍需手动阅读 SKILL.md |
| B: 独立 CLI 工具 | 可集成 CI | 需要用户切换工具，且没有对话引导 |
| **C: Claude Code Skill（已选）** | 渐进引导、中文友好、无缝集成 Claude Code | skill 质量依赖 Claude 模型能力 |

选择方案 C 的原因：TensorCast 项目已在使用 Claude Code，让用户通过 skill 自然地对话添加 profile 是最自然的交互方式。

---

## 3. 使用说明

### 3.1 调用方式

在 Claude Code 对话中直接使用：

```text
/device_config
```

### 3.2 典型使用流程

**用户**：我们有个 ATLAS_800 单卡，显存 96GB，带宽 3.2TB/s。BF16 是 800T，FP16 是 800T，INT8 是 1600T，FP32 是 100T。8 卡机器，卡间 400GB/s。

**第一轮（助手）**：复述已理解信息，提出下一批问题：

- 确认 profile 名称候选
- 追问 `gp_ops` 是否有官方峰值
- 追问 400GB/s 是单向还是双向，互联方式

**用户**：BF16 没有单独的 vector 算力，用留空吧。带宽不确定单向还是双向。

**第三轮（助手）**：展示将写入的字段和 `needs_calibration` 清单，确认后写入。

**最终输出**：

- 修改的文件 + 插入位置
- 已注册的 `DeviceProfile.name`
- `needs_calibration` 清单
- **可执行命令**：`python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device ATLAS_800_800T_96G`

### 3.3 使用约束

1. **粒度必须明确**：单卡、单 die、单 chiplet 只能选一个；整卡规格不能直接当单 die 值用
2. **带宽方向必须确认**：双向带宽不能静默当单向用
3. **gp_ops 缺失影响准确性**：没有官方 vector/general compute 峰值时，选择"留空"或"临时估值"
4. **needs_calibration 必须对用户可见**：所有默认值、估值、假设都必须在最终回复中列出

### 3.4 术语说明

| 术语 | 中文含义 |
|------|----------|
| `DeviceProfile` | TensorCast 里的一张"硬件能力卡片"，记录算力、显存、带宽和互联 |
| profile 粒度 | 这张能力卡片代表的最小硬件单元（单卡 / 单 die） |
| die / chiplet | 一张卡里的多个计算芯片小块 |
| `mma_ops` | 矩阵/张量计算峰值，影响 Linear、Attention 等大算子 |
| `gp_ops` | 通用/vector 计算峰值，影响 softmax、norm、激活等逐元素算子 |
| `needs_calibration` | 先跑通而暂用的默认值、估值、假设，需要后续校准 |

---

---

## 4. 参考资料

- [DeviceProfile 数据结构](../../tensor_cast/device.py)
- [repo 文档版 Skill](../../.agents/skills/device_config/SKILL.md)
- [RFC: 性能回归测试框架](../RFC/rfc_precision_protection.md)
