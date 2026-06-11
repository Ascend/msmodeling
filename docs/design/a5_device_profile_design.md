# 特性设计：A5 系列硬件 Device Profile 新增支持

## 修订记录

| Date (日期) | Version (修订版本) | Change Description (修改描述) | Author (作者) | RFC Document (RFC文档) |
| --- | --- | --- | --- | --- |
| 2026-06-09 | 1.0 | 初稿完成，支持 ATLAS 350（A350_112G / A350_84G）硬件建模 | huqixing | — |

---

## 1. 背景描述

### 1.1 需求背景

大模型的理论性能评估是推理部署决策中的关键环节，用于判断 NPU 和竞品的性能水平、预测模型在特定部署策略下的吞吐与延迟。传统手工建模方式效率低、准确性差，因此需要自动化的性能评估工具 msmodeling。

当前 msmodeling 已支持 ATLAS 800 系列（A2 / A3）硬件的性能建模。随着新一代 A5 系列硬件的推出，需将 A5 的硬件参数（算力、显存带宽、互联拓扑等）内置于框架中，使 msmodeling 能够对基于 A5 硬件的推理场景进行性能仿真。

### 1.2 核心价值

- 支持 ATLAS 350 系列 2 个硬件 Profile 的性能仿真
- 覆盖 PCIE + UB 混合组网部署拓扑
- 采用组合式架构设计，算力 / 显存 / 互联三维度解耦，易于扩展
- 零侵入现有代码，通过 `DeviceProfile.__post_init__` 自动注册

### 1.3 目标

用户可通过 CLI 或编程接口指定 A5 硬件，对任意模型进行性能评估，例如：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-8B \
    --num-queries 8 --query-length 1024 \
    --device ATLAS_350_425T_112G \
    --num-devices 2 --tp-size 2 --compile \
    --quantize-linear-action MXFP4
```

---

## 2. 方案设计

### 2.1 总体架构：组合模式解耦三维度

A5 系列硬件存在不同显存配置（112G / 84G），若为每种组合独立定义 Profile 会产生重复代码。

因此采用**组合模式**，将硬件参数拆分为三个正交维度：

```text
┌──────────────────────────────────────────────────────────────────┐
│                         DeviceProfile                            │
│    (name, vendor, static_cost, compute_efficiency,               │
│          memory_efficiency, comm_grid)                           │
└────────┬───────────────────┬──────────────────────┬──────────────┘
         │                   │                      │
   ┌─────▼──────┐     ┌──────▼───────┐     ┌────────▼───────────────┐
   │    Chip    │     │     Mem       │     │    Interconnect       │
   │   (算力)   │     │    (显存)     │     │     (通信拓扑)        │
   ├────────────┤     ├───────────────┤     ├───────────────────────┤
   │ C425T      │     │ M112G_1_4T    │     │ PCIE2_UB4             │
   └────────────┘     │ M84G_1_4T     │     └───────────────────────┘
                      └───────────────┘
```

每个 DeviceProfile 通过 Python 字典解包 `**Chip.C425T` + `**Mem.Mxxx` 组合出所需参数，再指定对应的 `comm_grid`，三行代码即可定义一个 Profile。

### 2.2 代码组织结构

所有新增代码位于单文件tensor_cast/device.py，以独立类 `class A5:` 组织：

```text
class A5:                    # 见 device.py
    STATIC_COST              # 静态调度开销（mma=5μs, gp=2μs, comm=5μs）
    class Chip:              # 算力规格（字典）
        C425T                # 425T bf16 芯片
    class Mem:               # 显存规格（字典）
        M112G_1_4T           # 112GB / 1.4TB/s
        M84G_1_4T            #  84GB / 1.4TB/s
    class Interconnect:      # 通信拓扑（CommGrid 对象）
        PCIE2_UB4            # 双 CPU PCIe + 4 卡 UB FullMesh，最大 16 卡
    # 2 个 DeviceProfile
```

### 2.3 芯片算力规格（Chip）

当前仅支持一种芯片规格：

| 规格 | bf16/half MMA | int8 / FP8 MMA | FP4 MMA | float32 GP | bf16/half GP | 计算效率 |
|------|--------------|----------------|---------|------------|-------------|---------|
| **C425T** | 378 TFLOPS | 756 TFLOPS | 1512 TFLOPS | 24 TFLOPS | 47 TFLOPS | 0.9 |

**设计要点**：

- C425T 的 float32 MMA 使用 HF32（189 TFLOPS），为 bf16 的一半
- FP8 算力为 bf16 的 2 倍，FP4 为 bf16 的 4 倍，遵循标准量化加速比
- GP ops（通用矢量计算）约为 MMA 算力的 1/8
- 计算效率统一设为 0.9，高于 ATLAS_800 的 0.7，体现新硬件架构改进

### 2.4 显存规格（Mem）

两种显存配置覆盖不同场景：

| 规格 | 容量 | 带宽 | 效率 | 使用方 |
|------|------|------|------|--------|
| **M112G_1_4T** | 112 GB | 1.4 TB/s | 0.8 | A350_112G |
| **M84G_1_4T** | 84 GB | 1.4 TB/s | 0.8 | A350_84G |

**设计要点**：

- 显存效率统一设为 0.8，高于 ATLAS_800 的 0.6

### 2.5 通信拓扑设计（Interconnect）

A5 当前支持 1 种通信拓扑。

#### 2.5.1 PCIE2_UB4 — 双路 PCIe + 4 卡 UB FullMesh

适用于 ATLAS 350 工作站级部署（单机最多 16 卡）。

```text
Grid: (2, 2, 4)  → 最多 16 卡
       │  │  │
       │  │  └── dim 2: 板内 4 卡间 3 路 UB FullMesh，单路 53 GB/s × 3 = 159 GB/s
       │  └───── dim 1: 两组 4 卡通过 2 路 PCIe x16 连接 CPU，有效带宽 32 GB/s
       └──────── dim 0: 双 CPU 间等效 3 路 PCIe x16，有效带宽 24 GB/s
```

| 层级 | 互联方式 | 带宽 | 延迟 | 效率 |
|------|---------|------|------|------|
| dim 2 | 3 路 UB FullMesh | 159 GB/s | 1.5 μs | 0.85 |
| dim 1 | 2 路 PCIe x16 到 CPU | 32 GB/s | 3.0 μs | 0.8 × 0.7 |
| dim 0 | CPU 间 3 路 PCIe x16 | 24 GB/s | 4.5 μs | 0.75 × 0.7 |

> PCIe 链路额外乘以 0.7 折扣因子，反映协议开销。

### 2.6 DeviceProfile 完整清单

全部 2 个 Profile 定义见 [device.py](../../tensor_cast/device.py)。

#### 2.6.1 ATLAS 350 系列（工作站级，PCIE + UB 组网）

| Profile 名称 | 芯片 | 显存 | 互联 | 最大卡数 |
|---|---|---|---|---|
| `ATLAS_350_425T_112G` | C425T | M112G_1_4T | PCIE2_UB4 | 16 |
| `ATLAS_350_425T_84G` | C425T | M84G_1_4T | PCIE2_UB4 | 16 |

### 2.7 与 ATLAS_800 系列的设计差异

| 维度 | ATLAS_800 | A5 |
|------|-----------|-----|
| 组织方式 | 类级别常量逐个定义（`A2_INTERCONNECT` 等） | 使用 `class Chip` / `class Mem` / `class Interconnect` 嵌套类 + 字典解包 |
| 算力 / 显存复用 | 每个 Profile 独立声明，参数重复 | 字典解包 `**Chip.C425T`，零重复 |
| 扩展性 | 新增规格需复制粘贴 | 新增 Chip/Mem 规格一行字典，组合即得 Profile |
| 计算效率 | 0.7 | 0.9 |
| 显存效率 | 0.6 | 0.8 |
| 静态成本 `comm_op_cost_s` | 10 μs | 5 μs |
| 最大设备规模 | 768（A3 die） | 16（PCIE2_UB4） |

### 2.8 DeviceProfile 命名规范

A5 系列采用统一的命名模式：

```text
ATLAS_{系列}_{芯片}_{显存}

其中：
  {系列} = 350
  {芯片} = 425T
  {显存} = 84G | 112G
```

### 2.9 影响范围

- **新增代码**：[device.py](../../tensor_cast/device.py)
- **无破坏性变更**：现有 ATLAS_800 系列硬件、API、CLI 均不受影响
- **自动注册**：通过 `DeviceProfile.__post_init__` 自动加入 `all_device_profiles`，CLI 和 Web UI 可直接使用
- **命名空间隔离**：A5 使用 `ATLAS_350` 前缀，与 ATLAS_800 系列的 `ATLAS_800` 前缀不冲突
- **新增设计文档**：`docs/design/a5_device_profile_design.md`（本文档）

---

## 3. 使用说明

### 3.1 CLI 使用

```bash
# ATLAS 350 工作站
python -m cli.inference.text_generate Qwen/Qwen3-8B \
    --device ATLAS_350_425T_112G \
    --num-devices 8 --tp-size 4 --dp-size 2 \
    --num-queries 8 --query-length 1024 --compile

# ATLAS 350（验收标准用例）
python -m cli.inference.text_generate Qwen/Qwen3-8B \
    --num-queries 8 --query-length 1024 \
    --device ATLAS_350_425T_112G \
    --num-devices 2 --tp-size 2 --compile \
    --quantize-linear-action MXFP4

# ATLAS 350 84GB 显存版本
python -m cli.inference.text_generate Qwen/Qwen3-8B \
    --device ATLAS_350_425T_84G \
    --num-devices 2 --tp-size 2 --compile
```

### 3.2 编程接口

```python
from tensor_cast.device import DeviceProfile
import torch

# 列出所有 A5 设备
a5_profiles = {
    k: v for k, v in DeviceProfile.all_device_profiles.items()
    if k.startswith("ATLAS_350")
}
print(f"共 {len(a5_profiles)} 个 A5 设备")  # 2

# 获取特定 Profile
p = DeviceProfile.all_device_profiles["ATLAS_350_425T_112G"]
print(p.vendor)                     # "HUAWEI"
print(p.mma_ops[torch.bfloat16])    # 3.78e14 (378 TFLOPS)
print(p.mma_ops[DTYPE_FP4])         # 1.512e15 (1512 TFLOPS)
print(p.memory_size_bytes)          # 120259084288 (112 GB)
print(p.memory_bandwidth_bytes_ps)  # 1.4e12 (1.4 TB/s)
print(p.compute_efficiency)         # 0.9
print(p.memory_efficiency)          # 0.8
print(p.comm_grid.grid.shape)       # (2, 2, 4) — 最大 16 卡
print(p.static_cost.comm_op_cost_s) # 5e-6 (5 μs)
```

### 3.3 约束与限制

1. **命名规范**：A5 设备名格式为 `ATLAS_350_425T_{显存}`，需完整匹配，大小写敏感。
2. **组网规模约束**：PCIE2_UB4 拓扑最大 16 卡，超出将导致仿真建模错误。

   | 拓扑 | 最大设备数 |
   |------|----------|
   | PCIE2_UB4 | 16 |

3. **效率因子待校准**：`compute_efficiency = 0.9` 和 `memory_efficiency = 0.8` 为初步估值，建议后续通过真实 profiling 数据校准。

---

## 4. 测试设计

### 4.1 单元测试

**测试文件**：[test_device.py](../../tests/regression/tensor_cast/test_device.py)

#### 1. A5 Chip 规格校验

验证 `A5.Chip.C425T` 字典：

- `mma_ops` 各 dtype（float32: 189T、bfloat16/half: 378T、fp8/int8: 756T、fp4: 1512T）算力值符合设计
- `gp_ops` 各 dtype（float32: 24T、bfloat16/half: 47T）算力值符合设计
- `compute_efficiency` 为 0.9

#### 2. A5 Mem 规格校验

验证 `A5.Mem` 下两种配置：

- M112G_1_4T：112 GB / 1.4 TB/s / efficiency 0.8
- M84G_1_4T：84 GB / 1.4 TB/s / efficiency 0.8

#### 3. A5 Interconnect 通信网格校验

对 PCIE2_UB4 通信拓扑逐一验证：

- grid shape 与维度正确
- grid ndim 与 topologies 数量一致
- 每层的带宽、延迟、效率符合 2.5 节设计值
- 每层 topology type（FULL_MESH 或默认 CLOS）正确

#### 4. DeviceProfile 属性正确性

参数化覆盖全部 2 个 A5 Profile，验证：

- Profile 已注册到 `DeviceProfile.all_device_profiles`
- `name`、`vendor = "HUAWEI"` 与定义一致
- `mma_ops`、`gp_ops` 与 C425T Chip 规格一致
- `memory_size_bytes`、`memory_bandwidth_bytes_ps` 与对应 Mem 规格一致
- `comm_grid` 引用指向正确的 Interconnect 对象
- `static_cost` 引用指向 `A5.STATIC_COST`
- `compute_efficiency`（0.9）、`memory_efficiency`（0.8）值正确合并

#### 5. 静态开销校验

验证 `A5.STATIC_COST`：

- `mma_op_cost_s` = 5 μs
- `gp_op_cost_s` = 2 μs
- `comm_op_cost_s` = 5 μs

#### 6. 注册逻辑校验

- 验证 A5 Profile 重复名称触发 `ValueError`
- 验证 A5 与 ATLAS_800 Profile 命名不冲突

### 4.2 集成测试

运行测试，验证新增 A5 硬件不影响已有 ATLAS_800 的测试用例：

```bash
python -m pytest tests/regression/tensor_cast/test_device.py -v
```

关注点：

- A5 新增测试全部通过
- 已有 ATLAS_800 测试无回归

### 4.3 端到端验证

使用需求文档中的验收标准用例：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-8B \
    --num-queries 8 --query-length 1024 \
    --device ATLAS_350_425T_112G \
    --num-devices 2 --tp-size 2 --compile \
    --quantize-linear-action MXFP4
```

成功标准：

1. 设备名 `ATLAS_350_425T_112G` 可被 CLI 正确识别并加载
2. `PCIE2_UB4` 的三层通信拓扑被正确解析
3. 仿真结果中算力按 C425T（378T bf16 × 0.9 效率）计算
4. 仿真结果中显存带宽按 M112G_1_4T（1.4 TB/s × 0.8 效率）计算
5. 量化行为 MXFP4 正确作用于 FP4 ops 路径
