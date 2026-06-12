# Design Document: A3 RoCE Device Profile 新增支持

## Revision History (修订记录)

| Date (日期) | Version (修订版本) | Change Description (修改描述) | Author (作者) | RFC Document (RFC文档) |
| --- | --- | --- | --- | --- |
| 2026-05-21 | 1.0 | 初稿完成，支持800I A3双机RoCE直连 | huqixing | — |

---

## 1. Background (背景描述)

当前 msmodeling 已支持 ATLAS 800 系列的 A2 和 A3 die 层级硬件建模，其中 A3 die 的通信互联基于 `A3_INTERCONNECT`，支持最多 768 个 die（48 节点），节点间通过 HCCS 互联，带宽为 196 GB/s。

在实际部署中，部分场景仅支持**双机**组网且使用 **RoCE**（RDMA over Converged Ethernet）替代 HCCS 作为节点间互联。RoCE 网络的实际带宽约为 HCCS 的 1/8（~24.5 GB/s）。

为准确建模此类双机 RoCE 组网场景，需新增对应的通信网格 `A3_INTERCONNECT_ROCE` 以及配套的设备 Profile：

- `A3_560T_128G_DIE_ROCE`：560T 算力 die + RoCE 互联

### 核心价值

- 支持双机 RoCE 组网场景的性能仿真
- 新增硬件遵循现有 `DeviceProfile` / `CommGrid` 框架，零侵入
- 提供开发者测试用例，保障后续维护

---

## 2. Design (方案设计)

### 2.1 总体思路

新增的 RoCE 硬件在算力、显存、效率等维度与原 A3 die Profile 完全一致，唯一差异在于通信网格。因此采用**复用算力参数 + 替换通信网格**的策略：
A3_560T_128G_DIE  ──(替换 comm_grid)──>  A3_560T_128G_DIE_ROCE

### 2.2 通信网格设计：A3_INTERCONNECT_ROCE

#### 对比：A3_INTERCONNECT vs A3_INTERCONNECT_ROCE

| 项目 | A3_INTERCONNECT (原始) | A3_INTERCONNECT_ROCE (新增) |
| --- | --- | --- |
| **grid 形状** | `(48, 8, 2)` | `(2, 8, 2)` |
| **最大设备数** | 768 dies | 32 dies（仅双机） |
| **tier 0（节点间）** | 两级 CLOS，196 GB/s，5.5 μs | **RoCE，24.5 GB/s，5.5 μs** |
| **tier 1（机内）** | 一级 CLOS，196 GB/s，0.5 μs | 一级 CLOS，196 GB/s，0.5 μs（不变） |
| **tier 2（板内 SIO）** | SIO，224 GB/s，0.2 μs | SIO，224 GB/s，0.2 μs（不变） |

#### 关键设计决策

1. **grid 第一维从 48 → 2**：限制最大节点数为 2（仅双机），从通信拓扑层面防止配置出超过双机的规模。
2. **tier 0 带宽降为 1/8**：`196 * 1e9 / 8 = 24.5 GB/s`，模拟 RoCE 网络实际带宽。使用浮点表达式 `196 * 1e9 / 8` 而非硬编码数值，保持可追溯性。
3. **tier 1 / tier 2 与原 A3_INTERCONNECT 一致**：机内和板内互联与标准 A3 相同，仅节点间互联有差异。

#### CommGrid 三维拓扑语义

grid = [2, 8, 2]
         │  │  │
         │  │  └── dim 2 (SIO):  板内 2 个 die 通过 SIO 互联
         │  └───── dim 1 (CLOS): 机内 8 个 SIO 组通过一级 CLOS 互联
         └──────── dim 0 (RoCE):  2 个节点通过 RoCE 互联

### 2.3 设备 Profile 定义

#### A3_560T_128G_DIE_ROCE

name:  "ATLAS_800_A3_560T_128G_DIE_ROCE"
vendor: "HUAWEI"
comm_grid: A3_INTERCONNECT_ROCE
mma_ops:  {float32: 75T, bfloat16: 245.8T, half: 280T, int8: 560T}
gp_ops:   {float32: 8T, bfloat16/half: 16T}
memory:   64 GB @ 1.6 TB/s
efficiency: compute=0.7, memory=0.6

### 2.4 代码组织结构

所有新增代码位于单文件 `tensor_cast/device.py` 的 `ATLAS_800` 类内：

| 新增项 | 行位置 | 说明 |
| --- | --- | --- |
| `A3_INTERCONNECT_ROCE` | ~L184 | RoCE 通信网格定义 |
| `A3_560T_128G_DIE_ROCE` | ~L383 | 560T die + RoCE |

### 2.5 影响范围

- **新增代码**：`tensor_cast/device.py`（3 个常量定义，~50 行）
- **新增测试**：`tests/test_tensor_cast/test_device.py`（参数化测试框架）
- **无破坏性变更**：现有硬件、API、CLI 均不受影响
- **自动注册**：通过 `DeviceProfile.__post_init__` 自动加入 `all_device_profiles`，CLI 和 Web UI 可直接使用

---

## 3. Usage Instructions (使用说明)

### 3.1 CLI 使用

```powershell
# 使用 560T RoCE 硬件进行文本生成仿真
python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --device ATLAS_800_A3_560T_128G_DIE_ROCE \
    --num-queries 2 \
    --query-length 3500
```

### 3.2 编程接口

```python
from tensor_cast.device import DeviceProfile
import torch

# 通过名称获取
profile = DeviceProfile.all_device_profiles["ATLAS_800_A3_560T_128G_DIE_ROCE"]
print(profile.comm_grid.grid.shape)  # (2, 8, 2)
print(profile.mma_ops[torch.bfloat16])  # 245.8e12
```

### 3.3 约束与限制

1. **仅双机**：`A3_INTERCONNECT_ROCE` 的 grid 第一维为 2，不支持超过 2 节点的规模配置。
2. **仅 die 层级**：与 `A3_560T_128G_DIE` 一致，基于单个 die 建模，不涵盖整芯片层级。
3. **不支持 RoCE 超 2 节点**：如需支持更多节点 RoCE，需新建 CommGrid（如 `grid=(N, 8, 2)`）。

---

## 4. Test Design (测试设计)

测试文件：`tests/test_tensor_cast/test_device.py`

### 4.1 单元测试

**1. A3_INTERCONNECT_ROCE 通信网格**

- 验证 grid shape 为 `(2, 8, 2)`，总设备数为 32（仅双机）。
- 验证 grid ndim 与 topologies 数量一致（均为 3）。
- 验证 tier 0（节点间 RoCE）带宽为 `196 * 1e9 / 8 = 24.5 GB/s`。
- 验证 tier 0 延迟为 `5.5 μs`，通信效率为 `0.7`。
- 验证 tier 1（机内 CLOS）和 tier 2（板内 SIO）的带宽、延迟与原始 `A3_INTERCONNECT` 一致。

**2. DeviceProfile 属性正确性**

通过参数化方式覆盖 `_DEVICE_PROFILE_SPECS` 中所有硬件：

- 验证 Profile 已注册到 `DeviceProfile.all_device_profiles`。
- 验证 `name`、`vendor`（HUAWEI）与 spec 一致。
- 验证 `comm_grid` 引用指向 `A3_INTERCONNECT_ROCE`。
- 验证各 dtype 的 `mma_ops` 和 `gp_ops` 算力值与 spec 一致。
- 验证 `memory_size_bytes`（64 GB）、`memory_bandwidth_bytes_ps`（1.6 TB/s）。
- 验证 `compute_efficiency`（0.7）、`memory_efficiency`（0.6）。
- 验证 `static_cost` 引用指向 `ATLAS_800.STATIC_COST`。

**3. 注册逻辑校验**

- 验证重复名称的 Profile 会触发 `ValueError`。

**4. CommGrid 参数校验**

- 验证 0 维 grid 抛出 `ValueError`。
- 验证 grid ndim 与 topologies 数量不匹配抛出 `ValueError`。
- 验证任一维度 < 2 的 grid 抛出 `ValueError`。

**5. InterconnectTopology 默认值**

- 验证 `comm_efficiency` 默认为 `1.0`。
- 验证 `type` 默认为 `InterconnectType.CLOS`。
- 验证可显式设为 `InterconnectType.FULL_MESH`。

**6. StaticCost 默认值与常量**

- 验证 `StaticCost()` 默认全为 0。
- 验证 `ATLAS_800.STATIC_COST` 的 mma/gp/comm 成本分别为 `5 μs`、`2 μs`、`10 μs`。

### 4.2 集成测试

运行 `tests/test_tensor_cast/` 下所有非 NPU 测试，验证新增硬件不影响已有功能：

```bash
python -m pytest tests/test_tensor_cast/test_device.py -v
```

验证结果为 `29 passed, 52 subtests passed`，覆盖 7 个测试类。

### 4.3 端到端验证

使用 RoCE 硬件执行 text generate CLI 命令，验证全链路可用：

```bash
python -m cli.inference.text_generate <model_id> \
    --device ATLAS_800_A3_560T_128G_DIE_ROCE \
    --num-devices 32 \
    --num-queries 32 \
    --query-length 4 \
    --num-mtp-tokens 3 \
    --context-length 4352 \
    --compile \
    --tp-size 16 \
    --dp-size 2 \
    --ep-size 32 \
    --quantize-linear-action W8A8_DYNAMIC
```

成功标准：

1. 设备名 `ATLAS_800_A3_560T_128G_DIE_ROCE` 可被 CLI 正确识别并加载。
2. `A3_INTERCONNECT_ROCE` 的三层通信拓扑被正确解析（tier 0 为 RoCE，tier 1 为 CLOS，tier 2 为 SIO）。
3. 仿真结果中节点间通信带宽按 RoCE 的 24.5 GB/s 计算，而非原始 HCCS 的 196 GB/s。
4. 与原始 `ATLAS_800_A3_560T_128G_DIE` 对比，相同模型和参数下，通信耗时占比应更高。
