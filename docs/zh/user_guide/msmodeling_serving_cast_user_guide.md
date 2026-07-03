# 服务仿真使用指南

## 简介

ServingCast 是一套用于系统级推理服务仿真与吞吐优化的工具集。

## 适用对象

本文适用于需要评估大模型服务部署方案的开发者、性能工程师和容量规划人员。阅读本文前，建议先完成《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》中的环境搭建。

## 组件

ServingCast 由两个主要组件构成：

| 组件 | 主要目标 | 推荐阅读 |
| --- | --- | --- |
| 吞吐优化器 | 在 TTFT / TPOT 等 SLO 约束下搜索最优并行策略、batch 和吞吐 | [吞吐优化指南](./msmodeling_throughput_optimizer_user_guide.md) |
| 服务仿真 | 基于 YAML 配置模拟多实例、多请求的端到端 serving 场景 | [服务仿真指南](./msmodeling_serving_cast_simulation_user_guide.md) |

### 1. 吞吐优化器

吞吐优化器可在指定 SLO 约束（例如 TTFT、TPOT 上限）下，自动搜索最优模型配置（并行策略、批大小），以最大化 token 吞吐。

**适用场景：**

- **硬件规划**：在部署前，针对特定硬件上的给定模型，确定最优并行策略（TP/DP/PP）与批大小
- **SLO 约束下的优化**：在满足延迟要求（TTFT/TPOT 限制）的前提下，寻找吞吐最大化的配置
- **PD 分离服务设计**：独立优化 Prefill 与 Decode 阶段，并计算最优 Prefill / Decode 实例配比

**主要特性：**

- PD 混部：优化 Prefill-Decode 合一的服务架构
- PD 分离：将 Prefill 与 Decode 阶段分离，进行独立优化
- PD 配比：计算使系统吞吐最大的 Prefill 与 Decode 实例比例

详见 [吞吐优化指南](./msmodeling_throughput_optimizer_user_guide.md)。

### 2. 服务仿真

服务仿真基于 YAML 配置文件，对多实例、多请求的端到端服务场景进行仿真，并输出系统级指标，如吞吐、延迟（TTFT、TPOT）。

**适用场景：**

- **系统行为验证**：在实际部署前，验证某套服务配置的预期性能
- **多实例基准测试**：仿真复杂服务拓扑，例如独立的 Prefill 与 Decode 集群
- **负载分析**：在特定请求模式与负载特征下评估系统性能
- **资源规划**：确定满足目标吞吐所需的实例数量及其配置

**主要特性：**

- 基于 YAML 的实例与负载配置
- 支持异构实例组
- 全面指标：端到端延迟、TTFT、TPOT、token 吞吐

详见 [服务仿真指南](./msmodeling_serving_cast_simulation_user_guide.md)。
