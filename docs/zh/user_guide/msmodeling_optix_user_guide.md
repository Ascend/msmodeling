# 服务化实测寻优使用指南

本文档介绍OptiX服务化实测寻优工具，涵盖公共环境准备、Agent方式与命令行方式两种寻优方式的选择以及参考资料，帮助用户在真实部署环境中启动服务、执行测评并迭代调整配置，找到满足时延要求、吞吐较高的部署方案，适用于在vLLM PD分离等场景下开展部署方案寻优的用户。

## 1. 简介

OptiX 是 msmodeling 的服务化实测寻优工具。它根据模型、设备、业务负载和性能目标，在真实部署环境中启动服务、执行测评并迭代调整配置，帮助用户找到满足时延要求、吞吐较高的部署方案。

OptiX 的主要能力包括：

- **覆盖多种部署场景**：支持多种服务化参数寻优场景，包括通过扩展插件实现的 vLLM 多机混部和 PD 分离寻优。PD 分离场景可评估 P/D 参数和实例配比，并生成部署相关产物。
- **适配多种模型**：OptiX 的模型范围取决于目标服务框架和部署环境，支持对框架能够部署的模型开展寻优。模型格式、量化方式、并行策略和 PD 通信能力需与所选框架及版本匹配。
- **支持多种工具**：支持接入 vLLM、MindIE 服务框架及 AISBench、vllm_benchmark 测评工具，也可通过插件扩展。使用前需在部署环境中安装所选框架和测评工具。
- **提供两种寻优方式**：Agent 方式由大模型结合调优知识库、环境约束和各轮实测结果，分析瓶颈并生成下一轮候选方案；命令行方式使用粒子群优化（PSO）算法，在配置的参数范围内根据实测结果迭代搜索。两种方式均基于真实服务测评，可按业务负载和吞吐、首 token 时延（TTFT）、后续 token 时延（TPOT）等目标寻优。
- **查看并复用结果**：寻优过程记录各轮配置和实测指标，完成后可查看推荐方案、报告及部署文件。实际生成的产物以所选流程和任务执行结果为准。

本文以 vLLM PD 分离为主要场景，推荐使用 Agent 方式。其他能力和插件的使用条件见[配置与参考](optix/plugin_and_modes.md)及[支持矩阵](support_matrix/support_matrix_user_guide.md)。

## 2. 准备公共环境

进行 PD 分离寻优时，先完成 [PD 分离寻优公共环境准备](../quick_start/optix_pd_environment.md)，已有环境可跳过已完成的步骤。

单机或 MindIE 等非 PD 场景可从[单机标准寻优指南](../quick_start/optix_single_node.md)开始，准备所选服务框架和测评工具，无需四机 PD 环境。

## 3. 选择寻优方式

本指南以 vLLM 和 AISBench 为例，两种方式都需要准备 OptiX、vLLM 部署环境和 AISBench 测评环境。

<a id="31-agent-方式"></a>
<a id="32-命令行方式"></a>

| 寻优方式 | 如何寻优 | 适合场景 |
|---|---|---|
| [Agent 方式](../quick_start/optix_agent_pd_disaggregation.md)（推荐） | 通过对话提供需求；大模型结合调优知识库和环境约束生成候选，OptiX 执行实测，Agent 根据结果分析瓶颈并决定下一轮方案。 | 希望通过对话开展寻优，结合实测分析调整方向并了解方案依据。需准备可执行命令的 Agent 工具及可访问的模型服务。 |
| [命令行方式](../quick_start/optix_pso_pd_disaggregation.md) | 手工配置搜索参数和范围，运行命令；OptiX 使用粒子群优化（PSO）算法生成候选，并根据实测结果迭代搜索。 | 适合希望通过配置文件和命令控制搜索过程，或无法访问 Agent 模型服务、使用内网或离线环境的场景。 |

## 4. 参考资料

- [插件、安装与寻优模式](optix/plugin_and_modes.md)
- [配置参数参考](optix/configuration.md)
- [命令与结果参考](optix/commands_and_results.md)
- [环境与故障排查](optix/troubleshooting.md)

<details open>
<summary>参考资料章节索引</summary>

按主题查找具体章节。

<a id="2-适用对象与阅读路径"></a>
[适用对象与阅读路径](#3-选择寻优方式)

**OptiX 插件、安装与寻优模式**

- <a id="3-插件支持"></a>[1. 插件支持](optix/plugin_and_modes.md#3-插件支持)
- <a id="4-环境准备与安装"></a>[2. 环境准备与安装](optix/plugin_and_modes.md#4-环境准备与安装)
- <a id="环境隔离原则"></a>[2.1 环境隔离原则](optix/plugin_and_modes.md#环境隔离原则)
- <a id="安装步骤"></a>[2.2 安装步骤](optix/plugin_and_modes.md#安装步骤)
- <a id="验证安装"></a>[2.3 验证安装](optix/plugin_and_modes.md#验证安装)
- <a id="卸载"></a>[2.4 卸载](optix/plugin_and_modes.md#卸载)
- <a id="5-快速入门"></a>[3. 快速入门](optix/plugin_and_modes.md#5-快速入门)
- <a id="6-寻优模式pso-与-agent"></a>[4. 寻优模式：PSO 与 Agent](optix/plugin_and_modes.md#6-寻优模式pso-与-agent)
- <a id="agent-模式工作方式"></a>[4.1 Agent 模式工作方式](optix/plugin_and_modes.md#agent-模式工作方式)

**OptiX 配置参数参考**

- <a id="7-配置文件说明"></a>[1. 配置文件说明](optix/configuration.md#7-配置文件说明)
- <a id="寻优参数"></a>[2. 寻优参数](optix/configuration.md#寻优参数)
- <a id="测评工具参数"></a>[3. 测评工具参数](optix/configuration.md#测评工具参数)
- <a id="aisbench-工具参数"></a>[3.1 AISBench 工具参数](optix/configuration.md#aisbench-工具参数)
- <a id="vllm_benchmark-工具参数"></a>[3.2 vllm_benchmark 工具参数](optix/configuration.md#vllm_benchmark-工具参数)
- <a id="服务化参数"></a>[4. 服务化参数](optix/configuration.md#服务化参数)
- <a id="vllm-服务化参数"></a>[4.1 vLLM 服务化参数](optix/configuration.md#vllm-服务化参数)
- <a id="vllm-自定义参数寻优"></a>[4.2 vLLM 自定义参数寻优](optix/configuration.md#vllm-自定义参数寻优)
- <a id="示例一vllm-环境变量寻优"></a>[4.2.1 示例一：VLLM 环境变量寻优](optix/configuration.md#示例一vllm-环境变量寻优)
- <a id="示例二命令行参数寻优"></a>[4.2.2 示例二：命令行参数寻优](optix/configuration.md#示例二命令行参数寻优)
- <a id="vllm-常用寻优字段"></a>[4.3 VLLM 常用寻优字段](optix/configuration.md#vllm-常用寻优字段)
- <a id="mindie-服务化参数"></a>[4.4 MindIE 服务化参数](optix/configuration.md#mindie-服务化参数)
- <a id="mindie-常用寻优字段"></a>[4.5 MindIE 常用寻优字段](optix/configuration.md#mindie-常用寻优字段)
- <a id="target_field-配置文件说明"></a>[4.5.1 target_field 配置文件说明](optix/configuration.md#target_field-配置文件说明)
- <a id="target_field-支持的-dtype-类型"></a>[4.5.2 target_field 支持的 dtype 类型](optix/configuration.md#target_field-支持的-dtype-类型)
- [5. 高级配置](optix/configuration.md#5-高级配置)
- <a id="部署环境"></a>[5.1 部署环境](optix/configuration.md#部署环境)
- <a id="结果存储与精调"></a>[5.2 结果存储与精调](optix/configuration.md#结果存储与精调)
- <a id="benchmark-早停"></a>[5.3 Benchmark 早停](optix/configuration.md#benchmark-早停)
- <a id="日志检测"></a>[5.4 日志检测](optix/configuration.md#日志检测)
- <a id="三元派生类型使用示例"></a>[6. 三元派生类型使用示例](optix/configuration.md#三元派生类型使用示例)
- <a id="场景一dp-由总卡数自动推导"></a>[6.1 场景一：dp 由总卡数自动推导](optix/configuration.md#场景一dp-由总卡数自动推导)

**OptiX 命令与结果参考**

- <a id="8-命令参数说明"></a>[1. 命令参数说明](optix/commands_and_results.md#8-命令参数说明)
- <a id="命令格式"></a>[1.1 命令格式](optix/commands_and_results.md#命令格式)
- <a id="注意事项"></a>[1.2 注意事项](optix/commands_and_results.md#注意事项)
- <a id="参数说明"></a>[1.3 参数说明](optix/commands_and_results.md#参数说明)
- <a id="使用示例"></a>[1.4 使用示例](optix/commands_and_results.md#使用示例)
- <a id="内置-pd-分离两步寻优"></a>[1.5 内置 PD 分离两步寻优](optix/commands_and_results.md#内置-pd-分离两步寻优)
- <a id="9-结果文件说明"></a>[2. 结果文件说明](optix/commands_and_results.md#9-结果文件说明)

**OptiX 环境与故障排查**

- <a id="10-附录"></a>[1. 附录](optix/troubleshooting.md#10-附录)
- <a id="环境变量与排障"></a>[1.1 环境变量与排障](optix/troubleshooting.md#环境变量与排障)
- <a id="日志级别与查看"></a>[1.2 日志级别与查看](optix/troubleshooting.md#日志级别与查看)
- <a id="故障排查"></a>[1.3 故障排查](optix/troubleshooting.md#故障排查)
- <a id="cli-退出码"></a>[1.4 CLI 退出码](optix/troubleshooting.md#cli-退出码)

</details>
