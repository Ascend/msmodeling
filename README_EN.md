<h1 align="center">MindStudio Modeling</h1>

<div align="center">
<p><b><span style="font-size:24px;">Ascend AI Model Performance Modeling and Simulation Tool</span></b></p>

 [![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](./docs/en/quick_start/tensorcast_throughput_optimizer_quick_start.md)
 [![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q%26A/DeepWiki/blue)](https://deepwiki.com/Ascend/msmodeling)
 [![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q%26A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
 [![Precise Search](https://badgen.net/badge/Exact%20Search/ReadTheDocs/blue)](https://mindstudio-docs-master.readthedocs.io)
 [![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/eng/developer/software/mindstudio)
 [![Report an Issue](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://gitcode.com/Ascend/msmodeling/issues)

</div>

English | [简体中文](./README.md)

## ✨ What's New

<span style="font-size:14px;">

We continue to support mainstream Chinese LLMs, including DeepSeek, Kimi, Qwen, GLM, and MiniMax. Click a series name to expand and view the newly added support entries.<br>

<details>
<summary><b>DeepSeek Series</b></summary>

🔹 **[Oct 20, 2026]**: msModeling adds support for the **DeepSeek-V4.1-Flash** model with text and image inputs.<br>
🔹 **[Jun 4, 2026]**: msModeling adds support for the **DeepSeek-V4** model.<br>
🔹 **[Apr 20, 2026]**: msModeling adds support for the **DeepSeek V3.2** model.<br>
🔹 **[Sep 6, 2025]**: msModeling adds support for the **DeepSeek V3** model.<br>

</details>

<details>
<summary><b>Kimi Series</b></summary>

🔹 **[Aug 27, 2026]**: msModeling adds support for the **Kimi-K3** model.<br>
🔹 **[Jun 11, 2026]**: msModeling adds support for the **Kimi-K2.6** model.<br>
🔹 **[May 27, 2026]**: msModeling adds support for the **Kimi-K2.5** model.<br>
🔹 **[Sep 6, 2025]**: msModeling adds support for the **Kimi-K2** model.<br>

</details>

<details>
<summary><b>Qwen Series</b></summary>

🔹 **[Aug 27, 2026]**: msModeling adds support for the **Qwen3.8 series** models.<br>
🔹 **[Jul 14, 2026]**: msModeling adds image-input support for **Qwen3.6**.<br>
🔹 **[Jul 14, 2026]**: msModeling adds text-input support for **Qwen3.6 Dense/MoE**.<br>
🔹 **[Apr 20, 2026]**: msModeling adds image-input support for **Qwen3.5**.<br>
🔹 **[Mar 31, 2026]**: msModeling adds text-input support for **Qwen3.5 Dense/MoE**.<br>
🔹 **[Dec 25, 2025]**: msModeling adds support for the **Qwen3 MoE** model.<br>
🔹 **[Sep 18, 2025]**: msModeling adds support for the **Qwen3-Next** model.<br>
🔹 **[Aug 18, 2025]**: msModeling adds support for the **Qwen3 Dense** model.<br>

</details>

<details>
<summary><b>GLM Series</b></summary>

🔹 **[Sep 16, 2026]**: msModeling adds support for the **GLM-5.3-Flash** model.<br>
🔹 **[Jul 7, 2026]**: msModeling adds support for the **GLM5.2** model.<br>
🔹 **[Jun 4, 2026]**: msModeling adds support for the **GLM5.1** model.<br>
🔹 **[Apr 30, 2026]**: msModeling adds support for the **GLM5** model.<br>
🔹 **[Mar 31, 2026]**: msModeling adds support for the **GLM-4 MoE** model.<br>

</details>

<details>
<summary><b>MiniMax Series</b></summary>
🔹 **[Jul 31, 2026]**: msModeling adds support for the **MiniMax M3** model.<br>
🔹 **[Jun 25, 2026]**: msModeling adds support for the **MiniMax M2.7** model.<br>
🔹 **[May 29, 2026]**: msModeling adds support for the **MiniMax M2.5** model.<br>
🔹 **[Dec 18, 2025]**: msModeling adds support for the **MiniMax M2** model.<br>

</details>

<details>
<summary><b>Step Series</b></summary>

🔹 **[Oct 26, 2026]**: msModeling adds support for the **Step 5 Preview** model with text and image inputs.<br>

</details>

<details>
<summary><b>Recent Feature Updates</b></summary>

🔹 **[Sep 15, 2026]**: Adds a calibrated analytical performance model that uses calibrated profiles to correct the Roofline latency estimates.<br>
🔹 **[Sep 14, 2026]**: Adds support for joint search of pipeline parallelism (PP) with MTP, DFlash, and DSpark speculative decoding.<br>
🔹 **[Sep 8, 2026]**: Adds an AI agent closed-loop optimization mode to OptiX.<br>
🔹 **[Sep 2, 2026]**: Adds pipeline parallelism (PP) search and explicit layer partitioning to serving throughput optimization.<br>
🔹 **[Aug 27, 2026]**: Adds query-driven shape grid generation and standalone MicroBench replay to the Profiling data tool.<br>
🔹 **[Aug 26, 2026]**: Adds support for Atlas A5 series device profiles.<br>
🔹 **[Aug 21, 2026]**: Adds support for DFlash and DSpark speculative decoding to model and serving simulation.<br>
🔹 **[Aug 4, 2026]**: Adds a Theory↔Runtime model diagnostic tool for structure and operator call reconciliation.<br>
🔹 **[Aug 3, 2026]**: Adds simulation and search for decode context parallelism (DCP).<br>

</details>

</span>

<span style="font-size:14px; display:block; margin-top:24px;">

Click a module name to expand the list of **supported features**.<br>

<details>
<summary><b>Model Inference Performance Simulation</b></summary>

🔹 Multi-hardware simulation for Ascend devices, including Atlas 800 A2/A3 and Atlas 350/850/850E/950 (A5), with support for custom device profiles<br>
🔹 Separate simulation of the LLM prefill and decode stages<br>
🔹 Prefix Cache simulation<br>
🔹 MTP, DFlash, and DSpark speculative decoding simulation<br>
🔹 Compilation and graph optimization, and multi-stream compute-communication overlap<br>
🔹 Quantization simulation, including `W8A8`, `W4A8`, `FP8`, and `MXFP4`<br>
🔹 Parallelism and MoE extensions, including TP, DP, EP, PP, and DCP, as well as fine-grained parallelism such as Embedding TP and Vision TP<br>
🔹 Switching among the Roofline, calibrated analytical, and Profiling performance models<br>
🔹 Chrome Trace and debugging<br>
🔹 DiT simulation for video generation, including Ulysses, CFG, and DiT Cache<br>

</details>

<details>
<summary><b>Serving Performance Simulation</b></summary>

🔹 Throughput optimization for LLM and VLM workloads under constraints such as TTFT, TPOT, and serving cost<br>
🔹 PD modes, including colocation, disaggregation, and ratio-based deployment<br>
🔹 Parallel strategy search, including TP, EP, MOE-DP, PP, and DCP<br>
🔹 MTP, DFlash, and DSpark speculative decoding configuration search, with support for joint search with PP<br>
🔹 Chunked prefill simulation<br>
🔹 Prefix Cache simulation<br>
🔹 Variable-length workload simulation<br>
🔹 Multi-stream compute-communication overlap<br>
🔹 Cross-hardware comparison<br>

</details>

<details>
<summary><b>Web UI</b></summary>

🔹 LLM and VL forward-pass simulation, and video generation simulation<br>
🔹 Throughput optimization experiments, including PD colocation, disaggregation, and ratio-based deployment<br>
🔹 Command preview and task caching<br>
🔹 Result display and export, including charts, tables, device memory and operator details, and Excel files<br>

</details>

<details>
<summary><b>Serving Optimization Based on Real-World Measurements</b></summary>

🔹 Serving framework optimization based on real-world measurements using particle swarm optimization (PSO) and Early Rejection, as well as AI agent closed-loop optimization<br>
🔹 Multi-engine support, including vLLM and MindIE, as well as evaluation strategies<br>
🔹 Custom optimization configurations and checkpoint-based resumption<br>

</details>

</span>

## ℹ️ Introduction

MindStudio Modeling (msModeling) is a neural network inference performance simulation and analysis framework designed for Ascend AI processors. It provides single-model performance simulation, service-level throughput optimization, automated parameter optimization for serving, and visual analysis. It helps developers predict model performance, identify bottlenecks, and optimize configurations when hardware is unavailable or before deployment.

## ⚙️ Features

msModeling provides modules for model inference performance simulation, serving performance simulation, and serving optimization based on real-world measurements. These modules cover the corresponding performance simulation and optimization scenarios. For coverage of supported models and features, see the [Model and Feature Support Matrix](./docs/en/user_guide/support_matrix/support_matrix_user_guide.md).

| Feature | Description |
|---------|--------|
| [**Model Inference Performance Simulation**](./docs/en/user_guide/msmodeling_tensor_cast_user_guide.md) | The model simulation module intercepts PyTorch computation graphs, simulates inference using specified device profiles, and outputs operator-level performance breakdowns, memory usage, operator shapes, and Chrome Trace. |
| [**Serving Performance Simulation**](./docs/en/user_guide/msmodeling_throughput_optimizer_user_guide.md) | The throughput optimization simulation module automatically searches for the optimal parallelism strategy and batch configuration under SLO constraints. It supports PD colocation, PD disaggregation, and PD ratio-based deployment. |
| [**Serving Optimization Based on Real-World Measurements**](./docs/en/user_guide/msmodeling_optix_user_guide.md) | Serving optimization based on real-world measurements uses the PSO algorithm to automatically search for optimal deployment parameters that satisfy latency constraints on real serving frameworks such as vLLM and MindIE. msModeling and OptiX must be installed in a uv virtual environment. vLLM, MindIE, and benchmark tools use the system deployment environment by default. |

## 🚀 Quick Start

To quickly run through the core process using model inference performance simulation and serving performance simulation as examples, see [msModeling Quick Start](./docs/en/quick_start/tensorcast_throughput_optimizer_quick_start.md). You can also use the [Web UI User Guide](./docs/en/user_guide/msmodeling_web_ui_user_guide.md) to configure tasks interactively and view results.

## 📦 Installation Guide

For environment dependencies and installation instructions, see the [msModeling Installation Guide](./docs/en/install_guide/msmodeling_install_guide.md).

## 📘 User Guide

For detailed usage instructions for each tool, refer to the `README` file in its source code repository, or jump directly via the links in the function introduction table above.

## 💡 Typical Use Cases

To help you understand and master the tools through typical problem scenarios, see the examples in [Model Inference Performance Simulation](./docs/en/user_guide/msmodeling_tensor_cast_user_guide.md) and [Serving Performance Simulation](./docs/en/user_guide/msmodeling_throughput_optimizer_user_guide.md).

## ❓ FAQ

For common issues and solutions, open an [issue](https://gitcode.com/Ascend/msmodeling/issues) or see the user guide for the relevant module.

## 🌌 Intelligent Search

To improve documentation search efficiency, we provide multiple efficient search methods:<br>
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/Ascend/msmodeling): Ask questions in natural language to quickly understand the project architecture and relationships between modules.<br>
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/master): Provides a better Chinese Q&A experience and accurately locates feature usage and details.<br>
🔹 [Precise Search (ReadTheDocs)](https://mindstudio-docs-master.readthedocs.io): Search the full text by keyword to find information about interfaces, parameters, and error messages directly.

## 🛠️ Contribution Guide

Contributions to the project are welcome. For detailed contribution procedures, coding standards, commit conventions, testing requirements, and more, see the [msModeling Contribution Guide](CONTRIBUTING.md). If you have questions, open an [issue](https://gitcode.com/Ascend/msmodeling/issues).

### Dev Container Development Environment

The repository provides a VS Code Dev Container based on Python 3.11. For the startup method, automatic initialization content, and the unified build and UT entry point, see the [Dev Container Guide](.devcontainer/README.md).

### AI Native Development

This repository supports AI-assisted issue drafting and review, end-to-end delivery from issue to PR/CI, multi-role and inline PR review, and
closed-loop handling of openLiBing CI failures. The unified entry point for AI is [AGENTS.md](AGENTS.md). For workflow descriptions and business examples, see
the [AI Native Development Guide](docs/ai-native/README.md).

- Use the `gitcode` CLI for all GitCode remote operations.
- By default, confirm each stage. You can also explicitly authorize automatic progress within a limited scope.
- Record key decisions and verification evidence in issue/PR comments.
- You can pause, modify the plan, take over, or resume at any time.
- AI does not automatically approve, merge, force push, or bypass gates.

Example: `Implement Issue 123. First provide the requirement analysis and solution, have me confirm each key stage, and leave an audit record in the issue/PR.`

## ⚖️ Related Information

🔹 [Release Notes](https://gitcode.com/Ascend/msmodeling/releases)<br>
🔹 [License Notice](./docs/en/legal/LICENSE)<br>
🔹 [Security Statement](./docs/en/legal/SECURITY.md)<br>
🔹 Disclaimer: The simulation and optimization results provided by this tool are for performance evaluation only. Use measurements from a real environment as the final reference for actual performance.

## 🤝 Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, open an [issue](https://gitcode.com/Ascend/msmodeling/issues), and we will respond as soon as possible. Thank you for your support.

**SIG Weekly Meeting**: The MindStudio Modeling Weekly Meeting is held every Wednesday from 10:00 to 12:00 (UTC+8). For meeting minutes and topics, see [sig-msit-modeling](https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling). You can also use the [Time Zone Converter](https://dateful.com/convert/gmt8?t=15) to view the local time.

| Instant Interaction (WeChat Group) | Official Updates (Official Account) | In-Depth Support (Assistant/Forum) |
|:--:|:--:|:--|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan to join the technical discussion group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan to follow the official account*</sub> | Scan the QR codes to join the group and follow the official account, providing the fastest way to connect with MindStudio users and developers:<br>**Ask questions quickly:** Discuss technical issues with community members in real time.<br>**Stay up to date:** Get first-hand notifications about releases and feature updates.<br>**Share experience:** Exchange best practices and hands-on experience with developers.<br><br>**More support channels:** 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by Huawei Technologies Co., Ltd. and developers.<br>
Thanks for every PR from the community, and contributions are welcome.
