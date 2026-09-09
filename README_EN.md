<h1 align="center">MindStudio Modeling</h1>

<div align="center">
<p><b><span style="font-size:24px;">Ascend AI Model Performance Modeling and Simulation Tool</span></b></p>

 [![Quick Start](https://badgen.net/badge/快速入门/QuickStart/blue)](./docs/en/quick_start/tensorcast_throughput_optimizer_quick_start.md)
 [![AI Q&A (DeepWiki)](https://badgen.net/badge/AI问答/DeepWiki/blue)](https://deepwiki.com/Ascend/msmodeling)
 [![AI Q&A (ZRead)](https://badgen.net/badge/AI问答/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
 [![Precise Search](https://badgen.net/badge/精确搜索/ReadTheDocs/blue)](https://mindstudio-docs-master.readthedocs.io)
 [![Ascend Community](https://badgen.net/badge/昇腾社区/Community/blue)](https://www.hiascend.com/cn/developer/software/mindstudio)
 [![Report an Issue](https://badgen.net/badge/报告问题/Issues/blue)](https://gitcode.com/Ascend/msmodeling/issues)

</div>

English | [简体中文](./README.md)

## ✨ What's New

<span style="font-size:14px;">

We continue to support mainstream Chinese LLMs, including DeepSeek, Kimi, Qwen, GLM, and MiniMax. Click a series name to expand and view the newly added support entries.

<details>
<summary><b>DeepSeek Series</b></summary>

🔹 **[Jun 4, 2026]**: msModeling adds support for the **DeepSeek-V4** model.  
🔹 **[Apr 20, 2026]**: msModeling adds support for the **DeepSeek V3.2** model.  
🔹 **[Sep 6, 2025]**: msModeling adds support for the **DeepSeek V3** model.

</details>

<details>
<summary><b>Kimi Series</b></summary>

🔹 **[Jun 11, 2026]**: msModeling adds support for the **Kimi-K2.6** model.  
🔹 **[May 27, 2026]**: msModeling adds support for the **Kimi-K2.5** model.  
🔹 **[Sep 6, 2025]**: msModeling adds support for the **Kimi-K2** model.

</details>

<details>
<summary><b>Qwen Series</b></summary>

🔹 **[Apr 20, 2026]**: msModeling adds image-input support for **Qwen3.5**.  
🔹 **[Mar 31, 2026]**: msModeling adds text-input support for **Qwen3.5 Dense / MoE**.  
🔹 **[Dec 25, 2025]**: msModeling adds support for the **Qwen3 MoE** model.  
🔹 **[Sep 18, 2025]**: msModeling adds support for the **Qwen3-Next** model.  
🔹 **[Aug 18, 2025]**: msModeling adds support for the **Qwen3 Dense** model.

</details>

<details>
<summary><b>GLM Series</b></summary>

🔹 **[Jun 4, 2026]**: msModeling adds support for the **GLM5.1** model.  
🔹 **[Apr 30, 2026]**: msModeling adds support for the **GLM5** model.  
🔹 **[Mar 31, 2026]**: msModeling adds support for the **GLM-4 MoE** model.

</details>

<details>
<summary><b>MiniMax Series</b></summary>

🔹 **[Dec 18, 2025]**: msModeling adds support for the **MiniMax M2** model.

</details>

</span>

<span style="font-size:14px; display:block; margin-top:24px;">

Click a module name to expand the list of **supported features**.

<details>
<summary><b>Model Inference Performance Simulation</b></summary>

🔹 Multi-hardware simulation for Ascend devices, including Atlas 800 A2/A3 and Atlas 350, with support for custom device profiles  
🔹 Separate simulation of the LLM prefill and decode stages  
🔹 Prefix Cache simulation  
🔹 MTP speculative decoding simulation  
🔹 Compilation and graph optimization, and multi-stream compute-communication overlap  
🔹 Quantization simulation, including `W8A8`, `W4A8`, `FP8`, and `MXFP4`  
🔹 Parallelism and MoE extensions, including TP, DP, EP, and fine-grained parallelism such as Embedding TP and Vision TP  
🔹 Switching between the Roofline and Profiling performance models  
🔹 Chrome Trace and debugging  
🔹 DiT simulation for video generation, including Ulysses, CFG, and DiT Cache

</details>

<details>
<summary><b>Serving Performance Simulation</b></summary>

🔹 Throughput optimization for LLM and VLM workloads under constraints such as TTFT, TPOT, and serving cost  
🔹 PD modes, including colocation, disaggregation, and ratio-based deployment  
🔹 Parallel strategy search, including TP, EP, and MOE-DP  
🔹 MTP configuration search  
🔹 Chunked prefill simulation  
🔹 Prefix Cache simulation  
🔹 Variable-length workload simulation  
🔹 Multi-stream compute-communication overlap  
🔹 Cross-hardware comparison

</details>

<details>
<summary><b>Web UI</b></summary>

🔹 LLM and VL forward-pass simulation, and video generation simulation  
🔹 Throughput optimization experiments, including PD colocation, disaggregation, and ratio-based deployment  
🔹 Command preview and task caching  
🔹 Result display and export, including charts, tables, device memory and operator details, and Excel files

</details>

<details>
<summary><b>Serving Optimization Based on Real-World Measurements</b></summary>

🔹 Serving framework optimization based on real-world measurements using particle swarm optimization (PSO) and Early Rejection  
🔹 Multi-engine support, including vLLM and MindIE, as well as evaluation strategies  
🔹 Custom optimization configurations and checkpoint-based resumption

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
| [**Serving Optimization Based on Real-World Measurements**](./docs/en/user_guide/optix_user_guide.md) | Serving optimization based on real-world measurements uses the PSO algorithm to automatically search for optimal deployment parameters that satisfy latency constraints on real serving frameworks such as vLLM and MindIE. |

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

Contributions to the project are welcome. For detailed contribution procedures, coding standards, commit conventions, testing requirements, and more, see [CONTRIBUTING.md](CONTRIBUTING.md). If you have questions, open an [issue](https://gitcode.com/Ascend/msmodeling/issues).

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
| <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan to join the technical discussion group*</sub> | <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan to follow the official account*</sub> | Scan the QR codes to join the group and follow the official account, providing the fastest way to connect with MindStudio users and developers:<br>**Ask questions quickly:** Discuss technical issues with community members in real time.<br>**Stay up to date:** Get first-hand notifications about releases and feature updates.<br>**Share experience:** Exchange best practices and hands-on experience with developers.<br><br>**More support channels:** 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by the following companies and departments, in no particular order:

🔹 **Huawei**<br>
&emsp;&emsp;Ascend Computing Products Department<br>
&emsp;&emsp;Zhanlu and AI Workload<br>
&emsp;&emsp;2012 Network Technology Laboratory and 2012 Markov Laboratory<br>
&emsp;&emsp;Xiaoqiaoling Strike Team and OTT Systems Department

🔹 **Ant Group**

🔹 **China Academy of Information and Communications Technology**

Thank you to all community contributors for every PR. Contributions are welcome.
