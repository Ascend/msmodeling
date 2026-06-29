<h1 align="center">MindStudio Modeling</h1>

<div align="center">
<p><b><span style="font-size:24px;">昇腾 AI 模型性能建模与仿真工具</span></b></p>

 [![快速入门](https://badgen.net/badge/快速入门/QuickStart/blue)](./docs/zh/quick_start/tensorcast_serving_cast_quick_start.md)
 [![AI问答(DeepWiki)](https://badgen.net/badge/AI问答/DeepWiki/blue)](https://deepwiki.com/Ascend/msmodeling)
 [![AI问答(ZRead)](https://badgen.net/badge/AI问答/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
 [![精确搜索](https://badgen.net/badge/精确搜索/ReadTheDocs/blue)](https://mindstudio-docs-master.readthedocs.io)
 [![昇腾社区](https://badgen.net/badge/昇腾社区/Community/blue)](https://www.hiascend.com/cn/developer/software/mindstudio)
 [![报告问题](https://badgen.net/badge/报告问题/Issues/blue)](https://gitcode.com/Ascend/msmodeling/issues)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.06.10]**：msModeling 新增 **DeepSeek-V4** 模型支持  
🔹 **[2026.04.02]**：msModeling 新增 **GLM5** 模型支持

</span>

## ℹ️ 简介

MindStudio Modeling（msModeling）是专为昇腾 AI 处理器打造的神经网络推理性能仿真与分析框架，提供单模型性能仿真、服务级吞吐优化、服务化参数自动寻优与可视化分析能力，帮助开发者在无物理硬件或部署前期预测模型性能、识别瓶颈并优化配置。

## ⚙️ 功能介绍

msModeling 提供 TensorCast、Throughput Optimizer、ServingCast、Web UI 和 OptiX 等功能模块，覆盖单模型性能仿真、吞吐优化、服务级仿真、可视化交互与服务化参数自动寻优等场景。模型与特性覆盖范围请参见《[模型支持与特性支持矩阵](./docs/zh/user_guide/support_matrix/support_matrix_user_guide.md)》。

| 功能名称 | 功能描述 |
|---------|--------|
| [**TensorCast**](./docs/zh/user_guide/msmodeling_tensor_cast_user_guide.md) | 算子仿真模块，拦截 PyTorch 计算图，在指定 DeviceProfile 上模拟推理过程，输出算子级性能分解、内存占用、算子 shape 及 Chrome Trace。 |
| [**Throughput Optimizer**](./docs/zh/user_guide/msmodeling_throughput_optimizer_user_guide.md) | 吞吐优化模块，在 SLO 约束下自动搜索最优并行策略与 batch 配置，支持 PD 混部、PD 分离、PD 配比三种模式。 |
| [**ServingCast**](./docs/zh/user_guide/msmodeling_serving_cast_user_guide.md) | 服务级推理仿真模块，基于 YAML 配置模拟多实例、多请求的端到端 serving 场景，输出吞吐、TTFT、TPOT 等系统级指标。 |
| [**Web UI**](./docs/zh/user_guide/msmodeling_web_ui_user_guide.md) | 可视化交互界面，支持通过页面配置模型、芯片、并行、量化和 workload 参数，并查看曲线、表格和导出结果。 |
| [**OptiX**](./docs/zh/user_guide/optix_user_guide.md) | 服务化参数自动寻优工具，基于 PSO 粒子寻优算法对 vLLM、MindIE 等服务框架进行参数寻优与验证。 |

## 🚀 快速入门

以 TensorCast 单模型仿真与 ServingCast 服务仿真为例，快速跑通核心流程，请参见《[TensorCast 与 ServingCast 快速入门](./docs/zh/quick_start/tensorcast_serving_cast_quick_start.md)》。

## 📦 安装指南

介绍工具的环境依赖与安装方法，请参见《[msModeling 安装指南](./docs/zh/install_guide/msmodeling_install_guide.md)》。

## 📘 使用指南

各工具的详细使用说明请参阅其源码仓库中的 README 文件，也可通过上方功能介绍表格中的链接直接跳转。

## 💡 典型案例

通过典型问题场景帮助用户理解并掌握工具使用，请参见《[吞吐优化指南](./docs/zh/user_guide/msmodeling_throughput_optimizer_user_guide.md)》与《[服务仿真指南](./docs/zh/user_guide/msmodeling_serving_cast_simulation_user_guide.md)》中的示例。

## ❓ FAQ

常见问题及解决方案，请提交 [Issues](https://gitcode.com/Ascend/msmodeling/issues) 或参见各模块使用指南。

## 🌌 智能检索

为提升文档查阅效率，我们提供多种高效检索方式：  
🔹 [AI 问答（DeepWiki）](https://deepwiki.com/Ascend/msmodeling)：自然语言问答，快速把握项目架构与模块关系。  
🔹 [AI 问答（ZRead）](https://zread.ai/mindstudio-docs/master)：中文问答体验更优，精准定位功能用法与细节。  
🔹 [精确搜索（ReadTheDocs）](https://mindstudio-docs-master.readthedocs.io)：关键词全文检索，直达接口、参数与报错等信息。

## 🛠️ 贡献指南

欢迎参与项目贡献！详细的贡献流程、代码规范、Commit 规范、测试要求等，请参见《[CONTRIBUTING.md](CONTRIBUTING.md)》。如有疑问，请提交 [Issues](https://gitcode.com/Ascend/msmodeling/issues)。

## ⚖️ 相关说明

🔹 《[版本说明](https://gitcode.com/Ascend/msmodeling/releases)》  
🔹 《[许可证声明](./docs/zh/legal/LICENSE)》  
🔹 《[安全声明](./docs/zh/legal/SECURITY.md)》  
🔹 免责声明：本工具仿真与优化结果仅供性能评估参考，最终性能表现请以真实环境实测为准

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/Ascend/msmodeling/issues)，我们会尽快回复。感谢您的支持。

**SIG 例会**：MindStudio Modeling Weekly Meeting 每周三 10:00-12:00（UTC+8）举行，会议纪要与议题请参见 [sig-msit-modeling](https://etherpad.ascend.osinfra.cn/p/sig-msit-modeling)，也可使用 [时区转换](https://dateful.com/convert/gmt8?t=15) 查看本地时间。

| 即时互动（微信群） | 官方资讯（公众号） | 深度支持（助手/论坛） |
|:--:|:--:|:--|
| <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*扫码加入技术交流群*</sub> | <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*扫码关注官方公众号*</sub> | 扫码入群并关注公众号，直达 MindStudio 用户与开发者最快捷的交流平台：<br>**快速提问：** 与社区小伙伴即时探讨技术问题<br>**掌握动态：** 第一时间获取版本发布与功能更新通知<br>**经验共享：** 与广大开发者交流最佳实践与实战心得<br><br>**更多支持渠道**：👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 致谢

本工具由以下公司/部门联合贡献：  
🔹 华为昇腾计算产品线 MindStudio 开发部  
🔹 华为2012网络技术实验室  
🔹 华为小巧灵突击队  
🔹 华为OTT5系统部  
🔹 湛卢  
🔹 AI Workload  
🔹 2012马尔科夫实验室

感谢来自社区的每一个 PR，欢迎贡献！
