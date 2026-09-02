# Web UI 使用说明

> **⚠ 安全须知**
>
> Web UI 服务**无认证机制**，绑定在回环地址（`127.0.0.1` / `::1`）。回环绑定仅阻止*远程*网络访问——TCP 回环对**本机所有用户可见**。同机上的任何用户都可以调用全部 API、提交任务、读取结果。
>
> **本服务仅限单用户、单机使用。** 请勿在多用户共享主机上运行。若服务进程以高权限（root / 管理员）运行，则构成本地提权风险。

本文档面向 Modeling 的日常使用者和即将接入项目的开发者，目标是帮助你快速理解工具能做什么、如何从 Web UI 或 CLI 发起仿真、如何解读结果，以及在不同业务场景下应该如何配置参数。

如果只想启动 Web UI，在仓库根目录（venv 已激活）运行启动器即可：

```bash
python web_ui/main.py
```

启动后在浏览器打开 `http://127.0.0.1:5173`。前端会自动将 `/api` 请求代理到后端 `http://127.0.0.1:8000`。

---

## 阅读导航

| 目标 | 推荐章节 |
| --- | --- |
| 第一次启动 Web UI | [3. Web UI 快速上手](#web-ui-quick-start) |
| 配置 LLM / VL 仿真 | [4. LLM / VL 仿真使用指南](#llm-vl-simulation) |
| 配置视频生成仿真 | [5. Video Generation 仿真使用指南](#video-generation-simulation) |
| 使用吞吐优化器 | [6. Optimizer 吞吐寻优使用指南](#optimizer-guide) |
| 解读结果与导出数据 | [7. 结果图和明细表怎么看](#results-guide) |

---

## 1. 工具定位

Modeling 是一个面向模型推理性能分析的仿真工具，核心能力包括：

- 在没有真实硬件或真实大模型完整运行环境的情况下，基于设备画像预测算子耗时、显存占用、通信开销和整体推理时间。
- 支持 LLM 文本推理、VL 多模态推理、视频生成 Diffusion Transformer 推理，以及服务化吞吐寻优。
- 支持多芯片横向对比，帮助判断同一个模型在不同设备上的性能差异。
- 支持并发、TP、量化、MTP、Prefix Cache、Ulysses、DiT Cache、PD 混部、PD 分离、PD 配比等参数组合分析。
- Web UI 提供可视化图表、明细表格、case 选择、CSV 导出、任务历史、Chrome trace 下载和历史缓存；CLI 适合脚本化批量实验。

仓库中与用户最相关的入口如下：

| 入口 | 作用 | 推荐使用场景 |
| --- | --- | --- |
| `python web_ui/main.py` | 启动 Vue 3 + FastAPI Web UI（单命令同时启动前后端） | 交互式配置、结果可视化、非开发用户使用 |
| `python -m cli.inference.text_generate` | LLM / VL 前向推理仿真 | 单次或脚本化 LLM/VL 性能分析 |
| `python -m cli.inference.video_generate` | 视频生成模型仿真 | Diffusion Transformer / Wan / HunyuanVideo 等场景 |
| `python -m cli.inference.image_generate` | 图像生成模型仿真（Transformer 去噪阶段） | Diffusion Transformer / FLUX / Qwen-Image-Edit 等场景 |
| `python -m cli.inference.throughput_optimizer` | 服务吞吐寻优 | 在 TTFT/TPOT/SLO 约束下寻找最优并行和 batch |

---

## 2. 环境准备

完整的环境搭建步骤（克隆仓库、创建虚拟环境、安装依赖、设置 `PYTHONPATH` 与 Hugging Face 访问）请参阅《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》。

若已完成环境搭建，从仓库根目录启动 Web UI 一般无需额外配置。工具会读取模型配置，常见来源包括 Hugging Face、ModelScope 或本地模型目录；若网络无法访问 Hugging Face，可在 Web UI 的 `remote-source` 中选择 `modelscope`，或按安装指南设置 `HF_ENDPOINT` 镜像。

### 2.1 Web UI 额外依赖

Web UI 采用前后端分离架构，除上述 Python 依赖外，还需要安装前端依赖：

**后端依赖**（已包含在仓库根 `pyproject.toml` 中，随主项目一起安装）：

- FastAPI、uvicorn、sqlmodel、alembic、pydantic 等

**前端依赖**（需要 Node.js ≥ 18 和 npm）：

```bash
cd web_ui/frontend
npm install
```

此命令安装 Vue 3、Element Plus、ECharts、Pinia 等前端库，只需执行一次。后续 `npm run dev` 会自动检测依赖变化。

> **Node.js 安装**：如未安装 Node.js，可从 [nodejs.org](https://nodejs.org/) 下载 LTS 版本，或使用 nvm / fnm 等版本管理工具。

---

<a id="web-ui-quick-start"></a>

## 3. Web UI 快速上手

### 3.1 启动本地页面

Web UI 采用前后端分离架构。

**首次启动前**，需安装前端依赖（只需一次）：

```bash
cd web_ui/frontend && npm install
```

同时确保已安装 Python 依赖（详见 [安装指南](../install_guide/msmodeling_install_guide.md)）：

```bash
uv sync  # 或 pip install -e .
```

然后在仓库根目录（venv 已激活）运行启动器：

```bash
python web_ui/main.py
```

启动器会并发启动前端（Vite dev server，默认端口 5173）和后端（FastAPI，默认端口 8000），两路输出以 `[backend]` / `[frontend]` 前缀合并显示，Ctrl+C 同时清理整个进程树。

浏览器打开：

```text
http://127.0.0.1:5173
```

前端 Vite dev server 会自动将 `/api` 请求代理到后端 `http://127.0.0.1:8000`。

### 3.2 Web UI 页面说明

Web UI 为单页应用（SPA），顶部导航栏提供以下功能：

| 导航按钮 | 说明 |
| --- | --- |
| 主页 | 返回工作台（Console） |
| 使用文档 | 内嵌本文档 |
| 历史记录 | 查看已提交任务的历史列表、状态和结果 |
| 语言切换 | 中文 / English 实时切换 |
| 主题切换 | 亮色 / 暗色 实时切换 |

主工作区 **Console（工作台）** 采用 **Tab + 上下分屏** 布局，三个模块共享同一页面：

| Tab | 能力 |
| --- | --- |
| 文本生成 (Text Generation) | LLM / VL 前向推理仿真，支持并发列表、TP 列表、量化、MTP、Prefix Cache、并行细分、算子和显存分析 |
| 视频生成 (Video Generation) | 视频生成模型推理仿真，支持 Ulysses、CFG、DiT Cache、Chrome Trace 等参数 |
| 吞吐优化 (Throughput Optimizer) | 服务吞吐寻优，支持 `PD 混部`、`PD 分离`、`PD 配比` 三种部署模式 |

每个 Tab 的工作区分为上下两部分：

- **上半部分**：配置表单（字段从 TypeScript 配置动态生成，分组折叠，鼠标悬停字段名可查看中英文说明）
- **下半部分**：结果面板（随任务状态变化：空闲占位 → 运行中 → 成功结果 / 失败详情）
- 中间有可拖拽的分割条，可调整表单和结果的显示比例

### 3.3 Web UI 的基本操作流程

1. 选择模型、主芯片和可选竞品芯片。
2. 填写卡数、并发、长度、量化、并行等参数。
3. 点击 **▶ 运行** 按钮提交任务。提交成功后会弹出 Toast 通知（含任务 ID）。
4. 结果面板自动切换为运行中状态，显示旋转图标和进度文本，可随时查看日志或取消任务。
5. 任务完成后，结果面板展示汇总结论、散点图/曲线图、显存分析、算子详情等。
6. 如果设置了多值字段（如并发列表、TP 列表），系统自动展开为多个 case，结果区以多用例视图分组展示。
7. 点击顶部 **历史记录** 可查看所有历史任务，点击可直接查看结果。

> **提示**：任务运行中不可切换 Tab（会弹出警告提示），需等待完成或取消后再切换。

---

<a id="llm-vl-simulation"></a>

## 4. LLM / VL 仿真使用指南

LLM 和 VL 仿真最终都调用：

```bash
python -m cli.inference.text_generate <model_id> [options]
```

其中 VL 是在 LLM 仿真基础上增加图像输入参数。

### 4.1 关键概念

| 概念 | 说明 |
| --- | --- |
| `num-queries` | 并发请求数，影响 batch、KV Cache、显存和吞吐 |
| `query-length` | 本次新增 token 数。prefill 通常较大，decode 通常为 1 或较小值 |
| `context-length` | 已有上下文长度，影响 KV Cache 和 attention 成本 |
| `decode` | 开启自回归 decode 模式 |
| `tp-size` | Tensor Parallel 数量 |
| `dp-size` | Data Parallel 数量，可在 Web UI 中填 `auto` |
| `ep-size` | Expert Parallel 数量，MoE 模型常用 |
| `num-mtp-tokens` | MTP token 数，DeepSeek 等支持 MTP 的模型可用 |
| `prefix-cache-hit-rate` | Prefix Cache 命中率，取值 `[0,1)`，用于估计 prefill token 复用收益 |
| `quantize-linear-action` | Linear 层量化方式，如 `W8A8_DYNAMIC`、`fp8`、`mxfp4` |
| `quantize-non-expert-linear-action` | 非专家 Linear 层量化覆盖项，主要用于 DeepSeek V4；作用于 attention projections、dense MLP 和 shared experts；routed MoE experts 仍使用 `quantize-linear-action` |
| `quantize-attention-action` | KV Cache / Attention 量化方式，如 `disabled`、`int8`、`fp8` |
| `image-height/image-width` | VL 图像尺寸 |

### 4.2 最小 LLM 示例：单芯片 decode

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 1 \
  --num-queries 32 \
  --query-length 1 \
  --context-length 4500 \
  --decode \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action disabled
```

适合快速观察某芯片在典型 decode 场景下的单设备推理时间、TPS/Device、显存和算子占比。

### 4.3 Prefill 示例：长输入吞吐和瓶颈分析

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 8 \
  --query-length 3500 \
  --context-length 0 \
  --compile \
  --tp-size 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action int8
```

此场景关注首段输入处理成本，适合比较：

- 不同 TP 下 prefill 是否受通信瓶颈影响。
- Attention 量化是否降低显存和带宽压力。
- `compile` 对图编译和执行耗时的影响。

### 4.4 并发列表示例：绘制并发曲线

Web UI 中可填写：

```text
并发列表: [16,32,64]
TP 并行数: 1
```

等价于多次执行不同 `--num-queries` 的实验。结果区会绘制并发数与推理时间、吞吐等变化关系，适合寻找最佳并发区间。

如果使用 CLI 批量实验，可用脚本循环：

```bash
for nq in 16 32 64; do
  python -m cli.inference.text_generate Qwen/Qwen3-32B \
    --device ATLAS_800_A2_280T_32G_PCIE \
    --num-devices 8 \
    --num-queries $nq \
    --query-length 8 \
    --context-length 4500 \
    --decode \
    --tp-size 1 \
    --quantize-linear-action mxfp4 \
    --quantize-attention-action disabled
done
```

### 4.5 TP 列表示例：同一模型遍历多个 TP

Web UI 中可填写：

```text
部署卡数: 8
请求并发数: 32
TP 列表: [1,2,4,8]
```

工具会在同一并发下遍历多个 TP，并输出 TP 数量与推理时间的变化图。横轴为 TP 数量，纵轴为推理时间。

适合回答：

- 增大 TP 后计算是否加速。
- 通信开销是否抵消了计算收益。
- 当前芯片和模型最适合的 TP 区间。

### 4.6 并发列表 + TP 列表示例

Web UI 中可填写：

```text
部署卡数: 8
并发列表: [16,32,64]
TP 列表: [1,2]
```

工具会按照每个 TP 遍历并发，并输出每个 TP 下的并发曲线。结果可理解为：

| TP | 会运行的 case |
| --- | --- |
| 1 | 并发 16、32、64 |
| 2 | 并发 16、32、64 |

后续结果面板会自动切换为 **多用例视图**：Summary 表格列出每个 case 的核心指标（如并发、TP、推理时间、显存等），点击某一行可 drill-down 到该 case 的完整结果（显存分布图、算子耗时表等）。

查看明细时请先选芯片，再选具体 case，否则容易混淆不同并发和 TP 的显存与算子数据。

### 4.7 DeepSeek / MTP 示例

```bash
python -m cli.inference.text_generate deepseek-ai/DeepSeek-R1 \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 32 \
  --query-length 3 \
  --context-length 3500 \
  --decode \
  --num-mtp-tokens 2 \
  --tp-size 8 \
  --ep-size 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --compile
```

注意：`query-length` 必须大于 MTP token 数，否则没有足够的生成 token 承载 MTP 分析。

### 4.8 Prefix Cache 示例

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 32 \
  --query-length 512 \
  --context-length 4096 \
  --prefix-cache-hit-rate 0.5 \
  --tp-size 4 \
  --quantize-linear-action W8A8_DYNAMIC
```

`prefix-cache-hit-rate=0.5` 表示按 token 级近似估算 50% prefix 命中。命中率越高，有效 prefill 长度越短，TTFT 和 prefill 侧显存压力通常会降低。

### 4.9 VL 示例：图像输入推理

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-235B-A22B-Instruct \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --num-queries 4 \
  --query-length 16 \
  --context-length 200 \
  --decode \
  --tp-size 8 \
  --image-batch-size 1 \
  --image-height 720 \
  --image-width 1080 \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action int8
```

VL 场景建议重点关注：

- 图像尺寸变化对显存的影响。
- 图像 batch 与文本并发叠加后的显存峰值。
- Vision tower 或多模态投影相关算子的耗时占比。

---

<a id="video-generation-simulation"></a>

## 5. Video Generation 仿真使用指南

视频生成入口：

```bash
python -m cli.inference.video_generate <model_id> [options]
```

该工具模拟 Diffusion Transformer 前向过程，常用于 Wan、HunyuanVideo 等视频生成模型的性能估算。

### 5.1 关键参数

| 参数 | 说明 |
| --- | --- |
| `--batch-size` | 视频生成 batch |
| `--seq-len` | 文本 prompt token 长度 |
| `--height / --width` | 视频分辨率 |
| `--frame-num` | 帧数 |
| `--sample-step` | denoise step 数 |
| `--dtype` | `float16`、`float32`、`bfloat16` |
| `--num-devices` | 总卡数 |
| `--ulysses-size` | Ulysses sequence parallel 大小，必须整除 `--num-devices` |
| `--use-cfg` | 启用 CFG |
| `--cfg-parallel` | 使用 CFG 并行 |
| `--dit-cache` | 启用 DiT block cache |
| `--cache-step-range` | DiT Cache 生效 step 范围，格式 `start,end` |
| `--cache-step-interval` | 每 N step 刷新一次 cache，`1` 等价于不复用 |
| `--cache-block-range` | block cache 范围，格式 `start,end` |

### 5.2 最小视频仿真示例

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 50 \
  --dtype float16 \
  --quantize-linear-action W8A8_DYNAMIC
```

### 5.3 Ulysses 并行示例

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 129 \
  --sample-step 50 \
  --num-devices 8 \
  --ulysses-size 4 \
  --dtype float16
```

配置要求：

```text
world-size % ulysses-size == 0
```

如果不满足，程序会报错。Web UI 中也会提前校验。

### 5.4 CFG 与 CFG Parallel 示例

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 30 \
  --num-devices 8 \
  --ulysses-size 4 \
  --use-cfg \
  --cfg-parallel
```

`--use-cfg` 会模拟 classifier-free guidance。`--cfg-parallel` 适合比较 CFG 对通信和并行效率的影响。

### 5.5 DiT Cache 示例

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --height 720 \
  --width 1280 \
  --frame-num 81 \
  --sample-step 50 \
  --dit-cache \
  --cache-step-range 10,40 \
  --cache-step-interval 5 \
  --cache-block-range 0,20
```

说明：

- `--cache-step-range 10,40` 表示第 10 到第 40 个 denoise step 尝试复用 cache。
- `--cache-step-interval 5` 表示每 5 个 step 刷新一次，其余 step 复用。
- `--cache-step-interval 1` 会使 cache 复用基本失效。

### 5.6 Chrome Trace 导出

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --batch-size 1 \
  --seq-len 128 \
  --chrome-trace-file trace/video.json
```

生成后可在 Chrome 浏览器打开：

```text
chrome://tracing
```

---

<a id="optimizer-guide"></a>

## 6. Optimizer 吞吐寻优使用指南

吞吐寻优入口：

```bash
python -m cli.inference.throughput_optimizer <model_id> [options]
```

Optimizer 不是只跑一个固定并行配置，而是在给定模型、设备、卡数、输入输出长度、SLO 约束和搜索空间后，自动搜索更优的并行方式、batch size、concurrency 和吞吐。

### 6.1 三种部署模式

Web UI 中部署模式名称为：

| Web UI 名称 | CLI 参数 | 适用场景 |
| --- | --- | --- |
| `PD 混部` | 默认，不加 `--disagg`，不加 `--enable-optimize-prefill-decode-ratio` | Prefill 和 Decode 在同一类实例中混合部署，先做基线和多芯片横向对比 |
| `PD 分离` | 加 `--disagg` | Prefill 和 Decode 分离分析，分别评估 TTFT 或 TPOT 约束下的能力 |
| `PD 配比` | 加 `--enable-optimize-prefill-decode-ratio`，并指定 P/D 单实例卡数 | 在 PD 分离架构下，寻找 Prefill 与 Decode 实例配比 |

### 6.2 PD 混部：离线吞吐寻优

不设置 TTFT/TPOT 约束时，工具会更关注最高吞吐：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action int8
```

适合回答：

- 给定 8 卡，这个模型理论最大吞吐是多少。
- 最优 TP/DP 和 batch 大概是什么。
- 多芯片横向对比时，哪张芯片的最优吞吐更高。

### 6.3 PD 混部：在线服务 SLO 约束

同时设置 TTFT 和 TPOT：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action int8 \
  --ttft-limit 2000 \
  --tpot-limit 50
```

适合在线服务容量评估：

- TTFT 是否能满足首 token 响应目标。
- TPOT 是否能满足持续生成速度目标。
- 在约束下最优 batch 和并发是多少。

### 6.4 限制 TP 搜索空间

默认情况下，Optimizer 会搜索可用 TP。你也可以手动限制：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --tp-sizes 1 2 4 8 \
  --batch-range 1 256 \
  --jobs 8
```

Web UI 中 `TP并行大小列表` 可填写：

```text
[1,2,4,8]
```

`batch-range` 支持两种含义：

| 写法 | 含义 |
| --- | --- |
| `--batch-range 256` | min 默认为 1，max 为 256 |
| `--batch-range 1 256` | min 为 1，max 为 256 |

### 6.5 PD 分离：Prefill 侧 TTFT 分析

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action disabled \
  --disagg \
  --ttft-limit 2000
```

该模式关注 Prefill 阶段在 TTFT 约束下能承载多少请求。

### 6.6 PD 分离：Decode 侧 TPOT 分析

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action disabled \
  --disagg \
  --tpot-limit 50
```

该模式关注 Decode 阶段在 TPOT 约束下的持续输出能力。

### 6.7 PD 配比：Prefill / Decode 实例比例寻优

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B \
  --device ATLAS_850_486T_112G_FM16 \
  --num-devices 8 \
  --input-length 500 \
  --output-length 100 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action disabled \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 4 \
  --decode-devices-per-instance 4 \
  --ttft-limit 10000 \
  --tpot-limit 3000 \
```

PD 配比的核心思想是分别计算 Prefill QPS 和 Decode QPS，再寻找更平衡的 Prefill / Decode 实例配比。

近似理解：

```text
Prefill QPS = prefill_concurrency / ttft_ms * 1000
Decode QPS  = decode_concurrency / (tpot_ms * output_length) * 1000
PD 配比    = Decode QPS / Prefill QPS
Balanced QPS = min(Prefill QPS, Decode QPS)
```

当 `PD 配比 > 1` 时，Decode 侧相对更强，可能需要更多 Prefill 实例；当 `PD 配比 < 1` 时，Decode 侧可能成为瓶颈。

### 6.8 Optimizer 输出解读

典型输出包含：

| 字段 | 说明 |
| --- | --- |
| `Best Throughput` | 当前约束下最优 token/s |
| `TTFT` | Time To First Token，首 token 延迟 |
| `TPOT` | Time Per Output Token，单输出 token 延迟 |
| `concurrency` | 最优配置对应的并发数 |
| `parallel` | 并行配置，如 `tp4pp1dp2` |
| `batch_size` | 最优 batch |
| `pd_ratio` | PD 配比模式下的实例配比 |
| `balanced_qps` | PD 配比模式下 P/D 平衡后的系统 QPS |

Web UI 还会展示：

- 各芯片最优吞吐对比。
- 各芯片最优 TTFT / TPOT 对比。
- 固定配置横向对比。
- PD 配比关键指标表。
- 单芯片 Pareto 明细。

---

<a id="results-guide"></a>

## 7. 结果图和明细表怎么看

Web UI 的结果展示采用模块化组件，不同模块有独立的结果视图。结果面板支持亮色 / 暗色主题，图表自动适配。

### 7.1 Text Generation 结果

结果面板从上到下依次展示：

1. **摘要指标卡**：batch_size / execution_time / peak_usage / total_device 等关键数值一目了然。
2. **仿真程序耗时**：独立显示模拟器 wall-clock 耗时（含编译，非模型执行时间）。
3. **TPS/设备柱状图**：多芯片对比时每设备一条柱。
4. **显存分布图**：total_device / model_weight / kv_cache / peak_usage / available 的可视化分解。
5. **算子瓶颈分布（OpBound）**：紧凑文本显示 memory bound / communication bound / compute bound 占比。
6. **算子耗时表**：Name / total / avg / # of Calls，按耗时降序排列，可展开 input shapes 和 bound 分析。
7. **Chrome Trace 下载**：按 case / seq 索引提供 JSON 下载链接（需启用 `--chrome-trace-file`）。

如果配置了多值字段（多设备、多量化、并发列表等），结果自动切换为 **多用例视图**：Summary 表格列出每个 case 的核心指标，点击可 drill-down 到单 case 完整结果。

### 7.2 Video 结果

重点关注：

- 总 analytic time 与 sample step 的关系。
- Ulysses 后通信算子的占比。
- CFG / CFG Parallel 是否引入额外 all-gather 或 batch 扩张。
- DiT Cache 是否显著减少重复 block 的计算耗时。
- 算子耗时表 / 图和 Chrome Trace 下载（与 Text Generation 共享相同的展示组件）。

多用例时同样展示 Summary 表 + drill-down。

### 7.3 Optimizer 结果

Optimizer 的结果根据部署模式不同展示不同视图：

**PD 混部（AggregatedView）**：

- 散点图：Throughput vs Concurrency / TPOT，按并行策略着色分组
- 跨设备最优吞吐对比柱状图（多设备时）
- Sweep 排序表：rank / throughput / TTFT / TPOT / concurrency / num_devices / parallel / batch_size
- CSV 导出

**PD 分离（DisaggregatedView）**：

- Prefill 表（TTFT-oriented）+ Best 配置卡
- Decode 表（TPOT-oriented）+ Best 配置卡
- CSV 导出

**PD 配比（PDRatioView）**：

- PD Ratio 表：PD Ratio / Balanced QPS / P/D QPS / TTFT / TPOT / 并行配置
- Best PD 配比卡

**散点图（OptimizerCurves）**：

- 数据来源为全量探索点（raw records），按并行策略着色
- 自动过滤内存溢出点（OOM）和重复行
- 模式感知：聚合 2 张图 / 分离 4 张图 / PD 配比 2 张图
- 亮色 / 暗色主题自动适配

**多用例视图（ThroughputMultiCaseResult）**：

- Summary 表（每 case 一行：设备 + 指标）
- 点击 drill-down 到单 case 完整结果（含散点图 + 模式视图）

### 7.4 任务日志

在工作区或任务状态页点击”日志”按钮，可打开日志抽屉（JobLogDrawer）：

| 功能 | 说明 |
| --- | --- |
| 全量日志 | 任务主日志（banner + 所有 case 交织输出） |
| Per-case 日志 | 按 case 过滤的独立日志（radio 切换） |
| 日志搜索 | 大小写不敏感行过滤（显示匹配行数 / 总行数） |
| ANSI 渲染 | 终端彩色输出 → HTML（保留 bold / color / italic / underline） |

### 7.5 历史记录

顶部导航点击 **历史记录** 进入 History 页：

| 功能 | 说明 |
| --- | --- |
| 任务列表 | 表格展示：Job ID / 模块 / 标签 / 状态 / 创建时间 / 完成时间 |
| 状态标签 | 颜色编码：成功(绿) / 失败(红) / 运行中(蓝) / 取消(黄) |
| 过滤 | 按模块 / 状态筛选，按 Job ID / 标签搜索 |
| 分页 | 可选每页 10 / 20 / 50 / 100 条 |
| 操作 | 查看结果（succeeded）/ 查看状态（running）/ 查看详情（failed） |

---

## 8. 参数选择建议

### 8.1 不知道从哪里开始时

LLM decode 初始值：

```text
num-devices: 8
num-queries: 32
query-length: 1
context-length: 4500
decode: true
tp-size: 8
quantize-linear-action: W8A8_DYNAMIC
quantize-attention-action: disabled
```

LLM prefill 初始值：

```text
num-devices: 8
num-queries: 8
query-length: 3500
context-length: 0
decode: false
tp-size: 8
quantize-linear-action: W8A8_DYNAMIC
quantize-attention-action: int8
```

Optimizer 在线服务初始值：

```text
input-length: 3500
output-length: 1500
ttft-limits: 2000
tpot-limits: 50
tp-sizes: [1,2,4,8]
batch-range: [1,256]
jobs: 8
```

### 8.2 TP 怎么选

经验规则：

- 模型权重太大放不下：优先增大 TP。
- 单卡算力瓶颈明显：增大 TP 可能收益明显。
- 通信占比高：继续增大 TP 可能收益下降。
- 小模型或小 batch：过大的 TP 可能因为通信和同步开销导致变慢。

建议先用 Web UI 的 TP 列表跑 `[1,2,4,8]`，再根据曲线缩小搜索范围。

### 8.3 并发怎么选

经验规则：

- 并发太低：设备利用率可能不足。
- 并发逐步增大：吞吐通常提升，但延迟和显存也会上升。
- 并发过高：可能触发显存瓶颈、KV Cache 过大或延迟不可接受。

建议用 `[16,32,64,128]` 做第一轮，然后在最优区间附近细扫。

### 8.4 量化怎么选

| 场景 | 建议 |
| --- | --- |
| 快速基线 | `W8A8_DYNAMIC` |
| 不希望引入量化影响 | `disabled` |
| 显存压力明显 | 尝试 `int8` attention 或 `fp8` |
| mxfp4 方案评估 | 使用 `mxfp4`，必要时调整 `mxfp4-group-size` |

注意：仿真工具关注性能和资源估计，不替代真实精度评估。量化后的模型质量仍需通过精度测试验证。

---

## 9. 开发者补充说明

如果你要修改 Web UI，建议先阅读设计文档：

```text
docs/design/web_ui_refactor_design.md
```

### 9.1 架构概览

Web UI 采用前后端分离架构：

```text
浏览器 (Vue 3 SPA)  ──HTTP/JSON──▶  FastAPI 后端  ──subprocess──▶  CLI 核心
```

- **前端**：Vue 3 + Element Plus + Pinia + ECharts + Vite，构建产物由后端 StaticFiles 挂载
- **后端**：FastAPI + SQLite（WAL mode）+ Alembic 迁移，任务在独立子进程中执行
- **前端源码**：`web_ui/frontend/`
- **后端源码**：`web_ui/backend/`

### 9.2 核心文件关系

**前端**：

```text
web_ui/frontend/src/
├── App.vue                    # 根组件（app-bar + router-view）
├── main.ts                    # 入口（Vue + Element Plus + Pinia）
├── router/index.ts            # 路由（Console / History / JobResult / Docs）
├── pages/                     # 路由页（Console / History / JobResult / JobStatus / Docs）
├── components/
│   ├── workspace/             # 工作区（ModuleWorkspace + ResultPane）
│   ├── form/                  # 动态表单（SchemaForm + SchemaFormItem）
│   ├── result/                # 结果组件（text / video / throughput 各子目录）
│   └── job-status/            # 任务状态卡 + 日志抽屉
├── composables/               # 组合式函数（useJobRunner / useFormValidation 等）
├── stores/                    # Pinia store（formState / telemetry）
├── services/                  # API 层（axios 封装）
├── config/forms/              # 表单配置 source of truth（.ts 文件）
└── styles/theme.css           # CSS 变量主题
```

**后端**：

```text
web_ui/backend/
├── main.py                    # FastAPI app + lifespan + uvicorn 入口
├── db.py                      # SQLite engine + Alembic 迁移
├── api/
│   ├── routers/               # API 路由（jobs / cases / modules / options）
│   ├── schemas.py             # Pydantic 响应模型
│   └── errors.py              # 异常处理
├── models/                    # 数据实体 + ORM 定义
├── services/
│   ├── job_manager.py         # 异步任务管理
│   ├── job_runner.py          # 任务执行（ThreadPoolExecutor + subprocess）
│   ├── result_view.py         # 结果组装（Top-N + SLO + 多用例）
│   ├── ranking.py             # 排名计算
│   ├── repositories.py        # 数据访问层
│   ├── schema_registry.py     # 表单 schema 快照 + hash
│   └── capture.py             # 日志捕获
├── runners/                   # Runner 适配器（text_generate / video_generate / throughput_optimizer）
└── migrations/                # Alembic 迁移
```

### 9.3 Web 启动

```bash
# 首次需安装前端依赖（只需一次）
cd web_ui/frontend && npm install

# 启动（单命令并发启动前后端）
python web_ui/main.py
```

### 9.4 表单配置开发

表单字段定义在 `web_ui/frontend/src/config/forms/*.ts` 中（source of truth），构建时通过 `npm run gen:schemas` 生成 data-only JSON 供后端 schema_registry 加载。修改字段后需 bump 版本号。

---

## 10. 快速命令索引

启动 Web UI：

```bash
# 首次需安装前端依赖（只需一次）
cd web_ui/frontend && npm install

# 启动（单命令并发启动前后端）
python web_ui/main.py
```

浏览器打开 `http://127.0.0.1:5173`。

LLM decode：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --num-queries 32 --query-length 1 --context-length 4500 --decode --tp-size 8
```

VL：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-VL-235B-A22B-Instruct --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --num-queries 4 --query-length 16 --context-length 200 --decode --tp-size 8 --image-batch-size 1 --image-height 720 --image-width 1080
```

Video：

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers --device ATLAS_800_A2_280T_32G_PCIE --batch-size 1 --seq-len 128 --height 720 --width 1280 --frame-num 81 --sample-step 50
```

Optimizer：

```bash
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --input-length 3500 --output-length 1500 --tp-sizes 1 2 4 8 --batch-range 1 256 --ttft-limit 2000 --tpot-limit 50
```
