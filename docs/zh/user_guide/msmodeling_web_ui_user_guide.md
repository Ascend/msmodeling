# Web UI 使用说明

本文档面向 Modeling 的日常使用者和即将接入项目的开发者，目标是帮助你快速理解工具能做什么、如何从 Web UI 或 CLI 发起仿真、如何解读结果，以及在不同业务场景下应该如何配置参数。

如果只想启动前端页面，直接执行：

```bash
python -m web_ui.web_ui_start --port 2345
```

启动后在浏览器打开 `http://127.0.0.1:2345`。

---

## 阅读导航

| 目标 | 推荐章节 |
| --- | --- |
| 第一次启动 Web UI | [3. Web UI 快速上手](#web-ui-quick-start) |
| 配置 LLM / VL 仿真 | [4. LLM / VL 仿真使用指南](#llm-vl-simulation) |
| 配置视频生成仿真 | [5. Video Generation 仿真使用指南](#video-generation-simulation) |
| 使用吞吐优化器 | [6. Optimizer 吞吐寻优使用指南](#optimizer-guide) |
| 解读结果与导出数据 | [7. 结果图和明细表怎么看](#results-guide) |
| 排查常见问题 | [9. 常见问题](#faq) |

---

## 1. 工具定位

Modeling 是一个面向模型推理性能分析的仿真工具，核心能力包括：

- 在没有真实硬件或真实大模型完整运行环境的情况下，基于设备画像预测算子耗时、显存占用、通信开销和整体推理时间。
- 支持 LLM 文本推理、VL 多模态推理、视频生成 Diffusion Transformer 推理，以及服务化吞吐寻优。
- 支持多芯片横向对比，帮助判断同一个模型在不同设备上的性能差异。
- 支持并发、TP、量化、MTP、Prefix Cache、Ulysses、DiT Cache、PD 混部、PD 分离、PD 配比等参数组合分析。
- Web UI 提供可视化图表、明细表格、case 选择、Excel 导出和历史缓存；CLI 适合脚本化批量实验。

仓库中与用户最相关的入口如下：

| 入口 | 作用 | 推荐使用场景 |
|---|---|---|
| `python -m web_ui.web_ui_start` | 启动 Gradio 前端 | 交互式配置、结果可视化、非开发用户使用 |
| `python -m cli.inference.text_generate` | LLM / VL 前向推理仿真 | 单次或脚本化 LLM/VL 性能分析 |
| `python -m cli.inference.video_generate` | 视频生成模型仿真 | Diffusion Transformer / Wan / HunyuanVideo 等场景 |
| `python -m cli.inference.throughput_optimizer` | 服务吞吐寻优 | 在 TTFT/TPOT/SLO 约束下寻找最优并行和 batch |

---

## 2. 环境准备

完整的环境搭建步骤（克隆仓库、创建虚拟环境、安装依赖、设置 `PYTHONPATH` 与 Hugging Face 访问）请参阅《[msModeling 安装指南](../install_guide/msmodeling_install_guide.md)》。

若已完成环境搭建，从仓库根目录启动 Web UI 一般无需额外配置。工具会读取模型配置，常见来源包括 Hugging Face、ModelScope 或本地模型目录；若网络无法访问 Hugging Face，可在 Web UI 的 `remote-source` 中选择 `modelscope`，或按安装指南设置 `HF_ENDPOINT` 镜像。

---

<a id="web-ui-quick-start"></a>

## 3. Web UI 快速上手

### 3.1 启动本地页面

```bash
python -m web_ui.web_ui_start --port 2345
```

适合本机使用。浏览器打开：

```text
http://127.0.0.1:2345
```

### 3.2 Web UI 页面说明

当前 Web UI 主要包含三类工作区：

| 页面 | 能力 |
|---|---|
| Simulator - LLM Forward | LLM 文本推理仿真，支持并发列表、TP 列表、量化、MTP、Prefix Cache、并行细分、算子和显存分析 |
| Simulator - VL Forward | 多模态 VL 推理仿真，在 LLM 参数基础上增加 image batch、height、width 等图像参数 |
| Video Generation | 视频生成模型推理仿真，支持 Ulysses、CFG、DiT Cache、Chrome Trace 等参数 |
| Optimizer | 服务吞吐寻优，支持 `PD 混部`、`PD 分离`、`PD 配比` 三种部署模式 |

### 3.3 Web UI 的基本操作流程

1. 选择模型、主芯片和可选竞品芯片。
2. 填写卡数、并发、长度、量化、并行等参数。
3. 点击“预览配置”或“预览命令”，确认将要执行的 CLI 命令。
4. 点击“开始运行”。
5. 查看汇总结论、曲线图、显存分析、带宽瓶颈、算子详情和导出结果。
6. 如果设置了并发列表或 TP 列表，在明细分析区选择具体 case，例如 `Concurrency=32 | TP=2`，再查看该 case 的显存和算子数据。

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
|---|---|
| `num-queries` | 并发请求数，影响 batch、KV Cache、显存和吞吐 |
| `query-length` | 本次新增 token 数。prefill 通常较大，decode 通常为 1 或较小值 |
| `context-length` | 已有上下文长度，影响 KV Cache 和 attention 成本 |
| `decode` | 开启自回归 decode 模式 |
| `tp-size` | Tensor Parallel 数量 |
| `dp-size` | Data Parallel 数量，可在 Web UI 中填 `auto` |
| `ep-size` | Expert Parallel 数量，MoE 模型常用 |
| `num-mtp-tokens` | MTP token 数，DeepSeek 等支持 MTP 的模型可用 |
| `prefix-cache-hit-rate` | Prefix Cache 命中率，取值 `[0,1)`，用于估计 prefill token 复用收益 |
| `quantize-linear-action` | Linear 层量化方式，如 `W8A8_DYNAMIC`、`FP8`、`MXFP4` |
| `quantize-non-expert-linear-action` | 非专家 Linear 层量化覆盖项，主要用于 DeepSeek V4；作用于 attention projections、dense MLP 和 shared experts；routed MoE experts 仍使用 `quantize-linear-action` |
| `quantize-attention-action` | KV Cache / Attention 量化方式，如 `DISABLED`、`INT8`、`FP8` |
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
  --quantize-attention-action DISABLED
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
  --quantize-attention-action INT8
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
    --quantize-linear-action MXFP4 \
    --quantize-attention-action DISABLED
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
|---|---|
| 1 | 并发 16、32、64 |
| 2 | 并发 16、32、64 |

后续显存、带宽、算子详情区域会出现 case 选择项，例如：

```text
并发=16 | TP=1
并发=32 | TP=1
并发=64 | TP=1
并发=16 | TP=2
并发=32 | TP=2
并发=64 | TP=2
```

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
  --quantize-attention-action INT8
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
|---|---|
| `--batch-size` | 视频生成 batch |
| `--seq-len` | 文本 prompt token 长度 |
| `--height / --width` | 视频分辨率 |
| `--frame-num` | 帧数 |
| `--sample-step` | denoise step 数 |
| `--dtype` | `float16`、`float32`、`bfloat16` |
| `--world-size` | 总卡数 |
| `--ulysses-size` | Ulysses sequence parallel 大小，必须整除 `world-size` |
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
  --world-size 8 \
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
  --world-size 8 \
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
  --chrome-trace trace/video.json
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
|---|---|---|
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
  --quantize-attention-action INT8
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
  --quantize-attention-action INT8 \
  --ttft-limits 2000 \
  --tpot-limits 50
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
|---|---|
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
  --quantize-attention-action DISABLED \
  --disagg \
  --ttft-limits 2000
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
  --quantize-attention-action DISABLED \
  --disagg \
  --tpot-limits 50
```

该模式关注 Decode 阶段在 TPOT 约束下的持续输出能力。

### 6.7 PD 配比：Prefill / Decode 实例比例寻优

```bash
python -m cli.inference.throughput_optimizer deepseek-ai/DeepSeek-V3.1 \
  --device ATLAS_800_A2_280T_32G_PCIE \
  --num-devices 16 \
  --input-length 3500 \
  --output-length 1500 \
  --compile \
  --quantize-linear-action W8A8_DYNAMIC \
  --quantize-attention-action DISABLED \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance 4 \
  --decode-devices-per-instance 2 \
  --ttft-limits 2000 \
  --tpot-limits 50 \
  --log-level info
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
|---|---|
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

### 7.1 LLM / VL 结果

建议按以下顺序阅读：

1. 汇总结论：先看总时间、TPS/Device、是否有失败或告警。
2. 推理时间曲线：看并发或 TP 增大后是否仍能降低耗时。
3. 显存分析：看模型权重、KV Cache、激活和预留显存占比。
4. 带宽瓶颈：看 memory bound、communication bound、compute bound。
5. 算子详情：按总耗时排序，定位最主要算子。
6. 算子分类统计：从 GEMM、Attention、Communication 等类别判断优化方向。

如果配置了并发列表或 TP 列表，查看明细前一定要选择 case。

### 7.2 Video 结果

重点关注：

- 总 analytic time 与 sample step 的关系。
- Ulysses 后通信算子的占比。
- CFG / CFG Parallel 是否引入额外 all-gather 或 batch 扩张。
- DiT Cache 是否显著减少重复 block 的计算耗时。

### 7.3 Optimizer 结果

推荐阅读顺序：

1. 推荐结论：看最优芯片、吞吐、并行方式、batch、并发。
2. 各芯片最优对比：用于竞品和主芯片横向比较。
3. 固定配置横向对比：确保比较是在同一配置下进行，而不是只比较各自最优点。
4. PD 配比：如果是 PD 分离架构，查看 Balanced QPS 和 Prefill / Decode 实例配比。
5. 单芯片 Pareto：判断是否存在“吞吐更高但延迟略差”的备选点。

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
quantize-attention-action: DISABLED
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
quantize-attention-action: INT8
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
|---|---|
| 快速基线 | `W8A8_DYNAMIC` |
| 不希望引入量化影响 | `DISABLED` |
| 显存压力明显 | 尝试 `INT8` attention 或 `FP8` |
| MXFP4 方案评估 | 使用 `MXFP4`，必要时调整 `mxfp4-group-size` |

注意：仿真工具关注性能和资源估计，不替代真实精度评估。量化后的模型质量仍需通过精度测试验证。

---

<a id="faq"></a>

## 9. 常见问题

### 9.1 Web UI 启动后浏览器打不开

检查：

- 是否使用了正确地址：`http://127.0.0.1:2345`。
- 端口是否被占用，可换成 `--port 2346`。

### 9.2 提示设备名不合法

`--device` 必须来自 `DeviceProfile.all_device_profiles`。Web UI 会从设备画像自动加载品牌和芯片列表。CLI 下可以查看报错中的 choices，或在 Web UI 中先选择可用芯片。

### 9.3 TP / DP / EP 配置不合法

常见原因：

- `num-devices` 不能被 `tp-size` 整除。
- `world-size` 不能被 `ulysses-size` 整除。
- `TP * DP * EP` 超过部署卡数。
- 某些细分 TP/DP 参数与总卡数不匹配。

处理建议：先用简单配置跑通，例如 `tp-size=1, dp-size=auto, ep-size=1`，再逐步增加并行复杂度。

### 9.4 Optimizer 没有可行方案

常见原因：

- TTFT 或 TPOT 约束过严。
- `max-batched-tokens` 小于有效输入长度。
- batch 搜索范围太小。
- 卡数不足或 TP 搜索空间不合适。
- 显存预留过大导致可用显存不足。

处理建议：

1. 先去掉 TTFT/TPOT 约束，看是否能找到离线最优。
2. 放宽 `tpot-limits` 或 `ttft-limits`。
3. 增大 `batch-range` 上限。
4. 检查 `tp-sizes` 是否包含可行值。
5. 降低 `reserved-memory-gb` 或使用更强设备画像。

### 9.5 结果来自缓存，想重新运行

Web UI 会根据 task hash 读取 `.msmodeling_ui/results.sqlite3` 和 `.msmodeling_ui/logs/` 中的缓存。若需要完全重跑，可以清理对应缓存目录，或调整一个会影响仿真的参数生成新 task hash。

### 9.6 图表标题遮挡内容

新版 Web UI 已将图标题放在图像区域外的独立标题位置，不再使用 Gradio 左上角覆盖式标题。如果仍看到旧样式，确认浏览器没有加载旧页面，并重启 Web UI。

---

## 10. 推荐工作流案例

### 10.1 案例 A：比较两张芯片的 LLM decode 能力

Web UI：

```text
模型: Qwen/Qwen3-32B
主芯片: ATLAS_800_A2_280T_32G_PCIE
竞品芯片: 选择另一张芯片
部署卡数: 8
并发列表: [16,32,64]
TP列表: [1,2,4,8]
生成token数量: 8
上下文长度: 4500
Decode模式: 开启
量化: MLP=W8A8_DYNAMIC, Attention=DISABLED
```

观察：

- 哪张芯片在同 TP 和同并发下推理时间更低。
- 是否存在某张芯片在高 TP 下通信瓶颈更明显。
- 显存和算子详情中瓶颈是否一致。

### 10.2 案例 B：评估 VL 图像尺寸影响

第一轮：

```text
image-height: 720
image-width: 1080
```

第二轮：

```text
image-height: 1024
image-width: 1024
```

保持其他参数不变，对比：

- 总推理时间变化。
- 显存占用变化。
- Vision 相关算子的耗时变化。

### 10.3 案例 C：视频生成 Ulysses 扩展性

依次测试：

```text
world-size=8, ulysses-size=1
world-size=8, ulysses-size=2
world-size=8, ulysses-size=4
world-size=8, ulysses-size=8
```

观察：

- 总耗时是否随 Ulysses 增大下降。
- 通信算子占比是否上升。
- 是否存在最优 Ulysses，而不是越大越好。

### 10.4 案例 D：在线服务容量评估

Web UI Optimizer：

```text
部署模式: PD 混部
模型: Qwen/Qwen3-32B
部署卡数: 8
输入长度: 3500
输出长度: 1500
TP并行大小列表: [1,2,4,8]
Batch范围: [1,256]
TTFT: 2000
TPOT: 50
量化: MLP=W8A8_DYNAMIC, Attention=INT8
```

输出中重点查看：

- 是否有可行解。
- 最优吞吐、TTFT、TPOT 是否同时满足目标。
- 最优 parallel 和 batch 是否符合部署预期。

### 10.5 案例 E：PD 配比部署规划

Web UI Optimizer：

```text
部署模式: PD 配比
部署卡数: 16
Prefill 单实例卡数: 4
Decode 单实例卡数: 2
输入长度: 3500
输出长度: 1500
TTFT: 2000
TPOT: 50
```

观察：

- Balanced QPS。
- Prefill QPS 与 Decode QPS 谁更低。
- 推荐 P/D 实例数量和总卡数是否匹配实际集群规划。

---

## 11. 开发者补充说明

如果你要修改 Web UI，建议先阅读：

```text
web_ui/README.md
```

核心文件关系：

```text
web_ui/__init__.py          包入口点，延迟暴露 launch_app
web_ui/app.py               页面布局和事件绑定
web_ui/components.py        复用组件和结果区域
web_ui/callbacks.py         表单构建、校验、运行、结果整理
web_ui/command_builder.py   CLI 命令和任务矩阵生成
web_ui/runner.py            缓存、子进程运行、进度流
web_ui/parsers.py           日志解析
web_ui/result_store.py      SQLite 和日志缓存
web_ui/charts.py            图表绘制
web_ui/styles.py            共享CSS、主题助手和头部样式
web_ui/schemas.py           构建器、运行器、解析器和存储之间共享的数据类
web_ui/utils.py             共享解析、哈希和标准化助手
web_ui/time_tracker.py      跟踪和显示仿真时间信息
web_ui/web_ui_start.py      Web UI服务器启动入口
```

修改前端功能后建议执行：

```bash
python -m py_compile web_ui/__init__.py web_ui/app.py web_ui/callbacks.py web_ui/command_builder.py web_ui/components.py web_ui/charts.py web_ui/parsers.py web_ui/result_store.py web_ui/runner.py web_ui/schemas.py web_ui/styles.py web_ui/time_tracker.py web_ui/utils.py web_ui/web_ui_start.py
```

---

## 12. 快速命令索引

启动 Web UI：

```bash
python -m web_ui.web_ui_start --port 2345
```

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
python -m cli.inference.throughput_optimizer Qwen/Qwen3-32B --device ATLAS_800_A2_280T_32G_PCIE --num-devices 8 --input-length 3500 --output-length 1500 --tp-sizes 1 2 4 8 --batch-range 1 256 --ttft-limits 2000 --tpot-limits 50
```
