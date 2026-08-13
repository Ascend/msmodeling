# RFC: 增加 FLUX.1-dev 图像生成仿真支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft（草案） |
| **作者** | `minghang_c` |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-09 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |
| **规范依赖** | [图像生成推理性能仿真架构 RFC：公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index) |

---

## 1. 概述

### 1.1 简介

本 RFC 提议在图像生成推理性能仿真公共架构之上，增加对 Hugging Face 上精确模型 `black-forest-labs/FLUX.1-dev` 的 Transformer 去噪阶段仿真支持。首版复用现有 `DiffusersTransformerModel` 的构建、meta-device 图追踪、设备性能统计和并行运行设施，新增 FLUX.1-dev 的静态模型 profile、输入 shape 适配、固定 dummy timestep/guidance 输入以及完整 Ulysses/true CFG 并行语义。公共 CLI 使用恰好一次的 `--output-image-size HEIGHT WIDTH` 和 `--sample-step N`，不暴露 guidance 数值。

本 RFC 只定义仿真契约与后续实现边界，不执行真实 prompt 编码、权重推理、VAE、图像输出或端到端延迟测量。其结果使用公共术语“Transformer 去噪阶段模拟时间”，并以 `FLUX.1-dev` profile 标识模型；不得解释为完整图片生成耗时或图片质量结论。

### 1.2 动机

FLUX.1-dev 是受限访问的 12B 级 rectified-flow 文生图模型。其 Diffusers pipeline 同时使用 CLIP/T5 两套文本条件、Flux Transformer 的 packed image tokens、上游 FlowMatch scheduler 和可选的 guidance embedding；本 simulator 只保留 Transformer 所需的输入 shape 与 timestep/guidance operator path，排除 scheduler 数值控制流。若仅按普通 `[B,C,H,W]` 图像输入或只识别 `FluxTransformer2DModel` 类名，无法得到正确的注意力序列长度、RoPE ID、operator path 和并行通信形态。

性能分析用户需要在不下载或加载完整权重的情况下，比较输出尺寸、文本条件长度、Transformer workload 迭代次数、Ulysses 大小和 true CFG 对 Transformer 去噪阶段的影响。因此必须将官方 checkpoint 的 pipeline/config fingerprint、2x2 packing、固定 dummy timestep/guidance 输入和并行边界固化为可验证的模型专属 profile，同时继承公共架构定义的阶段标记、结果字段和计时口径。

### 1.3 目标与非目标

#### 目标

1. 仅支持 canonical 远端模型 `(remote_source=huggingface, model_id=black-forest-labs/FLUX.1-dev)`，以及能够唯一匹配同一官方 Diffusers profile 的本地目录。
2. 在 `diffusers==0.38.0` 可执行基线上验证 `FluxTransformer2DModel` 及其基础组件的 config-only/meta 构建契约；官方 scheduler 仅作为上游 pipeline 证据，不进入首版模拟器。
3. 根据 VAE 和 Transformer config 构造 FLUX 的 latent、2x2 packed image tokens、文本条件、pooled projection、`img_ids`、`txt_ids` 和固定 timestep/guidance 的 meta 输入。
4. 保留 embedded guidance 与 true CFG 的独立语义，并准确统计每步一次或两次逻辑 Transformer forward；`--sample-step=N` 直接驱动 N 次 Transformer workload iteration。
5. 首版实现 FLUX 四类输入的完整 Ulysses 分片、attention collective 所需的分片形态、输出 gather、整除 fail-fast，以及 `U × C` Cartesian CFG/Ulysses 拓扑。
6. 提供 config-only、meta shape、N-step 调用计数、非法配置与并行拓扑失败场景，证明结果只覆盖 Transformer 去噪阶段。

#### 非目标与明确排除

- 不支持 `FLUX.1-schnell`、Fill、Redux、img2img、inpainting、ControlNet、IP-Adapter、LoRA 效果或其他当前/未来 FLUX checkpoint。
- 不实现真实 prompt/tokenizer/text encoder、随机噪声质量、权重下载或加载、VAE encode/decode、postprocess、图片保存和真实图片输出。
- 不将 FLUX 的支持扩展为动态插件、通用 model registry、任意 scheduler 抽象或跨模型 callback 系统。
- 不使用 padding 掩盖 Ulysses 不整除，不为不兼容输入隐式截断或改变文本/图像 token 数。
- 不修改 `video_generate`、视频参数、Web UI、历史任务或 Qwen-Image-Edit 适配。
- 不把本 RFC 重新定义为公共架构 RFC；公共 CLI、阶段/结果契约、Transformer workload 循环、CFG 编排和 `U × C` 组语义均规范性继承架构 RFC。

## 2. 适用范围与身份契约

### 2.1 精确远端 profile

远端解析只接受以下精确组合：

```text
remote_source = huggingface
model_id      = black-forest-labs/FLUX.1-dev
```

该仓库为 gated repository，访问需要完成 Hugging Face 登录、接受 FluxDev Non-Commercial License 和 Acceptable Use Policy，并按其要求共享联系信息。该模型采用 `flux-1-dev-non-commercial-license`；许可证与访问状态必须作为 profile/result metadata 记录，但不属于模拟设备时间。不得使用社区镜像替代官方配置，也不得因为能够访问同名 Transformer 类而放宽身份匹配。

远端 config-only 解析失败时，必须给出明确的 gated 访问或授权本地目录提示。不得自动回退镜像、下载完整权重或将网络失败解释为模型不兼容。

### 2.2 本地 profile fingerprint

本地 Diffusers 根目录必须存在 `model_index.json` 及 simulator 所需的下列组件配置，并唯一匹配 FLUX.1-dev profile：

- 根 `model_index.json._class_name == "FluxPipeline"`；
- `vae` 的类为 `AutoencoderKL`；
- `text_encoder` / `tokenizer` 分别为 `CLIPTextModel` / `CLIPTokenizer`；
- `text_encoder_2` / `tokenizer_2` 分别为 `T5EncoderModel` / `T5TokenizerFast`；
- `transformer` 的类为 `FluxTransformer2DModel`；
- `transformer/config.json`、`vae/config.json` 存在，且关键字段类型和数值满足 profile 校验。

官方 pipeline 的上游 scheduler 为 `FlowMatchEulerDiscreteScheduler`，但 scheduler 组件、类名和 config 都不属于 simulator 的本地 profile fingerprint：resolver 不读取或校验 scheduler config，也不因 scheduler 目录缺失或字段变化改变 Transformer workload contract。相关事实只保留在 research evidence 中。

关键 Transformer fingerprint 包括 `in_channels`、`out_channels`、`num_layers`、`num_single_layers`、`attention_head_dim`、`num_attention_heads`、`joint_attention_dim`、`pooled_projection_dim`、`guidance_embeds` 和 `axes_dims_rope`。基线契约值为：`in_channels=64`；若 raw config 的 `out_channels` 为 `None`，按 Diffusers 语义规范化为 `in_channels`，即 64；另有 19 个 dual-stream blocks、38 个 single-stream blocks、`attention_head_dim=128`、24 个 attention heads、`joint_attention_dim=4096`、`pooled_projection_dim=768`、RoPE axes 为 `(16,56,56)`、`guidance_embeds=true`；内部 attention width 为 `24×128=3072`。`guidance_embeds=false` 的 config 不属于本 RFC 的 FLUX.1-dev profile，必须按 fingerprint 不匹配拒绝。profile 还必须从官方实际 config 读取 VAE scaling/shift，不得用非官方镜像或通用构造器默认值硬编码替代。

本地目录只要存在 `FluxTransformer2DModel`，但 pipeline、VAE 或其它关键 fingerprint 不匹配，就必须稳定失败。首版不接受未知 pipeline、任意 FLUX 变体或不完整目录；scheduler 不参与接受或拒绝判断。

### 2.3 版本基线

实现和测试固定以本仓可执行的 `diffusers==0.38.0` 为基线，复用该版本 `FluxTransformer2DModel` 的 signature 与 context-parallel contract。若升级 Diffusers，必须重新核验 Transformer 的类、输入字段、packing 和分片行为后才能更新本 RFC 或实现；不能把公开文档当前指向的其它版本当作本仓现行契约。官方 pipeline/scheduler 事实仍可作为 research evidence，但不构成模拟器运行时契约。

## 3. 设计方案

<a id="flux1-dev-public-dependency"></a>

### 3.1 与公共架构的关系

本 RFC 规范性依赖[图像生成推理性能仿真架构 RFC 的公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index)。继承关系按以下章节固定，FLUX.1-dev 适配不得复制或改写这些公共定义：

| 继承的公共 contract | 规范章节 |
| :--- | :--- |
| CLI 与参数校验 | [公共架构 RFC 3.3](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-cli-contract) |
| exact-profile、静态组装与 adapter | [公共架构 RFC 3.4](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-profile-adapter-contract) |
| 固定 timestep 与 N-step workload | [公共架构 RFC 3.5](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-workload-contract) |
| Guidance 与 true CFG | [公共架构 RFC 3.6](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-guidance-cfg-contract) |
| `U × C` 拓扑与 critical path | [公共架构 RFC 3.7](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-topology-contract) |
| generated output span 与所有权 | [公共架构 RFC 3.8](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-output-ownership-contract) |
| 结果与 Chrome trace | [公共架构 RFC 3.9](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-result-trace-contract) |
| 现有模块复用与修改边界 | [公共架构 RFC 3.10](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-module-boundary) |

公共架构负责 `image-generate` 入口、公共参数与组合校验、静态 profile 分派、config-only resolver、meta Transformer builder、Runtime/设备计时、N-step workload 编排、true CFG 分支调度、`U × C` 组布局和结果阶段覆盖字段。FLUX adapter 只提供本节规定的 profile fingerprint、shape/ID 构造、固定 timestep/guidance 输入、分片计划、generated output span 和 FLUX 专属校验。

公共执行层将 `--sample-step=N` 直接解释为 N 次 FLUX Transformer workload iteration，不映射为 Diffusers `num_inference_steps`，不生成 timetable，不构造或调用 scheduler。仅 embedded guidance 时每次 iteration 一次逻辑 Transformer forward；启用 true CFG 时每次 iteration 两次，正负条件可有不同有效文本序列长度。两者同时启用仍为两次而不是四次。每次 iteration 使用固定 `[B]` meta/dummy timestep，关键路径止于必要 Transformer output、CFG/Ulysses communication 和 output ownership。

### 3.2 Pipeline 组件和阶段边界

官方 base `FluxPipeline` 的上游组件包括 `FlowMatchEulerDiscreteScheduler`、`AutoencoderKL`、`CLIPTextModel`、`CLIPTokenizer`、`T5EncoderModel`、`T5TokenizerFast` 和 `FluxTransformer2DModel`。首版只对其中的 Transformer workload 建模；scheduler 事实保留为上游 pipeline 证据，明确排除出模拟器：

| 阶段 | FLUX 首版行为 | 阶段标记 |
|---|---|---|
| 模型/config 解析 | host setup：读取本地目录或受支持远端的 manifest 与关键 config，不计模拟设备时间 | `excluded` |
| tokenizer/text encoder | 不执行；由用户提供实际进入 Transformer 的正/负有效文本序列长度，构造 meta embedding | `shape_only` |
| VAE encode/latent 准备 | 不读取图片；按请求尺寸和官方 VAE config 计算 latent/packed shape | `shape_only` |
| Transformer 构建 | host setup：使用 `DiffusersTransformerModel` 在 meta device 构建 `FluxTransformer2DModel`，不计模拟设备时间 | `excluded` |
| Scheduler 控制与数值更新 | 不读取 config、不构造 scheduler、不生成 timetable/sigma/shift、不调用 `scheduler.step` | `excluded` |
| 固定 dummy timestep/guidance | 构造 `[B]` meta/dummy tensor，只保留 timestep/guidance embedding operator path | `shape_only` |
| 每步 Transformer forward | 执行正向分支和必要的负向分支，包含真实 attention/collective 性能记录 | `measured` |
| Ulysses/CFG 通信 | 按实际 `U × C` 拓扑执行并由 Runtime 记录 | `measured` |
| CFG 合并与输出 ownership | 保留 shape、分支和所有权语义，不纳入设备测量总量 | `shape_only` |
| VAE decode、postprocess、保存 | 完全不执行 | `excluded` |

因此结果名称和说明必须使用公共术语“Transformer 去噪阶段模拟时间”，并通过 profile 字段标识 `FLUX.1-dev`。关键路径在完成所需 Transformer output、CFG/Ulysses communication 和 output ownership 后结束，不包含 scheduler update。不得另造模型专属结果类型，也不得宣称端到端生成延迟、prompt/VAE/文件 I/O 时间、真实显存权重占用、图片质量或生成图片。

### 3.3 FLUX meta 输入与 shape

设生成 batch 为 `B`，`--output-image-size HEIGHT WIDTH` 提供请求输出尺寸 `(H,W)`，正向有效文本序列长度为 `L_pos`，负向有效文本序列长度为 `L_neg`。该尺寸参数恰好出现一次，并同时驱动 requested/effective output spatial size、latent packing 和 `N_img`；FLUX 首版禁止 `--source-image-size`。`text_seq_len` 表示编码器输出进入 Transformer 的有效序列长度，不是原始字符数、tokenizer 输入长度或未经校验的 max length；FLUX 的文本序列长度上限为 512。启用 true CFG 时，`L_neg` 必须显式提供且独立校验。

adapter 从实际 `vae/config.json` 计算：

```text
vae_scale_factor = 2 ** (len(block_out_channels) - 1)
H_lat = 2 * floor(H / (2 * vae_scale_factor))
W_lat = 2 * floor(W / (2 * vae_scale_factor))
C_lat = transformer.in_channels / 4
N_img = (H_lat / 2) * (W_lat / 2)
```

标准八倍 VAE 且请求尺寸为 16 的整数倍时，`H_lat=H/8`、`W_lat=W/8`、`C_lat=16`，因此 `N_img=(H/16)×(W/16)`；1024×1024 的典型形状为 unpacked latent `[B,16,128,128]`，2x2 packing 后的 Transformer image input 为 `[B,4096,64]`。非 16 对齐尺寸必须同时记录 requested/effective size，并依照官方公式得到有效 latent 和 token 数，不得静默 padding。

FLUX adapter 构造以下 meta 输入：

```text
unpacked_latent:          [B, C_lat, H_lat, W_lat]       # shape_only
hidden_states:            [B, N_img, 64]
encoder_hidden_states:    [B, L_pos, 4096]
pooled_projections:       [B, 768]
img_ids:                  [N_img, 3]
txt_ids:                  [L_pos, 3]
timestep:                 [B]
guidance:                 [B]                            # 仅 guidance_embeds=true
```

负向 true CFG 分支使用相同的 image/pooled shape 和其自身的 `encoder_hidden_states=[B,L_neg,4096]`、`txt_ids=[L_neg,3]`。图像 position IDs 为 `[N_img,3]`，通道 0 为零，通道 1/2 表示 latent 行/列；文本 ID 和图像 ID 按 text-then-image 参与 RoPE。Transformer 的输入宽度是 packed token 的 64，而不是 VAE latent channel 或原始图像 channel。

FLUX 全部 image output 属于公共契约的 `generated output span`；Transformer 输出经过必要 gather 后，生成 image token 的完整 `[B,N_img,out_channels]` 作为最终 shape-only output ownership。不会截取 source token，也不会把任何未定义的额外 token 计入生成 span；不执行 scheduler update。

### 3.4 Fixed timestep 与 guidance

官方 `FlowMatchEulerDiscreteScheduler` 的类名、sigma/timestep、resolution-dependent shift 和 scheduler.step 语义仅作为上游 pipeline 证据保留；它们全部排除出首版模拟器。适配器不得构造或调用 scheduler，不读取 scheduler config fingerprint，不生成 sigma/timestep，不执行 scheduler.step，也不更新 latent state。不得把 t/1000 或任何实际 scheduler timestep 值作为模拟器契约。

公共 `--sample-step=N` 必须为正整数，直接驱动 N 次 FLUX Transformer workload iteration；不存在 Diffusers `num_inference_steps` 的内部映射。每次 iteration 向 Transformer 传入固定的 `[B]` meta/dummy timestep，dtype 必须与 hidden states 相容，并保留 checkpoint 所需的 timestep embedding/operator path。该输入不是实际 schedule 值，其数值不是 CLI 或结果字段；结果只报告 dummy timestep 的 shape/dtype。

`guidance_embeds=true` 是 FLUX.1-dev 的 exact fingerprint，不得根据 generic Transformer constructor 默认值推断。公共 CLI 不提供 embedded-guidance 数值参数：adapter 自动构造 `[B]` 的 meta/dummy guidance 输入，以保留 `CombinedTimestepGuidance...` operator path；该输入的数值既不是用户输入，也不是结果字段。embedded guidance 每次 iteration 仍只有一次 forward。

true CFG 仅由 `--use-cfg` 启用，并要求同时提供有效负向文本长度。每次 iteration 执行正向和负向各一次；FLUX 合并的数据依赖写为 `negative + s_cfg * (positive - negative)`，其中 `s_cfg > 1` 只是非 CLI 符号，组合在对应 local shard 上以 `shape_only` 方式保留，详见 3.6。embedded guidance 与 true CFG 可同时启用，但每个分支各携带 checkpoint 要求的 guidance，逻辑 forward 总数仍为每次 iteration 2 次。

关键路径在所需 Transformer output、CFG/Ulysses communication 和 output ownership 完成后结束；不包含 scheduler update 或 latent state update。

### 3.5 四类输入的完整 Ulysses 分片

设 Ulysses size 为 `U`。首版不做隐式 padding，首次 forward 前必须满足：

```text
N_img % U == 0
L_pos % U == 0
L_neg % U == 0                 # true CFG 时
num_attention_heads % U == 0
```

当 `U=1` 时不产生并行 collective。当 `U>1` 时，FLUX context-parallel contract 对四类入口按如下维度分片：

```text
hidden_states_local:          [B, N_img/U, 64]       # split dim 1
encoder_hidden_states_local:  [B, L/U, 4096]         # split dim 1
img_ids_local:                [N_img/U, 3]           # split dim 0
txt_ids_local:                [L/U, 3]               # split dim 0
```

`L` 分别取正向 `L_pos` 或负向 `L_neg`。adapter 必须保证所有 local shard 覆盖全局 token 一次且不重不漏，并使 dual-stream/single-stream attention 使用与 Diffusers 0.38.0 一致的四类输入分片和 all-to-all 语义。每个 rank 的 Transformer image output 为 `[B,N_img/U,out_channels]`；文本 token 不进入 `proj_out`。单分支执行（`C=1`）时，在该分支的 Ulysses group 内按 sequence 维 gather 后恢复 `[B,N_img,out_channels]`；CFG parallel（`C=2`）时采用 3.6 规定的 local-shard combine 和指定 positive group 单次 gather，作为最终 output ownership，不执行 scheduler 更新。

任何 `N_img`、有效文本长度或 attention head 数不能被 `U` 整除时，必须在构建并行计划或首次 forward 前失败，并明确指出不满足的量；禁止 padding、隐式 trim 或退化到错误的单卡 shape。

### 3.6 `U × C` Cartesian CFG 拓扑

定义 CFG parallel size `C`：未启用 CFG parallel 时 `C=1`，这包括无 true CFG 和 `--use-cfg` 的顺序 true CFG；同时传入 `--use-cfg` 与 `--cfg-parallel` 时 `C=2`。公共架构要求：

```text
world_size = U * C
rank = branch_id * U + ulysses_id
```

当 `C=2` 时：

```text
positive Ulysses group = [0, ..., U-1]
negative Ulysses group = [U, ..., 2U-1]
cfg group(u)            = [u, U+u]
```

正、负分支各自在自己的 Ulysses group 内完成四类输入分片、attention collective 和 image output local shard。随后每个相同 `ulysses_id` 的 CFG pair 在对应 local image shard 上按 `negative + s_cfg * (positive - negative)` 保留合并依赖，并将合并结果交给指定的 positive Ulysses group（`[0, ..., U-1]`）执行一次 sequence gather；`s_cfg > 1` 不是 CLI 或性能结果字段，negative Ulysses group 也不再执行第二次完整 output gather。这样既不把正负序列强行 batch-concat，也不改变正负文本长度的 shape；关键路径按公共架构定义为正负 branch 的较大 measured 时间，加上 CFG shard communication 和这一次 Ulysses output gather。

结果 metadata 必须区分 `logical_transformer_forwards_per_step=2`、`cfg_parallel_size=2`、`ulysses_size=U`、`world_size=2U` 与 critical-path 时间。错误 world size、错误 rank/group 布局、`--cfg-parallel` 未配合 `--use-cfg`，或启用 CFG 却未提供负向条件，必须在首次 forward 前失败。

### 3.7 既有 TensorCast 融合算子适配验收

FLUX Transformer 的受支持 profile 必须在锁定的 `diffusers==0.38.0` 源码基线上，对 forward 中的融合候选进行审计，并将“存在现有 TensorCast 融合算子精确匹配时完成适配”作为验收目标。审计重点包括 attention 中 Q/K 的 RMSNorm-like normalization、text/image RoPE 路径，以及锁定源码中发现的其它既有 TensorCast 融合候选；这些是审计位置，不得仅按模块名或算子名称自动判定兼容。审计必须捕获 dual-stream、single-stream 及合法 Ulysses-local 输入下的 meta FX graph，先证明上游 graph topology 确实命中现有 pattern，再声明候选适用。

只有数学语义、epsilon、affine/bias、dtype cast、layout/broadcast、输入/输出数量与 ownership、mutation、RoPE 轴/交错/partial rotary dimension，以及 Ulysses-local shape 均与既有 TensorCast 融合算子 contract 完全一致时，候选才标记为适用。RMS residual 候选必须区分 `add_rms_norm.default` 的单输出与 `add_rms_norm2.default` 的归一化输出加 updated-residual 双输出语义。语义不匹配或现有算子无法表达时，必须记录 `not_applicable` 及差异原因，不得为追求融合而改变模型语义。确认适用的融合必须通过现有 TensorCast wrapper、patch 或 compile replacement seam 接入。

当前基础设施中的 `fused_rope.default` 是独立 partial/3D 显式算子，未发现可供 FLUX 自动命中的通用 graph pattern、performance property 或 profiling `op_mapping`；因此本 RFC 不把它列为已支持 measured fusion。FLUX text/image 多轴 RoPE 只有在精确匹配 `apply_rope.default` 或 `apply_rope_single.default` 的现有 topology、layout 和 interleaving contract，并具备当前性能映射覆盖时才可标记为 `applicable`；否则保持分解形式并记录 `not_applicable` 或 profiling coverage gap。

采用模型 wrapper/patch 时必须在 `torch.compile` 前完成 source-faithful 替换；采用现有 compile replacement 时由既有 backend/pattern pipeline 在编译过程中执行。`--compile` 测试必须通过 captured/compiled graph 和 Runtime operator 事件显示 exact lowered operator overload、输入/输出与 mutation contract 和预期 TensorCast 高层融合节点，并证明 `CombinedTimestepGuidance...` operator path、text-then-image RoPE IDs、dual/single-stream layout 及 Ulysses-local shape 未被改写。若现有 pass 可能产生多个等价 form，必须事先列出 accepted equivalent 并证明语义、输出 ownership 和性能边界等价。该节点吸收的 primitive 不得继续独立出现或重复计量，也不得新增未声明的 graph break；显式启用 `--compile-allow-graph-break` 时必须列出预期 boundary 和原因。进入 `measured` 的 exact operator 还必须具有 performance property，并能由记录的 profiling database/version、mapping 和既有插值规则覆盖声明 shape range；缺口不得回填 `0 ms`。compile 关闭/开启前后的 shape、workload、调用次数和并行 contract 必须保持不变。本目标不新增 fusion op family、CLI flag、runtime result field、registry、scheduler 工作或第四份 RFC，也不改变 CFG/Ulysses 语义。

## 4. 编程与调用约束

### 4.1 公共入口

调用方式、公共参数和阶段/结果字段以架构 RFC 为准。FLUX profile 不新增独立的 pipeline API，也不修改视频入口。与 FLUX shape 或 forward 次数直接相关的公共输入包括：

- 精确 `model_id` 与本地/远端来源；
- `--batch-size`；
- 恰好一次的 `--output-image-size HEIGHT WIDTH`，驱动 requested/effective output spatial size、latent packing 和 `N_img`；FLUX 不接受 `--source-image-size`；
- `--text-seq-len`，以及启用 true CFG 时的 `--negative-text-seq-len`；
- `--sample-step N`，直接驱动 N 次 FLUX Transformer workload iteration；
- `--use-cfg`，启用 true CFG，并要求同时提供负向长度；
- 公共 `--device`、`--dtype`、量化、compile、`--world-size`、`--ulysses-size`、`--cfg-parallel` 和 trace 参数。

不接受 prompt 字符串、图片路径或像素作为首版仿真输入。序列长度必须由调用方提供为已编码条件的有效长度，不能在 adapter 内执行 tokenizer 或猜测长度。

### 4.2 兼容性

本 RFC 新增 FLUX image-generation profile，不改变 `cli.inference.video_generate` 的模块路径、命令名、参数、默认值、Web UI 或既有结果行为。图像入口只使用 `--sample-step N`，直接驱动 N 次 FLUX Transformer workload iteration；不为视频入口增加或修改参数。公共架构与 FLUX profile 均不承诺真实图片输出、端到端延迟或质量兼容性。

本 RFC 只承诺精确 canonical profile 和满足 fingerprint 的本地目录。其它 checkpoint 即使具有相同的高层 pipeline 名称或 Transformer 类，也必须明确失败。Diffusers 版本升级、官方 config 关键字段变化或 pipeline signature 变化都要求重新审计并更新 profile 测试。

## 5. 测试设计

测试不下载完整权重、不执行真实 prompt 编码、不运行 VAE、不产生图片。全部模型测试以官方授权 config-only 目录、脱敏 fixture 或等价的结构化 config fixture 配合 meta device 完成，并保留 gated 远端失败路径。

### 5.1 Profile 与加载矩阵

1. canonical 远端 pair 精确命中 `black-forest-labs/FLUX.1-dev`。
2. 未登录/未授权 gated config-only 解析失败时，错误信息明确要求授权或提供本地 Diffusers 配置目录。
3. 合法本地 manifest 命中 `FluxPipeline` 及六个 simulator-required 组件；缺少任一必需目录/config 时失败，scheduler 目录和 config 不参与该判断。
4. 错误 pipeline class、错误 VAE/文本 encoder class、关键 Transformer fingerprint 不匹配时失败；不因 scheduler 配置触发模拟器运行时依赖。
5. 仅存在 `FluxTransformer2DModel` 或使用社区 mirror/其它 FLUX variant 时失败。
6. 读取 `guidance_embeds`、VAE scaling/shift，并在 metadata 中保留来源与版本；官方 FlowMatch scheduler 字段仅作为 research evidence，不进入模拟器 metadata 或运行时依赖。

### 5.2 Shape、workload 与固定 timestep 矩阵

1. `--output-image-size 1024 1024` 产生标准 `[B,16,128,128]` unpacked latent、`[B,4096,64]` image tokens 和 `[4096,3]` image IDs；该参数恰好一次驱动 requested/effective size、packing 和 `N_img`，首版不执行 VAE decode 或图片输出。
2. 多个 16 对齐尺寸验证 `N_img=(H/16)×(W/16)`；非 16 对齐尺寸验证 requested/effective size 与官方 floor 公式，且不存在 padding。
3. 正向文本长度覆盖合法上限及超过 512 的失败场景；`--use-cfg` 覆盖正负长度不同、启用后负向缺失、未启用却传负向长度和非法长度。
4. 验证 `[B,L,4096]`、`[B,768]`、`[L,3]`、`[B]` guidance/timestep 以及 text-then-image ID 语义。
5. `--sample-step N` 直接对应 N 次 Transformer workload iteration 和每次 1 次/2 次逻辑 Transformer forward；验证不构造、不调用 scheduler，固定 dummy timestep 输入的 dtype 与 shape 正确；`N<=0` 在 Runtime 启动前失败。
6. 验证不存在 scheduler config fingerprint、sigma/timestep 生成、resolution-dependent shift、scheduler.step、latent state update 或 schedule 输出字段。

### 5.3 Guidance、Ulysses 与 CFG 矩阵

1. `guidance_embeds=true` 是 exact fingerprint；adapter 自动构造 `[B]` meta/dummy guidance 输入并保留 `CombinedTimestepGuidance...` operator path，计数为每步一次；该数值不属于用户输入或结果字段。
2. `--use-cfg` 的顺序 true CFG 覆盖 `C=1`、正负长度不同和每步 `2N` logical forwards；embedded guidance 与 true CFG 同时启用仍为 `2N`。
3. `U=1` 验证无 collective；至少一个合法 `U>1` 验证 hidden/text/img_ids/txt_ids 四类 shard 的覆盖、RoPE ID 和 output gather。
4. 验证每层 dual-stream/single-stream attention 的 Ulysses collective 输入和 head/sequence shape 与 `U` 一致。
5. 验证 `C=2` 时 `world_size=2U`、两组 Ulysses group、逐 `ulysses_id` 的 CFG group、local-shard CFG combine、随后 output gather，以及 `max(positive,negative)` critical path 口径。
6. 对 `N_img%U`、`L_pos%U`、`L_neg%U`、`num_attention_heads%U` 各自构造不可整除失败；验证错误 world size、错误 rank layout、`--use-cfg`/负向长度不完整以及 `--cfg-parallel` 未配合 `--use-cfg` 均在首次 forward 前失败。
7. 结果 metadata 同时记录 requested/effective size、token 数、guidance/CFG、U/C/world size、logical forward 数、fixed timestep shape/dtype、`scheduler_stage=excluded`、communication/bound breakdown 和阶段覆盖标记；不输出 schedule 字段。

### 5.4 回归与边界测试

- 公共架构 fake adapter 测试由[图像生成公共架构实现 PR](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)负责，本 RFC 不复制其实现；FLUX.1-dev 测试只验证本模型 profile 专属行为。
- FLUX 模型测试必须提供 `applicable / adapted / not_applicable` 融合候选清单，覆盖 Q/K RMSNorm-like normalization、text/image RoPE 及锁定 Diffusers 源码中发现的其它既有 TensorCast 候选；`not_applicable` 必须记录数学、graph topology、布局、broadcast、输入/输出 ownership、mutation、partial rotary dimension、RoPE 或 Ulysses-local shape 等语义差异。`fused_rope.default` 在当前基线不得仅凭名称列为 `applicable`。
- 对每个 `applicable` 候选，测试必须捕获 U=1 及至少一个合法 U>1 local shape 的 meta FX graph；wrapper/patch 在 `torch.compile` 前完成，现有 pattern replacement 在 compile backend 内完成。`--compile` 图和 Runtime operator 事件必须断言 exact overload、预期 TensorCast 高层融合节点、无预期 primitive decomposition、无新增 graph break，并验证 `CombinedTimestepGuidance...` operator path、text/image IDs、dual/single-stream layout 及 compile 开关前后的 shape/workload/并行 contract 不变。
- 进入 `measured` 的融合节点必须验证 performance property、profiling database/version、exact mapping 和声明 shape range 覆盖；mapping 或 shape-data gap 必须独立诊断，不能以 `0 ms` 通过。
- `video_generate` 既有测试保持通过，且不出现任何 FLUX 参数或行为变化。
- 不把 fixture 的 meta forward 成功表述为真实权重推理成功；不把 Transformer 去噪阶段模拟时间中的 measured 部分表述为端到端生成时间。

### 5.5 Canonical model ID simulator E2E 与 Chrome trace 一致性

这里的 E2E 只指“canonical model ID 到仿真结果与 Chrome trace”的 simulator 闭环，不包含 prompt 编码、VAE、真实权重或图片生成。FLUX.1-dev 适配 PR 必须提供一条 hermetic simulator E2E，用公共 `image-generate` CLI 或其同一公开 orchestration 入口完成以下完整路径，而不是分别 mock adapter、Runtime 和 trace exporter：

```text
model_id=black-forest-labs/FLUX.1-dev
  -> exact-profile resolver
  -> config-only fingerprint validation
  -> meta FluxTransformer2DModel builder
  -> N-step Runtime workload
  -> structured simulation result
  -> Chrome trace file
```

提交到仓库的测试不得依赖 gated Hub 登录或读取凭据。测试仍必须传入 canonical `model_id` 和 `remote_source=huggingface`，但在 resolver 的远端读取边界使用脱敏的官方 config-only fixture 或等价受控响应，使 model ID 分派、配置校验和后续公共执行链全部真实运行；不得直接把已经构造好的 adapter 注入 workload loop。基准请求固定为 `B=1`、`--output-image-size 1024 1024`、合法文本长度、`--sample-step=2`、`U=1,C=1`、不启用 true CFG，并提供临时 `--chrome-trace` 路径。

成功结果必须同时证明：canonical model/profile 与 config fingerprint 正确；logical Transformer forward 数恰好为 2；operator table 和 measured event 非空；Transformer critical path 大于 0；固定 dummy timestep/guidance shape 正确；`scheduler_stage=excluded`；prompt/text encoder/VAE/image I/O 未进入 measured stage；结果中的 trace 引用指向本次 Runtime 实际导出的文件。缺少 performance mapping、出现 0 个 measured event、只完成 resolver/meta build 或单独调用 trace exporter 均不算 E2E 成功。

Chrome trace 文件必须是可解析的 `{"traceEvents": [...]}` JSON，并包含 process/thread metadata 和非空 complete events。测试从锁定的 `diffusers==0.38.0` `FluxTransformer2DModel.forward`、本 RFC fingerprint 和同输入 meta FX graph 独立构造 normalized modeling fingerprint，再与 Chrome trace 中每条 stream 的有序 operator target、`simulation_shapes`、通信事件和调用倍数比较；不得把本次实际 trace 复制成 golden oracle。normalized fingerprint 必须证明每个 logical forward 覆盖 19 个 dual-stream block、38 个 single-stream block、一次固定 timestep/guidance path、正确的 text/image sequence 与 ID shape，以及最终 image projection；两次 workload 的结构必须按 N=2 完整重复，不能因没有 scheduler update 而跳过 Transformer block。

比较不要求 `ts`、`dur`、`pid` 数值或 JSON 字节级相等。compile 关闭时，primitive operator fingerprint 必须与 modeling graph 一致；compile 开启时，允许使用 3.7 已预先声明的 accepted equivalent，以一个 TensorCast 融合节点替代对应 primitive 子图，但禁止 fused node 与被吸收 primitive 同时出现。`U>1`、true CFG 和 CFG parallel 继续由 5.3 的矩阵测试看护，不扩张这条最小 E2E 用例。

## 6. 风险、替代方案与缓解

### 6.1 风险

1. **受限配置访问风险**：官方 gated config 在无授权环境不可读取。缓解方式是保留明确的 gated failure，并支持用户提供已获授权的本地 Diffusers 配置目录；绝不使用社区镜像静默替代。
2. **Diffusers 版本漂移风险**：公开文档和本地安装版本可能不同，输入 signature 或 context-parallel 细节可能变化。缓解方式是锁定 `diffusers==0.38.0`，并在升级时重新做 Transformer source/config 审计；scheduler 变化不属于模拟器运行时契约。
3. **shape 误建模风险**：忽略 VAE scale 或 2x2 packing 会将 image sequence length 放大或缩小，直接改变 attention 性能。缓解方式是将 VAE config、packing 公式、1024² fixture 和非对齐 fixture 作为 profile 验收条件。
4. **并行通信误计量风险**：正负分支 batch-concat 或只分片 hidden states 会改变真实 sequence/ID/collective 语义。缓解方式是固定四类输入分片、`U × C` Cartesian groups、local CFG combine 和 output gather 的测试矩阵。
5. **既有融合语义误判风险**：仅按模块名将 RMSNorm 或 RoPE 视为可融合，可能改变 epsilon、cast、layout 或 Ulysses-local shape 语义。缓解方式是按锁定 Diffusers 源码逐项审计并记录 `not_applicable`，适用项只接入现有 TensorCast seam，且以 compile graph 验证高层节点和 graph-break 不变。
6. **结果边界误读风险**：用户可能将 Transformer 去噪阶段模拟时间当成完整图片生成时间。缓解方式是每个结果输出阶段覆盖声明，并明确列出 tokenizer、VAE、I/O、权重和质量均未测量。
7. **许可证与部署误用风险**：仿真支持不等于获得 checkpoint 的商业部署权利。缓解方式是保存 license/access metadata，并在 profile 文档中明确 FluxDev 非商业许可限制。

### 6.2 替代方案

- **只按 Transformer 类名自动支持**：实现简单，但会误接受错误 pipeline 或其它 FLUX checkpoint，无法保证 shape 和许可证身份；不采用。
- **采用通用 FLUX/多 scheduler 插件抽象**：未来扩展性较强，但会扩大首版 API、掩盖版本差异并引入未验证的兼容承诺；首版固定一个 canonical profile，并将 scheduler 排除在模拟器之外，不采用。
- **用 padding 解决 Ulysses 不整除**：可提高部分尺寸的可运行率，但改变有效 token 数、attention 成本、位置 ID 和结果解释；首版采用 fail-fast，不采用。
- **将 true CFG 两分支拼成一次 batch forward**：可减少调度代码，但会改变正负文本长度、mask、分片和 critical-path 语义；首版采用显式分支，CFG parallel 仅在公共拓扑下并行，不采用拼接近似。
- **将 VAE/text encoder 纳入端到端测量**：更接近用户感知延迟，但超出当前公共 Transformer 去噪契约，也需要真实输入和权重；留待后续独立 RFC，不纳入本版。

<a id="flux1-dev-adaptation-boundary"></a>

## 7. FLUX.1-dev 适配 PR 边界

FLUX.1-dev 适配 PR 只实现本 RFC 的模型适配目标：

1. canonical FLUX.1-dev profile 与本地 fingerprint/config validation；
2. `DiffusersTransformerModel` meta 构建、VAE scale/latent shape、2x2 packing、position IDs 和 pooled/text inputs；
3. 固定 dummy timestep 的 FLUX timestep embedding/operator path；
4. embedded guidance、顺序 true CFG 和对应调用计数；
5. hidden/text/img_ids/txt_ids 四类完整 Ulysses sharding、attention collectives、image output gather 和整除 fail-fast；
6. `U × C` Cartesian topology、local-shard CFG combine 与 critical-path metrics；
7. config-only/meta/profile/并行回归测试；
8. 锁定 Diffusers 源码和 meta FX graph 的既有 TensorCast 融合候选审计、适用项接入顺序、exact overload/Runtime event、`--compile` 图及 measured performance mapping 验证；当前不扩展 `fused_rope.default` 支持。
9. canonical FLUX model ID 经公共入口完成 config-only resolver、meta build、N-step Runtime、结构化仿真结果和 Chrome trace 导出的 hermetic E2E；以独立 modeling fingerprint 验证 19 个 dual-stream block、38 个 single-stream block 及 accepted fusion equivalents。

FLUX.1-dev 适配 PR 仅依赖已合入的[图像生成公共架构实现 PR 边界](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)，与 Qwen-Image-Edit 适配 PR 无依赖关系。它只新增或修改 FLUX.1-dev 自身的 profile descriptor、adapter、fixture 和测试，通过公共静态组装 seam 接入；不得修改公共组装规则、Qwen-Image-Edit 文件、公共 CLI 参数语义、公共结果契约、视频入口或 Web UI。FLUX.1-dev 测试在 Qwen-Image-Edit profile、adapter、fixture 和测试均不存在时仍必须通过。

若发现公共字段、组装规则或 contract 不足，必须先通过独立的公共架构变更补齐并更新架构 RFC，再基于新的稳定 commit 更新 FLUX.1-dev 适配 PR；不得在该模型适配 PR 中静默扩张公共范围。该 PR 不新增 fusion op family、融合 registry、scheduler 工作或改变 CFG/Ulysses 语义，并可在公共依赖满足后独立评审和合入。

## 8. 验收结论

本 RFC 通过的最低条件是：精确远端/本地 profile 可诊断分派；无权重 config-only/meta 构建可验证；恰好一次 `--output-image-size HEIGHT WIDTH` 驱动的 requested/effective size、latent packing、`N_img`、标准和非对齐尺寸的 IDs 正确；`--sample-step=N` 直接驱动 N 次 Transformer workload iteration，固定 dummy timestep 的 shape/dtype 正确，且无 scheduler 构造、调用、latent update 或 schedule 字段；embedded guidance、true CFG、N-step 调用计数符合公共契约；embedded guidance 保留 `CombinedTimestepGuidance...` operator path 且不增加 forward；合法 `U>1` 下四类输入 shard、attention collective、output gather 和 `U × C` CFG 拓扑可验证；对锁定 Diffusers 源码中的融合候选完成逐项审计，精确匹配的既有 TensorCast 融合均通过现有 seam 接入，`not_applicable` 均有语义差异证据，`--compile` 图显示预期高层融合节点且无预期 primitive decomposition 或新增 graph break，compile 开关前后的 shape/workload/并行 contract 不变；非法 fingerprint、配置、参数、`--source-image-size`、整除关系和 world size 在首次 forward 前稳定失败；结果明确声明只测 Transformer 去噪阶段、关键路径止于必要 output/communication/ownership、首版不输出图片；`video_generate` 行为不变。

只有满足这些条件，FLUX.1-dev 才可被标记为本仓图像生成仿真的受支持 profile；config-only 成功不能被解释为已下载权重、已完成真实推理或已支持完整图像生成。
