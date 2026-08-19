# RFC：增加 FLUX.1-dev 图像生成仿真支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft（草案） |
| **作者** | `minghang_c` |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-13 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |
| **规范依赖** | [图像生成公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index) |

---

## 1. 概述

本 RFC 定义在既有图像生成公共架构上支持精确模型
`black-forest-labs/FLUX.1-dev` 的实现边界。实现只模拟 FLUX Transformer 的固定形状、固定 timestep 工作负载和必要的并行通信；不执行 prompt 编码、权重推理、VAE、scheduler 数值更新或真实图片生成。

命令行生命周期由 `cli/inference/image_generate.py` 持有，可复用模型代码放在
`tensor_cast/diffusers/flux_image.py`，静态 lazy 分派在
`tensor_cast/diffusers/image_dispatch.py` 中以 `flux1-dev` 分支完成。FLUX 分支不引入
`tensor_cast/image_generation`，不引入 request/result/stage 对象、通用 engine、plugin、registry 或 handler 对象。

输出是成功执行后的 Model compilation execution time 和 Runtime 表；用户请求 Chrome trace 时，在成功后导出 trace。输出表示 Transformer 工作负载仿真时间，不表示端到端生成延迟、图片质量、真实权重占用或真实图片。

## 2. 目标与排除项

### 2.1 目标

1. 只接受精确远端身份 `remote_source=huggingface`、`model_id=black-forest-labs/FLUX.1-dev`，以及严格匹配同一官方配置的本地 Diffusers 根目录。
2. 以可执行的 `diffusers==0.38.0` 为源码和 forward signature 基线，在 meta device 上建立 `FluxTransformer2DModel`。
3. 依据真实 VAE config 计算 latent 与 2x2 packed image-token 几何，并构造 FLUX 所需的文本、pooled projection、位置 ID、timestep 和 embedded guidance 输入。
4. 遵守公共 `image-generate` CLI、固定形状 N-step Transformer workload、普通 CFG、CFG parallel、Ulysses 和 Runtime 计时契约。
5. 将 FLUX 双流 `transformer_blocks` 与单流 `single_transformer_blocks` 的缓存建模作为本 PR 的完成门槛，而不是未来工作。
6. 通过 config-only、shape、并行、cache、compile 和首次完整 CLI 生命周期测试验证实现。

### 2.2 明确排除项

- `FLUX.1-schnell`、Fill、Redux、img2img、inpainting、ControlNet、IP-Adapter、LoRA 和其它 FLUX 变体。
- prompt 字符串、tokenizer、CLIP/T5 encoder、真实文本编码、权重下载或加载。
- source-image 输入、VAE encode/decode、scheduler、timetable、sigma、latent 数值更新、postprocess、保存文件和真实图片。
- 质量、显存权重占用、端到端延迟和网络访问性能。
- 对公共 CLI、公共输出契约、视频入口或 Qwen-Image-Edit 的改动。
- 以 padding、隐式 trim、类名猜测或社区镜像掩盖不兼容配置。

## 3. 公共契约依赖

本 RFC 规范性引用以下公共章节，不重复定义公共行为：

- [公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index)
- [公共 CLI contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-cli-contract)
- [静态分派 contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-dispatch-contract)
- [workload contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-workload-contract)
- [CFG contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-cfg-contract)
- [topology contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-topology-contract)
- [cache contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-cache-contract)
- [output contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-output-contract)
- [模块边界](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-module-boundary)
- [公共实现边界](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)

公共入口负责参数解析、生命周期、静态分派、Runtime、Model compilation execution time、Chrome trace 和既有视频行为。FLUX 专属实现只负责精确配置验证、FLUX 输入几何、FLUX forward 准备、Ulysses 分片和 FLUX Transformer 缓存。

## 4. 精确身份与配置验证

### 4.1 远端身份

远端只接受：

```text
remote_source = huggingface
model_id      = black-forest-labs/FLUX.1-dev
```

FLUX.1-dev 是 gated repository。config-only 远端读取失败时，错误必须说明配置路径、期望的 canonical identity 和实际错误，并提示用户使用已获授权的本地配置目录。不得自动下载权重、回退镜像、执行 remote code 或因同名类存在而放宽检查。

### 4.2 本地根目录

本地根目录是选择候选的唯一入口；严格验证决定是否接受。必须存在 `model_index.json` 以及 Transformer、VAE 和文本组件配置。验证至少包括：

- `model_index.json._class_name == "FluxPipeline"`；
- `vae` 为 `AutoencoderKL`；
- `text_encoder` / `tokenizer` 为 `CLIPTextModel` / `CLIPTokenizer`；
- `text_encoder_2` / `tokenizer_2` 为 `T5EncoderModel` / `T5TokenizerFast`；
- `transformer` 为 `FluxTransformer2DModel`；
- `transformer/config.json` 和 `vae/config.json` 存在；
- Transformer 关键字段与 FLUX.1-dev 精确匹配：`in_channels=64`、`num_layers=19`、`num_single_layers=38`、`attention_head_dim=128`、`num_attention_heads=24`、`joint_attention_dim=4096`、`pooled_projection_dim=768`、`axes_dims_rope=(16,56,56)`、`guidance_embeds=true`；
- raw `out_channels` 为 `None` 时按 Diffusers 语义规范化为 64，任何其它值必须明确比较并失败；
- VAE whitelist 精确为：`latent_channels=16`、`block_out_channels=(128,256,512,512)`、四个 `DownEncoderBlock2D`、四个 `UpDecoderBlock2D`、`layers_per_block=2`、`scaling_factor=0.3611`、`shift_factor=0.1159`、`latents_mean=None`、`latents_std=None`、`use_quant_conv=false`、`use_post_quant_conv=false`；这些值从实际 `vae/config.json` 读取并逐字段验证。

Scheduler 不参与接受或拒绝判断：不读取 scheduler config，不要求 scheduler 目录存在，也不因为 scheduler 类名变化改变 FLUX Transformer 工作负载。错误必须同时写出配置路径、期望值和实际值。只有存在 `FluxTransformer2DModel` 而其它字段不匹配时，仍必须稳定失败。

## 5. 模块与构建顺序

### 5.1 模块边界

- `cli/inference/image_generate.py`：持有 CLI 生命周期，调用公共 Runtime，成功后报告 Model compilation execution time 和 Runtime 表，并按请求导出 Chrome trace。
- `tensor_cast/diffusers/image_dispatch.py`：静态 lazy 分派 `flux1-dev`，不创建通用 registry 或动态 plugin。
- `tensor_cast/diffusers/flux_image.py`：持有严格配置验证、VAE-derived geometry、FLUX 输入、CFG duplication、Ulysses sharding、forward helper 和 cache spec。它直接返回绑定 `FluxTransformer2DModel` 的 `DiTBlockCacheSpec`；不在 import time 注册 FLUX 规格，也不依赖全局 image registry。
- `tensor_cast/diffusers/dit_cache_registry.py`：仅提供公共 `DiTBlockCacheSpec` 类型与既有 video cache 的 registry/helper；FLUX helper 返回的规格由调用方直接使用，不能通过 video fallback 接受错误 class。

### 5.2 预构建顺序

构建顺序固定为：

1. 仅配置的候选选择；
2. `flux1-dev` 静态 kind resolution；
3. 严格 pipeline、组件和字段验证；
4. 输入准备、普通 CFG duplication 和 Ulysses 分片计划；
5. 从已验证配置独立构造 baseline model，并在 compile 前应用 FLUX source-faithful model-specific patch；
6. 若请求 cache，则从同一份已验证配置独立构造第二个 cache model，应用相同 patch，再替换其 cache blocks；
7. 分别 compile baseline model 与 cache model，Runtime 按 per-run state 选择已编译实例。

真正的 cache build 必须是独立模型，不能从 baseline、已修改实例或已编译实例复制状态。cache replacement 只发生在第二个 cache model 上，并且发生在各自 compile 之前。

## 6. FLUX 输入与几何

设 batch 为 `B`，用户通过公共 `--output-image-size HEIGHT WIDTH` 提供请求尺寸 `(H,W)`，通过公共 `--text-seq-len` 提供有效文本长度 `L`。FLUX 不接受 source-image 输入。`--output-image-size` 必须恰好出现一次；requested size 与根据官方公式得到的 effective size 都要准确记录。

从实际 VAE config 读取：

```text
vae_scale_factor = 2 ** (len(block_out_channels) - 1)
H_lat = 2 * floor(H / (2 * vae_scale_factor))
W_lat = 2 * floor(W / (2 * vae_scale_factor))
C_lat = transformer.in_channels / 4
N_img = (H_lat / 2) * (W_lat / 2)
```

必须验证 `4 * C_lat == transformer.in_channels`。标准八倍 VAE 且尺寸为 16 的整数倍时，`H_lat=H/8`、`W_lat=W/8`、`C_lat=16`，故 1024×1024 对应 unpacked latent `[B,16,128,128]` 与 image tokens `[B,4096,64]`。非 16 对齐尺寸不得 padding；按公式得到的 effective size、latent shape 和 token 数必须与 requested size 分开说明。

`flux_image.py` 构造的 meta 输入为：

```text
hidden_states:          [B, N_img, 64]
encoder_hidden_states:  [B, L, 4096]
pooled_projections:     [B, 768]
img_ids:                [N_img, 3]
txt_ids:                [L, 3]
timestep:               [effective_batch]
guidance:               [effective_batch]  # guidance_embeds=true 时
```

`img_ids` 的通道 0 为零，通道 1/2 表示 latent 行列；`txt_ids` 与 `img_ids` 按 Diffusers 的 text-then-image RoPE 语义构造。FLUX Transformer 输出是完整 image-token 输出，不需要额外切片。

## 7. Forward、timestep 与 embedded guidance

FLUX forward 必须匹配 Diffusers 0.38.0 `FluxTransformer2DModel.forward` 的完整参数契约：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
): ...
```

本 workload 使用前七项模型输入，并保持其余参数为上述默认值；不得传 scheduler 参数。

不得传 scheduler 参数。`--sample-step=N` 是固定形状、固定 timestep 的 Transformer workload 次数；不生成 timetable，不调用 scheduler，不更新 latent。每步 timestep shape 等于 effective batch：无 CFG 与 CFG-parallel representative forward 为 `[B]`，普通 CFG 为 `[2B]`；dtype 与 hidden states 相容。N 次只意味着 N 次相同形状的 Transformer forward 工作负载。

FLUX.1-dev 的 `guidance_embeds=true` 是严格字段要求。`flux_image.py` 自动构造 `[B]` dummy guidance；普通 CFG 时先形成 effective batch 后构造对应 `[2B]` guidance。它保留 `CombinedTimestepGuidance...` operator path，不是 CLI 数值参数，不增加 forward，也不出现在 Runtime 结果中作为用户 scale。

## 8. 普通 CFG 与 CFG parallel

### 8.1 普通 `--use-cfg`

普通 `--use-cfg` 是模型专属的 batch-concat 输入语义：effective batch 为 `2B`，正负条件使用同一个 `--text-seq-len`，每步只执行一次 Transformer forward。必须在 batch 维准确复制：

- `hidden_states`；
- `encoder_hidden_states`；
- `pooled_projections`；
- `timestep`；
- `guidance`（如存在）。

不得复制 `img_ids [N_img,3]` 或 `txt_ids [L,3]`。普通 CFG 不接受独立的负向 sequence length，不做 numerical CFG combine 或 CFG scale。测试必须明确覆盖 effective batch `2B` 和每步一次 forward。

### 8.2 CFG parallel

CFG parallel 只在同时启用 `--use-cfg` 与 `--cfg-parallel` 时生效。设 Ulysses size 为 `U`：

- 无 CFG：world size=`U`，batch=`B`，每步一次 forward；
- 普通 `--use-cfg`：world size=`U`，effective batch=`2B`，每步一次 forward；
- CFG parallel：world size=`2U`；单个 representative rank 以 local batch=`B` 每步执行一次 forward，代表两个形状相同、并发的 CFG branch critical paths。

CFG parallel 的 Ulysses groups 为 `[0..U-1]` 和 `[U..2U-1]`；CFG group 为 `[u,U+u]`。每个 branch 先在自己的 Ulysses group 内完成 attention 所需的分片与 gather；随后 Ulysses gather 之后，CFG group 对 dim 0 执行 all-gather，使逻辑结果形状为 `2B`。模拟器不顺序执行两个 branch，也不把 representative rank 描述成两次 forward；非法 world size、rank layout 或未配对的 flags 必须在首次 forward 前失败。

## 9. Ulysses 分片

当 `U>1` 时，首次 forward 前必须满足：

```text
N_img % U == 0
text_seq_len % U == 0
num_attention_heads % U == 0
```

禁止 padding、trim 或降级到错误的单卡 shape。四类输入的分片固定为：

```text
hidden_states_local:         [B, N_img/U, 64]       # dim=1
encoder_hidden_states_local: [B, L/U, 4096]         # dim=1
img_ids_local:               [N_img/U, 3]            # dim=0
txt_ids_local:               [L/U, 3]                # dim=0
```

Transformer 输出按 dim=1 gather。所有 rank 必须恰好覆盖全局 image/text token；不得重复或遗漏。attention heads 也必须整除 U。Ulysses gather 必须先于 CFG parallel 的 dim-0 all-gather。

## 10. Transformer cache 完成门槛

`flux_image.py` 必须直接返回严格 class-bound 的 `DiTBlockCacheSpec`，其 `class_name` 精确为 `FluxTransformer2DModel`，而不是仅以模块名称接受。零 block、错误 class 或缺少任一 required block collection 时稳定失败。

cache discovery 和排序必须同时覆盖：

- dual-stream `transformer_blocks`；
- single-stream `single_transformer_blocks`。

缓存 wrapper 必须保留原始 callable 的精确 signature、输入输出语义、mutation 和 dtype/layout 语义，并支持 per-run update/reuse delta state。半开区间 `[start,end)` 必须只覆盖声明的 block；end 超过发现的 block 数时 clamp 到末端，clamp 后为空必须失败，零 replacement 也必须失败。双流与单流的发现顺序、层数和执行顺序必须可测试。

cache 测试是本 PR 的完成门槛，至少覆盖：双流/单流 discovery 与排序、wrapper signature、per-run update/reuse、半开 block range 的 end clamp 与 empty、普通 CFG `2B`、Ulysses 交互、严格 class failure、严格 zero-replacement 和 zero-block failure。未完成 cache 不得标记 FLUX 支持完成。

## 11. TensorCast 融合审计

FLUX 融合必须遵守现有 TensorCast 基础设施，遵循 source-faithful semantic matching：先从 Diffusers 0.38.0 源码和 meta FX graph 确认 topology，再逐项核对数学语义、epsilon、affine/bias、dtype cast、layout、broadcast、输入输出数量、输入输出、mutation、RoPE 轴/交错/partial rotary dimension 和 Ulysses-local shape。

模型 wrapper 或 patch 必须在 `torch.compile` 前完成；已有 compile replacement 才由既有 backend/pattern pipeline 在 compile 阶段替换。dual-stream、single-stream、U=1 和合法 U>1 local shape 都要审计。可适用的候选必须通过既有 wrapper、patch 或 replacement seam；不匹配项记录 `not_applicable` 及差异，不得改变模型语义。

不得声称 `fused_rope.default` 已获支持：当前基础设施没有 FLUX 自动命中的通用 graph pattern、performance property 或 profiling `op_mapping`。只有精确匹配现有 `apply_rope.default` 或 `apply_rope_single.default` 的 topology、layout、interleaving 和性能覆盖时才可适用，否则保持分解形式。

进入 measured 的融合节点必须有 performance property、记录 profiling database/version、exact mapping 和覆盖声明 shape range；缺口不得用 `0 ms` 填充。compile 开关前后 shape、workload、调用次数、并行契约和输出 输入输出 必须不变；被融合 primitive 不得继续出现或重复计量。

## 12. 输出与生命周期

公共 CLI 成功后必须使用公共 RFC 的精确文本输出：

```python
print(f"Model compilation execution time: {run_end - run_start}s")
print(runtime.table_averages(group_by_input_shapes=False))
if chrome_trace:
    runtime.export_chrome_trace(chrome_trace)
    print(f"Chrome trace written to: {chrome_trace}")
```

Chrome trace 只能在本次 Runtime 成功且用户请求时生成。Runtime 表只覆盖 Transformer forward 及必要的 Ulysses/CFG 通信；config 读取、meta build、prompt/tokenizer、VAE、scheduler、图片 I/O 和权重均不进入 measured time。

FLUX 输出是完整 image-token 输出的 shape-only 末端。无 CFG 时输出 batch 为 `B`；普通 CFG 时输出 batch 为 `2B`；CFG-parallel representative rank 在 CFG all-gather 前的 local batch 为 `B`，Ulysses gather 与后续 CFG dim-0 all-gather 完成后逻辑 batch 为 `2B`。序列形状均为 `[batch,N_img,out_channels]`。不执行 scheduler 或 latent update，不保存图片。不得引入模型专属 output object 或额外的结果封装层。

## 13. 测试设计

测试不访问 Hub、不读取凭据、不下载权重、不执行真实 prompt、VAE 或图片生成。canonical model ID 测试使用脱敏官方 config-only fixture 或等价受控响应，但仍通过真实公共入口和 `flux1-dev` 静态分派。

### 13.1 身份与配置

- canonical remote pair 精确命中；gated 失败信息包含配置路径、授权本地目录提示和实际错误；
- 合法 `FluxPipeline` 根目录及 required component classes 命中；scheduler 缺失不影响接受；
- pipeline、VAE、文本组件、Transformer fingerprint、VAE scaling/shift、`guidance_embeds` 任一错误都稳定失败；
- 仅有 `FluxTransformer2DModel`、社区镜像、其它 FLUX variant 或缺失 config 都失败；
- 错误信息明确列出 expected/actual。

### 13.2 几何、ID 与 guidance

- 1024×1024 产生 `[B,16,128,128]` latent geometry、`[B,4096,64]` image input、`[4096,3]` image IDs；
- 多种对齐与非对齐尺寸验证 requested/effective size、floor 公式、`4*C_lat` invariant 和无 padding；
- `[B,L,4096]`、`[B,768]`、`[L,3]`、`[B]` timestep/guidance 以及 text-then-image IDs 正确；
- forward kwargs 与 Diffusers 0.38.0 signature 完全一致且不含 scheduler args；
- `guidance_embeds=true` 构造 guidance，不增加 forward，不接受 CLI 数值 scale；
- FLUX 不接受 source-image 输入。

### 13.3 CFG、并行与 workload

- 无 CFG 为 world=`U`、batch=`B`、每步一次 forward；
- 普通 `--use-cfg` 为 effective batch=`2B`、同一 text length、每步一次 forward，且仅复制 batch-first tensors；
- CFG parallel 为 world=`2U`、两个 Ulysses groups、`cfg group(u)=[u,U+u]`、Ulysses gather 先于 CFG dim-0 all-gather；
- Ulysses 覆盖四类输入的分片和 output gather；`N_img`、text length、heads 不整除分别失败；
- `--sample-step=N` 恰好执行 N 次固定形状 workload；不构造 scheduler、不生成 schedule、不更新 latent；
- compile off/on 保持形状、workload、通信和 forward 计数一致。

### 13.4 Cache 与完整 CLI 生命周期

- cache 测试覆盖双流/单流 discovery/order、严格 class、zero-block、wrapper signature、update/reuse、半开范围、普通 CFG 2B 和 Ulysses；
- canonical ID 通过首次完整 `image-generate` 生命周期：静态分派、严格验证、meta build、FLUX patch、cache、compile、N-step Runtime、Model compilation execution time、Runtime 表、Chrome trace；
- 测试不直接注入已构造的模型或绕过分派；
- trace 是可解析 `{"traceEvents": [...]}` JSON，包含非空 complete events；dual-stream 与 single-stream 的 block 数量和顺序必须从已验证的 exact config/fixture 派生核对，不使用未验证的硬编码数量；同时核对固定 timestep/guidance、ID shapes、最终 image projection 与 N 次重复；
- 没有 measured event、只完成 config/meta build、或单独调用 trace exporter 都不算成功；
- 既有 `video_generate` 测试通过且无 FLUX 参数或行为变化。

## 14. 风险与替代方案

1. **gated 配置不可读**：保留可诊断失败，提示已获授权的本地配置目录；绝不下载权重或使用镜像。
2. **Diffusers 漂移**：锁定 0.38.0；升级必须重新审计 class、signature、geometry、RoPE 和分片行为。
3. **几何错误**：用实际 VAE config、2x2 packing、1024 fixture、非对齐 fixture 和 `4*C_lat` 检查约束。
4. **并行形状错误**：固定四类分片维度、整除 fail-fast、Ulysses-first 顺序和 CFG group 布局。
5. **融合误判**：以源码和 FX graph 的逐项语义审计为准；没有精确现有基础设施就标为 not applicable。
6. **输出边界误读**：Runtime 表和 trace 明确只覆盖 Transformer 工作负载；不声称图片或端到端延迟。

不采用以下替代方案：只按类名支持、通用 FLUX registry/plugin、scheduler 抽象、padding、额外 CFG 数值组合、真实 prompt/VAE/图片流程，以及把 cache 推迟到后续 PR。

## 15. PR 边界与验收结论

本 PR 可以新增 `tensor_cast/diffusers/flux_image.py`，增加 `image_dispatch.py` 的精确 `flux1-dev` 分支，并更新 FLUX fixture/test。它依赖公共 Core，独立于 Qwen；cache 是合入门槛。不得修改视频行为、公共 CLI 合同、公共输出合同、公共组装规则或 Web UI。

最低验收条件是：canonical identity 和本地 strict config validation 可诊断；VAE-derived geometry、packing、IDs、guidance 和 forward kwargs 与 Diffusers 0.38.0 一致；普通 CFG 为 `2B` 单 forward；CFG parallel 为 `2U` 并发代表性 forward 且通信顺序正确；Ulysses 四类分片、gather 和 fail-fast 正确；N-step 为固定 Transformer workload；双流/单流 cache 完成并通过全部门槛；source-faithful fusion audit 不越过现有 TensorCast infra；compile off/on、首次完整 CLI 生命周期和 hermetic Chrome trace 测试通过。

满足上述条件后，FLUX.1-dev 才可标记为受支持的图像生成 Transformer 仿真 kind。config-only 成功不代表权重已下载、真实推理已完成或真实图片生成已支持。
