# RFC：增加 Qwen-Image-Edit 图像编辑仿真支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft（草案） |
| **作者** | minghang_c |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-13 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |
| **规范依赖** | [图像生成推理性能仿真架构 RFC](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index) |

---

## 1. 概述

本 RFC 为既有图像生成公共架构增加 Qwen-Image-Edit 的固定形状 Transformer 仿真支持。首版不是图片编辑器，也不是端到端推理器；它只在 config-only、meta-device 和 Runtime 边界内重现 Qwen 去噪 Transformer 的输入布局、调用次数、并行通信、缓存路径和设备执行时间。

支持范围严格限定为三个 canonical model ID：

- `Qwen/Qwen-Image-Edit`，模型 kind `qwen-image-edit`；
- `Qwen/Qwen-Image-Edit-2509`，模型 kind `qwen-image-edit-2509`；
- `Qwen/Qwen-Image-Edit-2511`，模型 kind `qwen-image-edit-2511`。

生命周期由 `cli/inference/image_generate.py` 持有；可复用的 Qwen 形状和调用逻辑位于 `tensor_cast/diffusers/qwen_image_edit.py`；静态 lazy 分派位于 `tensor_cast/diffusers/image_dispatch.py`。本适配不创建 `tensor_cast/image_generation`，不创建通用 image engine、动态 plugin/registry、请求/结果/stage 对象体系，也不增加兼容层。公共入口、Runtime 和输出边界均继承公共 RFC，Qwen 文件只实现 Qwen 专属规则。

本 RFC 的公共锚点是：

- [image-generation-public-contract-index](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index)
- [image-generation-public-cli-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-cli-contract)
- [image-generation-dispatch-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-dispatch-contract)
- [image-generation-workload-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-workload-contract)
- [image-generation-cfg-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-cfg-contract)
- [image-generation-topology-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-topology-contract)
- [image-generation-cache-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-cache-contract)
- [image-generation-output-contract](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-output-contract)
- [image-generation-module-boundary](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-module-boundary)
- [image-generation-public-implementation-boundary](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)

若本 RFC 与公共 RFC 冲突，以公共 RFC 为准；本 RFC 不复制公共 contract，也不把 Qwen 专属字段扩展为公共 API。执行和验证基线固定为 `diffusers==0.38.0`。Diffusers 升级后必须重新审核 pipeline、Transformer forward、配置 fingerprint、缓存 block signature 和 `_cp_plan`。

## 2. 目标与非目标

### 2.1 目标

1. 对三个 canonical ID 做完整字符串匹配，并把它们静态分派到三个精确 kind；拒绝前缀、未来变体、镜像和未登记来源。
2. 对远端和本地目录执行相同的 config-only pipeline、component、Transformer、VAE 和 canonical kind/fingerprint 校验；不下载权重、不执行 remote code、不回退到源码或其他来源。
3. 使用现有 Diffusers Transformer 构建和 Runtime seam，在 meta device 上构造 `QwenImageTransformer2DModel` 的固定形状 forward。
4. 复现 original 与 Plus pipeline 的 source cardinality、有效尺寸、latent geometry、2x2 packing、`generated | sources` 顺序和嵌套 Python `img_shapes`。
5. 让 `--sample-step=N` 表示恰好 N 个重复的固定形状、固定 timestep Transformer workload；不创建 scheduler，不更新 latent。
6. 明确定义普通 `--use-cfg` 和 Qwen CFG-parallel 的 batch、world、forward 和通信语义。
7. 复现 2511 的 `zero_cond_t`、`[t, 0]` 内部扩展和 `modulate_index`，并让缓存覆盖成为适配 PR 的完成门槛。
8. 对现有 TensorCast 融合候选完成 source-faithful 审计和验证，不因名称相似而宣称支持不存在的基础设施。
9. 以 config fixtures 完成三个 canonical ID 的 hermetic CLI 生命周期测试，覆盖 Runtime 和可选成功 Chrome trace，不访问 Hub 或权重。

### 2.2 非目标

以下内容不属于本 RFC 或 Qwen 适配 PR：

- 真实 prompt、tokenizer、processor、文本/视觉 encoder、图片像素读取、resize 执行或 VAE encode/decode；
- scheduler、timetable、sigma、latent 数值更新、图片保存、输出图片、质量和端到端延迟；
- ControlNet、LoRA、inpainting、未来 checkpoint、社区 mirror 或任意未登记 pipeline；
- `U>1` 的 Ulysses/context parallel；
- 公共 generic engine、plugin、registry、handler、结果 schema、生命周期对象或 stage object；
- 对视频入口、公共 CLI/Runtime/output contract 的改写；
- 新增通用融合算子、`fused_rope.default` 基础设施或为单一模型建立兼容层。

## 3. 精确模型分派与 config-only 校验

### 3.1 远端 ID 和 kind

当 `remote_source=huggingface` 时，允许的 `(remote_source, model_id)` 只有：

| model_id | kind | pipeline | source 数量 | `zero_cond_t` |
| :--- | :--- | :--- | :---: | :--- |
| `Qwen/Qwen-Image-Edit` | `qwen-image-edit` | `QwenImageEditPipeline` | 恰好 1 | 缺省或显式 `false` |
| `Qwen/Qwen-Image-Edit-2509` | `qwen-image-edit-2509` | `QwenImageEditPlusPipeline` | 1–3 | 缺省或显式 `false` |
| `Qwen/Qwen-Image-Edit-2511` | `qwen-image-edit-2511` | `QwenImageEditPlusPipeline` | 1–3 | 必须为 `true` |

不得接受 `Qwen/Qwen-Image-Edit-*` 前缀、未来后缀、第三方镜像、其他 `remote_source` 或同名 Transformer 自动推断。远端访问只读官方 config-only snapshot；不能读取权重、执行 remote code、使用 source fallback 或从网络下载图片。失败诊断必须包含配置路径、期望值和实际值；对于需要授权的本地替代，应提示用户提供已授权的本地 Diffusers 目录，而不是暗中切换来源。

本地输入必须是根目录含 `model_index.json` 的已授权 Diffusers 目录。local pipeline class、Transformer class/config、VAE config 和组件类必须共同匹配上述一种 kind；只匹配类名不构成支持。`model_index.json` 的 pipeline class 只选择候选，严格 fingerprint 才决定通过或失败；额外 variant override 不参与候选选择且必须按下文拒绝。

### 3.2 必须验证的组件与 fingerprint

三种 kind 都必须验证以下组件类：

- `AutoencoderKLQwenImage`；
- `Qwen2_5_VLForConditionalGeneration`；
- `Qwen2Tokenizer`；
- `Qwen2VLProcessor`；
- `QwenImageTransformer2DModel`。

Transformer fingerprint 至少包含以下已审计字段，并且错误信息同时报告配置路径、expected 和 actual：

```text
patch_size=2
in_channels=64
out_channels=16
num_layers=60
attention_head_dim=128
num_attention_heads=24
joint_attention_dim=3584
axes_dims_rope=(16, 56, 56)
guidance_embeds=false
```

VAE 必须提供并验证 `z_dim`、`temperal_downsample`、`latents_mean` 和 `latents_std` 的存在及可用类型。scheduler 不属于 simulator contract，不解析、不校验、不实例化；不能以视频 VAE stride 或其他模型字段作为缺失配置 fallback。这里的 variant 指 `model_id` 对应的三个显式 kind，而不是额外可缺省的配置字段：canonical config 不得通过未声明的 variant override 改写 pipeline/component/Transformer/VAE fingerprint。若本地 manifest/config 声明额外 variant，必须报告 path/expected=`absent`/actual 并失败。

`zero_cond_t` 必须来自 Transformer config：original 和 2509 规范化为 `false`，2511 必须为 `true`。original/2509 显式为 `true`，或 2511 缺失、显式为 `false`，均在 build/forward 前失败。kind、pipeline、component、额外 variant override 和 `zero_cond_t` 的诊断必须彼此可区分。

## 4. CLI 生命周期和请求边界

公共 CLI 生命周期仍由 `cli/inference/image_generate.py` 负责：解析参数，检查组合，调用静态 lazy dispatch，建立 Qwen workload，执行 Runtime，导出 Runtime 时间和可选 Chrome trace。`tensor_cast/diffusers/qwen_image_edit.py` 不拥有 CLI 生命周期，不返回通用生命周期或结果对象。

Qwen 只使用公共参数：

- `model_id` 和 `--remote-source`；
- `--batch-size`；
- 恰好一次的 `--output-image-size HEIGHT WIDTH`；
- `--source-image-size HEIGHT WIDTH`，按 source 重复传入；
- `--text-seq-len`；
- `--sample-step N`；
- `--use-cfg`、`--cfg-parallel`；
- 公共 device、dtype、量化、compile、`--num-devices`（别名 `--world-size`）、`--ulysses-size` 和 `--chrome-trace-file`（别名 `--chrome-trace`）。

不增加 `--negative-text-seq-len`；不提供独立正/负 shape 参数，不提供 embedded guidance 数值参数、source timestep 参数或 Qwen 专属公共开关。`--output-image-size` 缺失、重复、格式非法或非正值必须失败，不能从 source size 推断。original 必须恰好一个 source size；2509/2511 必须一至三个。source size 是同一组 CLI 几何条件，不能使用嵌套 per-batch 语法。

`--text-seq-len` 是进入 Transformer 的有效文本序列长度，正负条件使用同一个长度。它不是字符数、tokenizer 输入长度或模板长度；本首版不执行文本编码。`--sample-step` 必须为正整数。所有校验在 Runtime/model execution 之前完成。

## 5. Qwen 输入几何与 packing

### 5.1 输出和 source 有效尺寸

从 VAE config 读取：

```text
S     = 2 ** len(vae.temperal_downsample)
C_lat = vae.z_dim
```

默认审计配置为 `S=8`、`C_lat=16`，但实现必须使用 config 值，并验证 `4 * C_lat == transformer.in_channels` 以及 `patch_size² * out_channels` 与 packed output width 一致。

一次 `--output-image-size H_requested W_requested` 形成生成图 requested size；按 Diffusers 规则分别向下对齐到 `2S` 倍数，得到 `H_effective W_effective`。source requested size 同样只作为 shape 输入，不读取像素：

- original：每张 source 按约 `1024²` target area 计算 aspect-ratio-preserving effective size，宽高分别 round 到 32 的倍数；该尺寸同时用于 VL 条件 metadata 和 source VAE token shape。
- Plus（2509/2511）：每张 source 分别计算约 `384²` 的 VL condition effective size和约 `1024²` 的 VAE effective size，均 round 到 32 的倍数；VAE effective size 决定 source token 数。

requested/original/VL/VAE metadata 必须按输入顺序保存。source geometry 不随 batch 中样本变化，所有 B 个样本复用这一组 source geometry。

### 5.2 latent 和 token 公式

生成 latent 和 packed token 为：

```text
H_gen_lat = 2 * floor(H_effective / (2S))
W_gen_lat = 2 * floor(W_effective / (2S))

generated latent = [B, 1, C_lat, H_gen_lat, W_gen_lat]
N_gen = (H_gen_lat / 2) * (W_gen_lat / 2)
generated packed = [B, N_gen, 4 * C_lat]
```

每个 source `i` 先按对应 pipeline 的 VAE target area 得到 effective size，再推导 latent：

```text
ratio_i = W_requested_i / H_requested_i
W_src_i_effective = round(sqrt(A_vae * ratio_i) / 32) * 32
H_src_i_effective = round((sqrt(A_vae * ratio_i) / ratio_i) / 32) * 32
A_vae = 1024 * 1024

H_src_i_lat = 2 * floor(H_src_i_effective / (2S))
W_src_i_lat = 2 * floor(W_src_i_effective / (2S))
source_i latent = [B, 1, C_lat, H_src_i_lat, W_src_i_lat]
N_src_i = (H_src_i_lat / 2) * (W_src_i_lat / 2)
source_i packed = [B, N_src_i, 4 * C_lat]
```

original 的 VL 条件和 VAE 使用同一组 `1024²` effective size；Plus 另以 `A_condition=384*384` 和同一公式计算 VL condition size，但 source VAE/token 仍使用上面的 `1024²` size。所有 round 后尺寸必须为正且能经 `2S` floor 产生非空、偶数 latent 高宽；否则在 input preparation 前失败。

Transformer 的 image sequence 固定按 generated first、sources in input order 拼接：

```text
hidden_states = generated | source-1 | source-2 | source-3
              [B, N_gen + sum(N_src_i), 4 * C_lat]
```

`img_shapes` 是 nested Python metadata，不是可用 tensor `shape[0]` 推断的对象：

```python
sample_img_shapes = [
    (1, H_gen_lat // 2, W_gen_lat // 2),
    (1, H_src1_lat // 2, W_src1_lat // 2),
    # ... one entry per source
]
img_shapes = [list(sample_img_shapes) for _ in range(B)]
```

实现可以避免别名副作用，但结构上必须是每个 batch sample 一个相同的 descriptor，且 descriptor 首项为 generated、之后为 source。普通 CFG 复制时也必须复制该嵌套结构，不能把它当作普通 tensor 广播。

`--batch-size=B` 表示同一组 source/text 条件下的 `num_images_per_prompt=B`。Plus pipeline 的 source 数量不是 prompt batch；source count 不改变 batch 语义。

## 6. Transformer forward 契约和 workload

`qwen_image_edit.py` 负责配置校验、source cardinality、几何推导、packing、文本 mask、固定 timestep、CFG batch duplication、generated-prefix slicing、`zero_cond_t` metadata 和缓存 spec。`forward_image_model` 必须对齐 Diffusers 0.38.0 `QwenImageTransformer2DModel.forward` 的完整参数契约：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
): ...
```

`encoder_hidden_states_mask` 是实际构造的主文本 mask seam；实现传入 `[effective_batch, text_seq_len]` 的布尔全有效 mask（而不是模糊的 metadata mask），并在测试中验证其 shape、dtype 和 batch duplication。`txt_seq_lens=None` 保留在 0.38.0 完整 signature 中，但它已 deprecated，不能作为 primary interface。`guidance`、`attention_kwargs`、`controlnet_block_samples`、`additional_t_cond` 和 `return_dict` 保持上述默认值，除非锁定源码审计证明 workload 需要其它值。不得假设只要 `shape[0] == B` 就能复制所有 Qwen 输入。

每个 `--sample-step=N` 都是相同 shape、相同 dummy timestep、相同 source condition 的重复 Transformer workload：

1. 以 generated packed tensor 加 source packed tensors 构成完整 image sequence；
2. 构造 text hidden states、`encoder_hidden_states_mask`、固定 timestep 和嵌套 `img_shapes`；
3. 调用一次 Qwen Transformer；
4. `forward_image_model` 立即返回 `output[:, :generated_token_count]` 作为 generated prefix；完整 output 只作为该切片操作的瞬时中间值，不由模块保留；
5. 不执行 scheduler，不把输出数值写回下一步，不更新 generated latent；source latent、source metadata 和 source condition 在所有步骤保持不变。

generated-prefix slicing 是 `qwen_image_edit.py` 的内部形状语义。CFG-parallel 在 slicing 后对 generated prefix 执行 CFG-group communication；普通 CFG 保留 effective `2B` 的 model output，不做 numerical combine。source token output 不进入 CFG communication，也不成为下一步输入的数值更新。N 次 workload 不得隐藏 warmup 或把 compile/setup 从 N 中扣除。

## 7. Qwen CFG、2511 timestep 和拓扑

### 7.1 普通 `--use-cfg`

Qwen 普通 CFG 不做两次 sequential true-CFG forward，也不做数值 CFG combine/norm rescale。启用 `--use-cfg` 时，Qwen-specific batch-concat 一次性把有效 batch 扩为 `2B`，仍使用同一个 `--text-seq-len`，每个 step 只执行一次 Transformer forward：

```text
hidden_states                 [2B, N_image, 4*C_lat]
encoder_hidden_states         [2B, L, joint_attention_dim]
encoder_hidden_states_mask    [2B, L]
timestep                      [2B]
img_shapes                    nested structure with 2B entries
source latent/metadata        duplicated per batch entry
```

所有 batch-first tensor、generated/source hidden states、encoder states/mask、timestep、source latent 或其他 per-batch metadata 都必须显式复制；nested Python `img_shapes` 也必须按 effective batch 复制。不得复制后仍依赖 generic tensor `shape[0] == B`。普通 CFG 的有效 Transformer batch 是 `2B`，每个 step 的 forward 次数为 1。

### 7.2 CFG-parallel

CFG-parallel 只在 `--use-cfg` 同时启用时有效，且只支持 `U=1、world_size=2`。Qwen CFG-parallel 是单 rank representative simulator：每步执行一次本地代表性 `B` forward，代表两个相同并发分支；随后在 CFG group 上对 generated-prefix 按 dim 0 做 `all_gather`，形成逻辑 `2B` 的 CFG 结果。它不是两条 sequential branch，也不执行数值 combine。每步一个代表性 B forward 加一个 measured CFG-group collective；通信进入 Runtime critical path。

### 7.3 Ulysses 限制

Qwen 首版强制 `ulysses_size=U=1`。`U>1` 必须在 Runtime/model execution 前失败，并给出 Qwen-specific message，说明 Diffusers 0.38 `_cp_plan` 所需的 sequence sharding、text/image boundary 和 projection gather 尚未实现。不能把 Ulysses 当作静默 fallback，也不能因 CFG-parallel 成功而绕过此限制。

无 CFG：`B` effective batch、`world_size=1`、每步一次 forward。

普通 CFG：`2B` effective batch、单 rank、每步一次 forward。

CFG-parallel：单 rank 执行代表性的 `B` forward（代表 `world_size=2` 的两个相同并发分支），每步一次 forward 加 CFG-group all-gather。

所有非法 world/CFG/U 组合在 Runtime 前失败。

### 7.4 2511 `zero_cond_t`

2511 的 `zero_cond_t=true` 是模型内部条件，不是外部 CFG batch。外部传给 Transformer 的 timestep shape 始终等于当前 effective batch：无 CFG 与 CFG-parallel representative forward 为 `[B]`，普通 CFG 为 `[2B]`。模型内部沿 batch 维执行 `torch.cat([timestep, timestep * 0])`，所以内部 timestep embedding shape 分别为 `[2B]` 或 `[4B]`；随后每个 block 的 text stream 使用前半 effective-batch embedding，image stream 依据 `modulate_index` 在 generated/source token 上选择生成或 zero-condition embedding。`modulate_index` 的第一维必须等于 effective batch，每个 sample 的 generated positions 为 0、source positions 为 1。

普通 CFG 的外部 `2B` duplication 不能替代 2511 内部 `[t,0]` 扩展；两者叠加时必须保持上述 `[2B] -> [4B]` 映射。original/2509 不得进入 `zero_cond_t` 分支。该映射必须在 forward trace、compile off/on 和缓存测试中可观察。

## 8. 缓存契约与完成门槛

Qwen 适配必须提供显式、严格绑定 `QwenImageTransformer2DModel` 的 `DiTBlockCacheSpec`。缓存不是非目标，而是 Qwen PR completion gate。Diffusers 0.38.0 的 Qwen Transformer 只有一个 `transformer_blocks` 集合；每个 `QwenImageTransformerBlock` 同时更新 text/image 双流并返回 `(encoder_hidden_states, hidden_states)`。因此 Qwen cache wrapper 必须覆盖该真实双流 block 契约，不得虚构 FLUX 风格的 `single_transformer_blocks` 集合。

缓存 wrapper 必须覆盖：

- `transformer_blocks` 的真实 Diffusers 0.38.0 signature、调用顺序、双输出语义和 block range；
- generated/source sequence 及对应 mask；
- source cardinality 和 source geometry；
- 2511 `zero_cond_t`、`[t,0]` 和 `modulate_index`；
- cache update/reuse 生命周期；
- 半开 block range `[start, end)`；
- 普通 CFG 的 `2B` effective batch；
- strict Transformer class/block signature failure；不存在或为空的 block collection、非法空 range、zero replacement、每次运行的 state reset 也必须显式失败或可观测。

cache hit、miss、update、reuse 的 shape、dtype、layout、batch、source count、mask、timestep semantics 必须在 wrapper 边界验证。cache spec 由 Qwen module 直接返回，不使用 import-time global registry。不能按 block 名称、参数数量或可选 kwargs 猜测兼容。缓存失败必须在执行前清楚报告 expected/actual class、block signature 或 spec；不能自动降级到未审计的通用缓存。

模型生命周期顺序固定为：从同一份已验证 config 独立构造 baseline model 与 cache model；对两者应用相同的 source-faithful Qwen patch/wrapper；只在 cache model 上按显式 spec 替换 blocks；然后分别执行 `torch.compile()`。patch/wrapper 必须早于 cache replacement，cache replacement 必须早于 compile。每次 `run_inference()` 创建独立 cache state，Runtime 在 cache window 外使用 baseline model、窗口内使用 cache model。

缓存的 update/reuse 不得改变 generated-prefix slicing、source immutability、2511 `[t,0]`、普通 CFG 2B 或 CFG-parallel representative B 语义。测试必须证明半开范围不会漏 block、重复 block 或覆盖错误 block。

## 9. TensorCast 融合审计

三个 kind 共享 `QwenImageTransformer2DModel` 路径，但共享类名不等于共享融合保证。适配 PR 必须基于锁定的 Diffusers 0.38.0 source 和三个 kind 的 meta FX graph，逐项审核现有 TensorCast fusion candidates，至少包括：

- attention Q/K normalization；
- Qwen image/text multi-axis RoPE；
- 其他已存在且有明确 TensorCast seam 的候选。

每个候选必须核对数学、epsilon、affine/bias、dtype cast、layout/broadcast、exact graph/overload、输入输出数量、输入输出 alias/mutation contract、RoPE axis/interleave/partial rotary dimension 以及 generated/source sequence layout。只有 source-faithful semantic match 才能标为 applicable；不匹配必须记录 `not_applicable` 及具体差异。wrapper/patch 必须在 compile 前完成；已有 pattern replacement 才能由 compile backend 执行。

不得声称 `fused_rope.default` 已支持，除非仓库已有所需通用 pattern、performance property、exact mapping 和 profiling `op_mapping` 基础设施。`add_rms_norm` 与 `add_rms_norm2` 必须分别证明单输出和双输出的 alias/mutation contract。编译后必须用 graph 和 Runtime operator 事件确认 exact overload、layout、dtype、mutation 和性能映射；融合节点不能与已吸收 primitive 同时出现，不能新增未声明 graph break。语义不匹配、mapping 缺失和 shape-data 缺失必须使用不同诊断。

## 10. 测量与输出

结果只称为“Transformer 去噪阶段模拟时间”。Runtime 表至少包括：

| 项目 | 口径 |
| :--- | :--- |
| Transformer forward | 实际 measured Runtime 时间；无 CFG/普通 CFG 每步一次，CFG-parallel 为本地代表性 forward |
| Qwen/CFG communication | 实际发生的 Transformer 内部通信或 CFG-group all-gather |
| logical workload | `N` 个 workload；普通 CFG 仍每步一个 `2B` forward；CFG-parallel 记录代表性 forward 加 collective |
| critical path | 按公共 topology contract 计算的 Runtime critical path |
| operator/call breakdown | Runtime 实际记录的 operator 和调用事件 |
| cache | update/reuse 命中路径和 block range 事件 |

可选 `--chrome-trace-file`（别名 `--chrome-trace`）成功时必须导出可解析 Chrome trace；trace 是 Runtime 的实际输出，不是结果 schema 的替代。失败或未请求 trace 时不得伪造路径或成功状态。

输出只报告公共 output contract 所允许的 Runtime 时间、operator/call、通信和必要 shape audit 信息。Qwen 内部必须可审计 generated/source token counts、`img_shapes`、text mask、effective batch、forward count、固定 timestep shape/dtype、source immutability、`zero_cond_t` 和 cache state；不得建立独立结构化 JSON/result schema，不得把 shape-only 作为 measured 时间，不得报告 prompt/VAE/image I/O、质量、权重显存或端到端编辑结论。

## 11. 测试设计

所有测试使用 config fixtures、config-only resolver、meta device 或 hermetic Runtime，不访问 Hub、不下载权重、不读取凭据、不读取真实图片。

### 11.1 精确 identity 和 fingerprint

- 三个完整 remote ID 精确映射到三个 kind；前缀、未来 ID、错误 remote source 和 mirror 拒绝。
- model index pipeline class、所有组件 class、Transformer fingerprint、VAE 字段和 `zero_cond_t` 逐字段验证；kind 只由三个 canonical ID/config 选择，额外 variant override 必须 expected=`absent`/actual 诊断并失败。
- 错误诊断包含 config path、expected、actual；不构造 scheduler，不执行 remote code。
- local fixture 与 canonical remote fixture 使用相同校验路径；仅同名 Transformer 的目录失败。

### 11.2 source、尺寸和 packing

- original 的 0、2、3 source 失败，恰好 1 成功；2509/2511 的 0、4 失败，1、2、3 成功。
- output size 必须恰好一次；requested/effective 和 VAE-derived latent 公式正确；source size 不改变 output geometry。
- original 的 VL/VAE effective size 与 Plus 的双 effective size 规则、aspect ratio、32 对齐和 input order 正确。
- 1/2/3 source 的 `N_src_i`、总 token、generated first 顺序和 nested `img_shapes` 正确。
- `B=1` 与 `B>1` 复用同一 source 条件组；不使用 nested per-batch CLI syntax；source metadata 跨 N steps 不变。

### 11.3 forward、mask、N-step 和 CFG

- forward kwargs 含 `hidden_states`、`encoder_hidden_states`、`encoder_hidden_states_mask`、`timestep`、`img_shapes`；`txt_seq_lens` 不作为主 seam。
- text mask shape 和有效位置正确；正负不分离 text length，`--text-seq-len` 是唯一有效长度。
- `sample-step=N` 恰好产生 N 次固定形状 forward workload；无 scheduler、timetable、latent update。
- 无 CFG 为 B；普通 CFG 为 2B 且每步一次 forward，所有 tensor 和 nested `img_shapes` 都复制；不得依赖 `shape[0] == B`。
- CFG-parallel 为单 rank 代表性 B forward 加 generated-prefix 的 dim-0 all-gather，验证 communication 和 critical path；不得退化为 sequential 两 forward 或 numerical combine。
- `forward_image_model` 每步返回 `output[:, :generated_token_count]`；ordinary CFG 保留 `2B` model output，不做 numerical combine；source token output 不进入 CFG communication 或下一步输入。

### 11.4 2511、拓扑和缓存

- original/2509 不进入 zero condition；2511 保留 `[t,0]`、`modulate_index` 和 source/generated boundary，并覆盖普通 CFG 外部 `[2B]` timestep 到内部 `[4B]` embedding 的精确映射。
- `U>1` 在 Runtime 前以 Qwen-specific `_cp_plan` unsupported 错误失败；world size、CFG group 和 `U=1` 组合正确。
- cache wrapper 覆盖真实 `transformer_blocks` 双流 signature/双输出、source count、source/mask、2511、update/reuse、半开 block ranges、2B CFG、class mismatch、缺失/空 block collection、empty range、zero replacement 和 per-run state reset 的 strict failures；并验证不存在虚构的 `single_transformer_blocks` 依赖。
- compile off/on 保持输入输出 shape、forward count、packing、`img_shapes`、mask、prefix slicing、source immutability、CFG 和 zero condition 语义不变。

### 11.5 融合和 hermetic CLI 生命周期

- 三个 kind 均完成候选 fusion audit、meta FX graph、compile graph、Runtime exact operator/mapping 证据；`not_applicable`、mapping gap 和 semantic mismatch 分开诊断。
- 通过公共 `image-generate` CLI 或同一公开 orchestration 入口，三个 canonical ID 都完成：exact dispatch、config-only validation、meta build、N-step Runtime、Runtime output 和可选 Chrome trace。
- 最小 hermetic case 使用 `B=1`、一个 1024x1024 source、1024x1024 output、合法 text length、`sample-step=2`、`U=1`、无 CFG；每个 case 必须证明 logical forward 恰为 2、measured operator/event 非空、critical path 大于 0、scheduler 未执行且 trace（若请求）来自本次 Runtime。
- CLI 输出必须保持视频入口采用的可审计文本打印形态，例如：

  ```python
  print(f"Model compilation execution time: {run_end - run_start}s")
  print(runtime.table_averages(group_by_input_shapes=False))
  if chrome_trace:
      runtime.export_chrome_trace(chrome_trace)
      print(f"Chrome trace written to: {chrome_trace}")
  ```

  其中 `runtime.export_chrome_trace` 只在 Runtime 成功且请求 trace 时调用；失败或未请求时不得打印伪造路径或 JSON/result schema。
- 测试不依赖旧 fake model layer、旧 result schema 或修改视频路径；公共 Core 回归和 `video_generate` 行为保持不变。

## 12. 适配 PR 边界

Qwen 适配 PR 只包含：

1. `tensor_cast/diffusers/qwen_image_edit.py`；
2. `tensor_cast/diffusers/image_dispatch.py` 中三个静态 lazy branches；
3. 对应 config fixtures 和测试；
4. 缓存 wrapper/spec、fusion audit 及 hermetic CLI/Chrome trace 验收；真实 `transformer_blocks` 双流 wrapper/spec、2511、ordinary CFG 2B、source/mask 和 strict failure tests 是 PR merge completion gate，不得以“后续补齐”替代。

它依赖已合入的 Core/public image-generation implementation boundary，与 FLUX 适配独立。不得修改公共 CLI、视频入口、Runtime contract、output contract、Web UI 或公共组装规则；若公共 contract 不足，先提交独立 Core 变更并更新公共 RFC。不得把 Qwen 需求隐藏到通用 engine、plugin、registry、handler 或 compatibility layer 中。

## 13. 取舍与风险

Qwen original、2509 和 2511 共享 Transformer，但 pipeline class、source cardinality、VL/VAE geometry 和 2511 zero condition 不同，因此采用三个 exact kind、一个 Qwen module 和静态 dispatch branches，而不是 class-only 自动接受或三个重复实现。

普通 Qwen CFG 采用 `2B` batch-concat 的单次 forward，保持同一 text length 和完整 batch-first/nested metadata；CFG-parallel 采用 U=1 的代表性 B forward 加 CFG-group all-gather。这样避免两次 sequential forward、数值 CFG combine 和不受控的 generic video semantics，同时使调用次数和通信可审计。

`U>1` 明确失败而不是近似运行，因为 `_cp_plan` 的 text/image sequence sharding、modulation split 和 output gather 尚未验证。Diffusers、Qwen config 或 block signature 漂移时，strict fingerprint 和 strict cache spec 必须拒绝执行，不能以同名类或默认值继续。

---

## 14. 参考资料

1. [Qwen-Image-Edit 官方介绍](https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit.md)
2. [Qwen-Image-Edit-2509 官方说明](https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit-2509.md)
3. [Qwen/Qwen-Image-Edit 模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit)
4. [Qwen/Qwen-Image-Edit-2509 模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
5. [Qwen/Qwen-Image-Edit-2511 模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
6. [Diffusers 0.38.0 original edit pipeline](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py)
7. [Diffusers 0.38.0 Plus pipeline](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)
8. [Diffusers 0.38.0 Qwen Transformer](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/transformers/transformer_qwenimage.py)
