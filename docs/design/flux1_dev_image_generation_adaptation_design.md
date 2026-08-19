# 特性设计：TensorCast 适配 FLUX.1-dev 图像生成仿真

## 修订记录

| 日期 | 版本 | 修改描述 | 作者 | 关联文档 |
| --- | --- | --- | --- | --- |
| 2026-08-12 | 0.1 | 锁定 FLUX.1-dev 模型专属实现与验收范围 | `minghang_c` | [FLUX.1-dev RFC](../RFC/rfc_add_flux1_dev_support_zh.md) |
| 2026-08-13 | 0.2 | 按更新后的 procedural image-generation Core 重写模块、CFG、cache 与生命周期设计 | `minghang_c` | [FLUX.1-dev RFC](../RFC/rfc_add_flux1_dev_support_zh.md) |

## 1. 背景与目标

本设计在图像生成公共 Core 基线
`bcfd4fa27a2e293ac62008ffae404e49f276f768` 上适配精确模型
`black-forest-labs/FLUX.1-dev`。仿真只覆盖 FLUX Transformer 的固定形状、固定
`timestep` 工作负载及必要通信，不执行真实图片生成。

实现边界：

- 不下载或加载权重；
- 不执行 prompt 编码、tokenizer、CLIP/T5 encoder、VAE、scheduler、latent 数值更新或图片 I/O；
- 不把 Transformer 仿真耗时称为端到端图片生成耗时；
- 不引入 `tensor_cast/image_generation`；
- 不引入 `ImageGenerationRequest`、`ImageGenerationResult`、`ImageModelProfile`、
  `ImageProfileCollection`、`ImageGenerationAdapter`、`CFGBranch`；
- 不引入通用 engine、registry、plugin、handler 或模型 ID 动态注册；
- 不修改公共 CLI 生命周期、Runtime 输出契约、video generation 或其它模型行为。

## 2. 模块边界

### 2.1 生产文件

```text
cli/inference/image_generate.py
  公共 CLI 生命周期、参数校验、模型构建、cache/compile、Runtime、输出与 trace

tensor_cast/diffusers/image_dispatch.py
  frozen procedural API；通过静态 lazy `flux1-dev` 分支调用模型 helper

tensor_cast/diffusers/flux_image.py
  FLUX 身份、配置验证、输入几何、CFG、Ulysses、模型 patch、forward 与 cache spec

tests/assets/model_config/FLUX.1-dev/
  config-only fixture、provenance 与 SHA256SUMS
```

FLUX 模块导入时不得解析 fixture、构造模型、访问网络或修改全局 registry。
`image_dispatch.py` 只在命中远端候选或本地
`FluxTransformer2DModel` 配置时 lazy import `flux_image`。未支持 kind 继续 fail closed。

### 2.2 Frozen procedural seams

公共生命周期只通过以下平面函数调用模型逻辑：

```text
resolve_image_model_kind
validate_image_config
prepare_image_inputs
apply_image_cfg
shard_image_inputs
prepare_image_model
forward_image_model
image_cache_spec
```

不返回结构化 image-generation result。成功边界仍是公共 CLI 打印的
`Model compilation execution time`、Runtime operator table，以及可选 Chrome trace 完成信息。

## 3. 身份与 config-only 验证

### 3.1 远端身份

远端只接受：

```text
remote_source = huggingface
model_id      = black-forest-labs/FLUX.1-dev
```

`FLUX.1-schnell`、Fill、Redux、社区 mirror、近似 ID 和其它 source 均失败，错误包含
expected/actual。

`FLUX.1-dev` 是 Hugging Face gated model。真实远端 config-only 解析要求用户先在官方页面接受条款，并通过标准 Hugging Face 凭据存储提供有读取权限的 token。源码、命令、fixture、日志和测试不得记录 token。hermetic 测试只使用审核后的本地 config-only fixture，不访问 Hub 或个人凭据。

### 3.2 本地根目录

本地候选先由已加载 Transformer 配置的
`_class_name == "FluxTransformer2DModel"` 进入 FLUX 静态分支，再由完整 fingerprint 决定是否接受。单独类名不能通过验证。

必须验证 `model_index.json`：

```text
_class_name    FluxPipeline
vae            [diffusers, AutoencoderKL]
text_encoder   [transformers, CLIPTextModel]
tokenizer      [transformers, CLIPTokenizer]
text_encoder_2 [transformers, T5EncoderModel]
tokenizer_2    [transformers, T5TokenizerFast]
transformer    [diffusers, FluxTransformer2DModel]
```

Transformer 字段锁定为：

```text
_class_name           FluxTransformer2DModel
patch_size            1
in_channels           64
num_layers            19
num_single_layers     38
attention_head_dim    128
num_attention_heads   24
joint_attention_dim   4096
pooled_projection_dim 768
guidance_embeds       true
axes_dims_rope        缺失时按 Diffusers 0.38.0 默认值 (16, 56, 56)
out_channels          缺失或 null 时规范化为 64
```

VAE 字段锁定为：

```text
_class_name         AutoencoderKL
latent_channels     16
block_out_channels  [128, 256, 512, 512]
down_block_types    4 * DownEncoderBlock2D
up_block_types      4 * UpDecoderBlock2D
layers_per_block    2
scaling_factor      0.3611
shift_factor        0.1159
latents_mean        null
latents_std         null
use_quant_conv      false
use_post_quant_conv false
```

文本配置分别要求：

```text
text_encoder/config.json   architectures == [CLIPTextModel]
text_encoder_2/config.json architectures == [T5EncoderModel]
```

当前运行时严格要求 `diffusers==0.38.0`。缺失文件、非法 JSON、错误 pipeline/component/字段或版本漂移均在 meta Transformer 构建前失败。只有完整验证成功后才设置
`model_config.image_dispatch_validated = True`；builder 以此作为构建门禁。

scheduler 不参与验证、构建或 workload。fixture 可保留 root manifest 中的 scheduler provenance，但不包含 scheduler config。

## 4. 输入几何与 FLUX ID

从已验证 VAE/Transformer 配置计算：

```text
vae_scale_factor = 2 ** (len(block_out_channels) - 1)
H_lat = 2 * floor(H / (2 * vae_scale_factor))
W_lat = 2 * floor(W / (2 * vae_scale_factor))
C_lat = transformer.in_channels / 4
N_img = (H_lat / 2) * (W_lat / 2)
```

1024×1024 请求得到：

```text
unpacked latent [B, 16, 128, 128]
packed hidden   [B, 4096, 64]
img_ids         [4096, 3]
```

非对齐尺寸按上述 floor 公式处理，不 padding、不隐式 trim。小到无法形成正 latent 的尺寸失败。source-image 输入不支持。

2×2 packing 必须使用 Diffusers source-faithful 顺序：

```python
latents.view(B, C, H_lat // 2, 2, W_lat // 2, 2) \
    .permute(0, 2, 4, 1, 3, 5) \
    .reshape(B, (H_lat // 2) * (W_lat // 2), C * 4)
```

meta 输入：

```text
hidden_states         [B, N_img, 64]   Transformer dtype
encoder_hidden_states [B, L, 4096]     Transformer dtype
pooled_projections    [B, 768]         Transformer dtype
img_ids               [N_img, 3]       FP32
txt_ids               [L, 3]           FP32
timestep              [B]              FP32
guidance              [B]              FP32
```

`img_ids[:, 0]` 为零，后两列按 row-major 表示 image-token 行列；`txt_ids` 全零。Diffusers 模型内部按 text-first 语义组合 text/image IDs。`timestep` 与 `guidance` 直接传入；Diffusers 0.38.0 自行 cast 并乘 1000，模型 helper 不预乘，也不引入 scheduler 数值语义。

## 5. CFG 与 Ulysses

### 5.1 普通 CFG

`--use-cfg --no-cfg-parallel` 使用 batch-concat：

- effective batch 为 `2B`；
- 每 step 只有一次 Transformer forward；
- 沿 dim 0 复制 `hidden_states`、`encoder_hidden_states`、`pooled_projections`、
  `timestep`、`guidance`；
- 不复制共享的 `img_ids [N_img,3]` 和 `txt_ids [L,3]`；
- 不接受独立负向 text length，不做 numerical CFG combine 或 CFG scale。

### 5.2 CFG parallel

CFG parallel 只在 `use_cfg=true` 时有效：

```text
world_size = 2U
representative local batch = B
每 step 每个 representative rank 一次 forward
cfg group(u) = [u, U + u]
```

该 workload 表示两个形状相同、并发 CFG branch 的 local critical path。Transformer local output 先完成 Ulysses dim-1 gather，再执行 CFG group dim-0 all-gather。公共生命周期不返回 combine 后 tensor，也不报告图片结果。

### 5.3 Ulysses

首次 forward 前检查：

```text
N_img % U == 0
text_seq_len % U == 0
num_attention_heads % U == 0
```

`U>1` 固定分片：

```text
hidden_states         dim 1
encoder_hidden_states dim 1
img_ids               dim 0
txt_ids               dim 0
```

`forward_model` 返回的本地输出 sequence 长度与分片后的 `hidden_states` 一致，为 `N_img/U`；公共 CLI 再沿 dim 1 gather，恢复全局 `N_img`。`U=1` 不分片、不设置 output gather dim。Runtime 期间按 active model 设置 Diffusers sequence-parallel group，并在成功或异常退出时恢复为 `None`。

## 6. 模型构建、patch 与 compile

顺序固定为：

1. 解析 config-only 模型选择与 pipeline manifest；
2. 加载 Diffusers config；
3. 静态解析 `flux1-dev` 并严格验证；
4. 准备输入、CFG 和 Ulysses shape；
5. 从已验证配置构造 baseline meta `FluxTransformer2DModel`；
6. 在 compile 前执行 FLUX model patch；
7. 若启用有效 cache window，从同一配置独立构造 cache model、执行相同 patch，再替换 cache blocks；
8. 分别 compile baseline/cache model；
9. Runtime 中按 step 选择 baseline 或 cache model。

baseline 与 cache model 必须是独立构造实例。不得从已修改或已编译 baseline 复制 cache model。

### 6.1 Q/K RMSNorm patch

唯一首版融合适配是 FLUX attention 的 Q/K RMSNorm：

```text
19 dual blocks * 4 norms + 38 single blocks * 2 norms = 152
```

目标 module：

```text
norm_q
norm_k
norm_added_q
norm_added_k
```

source module 必须是 `torch.nn.RMSNorm` 或已经 patch 的
`RMSNormFusedWrapper`，权重 shape 为 `(128,)`，`eps == 1e-6`。只有全部 152 个目标预检成功后才修改模型，避免部分 patch。重复调用幂等。

compile 验收应证明 patch 在 `torch.compile` 前发生、export/compiled graph 使用
`tensor_cast.rms_norm.default`、无非预期 graph break，并保持 eager/compiled 输出 shape。未实际执行的 graph/event 断言不得写成已完成结论。

## 7. FLUX DiT cache

`flux_image.cache_spec()` 直接返回 class-bound `DiTBlockCacheSpec`：

```text
class_name = FluxTransformer2DModel
model_type = flux1-dev
```

不注册到 video cache registry，不通过 fallback 接受错误 class。

block discovery 顺序必须与 FLUX forward 顺序一致：

1. 19 个 dual-stream `transformer_blocks`；
2. 38 个 single-stream `single_transformer_blocks`。

缺 collection、数量不是 19/38、空 inventory 或 class mismatch 均失败。双流 block 的 Diffusers output 是
`(encoder_hidden_states, hidden_states)`；generic cache agent 以 hidden-first 处理，因此 wrapper 在进入和退出 cache agent 时各做一次顺序适配，外部签名与 Diffusers 保持一致。

cache 生命周期：

- 仅 `dit_cache=true` 且 `cache_step_interval>1` 时构建第二模型；
- step range 先 clamp 到 `[0, sample_step-1]`，clamp 后为空则在第二模型构建前失败；
- block range 使用半开区间 `[start,end)`，在实际 57 blocks 上 clamp；
- 未替换任何 block 必须失败；
- `enable_dit_block_cache` 每次运行创建新的 `CacheState`；
- cache window 内按 interval 更新 `cache_state.reuse`，窗口外使用 baseline；
- Chrome trace 只在整个 Runtime 成功完成后导出。

## 8. Forward 与公共输出

FLUX helper 调用 Diffusers 0.38.0 forward 的模型输入只有：

```text
hidden_states
encoder_hidden_states
pooled_projections
timestep
img_ids
txt_ids
guidance
return_dict=False
```

不传 scheduler、ControlNet、IP-Adapter 或其它排除项。输出必须是单一 rank-3 tensor，且 pre-gather shape 与本地 `hidden_states` 完全一致；全局 `generated_token_count` 必须是该本地 sequence 长度的正整数倍。公共 CLI 完成 Ulysses dim-1 gather 后，logical sequence 维才等于全局 `generated_token_count`。异常输出 fail closed。

`sample_step=N` 恰好执行 N 次固定 shape forward。无 scheduler read/call、schedule、sigma 或 latent update。

成功输出由 `cli/inference/image_generate.py` 持有：

```python
print(f"Model compilation execution time: {run_end - run_start}s")
print(runtime.table_averages(group_by_input_shapes=False))
if chrome_trace:
    runtime.export_chrome_trace(chrome_trace)
    print(f"Chrome trace written to: {chrome_trace}")
```

函数返回 `None`。配置解析、模型构建、Runtime forward 或 Runtime exit 失败时不得导出 Chrome trace，也不得打印成功完成信息。

## 9. Fixture 与 provenance

fixture 位于 `tests/assets/model_config/FLUX.1-dev/`，来源 revision：

```text
3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
```

只保留 config-only 构建与 exact validation 所需公开事实。禁止包含：

- 权重；
- tokenizer vocabulary/merges；
- prompt、token、图片或生成结果；
- 用户名、credential path、HF token；
- scheduler config；
- Runtime trace 或 trace-derived fingerprint。

`provenance.json` 记录 canonical ID、remote source、source revision、Diffusers baseline、captured paths、redaction/access boundary、scheduler exclusion 和各 component JSON hash。
`SHA256SUMS` 还包含 `provenance.json` 自身 hash。fixture 更新必须同步重新计算两处 hash，并运行 JSON/hash regression。

## 10. 测试设计

### 10.1 模型单元与集成测试

`test_image_generation_flux1_dev.py` 至少覆盖：

- exact remote identity 与近似 ID/source rejection；
- valid local fixture 与 pipeline/component/Transformer/VAE/text mismatch；
- missing/unparseable JSON 和 Diffusers version mismatch；
- validation-before-build 门禁；
- 1024²、非对齐和非法几何；
- source-faithful packing、row-major image IDs、zero text IDs；
- ordinary CFG effective `2B` 且 IDs 不复制；
- CFG parallel local `B`；
- 四类 Ulysses sharding 与三类整除失败；
- exact forward kwargs/output contract；
- 152 RMSNorm patch、原子性与幂等性；
- 19+38 cache discovery/order、signature、output-order adaptation、update/reuse；
- real `enable_dit_block_cache` 的 class、range、replacement 与 fresh `CacheState`。

### 10.2 Canonical-ID hermetic CLI

`test_image_generation_flux1_dev_e2e.py` 必须通过公共 `run_inference`，使用 canonical model ID 和本地 config-only fixture，同时保留真实：

- `resolve_image_model_kind`；
- `validate_image_config`；
- `build_diffusers_transformer_from_config` 的 meta `FluxTransformer2DModel` 构建；
- `prepare_image_model` 的 152 RMSNorm patch；
- 输入准备、CFG/Ulysses shape 和 forward seam。

只允许替换设备性能模型、Runtime 观察器、compile backend 或昂贵 model execution，使测试 hermetic 且不访问网络/权重。测试验证 `sample_step=N` 为 N 次 forward、真实 dispatch/config/meta build 未被绕过，并覆盖 cache/compile/trace 生命周期。

### 10.3 公共回归

验证批次包括：

```text
tests/regression/tensor_cast/test_image_generation_flux1_dev.py
tests/regression/tensor_cast/test_image_generation_flux1_dev_e2e.py
tests/regression/tensor_cast/test_image_dispatch.py
tests/regression/cli/test_image_generate.py
Diffusers resolver/builder regressions
DiT cache regressions
受影响 video generation regressions
```

同时运行可用静态检查、`git diff --check`，并确认：

- `uv.lock` 无 diff；
- `report.log` 保持未跟踪且未修改；
- 无 legacy `tensor_cast.image_generation` 或旧 request/result/profile/adapter 符号；
- 未提交、未推送、未创建 PR。

## 11. 完成定义

FLUX.1-dev 只有在以下条件全部成立后，才能称为已适配到更新后的 Core：

- 分支基于 `bcfd4fa27a2e293ac62008ffae404e49f276f768`；
- exact remote/local identity、strict config-only validation 与 meta build 通过；
- geometry、packing、IDs、timestep/guidance、ordinary CFG、CFG parallel 与 Ulysses 通过；
- 152 RMSNorm patch 与 compile/export 验收通过；
- dual/single-stream cache 真实集成、range 和 per-run state 通过；
- canonical-ID hermetic CLI 覆盖真实 dispatch、validation、meta build、patch 与 N-step lifecycle；
- Runtime 文本/table 输出及 success-only Chrome trace 契约通过；
- 全部受影响 regression 与仓库可用门禁通过；
- `uv.lock`、`report.log` 和其它模型行为未被混入。

该状态仍只表示 FLUX Transformer 性能仿真支持，不表示权重推理、真实图片生成或端到端图片生成性能已支持。
