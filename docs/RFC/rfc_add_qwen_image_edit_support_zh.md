# RFC: 增加 Qwen-Image-Edit 图像编辑仿真支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft（草案） |
| **作者** | minghang_c |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-09 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |
| **规范依赖** | [图像生成推理性能仿真架构 RFC：公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index) |

---

## 1. 概述

本 RFC 提议在图像生成推理性能仿真公共架构的基础上，增加 Qwen-Image-Edit 的配置驱动适配器。首版不是端到端图片编辑器，而是对 Diffusers 去噪阶段 Transformer 执行路径进行可复现的形状构造、调用计数、并行路径和设备时间仿真。

本提案使用一个适配器覆盖三个明确的官方 checkpoint：`Qwen/Qwen-Image-Edit`、`Qwen/Qwen-Image-Edit-2509` 和 `Qwen/Qwen-Image-Edit-2511`。适配器按 pipeline class 和 config fingerprint 区分原版单图契约与 Plus 多图契约，并按 `zero_cond_t` 处理 2511 的源图条件 timestep 差异。所有公共执行、测量、结果和失败语义继承并服从《图像生成推理性能仿真公共架构 RFC》，本 RFC 不复制或改写该公共契约。

<a id="qwen-image-edit-public-dependency"></a>

### 1.1 依赖与范围声明

本 RFC **规范性依赖**[图像生成推理性能仿真架构 RFC 的公共 contract 索引](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-contract-index)。继承关系按以下章节固定，Qwen-Image-Edit 适配不得复制或改写这些公共定义：

| 继承的公共 contract | 规范章节 |
| :--- | :--- |
| CLI 与参数校验 | [公共架构 RFC 3.3](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-cli-contract) |
| exact-profile、静态组装与 adapter | [公共架构 RFC 3.4](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-profile-adapter-contract) |
| 固定 timestep 与 N-step workload | [公共架构 RFC 3.5](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-workload-contract) |
| Guidance 与 true CFG | [公共架构 RFC 3.6](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-guidance-cfg-contract) |
| `U × C` 拓扑与 critical path | [公共架构 RFC 3.7](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-topology-contract) |
| generated/source output 所有权 | [公共架构 RFC 3.8](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-output-ownership-contract) |
| 结果与 Chrome trace | [公共架构 RFC 3.9](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-result-trace-contract) |
| 现有模块复用与修改边界 | [公共架构 RFC 3.10](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-module-boundary) |

公共架构定义 `image_generate` 入口、config-only resolver、meta-device builder、`--sample-step` 驱动的 Transformer workload、embedded guidance 与 true CFG 计数、`U × C` 并行拓扑、Runtime 测量及 `measured` / `shape_only` / `excluded` 阶段标记。若两个 RFC 的描述冲突，以链接章节中的公共 contract 为准；本 RFC 只提供 Qwen 专属 profile、输入形状和执行边界。

本 RFC 的可执行依赖基线固定为 `diffusers==0.38.0`。该版本由仓库锁文件固定；升级 Diffusers 后，必须重新核验 pipeline、`QwenImageTransformer2DModel`、forward signature 以及 `_cp_plan`，不得仅凭类名继续接受模型。

## 2. 动机

Qwen-Image-Edit 的源图 token 会与生成图 token 一起进入每个 Transformer workload iteration，但输出只保留生成图前缀；源图条件在迭代间保持不变。原版和 Plus pipeline 还分别具有单源图与多源图的输入语义；2511 在相同核心 Transformer 架构上启用 `zero_cond_t`，使源图 token 的 timestep conditioning 不再等同于生成图 token。若把它们简单当作普通 text-to-image 模型，或者复用视频入口的 batch-concat CFG 路径，会得到错误的 sequence shape、generated/source output ownership、调用次数或并行时间。

当前项目需要一个无需下载权重、不读取真实图片、但能由 checkpoint 配置准确推导 Transformer 去噪形状的适配边界。这样可以在统一的图像仿真入口中比较 batch、分辨率、源图数量、去噪步数、true CFG 和 CFG 并行拓扑，同时明确结果只代表 Transformer 去噪阶段，而不是完整的图片编辑延迟。

## 3. 目标与非目标

### 3.1 目标

1. 精确支持以下 canonical model ID，禁止前缀匹配或自动接受未来变体：
   - `Qwen/Qwen-Image-Edit`：`QwenImageEditPipeline`，恰好一张源图。
   - `Qwen/Qwen-Image-Edit-2509`：`QwenImageEditPlusPipeline`，一至三张源图。
   - `Qwen/Qwen-Image-Edit-2511`：`QwenImageEditPlusPipeline`，一至三张源图，要求 `zero_cond_t=true`。
2. 对远端 config-only snapshot 和本地 Diffusers 目录执行相同的 pipeline/component/config fingerprint 校验。
3. 复用公共 `DiffusersTransformerModel` wrapper 和 meta-device 构建路径，不复制一套 Qwen 权重加载或 Transformer 包装层。
4. 根据 Qwen VAE config、恰好一次的 `--output-image-size` 和可重复的 `--source-image-size` 计算 output requested/effective size、source original/VL/VAE effective metadata、latent shape、packed token 数及 `img_shapes`，而不读取图片像素；source size 不隐式推断 output size。
5. 保留生成图与源图 token 的拼接顺序、源图条件在迭代间不变，以及 Transformer 输出 generated-prefix 截取的语义。
6. `--sample-step=N` 直接驱动恰好 N 次 Transformer workload iteration；不在适配器内部映射 Diffusers `num_inference_steps`，也不构造 scheduler timetable。
7. 将三个 profile 的 `guidance_embeds=false` 作为 fingerprint；adapter 不构造 embedded guidance。独立实现 Qwen true-CFG norm rescale，并由 `--use-cfg` 表示 true CFG。
8. 首版支持 `U=1` 的顺序执行，以及 `U=1、C=2` 的 true-CFG parallel；对 `U>1` 在执行前稳定失败。
9. 建立 config-only/meta 测试矩阵，覆盖三个 profile、1/2/3 源图、尺寸元数据、N-step、true CFG、2511 `zero_cond_t` 和失败路径。
10. 将三个 supported profile 的共享 Transformer 融合候选纳入适配验收：审计 attention Q/K RMSNorm-like normalization、Qwen image/text multi-axis RoPE 及锁定 Diffusers 0.38.0 源码中的其他 TensorCast 融合候选；若与既有算子 exact match，必须适配，不能仅以模块名判定兼容。

### 3.2 非目标

以下内容不属于本 RFC 或 Qwen-Image-Edit 适配 PR：

- Qwen 未来 checkpoint、社区 mirror、任意 `Qwen/Qwen-Image-Edit-*` 前缀模型；
- 超过三张源图；三张源图是首版文档化支持上限，不推断 Diffusers 当前可变 list API 的未来兼容性；
- Qwen `U>1` 的 sequence/context parallel 或 Ulysses 实现；
- 真实 prompt、tokenizer、processor、视觉/文本 encoder 执行；
- 真实图片读取、RGB 转换、resize 像素处理和源图 VAE encode；
- VAE decode、postprocess、图片保存、输出质量或端到端编辑延迟；
- ControlNet、inpainting mask、LoRA 效果、缓存和其他质量能力；
- 视频入口、Web UI、历史 `video_generate` 参数和结果行为；
- plugin/动态模型注册系统、通用 scheduler 层次或为单一适配器预留的抽象；
- 通过下载完整权重、执行真实推理或以真实显存占用证明兼容性。

## 4. 方案设计

### 4.1 精确 profile 与加载分派

适配器提供一个静态 config-driven profile 集合，而不是动态 registry 或 provider discovery。远端 lookup 使用精确的 `(remote_source, model_id)`；本 RFC 登记的远端来源为 `remote_source=huggingface`，model ID 只能是上面三个完整字符串。本地目录必须由根目录 `model_index.json`、组件目录和结构 fingerprint 唯一匹配 profile。只发现同名 `QwenImageTransformer2DModel` 不构成支持条件。

profile 的规范分派如下：

| Profile | `model_index.json._class_name` | 源图数量 | `zero_cond_t` |
| :--- | :--- | :---: | :---: |
| `qwen-image-edit-original` | `QwenImageEditPipeline` | 恰好 1 | 缺省规范化为 `false` |
| `qwen-image-edit-2509-plus` | `QwenImageEditPlusPipeline` | 1–3 | 缺省规范化为 `false` |
| `qwen-image-edit-2511-plus` | `QwenImageEditPlusPipeline` | 1–3 | 必须为 `true` |

三个 profile 必须同时具有下列组件类：

- `AutoencoderKLQwenImage`；
- `Qwen2_5_VLForConditionalGeneration`；
- `Qwen2Tokenizer`；
- `Qwen2VLProcessor`；
- `QwenImageTransformer2DModel`。

Transformer 的结构 fingerprint 至少必须验证：`patch_size=2`、`in_channels=64`、`out_channels=16`、`num_layers=60`、`attention_head_dim=128`、`num_attention_heads=24`、`joint_attention_dim=3584`、`axes_dims_rope=(16,56,56)` 和 `guidance_embeds=false`。适配器还必须验证 VAE 的 `z_dim`、`temperal_downsample`、`latents_mean` / `latents_std` 存在且类型可用；scheduler 配置不属于 simulator contract，不能用视频 VAE stride 作为缺失字段的 fallback。

2511 的 `zero_cond_t` 必须来自 transformer config。adapter 不增加第二套 source timestep CLI 参数，也不根据 model ID 之外的猜测伪造该分支。原版或 2509 明确设置为 `true`、2511 缺少或设置为 `false`，都在 build/forward 前失败。

模型经过 resolver 验证后，使用现有 `DiffusersTransformerModel` wrapper 在 meta device 上构造 `QwenImageTransformer2DModel`。wrapper 负责公共 device、dtype、量化、compile 和 Runtime seam；Qwen adapter 只负责传入 Qwen 需要的 meta tensors、`encoder_hidden_states_mask=None`、`img_shapes`、timestep 和 profile-specific metadata。

### 4.2 请求尺寸、有效尺寸与 latent geometry

适配器从 VAE config 读取：

```text
S     = 2 ** len(vae.temperal_downsample)
C_lat = vae.z_dim
```

默认 Qwen config 下 `S=8`、`C_lat=16`。公共 CLI 恰好接受一次 `--output-image-size HEIGHT WIDTH`，形成输出请求尺寸 `H_requested × W_requested`；adapter 禁止从 source size 隐式推断 output size。输出请求尺寸再按 Diffusers 规则向下对齐到 `2S` 的倍数，得到 `H_effective × W_effective`；requested 与 effective 必须同时写入结果 metadata，并共同驱动 generated latent、`N_gen` 和 output metadata。随后生成 latent 形状为：

```text
H_lat = 2 * floor(H_effective / (2S))
W_lat = 2 * floor(W_effective / (2S))
generated latent  = [B, 1, C_lat, H_lat, W_lat]
N_gen             = (H_lat / 2) * (W_lat / 2)
generated packed  = [B, N_gen, 4 * C_lat]
```

adapter 必须验证 `4 * C_lat == transformer.in_channels`，并验证 `patch_size² * out_channels` 是 generated packed output width。默认值因此是 64，但 shape 必须来自 config，而不是硬编码为唯一合法值。

请求的源图尺寸通过可重复的 `--source-image-size HEIGHT WIDTH` 传入。该参数描述原始尺寸，不接收路径或像素。源图预处理只在 shape/meta 层重现官方尺寸规则：

- 原版 pipeline：单张源图按 aspect ratio 映射到约 `1024²` target area，宽高分别 round 到 32 的倍数；该 effective size 同时决定 VL 条件 metadata 和源图 VAE token shape。
- Plus pipeline：每张源图分别记录约 `384²` 的 VL condition effective size，以及约 `1024²` 的 VAE effective size，两者均 round 到 32 的倍数；VAE effective size 决定 source token 数。

所有源图的 requested/effective metadata 必须按输入顺序保留，且区分 `vl_condition_size` 与 `vae_condition_size`。本 RFC 不声称这些 resize、processor 或 VAE encode 已被设备测量。

每张源图独立得到：

```text
source_i latent  = [B, 1, C_lat, H_src_i_lat, W_src_i_lat]
source_i packed  = [B, N_src_i, 4 * C_lat]
```

源图 packed sequence 按 token 维拼接为 `[B, sum(N_src_i), 4*C_lat]`。每次 Transformer 输入固定为：

```text
hidden_states = generated | source-1 | source-2 | source-3
               [B, N_gen + sum(N_src_i), 4*C_lat]
```

`img_shapes` 必须按 batch 嵌套；每个 sample descriptor 使用同一 segment 顺序，首项为生成图，之后逐一为源图：

```text
sample_img_shapes = [
  (1, H_gen_lat / 2, W_gen_lat / 2),
  (1, H_src1_lat / 2, W_src1_lat / 2),
  ...
]
img_shapes = [sample_img_shapes] * B
```

batch 内复用同一组 generated/source shape metadata，因此外层长度为 `B`，每个元素的 segment 顺序与内容相同。

公共 `--batch-size=B` 表示同一条件组的生成样本数，对应官方 pipeline 的 `num_images_per_prompt=B` 语义；Plus pipeline 的 prompt batch size 仍为 1，本 RFC 不承诺多 prompt batching。adapter 将同一组 source、text 和 `img_shapes` 条件复制或广播到有效 Transformer batch。

`img_shapes`、`N_gen`、每个 `N_src_i`、source token 总数和总 image sequence length 必须出现在结果 metadata 或可审计的调用记录中。不得把源图数量或 prompt 数量误当 batch 维。

### 4.3 文本条件

首版不运行 Qwen processor、tokenizer 或 `Qwen2_5_VLForConditionalGeneration`。公共 `--text-seq-len` 与 `--negative-text-seq-len` 表示进入 Transformer 的有效 encoder 输出序列长度，而不是字符数、tokenizer 输入长度或模板长度。

```text
encoder_hidden_states_pos = [B, L_pos, 3584]
encoder_hidden_states_neg = [B, L_neg, 3584]  # true CFG 时
```

其中 3584 必须从 `joint_attention_dim` fingerprint/config 得到。`L_pos` 与 `L_neg` 均为有效输出长度，取值范围为 `(0,1024]`；true CFG 允许正向与负向 sequence length 相同或不同，两条分支分别使用各自的有效长度。同一组条件在 batch 内复用，所有位置均有效，传入 `encoder_hidden_states_mask=None`，不新增 padding 或 mask CLI。原版 pipeline 的单图模板与 Plus pipeline 的多图标记只影响 shape metadata 和 source cardinality，不被误报为已测文本编码时间。超过范围的有效文本长度在 builder 阶段失败；公共 contract 采用 Diffusers 0.38 的最大长度校验。

### 4.4 Transformer workload 与每步数据所有权

`--sample-step=N` 直接定义 N 次 Transformer workload iteration；N 必须为正整数，适配器不得把它映射为 Diffusers `num_inference_steps`，不得构造 sigma/timestep timetable，也不得实例化或调用 scheduler。每次 iteration 使用固定的 `[B]` meta/dummy timestep，dtype 与 Transformer 的 timestep 输入要求一致；该 timestep 只承担形状与数据依赖，不表达上游 scheduler 轨迹。

每个 iteration 的所有权顺序为：

1. 读取当前 generated packed meta tensor，并与保持不变的 source packed meta tensor 按 `generated | source...` 拼接；generated tensor 不因前一 iteration 的 Transformer output 数值而更新；
2. 构造该 iteration 的 `img_shapes`、text condition、`encoder_hidden_states_mask=None` 和固定 dummy timestep；
3. 执行 Qwen Transformer forward；
4. 对完整输出只截取 generated prefix：

   ```text
   full_output = [B, N_gen + sum(N_src_i), 64]
   generated_output = full_output[:, :N_gen, :]
   ```

5. 将 generated prefix 交给后续 CFG communication；实际 CFG-parallel collective 进入 `measured`，combine 与 output ownership 只保留 `shape_only` 依赖；不执行 scheduler update，不把 output 写回下一 iteration 的 input；
6. source packed latent、source shape metadata 和 source condition object 在所有 iteration 保持不变。

source 构造、prefix slicing、CFG combine 和 output ownership 依赖标记为 `shape_only`；Transformer forward、Transformer 内部通信和实际 CFG-parallel communication 属于 `measured` 并由 Runtime 记录。关键路径在所需 Transformer output/generated-prefix/CFG communication 完成后结束。

### 4.5 Guidance 与 true CFG

Qwen 的 embedded guidance 和 true CFG 是两个独立概念。三个目标 profile 的 `guidance_embeds=false` 是 profile fingerprint；adapter 不构造 embedded guidance，也不向 Transformer 提供 guidance input。公共 CLI 不暴露 embedded-guidance 数值参数。

true CFG 由 `--use-cfg` 启用，并要求提供负向有效序列长度。每个步骤执行正向、负向各一次，共两次逻辑 Transformer forward；顺序模式不使用视频入口的 batch-concat 近似，因正负分支可能有不同 text sequence length，但两者均传入 `encoder_hidden_states_mask=None`。组合依赖与 Qwen 官方 pipeline 一致，其中 `s_cfg > 1` 仅作为非 CLI 符号：

```text
combined = negative + s_cfg * (positive - negative)
noise_pred = combined * norm(positive) / norm(combined)
```

`s_cfg` 和 norm rescale 的数值操作都属于 `shape_only` 的控制依赖，不作为 CLI 或性能结果字段，但分支输入 shape、调用次数和 forward 设备时间必须保留。三个 profile 均由 `guidance_embeds=false` 驱动不构造 embedded guidance；true CFG 仅由 `--use-cfg` 启用。

### 4.6 2511 `zero_cond_t`

2511 的 `zero_cond_t=true` 由 Transformer config 驱动。adapter 输入 timestep 为 `[B]`；底层模型内部扩展为 `[2B] = [t, 0]`，并构造 `modulate_index` `[B, N_gen + sum(N_src_i)]`，其中 generated positions 全部为 0、source positions 全部为 1。对于该 profile，forward trace 必须可观察到与配置一致的 `[t, 0]` 条件语义；adapter 不在 CLI 暴露第二个 source timestep，也不提前把源图拆成另一套 scheduler 流程。

原版和 2509 采用普通单 timestep 语义，不得进入 2511 分支。三者即使共享 Transformer class 和核心维度，也不能因为 class 相同而忽略该 config difference。

### 4.7 首版并行边界

Qwen joint attention 在每层按 `[text, image]` 组织 QKV，并在 attention 后按 local text boundary 拆回。Diffusers 0.38 的 `QwenImageTransformer2DModel._cp_plan` 对 context parallel 的切分/聚合计划要求：首个 Transformer block 对 `hidden_states` 和 `encoder_hidden_states` 沿 sequence 切分，每个 block 对 `modulate_index` 沿 image sequence 切分，`pos_embed` 沿 sequence 切分，`proj_out` 对 image output 执行 gather。当前 Qwen 适配器没有实现这些 collectives 和边界所有权，因此不能把 `U>1` 当作安全的通用 fallback。

首版明确规定：

```text
ulysses_size = U = 1
C = 1：顺序执行，不启用 CFG parallel
C = 2：同时传入 `--use-cfg` 与 `--cfg-parallel`，world_size = 2
U > 1：在 builder/parallel-group 建立前 fail-fast
```

当 `C=2` 时，正负分支分配到两个 CFG rank；Runtime 将两条 branch 的 measured forward 和通信依赖组织为并行 critical path。逻辑 work 仍计为两次 forward，critical path 使用 `max(positive_branch, negative_branch)` 加上必要的 generated-prefix CFG communication 和 shape-only combine dependency；该路径不包含 Ulysses output gather。`C=1` 的 true CFG 为顺序正/负调用；不因 batch 维伪造并行。

world size、`ulysses_size`、`--use-cfg` 和 `--cfg-parallel` 的组合在 forward 前验证。`U>1` 的失败信息应指出当前 `_cp_plan` 所需的 sequence sharding/gather 尚未属于首版契约，而不是报告笼统的 unsupported model。

融合图验证与并行能力正交：首版只对 `U=1` 的 original/2509/2511 路径捕获并验证 compiled graph；成功融合不能绕过 `U>1` fail-fast。2511 的 compile off/on 对比必须证明 `[t,0]` 扩展、`modulate_index`、`[text,image]` joint-attention 边界、`img_shapes` 和 generated/source span 未被 wrapper、patch 或 pattern rewrite 改变。graph verification failure 与 `_cp_plan` unsupported 必须使用不同诊断。

### 4.8 测量边界与结果声明

结果只称为“Transformer 去噪阶段模拟时间”，不得称为端到端图片生成/编辑延迟。阶段覆盖必须明确标记：

| 阶段 | 首版处理 | 覆盖标记 |
| :--- | :--- | :--- |
| model/config 解析、exact profile 校验 | 执行 config-only 解析 | `excluded`（host setup） |
| Transformer 构建 | meta-device 构建并包装 | `excluded`（构建耗时） |
| Scheduler 控制与数值更新 | 不读取 config、不构造 scheduler、不生成 timetable/sigma/shift、不调用 `scheduler.step` | `excluded` |
| prompt、processor、tokenizer、text encoder | 用户提供有效序列长度，构造 meta embeddings，并传入 `encoder_hidden_states_mask=None` | `shape_only` |
| 源图 resize metadata、VL 尺寸和 VAE source shape | 由尺寸计算，不读图片 | `shape_only` |
| source latent、packing、`img_shapes` | 构造 meta tensor 和 metadata | `shape_only` |
| fixed dummy timestep | 每个 iteration 构造 `[B]` meta/dummy timestep，不构造 timetable | `shape_only` |
| 每步 Qwen Transformer forward | 执行正负 branch 及允许的 CFG parallel | `measured` |
| Transformer 内部通信 | 按实际允许拓扑由 Runtime 记录 | `measured` |
| CFG-parallel communication | `C=2` 时在 CFG group 传递 generated prefix，并由 Runtime 记录 | `measured` |
| CFG combine、generated-prefix 截取、output ownership 依赖 | 保留依赖和 shape，不计轻量算术设备时间 | `shape_only` |
| VAE decode、postprocess、图片保存 | 完全排除 | `excluded` |

输出至少包含 output requested/effective size、按输入顺序保留的 source requested/original size 及推导出的 VL/VAE effective size、profile/model ID、source cardinality、`img_shapes`、generated/source token counts、有效正负文本长度、`sample-step`/Transformer workload iterations、固定 dummy timestep shape/dtype、每步和总 logical forward 次数、`guidance_embeds=false` profile 状态与 `--use-cfg` 状态、`U/C` 拓扑、source immutability、Transformer 去噪阶段模拟时间（Runtime critical path）、operator/call count、通信和 boundary breakdown，以及 scheduler stage=`excluded` 的阶段覆盖声明。不得输出 guidance 数值 scale。不得输出或暗示真实图片、prompt/VAE/文件 I/O 时间、真实权重显存、图片质量或端到端编辑结论。

## 5. 编程与调用设计

### 5.1 公共调用约束

本 RFC 使用公共架构 RFC 的 `image-generate` 入口和参数语义。Qwen 相关输入为：

- `model_id`：三个精确远端 ID 或已校验的本地 Diffusers 根目录；
- `--batch-size`：生成样本数；
- 恰好一次的 `--output-image-size HEIGHT WIDTH`：请求输出尺寸，驱动 generated latent、`N_gen` 以及 output requested/effective metadata；不从 source size 隐式推断；
- `--text-seq-len`：正向 Transformer encoder 输出长度；
- `--negative-text-seq-len`：启用 true CFG 时的负向输出长度；
- `--sample-step N`：正整数 N，直接驱动 N 次 Transformer workload iteration；
- 可重复的 `--source-image-size HEIGHT WIDTH`：每张源图原始尺寸；original 恰好一次，2509/2511 为一至三次，并推导 VL/VAE effective size；
- 不暴露 embedded-guidance 数值参数；三个 profile 的 `guidance_embeds=false` 由 profile config 驱动，不构造 guidance input；
- `--use-cfg`：启用 true CFG，并要求同时提供负向长度；
- 公共 `--device`、`--dtype`、量化、compile、`--world-size`、`--ulysses-size`、`--cfg-parallel`、`--remote-source` 和 `--chrome-trace`。

`--output-image-size` 必须恰好提供一次；缺失、重复、非正值、非整数或格式错误均失败，且不得从 source size 推断 output size。原版必须提供且仅提供一张 source size；2509/2511 必须提供一至三张。源图尺寸数量与 profile 不匹配、`--sample-step` 缺失或非法、未传 `--use-cfg` 却提供负向长度、传入 `--use-cfg` 却缺少负向长度、`--cfg-parallel` 未配合 `--use-cfg`、`U>1` 或错误 world size，均须在 resolver/builder/Runtime 启动前失败。Qwen profile 不增加额外公共参数。

### 5.2 Wrapper 与 adapter 责任

公共层负责入口、参数组合校验、profile lookup seam、Diffusers resolver、`DiffusersTransformerModel` wrapper、N 次 workload orchestration、Runtime streams/dependencies、CFG group 和结果字段。Qwen adapter 负责：

- 三个 exact profile 和 original/Plus dispatch；
- component/config fingerprint 和 `zero_cond_t` 校验；
- VAE-derived effective size、source metadata、latent/packing shape；
- generated/source token 顺序及 `img_shapes`；
- text width/length 与 `encoder_hidden_states_mask=None` 契约；
- Qwen 固定 dummy timestep、generated-prefix ownership；
- `guidance_embeds=false` profile handling（不构造 guidance input）、true-CFG norm rescale 和 2511 zero-timestep metadata；
- Qwen 首版 `U=1` 并行限制及失败诊断；
- 在锁定的 `diffusers==0.38.0` 源码中审计三个 profile 共享的 QwenImage Transformer 语义位置，与 TensorCast 既有融合算子逐项匹配；重点覆盖 attention Q/K 的 RMSNorm-like normalization、Qwen image/text multi-axis RoPE，以及其他已有 TensorCast 融合候选。只有数学、graph topology、epsilon、affine/bias、dtype cast、layout/broadcast、输入/输出数量与 ownership、mutation、RoPE 轴/交错/partial rotary dimension、generated/source sequence layout 全部一致时才适用；适用项必须通过既有 wrapper、patch 或 compile graph replacement seam 接入，不适用项记录 `not_applicable` 及语义差异原因，不得为融合而改变 original/2509/2511 共享路径或 2511 `zero_cond_t` 语义。
- 对声明 `applicable` 的候选捕获各 profile 的 U=1 meta FX graph。采用 wrapper/patch 时在 `torch.compile` 前完成 source-faithful 替换；采用现有 pattern replacement 时由既有 compile backend 执行。编译后以 compiled graph 和 Runtime operator 事件验证 exact overload、shape/layout/dtype、输入输出 ownership 与 mutation。当前 `fused_rope.default` 缺少可供 Qwen 自动命中的通用 pattern、performance property 和 profiling `op_mapping`，不得仅因多轴 RoPE 名称相似而列为已支持 measured fusion；`add_rms_norm`/`add_rms_norm2` 候选必须分别证明单输出/双输出 ownership。

Qwen adapter 不修改视频 `process_input`、视频 CFG batch-concat、视频 VAE stride fallback、公共 Web UI 或公共结果格式，不复制 `DiffusersTransformerModel`。

## 6. 测试设计

所有测试均应以 config-only 或 meta-device 为主，不下载权重、不执行真实图片推理。测试必须区分 adapter shape correctness 与 Runtime measured path；通过 fake tensor 不能声称真实 Qwen checkpoint 已完成质量或端到端验证。

### 6.1 Profile 与 fingerprint

1. 三个精确远端 ID 分别映射到正确 profile、pipeline class 和 source cardinality。
2. 三个 profile 的本地 `model_index.json` 和 component directories 能 config-only 解析。
3. 错误 pipeline class、组件 class、缺少 config、结构 fingerprint、VAE 字段或 `zero_cond_t` 均在构建前失败；不构造或调用 scheduler。
4. 未登记的 future checkpoint、前缀相似 ID、错误 remote source 和社区 mirror 不被接受。
5. 三 profile 都能通过 `DiffusersTransformerModel` 在 meta device 构建 `QwenImageTransformer2DModel`。

### 6.2 Source cardinality 与尺寸

1. 原版 0 张、2 张或 3 张 source 失败；恰好 1 张成功。
2. 2509/2511 的 0 张、4 张失败；1、2、3 张成功。
3. original 单图的约 1024² effective size 同时决定 VL/VAE metadata；Plus 1/2/3 张、不同 aspect ratio source，分别产生约 384² VL metadata 与约 1024² VAE metadata。
4. 恰好一次 `--output-image-size` 的非 `2S` 对齐请求同时报告 output requested/effective size，并产生公式对应的 generated latent、`N_gen` 与 packed shape；source size 不能改变 output size。
5. 每张 source 的 `N_src_i`、source 总 token 数、`img_shapes` 顺序与 generated-prefix 边界正确。
6. `B=1` 与 `B>1` 下同一 source 条件组复用；`B>1` 明确映射为 `num_images_per_prompt`，不被误解为 Plus prompt batch；source count 不改变 batch 语义，source meta object 在全部步骤保持稳定。

### 6.3 Transformer workload、固定 timestep 与 guidance

1. 正向输入具有 `[B, N_gen + sum(N_src_i), 64]` 和 `[B, L_pos, 3584]`（宽度由 config 校验）。
2. 每个 iteration 完整输出先截 generated prefix；source 输出不进入 CFG communication，也不进入下一 iteration 输入。
3. `--sample-step=N` 恰好产生 N 次正向逻辑 forward；不生成 timetable、不执行 generated update。
4. `--use-cfg` 且提供负向长度时恰好为 `2N` 次逻辑 forward；正负文本长度可不同，且两个分支均传入 `encoder_hidden_states_mask=None`。未启用却传负向长度、启用却缺负向长度都稳定失败。
5. 每个 branch 都接收固定 `[B]` dummy timestep，dtype 正确；改变 source count 不改变同一 generated geometry 的 timestep shape。
6. 公共 help 只显示已声明参数；三个 `guidance_embeds=false` profile 的 Transformer 输入不包含 guidance input；true CFG norm rescale 的依赖顺序和 shape 正确。
7. source packed shape、`img_shapes` 和 source condition 在每一步不变。

### 6.4 2511 与并行

1. 原版和 2509 不进入 `zero_cond_t` 分支；2511 的 trace 具有生成 timestep 与 source zero timestep 的 `[t, 0]` 语义。
2. `U=1,C=1` 正常顺序 forward；`U=1,C=2` 仅在 `--use-cfg` 时建立两个 CFG rank 并记录并行 critical path，`--cfg-parallel` 不能自行启用 CFG。
3. `C=1` 的 true CFG 顺序调用为两次逻辑 forward，`C=2` 的 logical work 仍为两次，critical path 不错误相加两个并行 branch 的完整时间。
4. `U>1`、world size 不匹配、错误 CFG group 和非支持的 Ulysses 配置在 Runtime 前 fail-fast，并指出 `_cp_plan` sharding/gather 边界尚未支持。
5. 三个 profile 至少各完成一次 config-only/meta forward shape trace；不使用真实图片或权重。
6. 建立 `applicable / adapted / not_applicable` 融合覆盖清单；`not_applicable` 必须记录数学、graph topology、epsilon、affine/bias、dtype cast、layout/broadcast、输入/输出 ownership、mutation、partial rotary dimension、RoPE 轴/交错规则或 generated/source sequence layout 的具体差异。当前 `fused_rope.default` 不得仅凭名称列为 `applicable`。
7. 每个 exact match 候选先捕获 original/2509/2511 的 U=1 meta FX graph；wrapper/patch 在 `torch.compile` 前完成，现有 pattern replacement 在 compile backend 内执行。`--compile` 图和 Runtime operator 事件必须确认 exact lowered operator overload、输入/输出与 mutation contract，并保留预期 TensorCast 高层融合节点；RMS residual 候选必须区分单输出与双输出 ownership。若现有 pass 可能产生多个等价 form，必须事先列出 accepted equivalent 并证明语义、输出 ownership 和性能边界等价。该节点吸收的 primitive 不得继续独立出现或重复计量，也不得新增未声明的 graph break；显式启用 `--compile-allow-graph-break` 时必须列出预期 boundary 和原因。
8. 进入 `measured` 的融合节点必须验证 performance property、profiling database/version、exact mapping 和声明 shape range 覆盖；mapping 或 shape-data gap 与 semantic `not_applicable` 分开诊断，不能以 `0 ms` 通过。
9. compile off/on 对三个 profile 的输入/output shape、Transformer workload 次数、`[text,image]` joint-attention 边界、`img_shapes`、generated-prefix/source ownership 及 CFG contract 保持不变；测试确认 original/2509/2511 共享路径和 2511 的 `[t,0]`、`modulate_index`、`zero_cond_t` 未被融合适配改变。graph verification failure 与 `U>1` `_cp_plan` unsupported 使用不同诊断。

### 6.5 公共回归

[图像生成公共架构实现 PR](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)的 fake-adapter N-step、CFG、Runtime 和结果契约测试必须通过；`video_generate` 模块、命令、参数、默认值、Web UI 和已有结果行为保持不变。Qwen 测试不得通过修改视频路径来复用其未验证的 batch-concat 逻辑。

### 6.6 融合适配验收

三个 supported profile 的共享 QwenImage Transformer 必须完成候选审计。对 exact match 的 TensorCast 既有融合算子，必须按既有 seam 的正确顺序完成适配，并提供 meta FX graph、`--compile` graph、Runtime exact operator 事件及 measured performance mapping 证据；对不匹配候选提供 `not_applicable` 记录，对 mapping/shape-data 缺口提供独立诊断。验收同时确认 compile off/on 的 shape、workload、`[text,image]`/`img_shapes`/generated-source ownership 和 CFG contract 不变，original/2509/2511 共享路径及 2511 `[t,0]`、`modulate_index`、`zero_cond_t` 保持完整。该验收不新增融合算子族、CLI flag、Runtime 结果字段、动态 registry、第四份 RFC、scheduler 工作、`fused_rope.default` 基础设施或 `U>1` 支持。

### 6.7 Exact-profile simulator E2E 与 Chrome trace 一致性

这里的 E2E 只指“canonical model ID 到仿真结果与 Chrome trace”的 simulator 闭环，不包含 prompt/VL 编码、VAE、真实权重或图片生成。Qwen-Image-Edit 适配 PR 必须提供一条按三个 canonical model ID 参数化的 hermetic simulator E2E。每个 case 都必须从公共 `image-generate` CLI 或其同一公开 orchestration 入口开始，走完真实 resolver、adapter、meta model、Runtime 和 trace exporter，而不是分别证明各组件可调用：

```text
Qwen/Qwen-Image-Edit
Qwen/Qwen-Image-Edit-2509
Qwen/Qwen-Image-Edit-2511
  -> exact-profile resolver
  -> config-only fingerprint validation
  -> meta QwenImageTransformer2DModel builder
  -> N-step Runtime workload
  -> structured simulation result
  -> Chrome trace file
```

提交到仓库的测试不访问 Hub、不下载权重、不读取凭据。每个 case 仍传入完整 canonical `model_id` 和 `remote_source=huggingface`，但在 resolver 的远端读取边界返回对应的脱敏官方 config-only fixture；不得绕过 exact ID 分派或直接注入已经构造的 Qwen adapter。三个 case 使用共同的最小合法请求：`B=1`、一个 `--source-image-size 1024 1024`、`--output-image-size 1024 1024`、各 profile 均合法的文本长度、`--sample-step=2`、`U=1,C=1`、不启用 true CFG，并分别提供临时 `--chrome-trace` 路径。original 与 Plus 的其它 source cardinality、true CFG 和 `C=2` 继续由 6.2–6.4 矩阵测试覆盖。

每个成功结果必须证明：canonical model ID 精确解析到 original、2509 或 2511 profile；pipeline/cardinality/`zero_cond_t` fingerprint 正确；logical Transformer forward 数恰好为 2；operator table 和 measured event 非空；Transformer critical path 大于 0；generated/source token 数、`img_shapes`、generated-prefix ownership 和 fixed dummy timestep shape 正确；`scheduler_stage=excluded`；prompt/VL encoder/VAE/image I/O 未进入 measured stage；trace 引用指向本次 Runtime 实际导出的文件。缺少 performance mapping、出现 0 个 measured event、只完成 resolver/meta forward 或单独调用 trace exporter 均不能通过。

每个 Chrome trace 必须是可解析的 `{"traceEvents": [...]}` JSON，并包含 process/thread metadata 和非空 complete events。测试从锁定的 `diffusers==0.38.0` `QwenImageTransformer2DModel.forward`、对应 profile fingerprint 和同输入 meta FX graph 独立构造 normalized modeling fingerprint，再与 trace 中每条 stream 的有序 operator target、`simulation_shapes` 和调用倍数比较；不得把本次实际 trace 保存为 golden oracle。每个 logical forward 必须覆盖 60 个 Transformer block，并在所有 block 中保持 generated/source packed sequence、`img_shapes` 和 text/image joint-attention shape；Transformer 返回完整 image sequence 后只把 generated prefix 归入后续 CFG/output ownership，第二次 workload 不得出现 scheduler、generated-latent 或 source-token 数值更新路径。2511 case 还必须保留 `[t,0]`、`modulate_index` 和 `zero_cond_t=true` 路径；original/2509 不得出现该分支。

比较不要求 `ts`、`dur`、`pid` 数值或 JSON 字节级相等。compile 关闭时，primitive operator fingerprint 必须与 modeling graph 一致；compile 开启时，只允许使用 6.6 已声明并证明的 accepted equivalent，以 TensorCast 融合节点替代对应 primitive 子图，禁止 fused node 与被吸收 primitive 同时出现。首版三个 E2E case 均固定 `U=1,C=1`；Qwen 不得因 E2E trace 成功而绕过 `U>1` fail-fast。

## 7. 替代方案与取舍

### 7.1 为原版、2509、2511 分别实现三个 adapter

该方案能把 pipeline 差异写在三个独立文件中，但三个 checkpoint 共享组件 class、核心维度和 packing。三份实现会重复 source packing、prefix ownership 和 CFG 逻辑，并使 2511 的唯一 forward difference 难以审计。采用一个 config-driven adapter，通过 profile 数据驱动 pipeline/cardinality/`zero_cond_t`，在保留差异的同时避免错误复制。

### 7.2 仅按 Transformer class 自动接受模型

该方案实现简单，但会把错误 pipeline、未来 checkpoint、不同 `zero_cond_t` 当作兼容模型。Qwen original/Plus 的 source cardinality 和 2511 forward behavior 都证明 class-only dispatch 不足。采用 exact model ID 加 local fingerprint；未知模型明确失败。

### 7.3 复用视频 CFG batch-concat

视频路径的 batch-concat 近似可能减少调度代码，但 Qwen 正负文本长度可以不同，且 source/generated sequence 和 mask 不是视频输入契约。顺序 CFG 与显式 `C=2` CFG parallel 更能保持 shape 和 critical-path 语义，故不复用视频 batch-concat。

### 7.4 为 simulator 引入 scheduler timetable 或 generated update

source token 确实进入 Transformer input，但本 simulator 只需复现 Transformer workload、generated-prefix 选择和 source immutability。引入 scheduler timetable 或 generated update 会把上游数值控制面错误地纳入仿真，故明确排除。

### 7.5 首版实现 Qwen `U>1`

`_cp_plan` 需要首 block sequence split、逐层 modulation/pos split 和 projection gather；在没有验证这些 collective 及 text/image local boundary 之前，开放 `U>1` 会产生看似可运行但错误的性能路径。首版采用 `U=1` fail-fast，后续若有独立需求再做新的并行设计与 RFC，而不是在本 RFC 中假设 fallback。

## 8. 风险与兼容性

### 8.1 Diffusers 版本漂移

pipeline signature、Transformer forward 或 `_cp_plan` 变化都可能影响 shape 和调用语义。通过锁定 `diffusers==0.38.0`、记录 source 版本和在升级时重新执行 fingerprint/meta 矩阵缓解。不得把升级后的同名 class 自动视为兼容。

### 8.2 官方文档能力与仿真边界混淆

2509/2511 官方材料提到多图、ControlNet 或质量改进，但本 RFC 只实现有明确 Diffusers shape contract 的基础 edit pipeline。结果必须逐阶段声明覆盖状态，不得把 `shape_only` 的 processor/VAE 或 `excluded` 的图片输出包装成已测能力。

### 8.3 Plus 源图数量的语义

Diffusers Plus 当前可能接受任意长度 list，而官方 2509 材料将 1–3 张描述为最佳结果范围。本 RFC 将 1–3 张作为首版明确支持边界，以保证 shape、测试和结果含义稳定；超过三张不是兼容承诺，必须显式失败。若未来需要更高 cardinality，应重新评估显存、文本模板、结果契约和 RFC 边界。

### 8.4 尺寸和文本长度差异

不同 pipeline 的 effective-size 规则、VL 与 VAE 双尺寸以及 512/1024 文本长度上限可能导致用户将原始输入尺寸误认为模型输入 shape。metadata 必须同时输出 requested/effective、VL/VAE source size、latent/packed shape 和有效 encoder sequence length，错误组合必须在设备执行前诊断。

### 8.5 并行支持不足

`U>1` fail-fast 限制首版可覆盖的集群拓扑，但比静默使用错误 shard 语义更安全。`U=1,C=2` 的 CFG parallel 仍提供首版 true CFG 的合法 critical-path 选择；后续扩大 context parallel 必须以 `_cp_plan`、collective shape 和 output ownership 的独立验证为前提。

### 8.6 既有融合算子语义不匹配

Qwen Transformer 的模块名称或局部算子形态不能单独证明与 TensorCast 既有融合算子兼容。适配必须在锁定的 Diffusers 0.38.0 源码中逐项核对数学、epsilon、affine/bias、dtype cast、layout/broadcast、RoPE 轴与交错规则，以及 generated/source sequence layout；任一项不一致都记录为 `not_applicable` 并保留原始语义，不能为获得融合节点而改写 original/2509/2511 共享路径或 2511 `zero_cond_t`。对确认适用但未接入既有 wrapper、patch 或 compile replacement seam 的候选，Qwen-Image-Edit 适配 PR 不能通过验收。

<a id="qwen-image-edit-adaptation-boundary"></a>

## 9. Qwen-Image-Edit 适配 PR 边界

Qwen-Image-Edit 适配 PR 只实现本 RFC 的模型适配目标：

1. 注册三个静态 Qwen profile，完成 exact remote/local fingerprint 和 Diffusers 0.38.0 config-only resolver。
2. 使用现有 `DiffusersTransformerModel` 在 meta device 构建 Qwen Transformer，补齐 VAE-derived size、packing、`img_shapes` 和 source metadata。
3. 接入公共 N-step Transformer workload orchestration，提供固定 dummy timestep、generated-prefix 输出所有权和 source immutability。
4. 实现 Qwen 不构造 guidance input 的 profile 分支、true-CFG norm rescale、2511 `zero_cond_t` trace 和 `U=1/C=2` 规则；对 `U>1` fail-fast。
5. 增加本 RFC 的 config-only/meta 测试矩阵和结果覆盖声明。
6. 完成共享 Transformer 的 TensorCast 既有融合候选审计；对 exact match 的候选按 wrapper/patch-before-compile 或现有 compile pattern 的顺序接入，并提供三个 profile 的 U=1 meta FX graph、`--compile` exact operator、高层融合节点、primitive 未分解、无新增 graph break、performance mapping 及 compile off/on model contract 不变的证据。候选不适用时记录 `not_applicable` 语义差异；mapping/shape-data 缺口独立失败；Qwen-Image-Edit 适配 PR 不扩展 `fused_rope.default`。
7. 三个 canonical Qwen model ID 参数化经过公共入口完成 config-only resolver、meta build、N-step Runtime、结构化仿真结果和 Chrome trace 导出的 hermetic E2E；以独立 modeling fingerprint 验证每次 forward 的 60 个 Transformer block、generated/source ownership、2511 `[t,0]` 路径及 accepted fusion equivalents。

Qwen-Image-Edit 适配 PR 仅依赖已合入的[图像生成公共架构实现 PR 边界](./rfc_image_generation_inference_performance_simulation_zh.md#image-generation-public-implementation-boundary)，与 FLUX.1-dev 适配 PR 无依赖关系。它只新增或修改 Qwen-Image-Edit 自身的 profile descriptor、adapter、fixture 和测试，通过公共静态组装 seam 接入；不得修改公共组装规则、FLUX.1-dev 文件、公共 CLI/Runtime contract、视频入口或 Web UI。Qwen-Image-Edit 测试在 FLUX.1-dev profile、adapter、fixture 和测试均不存在时仍必须通过。

若发现公共字段、组装规则或 contract 不足，必须先通过独立的公共架构变更补齐并更新架构 RFC，再基于新的稳定 commit 更新 Qwen-Image-Edit 适配 PR；不得在该模型适配 PR 中静默扩大公共范围。该 PR 不下载权重、不执行真实图片、不提供 output quality 或端到端延迟结论，并可在公共依赖满足后独立评审和合入。

## 10. 参考资料

1. [Qwen-Image-Edit 官方介绍](https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit.md)：原版编辑能力与单图示例。
2. [Qwen-Image-Edit-2509 官方说明](https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit-2509.md)：多图输入及 1–3 张质量范围。
3. [Qwen/Qwen-Image-Edit 模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit)：原版 `QwenImageEditPipeline`。
4. [Qwen/Qwen-Image-Edit-2509 模型卡与仓库](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)：Plus pipeline 与组件目录。
5. [Qwen/Qwen-Image-Edit-2511 模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)：2511 Plus pipeline 与多图示例。
6. [原版 `model_index.json`](https://huggingface.co/Qwen/Qwen-Image-Edit/raw/main/model_index.json)、[原版 Transformer config](https://huggingface.co/Qwen/Qwen-Image-Edit/raw/main/transformer/config.json)。
7. [2509 `model_index.json`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/raw/main/model_index.json)、[2509 Transformer config](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/raw/main/transformer/config.json)。
8. [2511 `model_index.json`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/raw/main/model_index.json)、[2511 Transformer config](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/raw/main/transformer/config.json)。
9. [Diffusers v0.38.0 原版 edit pipeline](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py)：原版尺寸、packing、prefix 和 CFG contract；scheduler 细节仅作上游证据。
10. [Diffusers v0.38.0 Plus pipeline](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)：多图尺寸、packing、`img_shapes` 和 Plus loop。
11. [Diffusers v0.38.0 Qwen Transformer](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/transformers/transformer_qwenimage.py)：forward、`zero_cond_t`、joint attention 与 `_cp_plan`。
