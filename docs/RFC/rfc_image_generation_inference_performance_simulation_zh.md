# RFC: 图像生成推理性能仿真架构

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft（草稿） |
| **作者** | minghang_c |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-09 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |

---

## 1. 概述

### 1.1 简介

本 RFC 定义在 MindStudio-Modeling 中增加图像生成推理性能仿真的公共架构。架构提供独立的 `image-generate` CLI（Python 模块为 `cli.inference.image_generate`），以配置解析、meta 输入构造、Transformer workload 循环和 Runtime 统计为边界，支持 text-to-image 与 image-conditioned editing 两类图像生成入口。

本提案不把图像生成建模为端到端图片生成。它明确区分实际进入模拟设备的 Transformer 去噪阶段、只构造形状和数据依赖的 `shape_only` 阶段，以及完全排除的 `excluded` 阶段，从而使结果可解释、可复现，并避免把文本编码、VAE、文件 I/O 或权重加载时间误报为设备推理时间。公共架构由后续 FLUX.1-dev 和 Qwen-Image-Edit 模型 RFC 通过静态 adapter 接入；本 RFC 不包含任何生产模型 profile 数据。

执行与验证基线为 `diffusers==0.38.0`。该版本变更后必须重新核对 pipeline 和 Transformer 的调用签名；官方 scheduler 仅作为上游 pipeline 背景，不属于运行时仿真，不能以类名相同作为兼容性证明。

### 1.2 动机

现有性能仿真入口主要面向视频或文本生成场景。图像生成模型虽然同样包含 Transformer/DiT、scheduler、并行和 Runtime 统计，但其输入形状、二维 latent packing、图像条件序列、文本条件以及生成输出所有权各不相同。若为每个模型各自实现 CLI 和执行循环，容易产生以下问题：

1. 不同入口对 batch、步数、CFG 和输出时间的解释不一致；
2. prompt、VAE 和图像文件处理被误计入设备执行时间，结果无法与 operator/通信 trace 对齐；
3. image-conditioned editing 的 source token 与 generated token 混淆，生成输出与源图条件的所有权错误，导致源图条件被错误纳入数值状态推进；
4. 只根据 Diffusers 类名自动加载，导致未验证的 checkpoint 进入错误的模型适配路径；
5. 并行配置、CFG 分支和 critical path 的统计口径不清，用户无法判断 `N` 步到底执行了多少逻辑 forward。

公共架构需要先把这些跨模型的语义固定下来，再由独立模型 RFC 负责模型专属公式、fingerprint、输入构造和并行实现。这样既能复用已验证的 resolver、builder、设备、量化、compile、ParallelGroup 和 Runtime 能力，也能把真实模型支持限制在可审计的精确 profile 内。

### 1.3 目标

本 RFC 的目标如下：

1. 增加独立的 `cli.inference.image_generate` 模块和 `image-generate` 命令，不改变 `video_generate` 的名称、参数、默认值、Web UI 绑定和结果行为。
2. 定义统一的公共 CLI 参数、参数组合校验、错误语义和静态 exact-profile 分派边界。
3. 复用现有模型 resolver、Diffusers config-only/meta-device builder、设备/dtype、量化、compile、并行和 Runtime 能力；只在缺少必要 seam 时做最小修改。
4. 定义 `measured`、`shape_only`、`excluded` 三类 pipeline 阶段，并将结果明确命名为“Transformer 去噪阶段模拟时间”。
5. 用 `--sample-step=N` 精确定义 N 次 Transformer workload 迭代、逻辑 forward 次数和通信统计；每次迭代使用模型要求 shape/dtype 的固定 meta/dummy timestep，不构造或推进真实 scheduler 状态。
6. 统一 embedded guidance 与 true CFG 的调用计数和数据依赖，不把两者混为一个开关。
7. 定义 `U × C` 并行拓扑、CFG 分支的 Runtime stream/dependency 以及 critical-path 统计口径。
8. 定义窄的静态 `ImageModelProfile`/adapter contract 和结果 contract，使 FLUX 与 Qwen 能独立提供模型专属实现。
9. 提供不下载权重、不执行真实图片生成的测试设计，包括 test-only fake adapter；未注册生产 profile 必须稳定且可诊断地失败。
10. 为图像生成公共架构实现、FLUX.1-dev 适配实现和 Qwen-Image-Edit 适配实现划定依赖清晰、可独立评审的边界，防止图像生成公共架构实现 PR 演变为动态插件系统或 image/video executor 重构。
11. 将适用的 TensorCast 既有融合算子纳入模型适配验收：模型 adapter 必须按数学语义、layout、dtype、epsilon、affine 和 RoPE 轴规则精确匹配并复用既有融合 seam，并用 `--compile` 图证明确实保留高层融合节点。

#### 非目标

以下内容不属于本 RFC：

1. 不定义或注册 FLUX、Qwen 或其他生产模型的 profile 常量、模型公式或 checkpoint 专属 shape 规则。
2. 不支持依据 Transformer 类名、模型 ID 前缀或模糊相似度自动推断 profile。
3. 不实现 tokenizer、prompt template、text encoder、真实图片读取、源图 VAE encode、VAE decode、postprocess 或文件保存。
4. 不给出端到端图像生成延迟、图片质量、真实权重显存占用或吞吐结论；不执行 scheduler 数值更新或生成 latent 状态推进。
5. 不修改 `cli.inference.video_generate`、`video-generate`、其 `--sample-step` 语义、既有默认值、历史任务或 Web UI。
6. 不抽取通用 image/video generation executor；在两个入口出现经验证的重复前，不进行跨入口重构。
7. 不建立动态插件、provider discovery、通用 registry、scheduler class hierarchy、cache 抽象或任意 output callback 扩展点。
8. 不支持未来 checkpoint、未登记远端模型、任意 Diffusers pipeline 或未在对应模型 RFC 中验证的 Transformer contract。
9. 不在图像生成公共架构实现 PR 中加入真实模型 profile；该实现只能通过 test-only fake adapter 证明公共 contract。
10. 不新增通用融合算子类型、动态 fusion registry、运行时 callback 或 CLI 开关；本 RFC 只定义既有 TensorCast 融合算子的适配验收标准。

## 2. 用例分析

### 2.1 Text-to-image 性能仿真

用户选择一个已登记的图像模型 profile，给出输出尺寸、生成 batch、进入 Transformer 的正向文本条件长度、步数和并行参数。系统从本地 Diffusers 配置目录或受支持的 config-only 远端 snapshot 读取模型配置，以 meta device 构造必要输入，执行 N 次 Transformer workload iteration 并输出 operator、通信、bound 和 critical-path 结果。

输入不接受真实 prompt 来推断序列长度。`--text-seq-len` 表示文本 encoder 输出、实际传入 Transformer 的有效条件序列长度，因此结果可以在无 tokenizer、无网络权重和无文本编码执行的环境中稳定复现。

### 2.2 Image-conditioned editing 性能仿真

用户提供每张源图的原始高度和宽度，而不是图片路径或像素。模型 adapter 按其官方配置和规则计算有效尺寸、latent shape、packed source token 数和 generated token 数，构造 meta source 条件并重复执行 Transformer workload。

源图条件仅作为每步 Transformer 的输入，不参与任何数值状态更新。结果必须区分 generated span 与 source span，避免把源图 token 的形状或数量误当成生成输出，也不将源图读取或 VAE encode 时间计入测量。

### 2.3 True CFG 与 embedded guidance

embedded guidance 是 checkpoint 要求随一次 forward 传入的条件，它不增加 forward 数量。true CFG 则需要正向和负向两条逻辑条件路径，每个 workload iteration 为 2 个逻辑 Transformer forward。两者可以同时启用，但每个分支都各自携带 checkpoint 要求的 embedded guidance，调用次数仍为 2 而不是 4。

首版 true CFG 由 `--use-cfg` 启用，并默认采用顺序分支调用，以保持正向、负向 mask 和 sequence shape 的独立语义。用户同时启用 `--use-cfg` 与 `--cfg-parallel`、使 `C=2` 时，两个分支可以放入 CFG 并行拓扑，但 logical work 和 critical path 必须分别记录。

### 2.4 配置解析与失败场景

本地目录必须由 `model_index.json`、pipeline class、Transformer class 和关键配置 fingerprint 唯一匹配一个已登记 profile；官方 scheduler 仅作为上游 pipeline 背景，不解析或校验。远端只接受模型 RFC 登记的 canonical `(remote_source, model_id)` 对；远端 config-only 解析失败时，应提示用户提供已经获授权的本地 Diffusers 配置目录。

未登记生产 profile、错误 pipeline、错误 Transformer、缺少必需配置或不合法并行拓扑均在 resolver/builder/Runtime 启动前失败，不能静默回退、半初始化或依据同名类继续执行；scheduler 不解析、不实例化、不调用。

### 2.5 DFX 要求

- **兼容性**：新增图像入口与视频入口隔离；现有 `video_generate` 的行为不变；公共架构只依赖当前已验证的 Diffusers `0.38.0` contract。
- **可维护性**：公共层只编排公共语义；模型专属公式和输入布局留在模型 adapter；静态 profile 便于审计和代码搜索。
- **可测试性**：所有公共计数、校验、阶段覆盖、结果字段和并行 dependency 都可用 test-only fake adapter 验证，不依赖网络或模型权重。
- **可靠性**：非法输入在构建或执行前失败；结果标明覆盖范围和失败原因；不会以零值伪装未测阶段。
- **可观测性**：保留 Runtime 的 operator、通信、bound breakdown 和可选 Chrome trace；结果包含 logical work 与 critical path 的区分。
- **资源约束**：只允许 config-only 远端解析，不下载完整权重；不读取真实图片，不执行 VAE 或文本 encoder。
- **可复现性**：workload 迭代次数、固定 dummy timestep shape/dtype、输入 shape、拓扑、CFG 模式和 profile fingerprint 必须出现在结果中；不隐藏 warmup 或从 N 中扣除 compile/setup。

## 3. 方案设计

### 3.1 总体架构

新增入口的数据流如下：

```text
image-generate CLI
  -> public argument validation
  -> static exact-profile resolver
  -> config-only model resolver
  -> meta-device Transformer builder
  -> ImageModelProfile.adapter
       -> shape/input metadata
       -> fixed dummy timestep metadata
       -> one-step forward contract
       -> generated output span
       -> model-specific wrapper/patch preparation (when applicable)
  -> existing compile backend / pattern passes (when --compile)
  -> Runtime / ParallelGroup execution
  -> stage coverage + result contract
```

公共层负责参数解析、组合校验、profile 查找、N 次 workload 编排、固定 dummy timestep 传递、CFG 分支调度、并行拓扑验证、Runtime 依赖和结果封装。adapter 负责读取已经校验的模型配置，生成模型要求的 meta 输入，计算 effective size 和 token span，并提供模型专属的一步 forward 所需结构。

公共层不得通过字符串拼接、反射类名或任意 `**kwargs` 试探模型接口。每个生产 profile 必须静态注册在内建 profile collection 中，并由模型 RFC 提供精确的 pipeline/Transformer/config fingerprint；官方 scheduler 仅作为上游 pipeline 背景，不属于运行时 contract。公共架构在没有生产 profile 时仍可编译和测试，但任何真实模型 ID 都应失败为“当前构建未注册该图像模型 profile”。

### 3.2 Pipeline 阶段与时间口径

首版阶段契约如下：

| 阶段 | 首版处理 | 结果分类 | 是否进入模拟设备时间 |
| --- | --- | --- | --- |
| 模型与 config 解析 | 解析本地目录或 config-only 远端 snapshot，读取 `model_index.json`、Transformer、VAE 配置；官方 scheduler 仅作为上游 pipeline 背景，不解析其配置 | `excluded` | 否，属于 host setup |
| Transformer 构建 | 按 config 在 meta device 构建并包装 Transformer | `excluded` | 否，构建耗时不是模型推理耗时 |
| Prompt/tokenizer/text encoder | 不执行；由用户提供实际进入 Transformer 的条件序列长度，构造 meta embeddings、mask 和 pooled states | `shape_only` | 否 |
| 源图预处理与 VAE encode | 不读取图片、不执行 VAE；根据原始尺寸及 adapter 规则计算 effective size 和 source token shape | `shape_only` | 否 |
| 目标 latent 初始化、packing、position IDs | 只构造 meta tensor 和 shape metadata | `shape_only` | 否 |
| Scheduler 控制与数值更新 | 不读取 config、不构造 scheduler、不生成 timetable/sigma/shift、不调用 `scheduler.step` | `excluded` | 否，属于排除的控制面 |
| 固定 dummy timestep | 按模型要求构造固定 shape/dtype 的 meta/dummy tensor，只保留 Transformer timestep operator path | `shape_only` | 否 |
| 每步 Transformer/DiT forward | 执行正向分支；true CFG 再执行负向分支或其 CFG-parallel 等价路径 | `measured` | 是 |
| Ulysses/CFG 通信 | 按实际并行拓扑执行并由 Runtime 记录 | `measured` | 是 |
| CFG 合并、generated-prefix 截取、结果 ownership | 保留必要的数据依赖、shape 和所有权语义，但不执行 scheduler latent update 或其它数值状态更新 | `shape_only` | 否 |
| VAE decode、postprocess、保存 | 完全排除 | `excluded` | 否 |

因此结果只能称为“Transformer 去噪阶段模拟时间”。模型与 config 解析、Transformer 构建属于执行所需的 host setup，但仍 `excluded from device timing`；scheduler config/实例/控制流则完全不执行，官方 scheduler 仅作为上游 pipeline 背景。同属 `excluded` 的阶段必须进一步标明 `executed_host_setup` 或 `not_executed`：前者用于已执行但不计设备时间的 host setup，后者用于 scheduler、VAE decode、postprocess、保存等未执行阶段。Prompt encoding、源图 VAE encode、VAE decode、权重下载/加载、文件 I/O 和图像质量都不得以 `0 ms` 表示已经测量。

<a id="image-generation-public-cli-contract"></a>

### 3.3 公共 CLI 与参数校验

统一命令为：

```bash
python -m cli.inference.image_generate MODEL_ID [OPTIONS]
msmodeling image-generate MODEL_ID [OPTIONS]
```

模块名使用下划线，子命令名使用连字符，与仓库已有约定一致。`image_generate` 同一个入口覆盖 text-to-image 和 image-conditioned editing；不增加 `image-edit` 入口。

公共参数如下：

| 参数 | 语义与校验 |
| --- | --- |
| `model_id` | 本地 Diffusers 根目录或模型 RFC 登记的精确远端模型 ID；不接受未注册 profile。 |
| `--batch-size` | 一次 Transformer 调用中的生成样本数，必须为正整数；它不是 prompt 数或源图数。Qwen 等 editing 模型的一组 source 条件在 batch 内复用。 |
| `--output-image-size HEIGHT WIDTH` | 恰好一次；最终生成图片空间尺寸，两个值必须为正整数。模型 adapter 据此推导 generated latent/token shape 和 effective size；首版不执行 decode 或生成真实图片。 |
| `--text-seq-len` | 正向条件经过专属 tokenizer/template/encoder 后、实际进入 Transformer 的有效序列长度，必须为正整数并通过 profile 上限校验；不是原始字符数。 |
| `--negative-text-seq-len` | true CFG 负向分支的有效序列长度；只有启用 true CFG 时允许传入，必须为正整数并通过 profile 上限校验。 |
| `--sample-step` | 必须为正整数；复用 `video_generate` 的参数名。图像入口精确定义为 N 次 Transformer workload 迭代，不将 N 映射为 Diffusers `num_inference_steps`，只执行 N 次循环并接收固定 dummy timestep；不提供其它公共步数名。 |
| `--source-image-size HEIGHT WIDTH` | 可重复参数；每次描述一张源图的原始尺寸，必须为两个正整数；不接受图片路径或像素。仅 editing profile 允许使用。结果同时记录每张源图的 requested/effective size，不能从 source size 推断 output size。 |
| embedded guidance | 不是公共 CLI 参数；由 exact profile 的 `guidance_embeds` config 决定。为 true 时 adapter 构造 `[B]` meta/dummy guidance 以保留 operator graph；为 false 时不构造。结果只报告 embedded-guidance 状态。 |
| `--use-cfg` | 启用 true CFG；必须同时提供 `--negative-text-seq-len`。该布尔参数只控制是否创建正、负两个逻辑分支，不携带或推断 CFG 数值 scale。 |
| `--device`、`--dtype` | 复用现有设备和 dtype 语义。 |
| `--quantize-linear-action` | 复用现有线性层量化动作参数和校验。 |
| `--quantize-attention-action` | 复用现有 attention 量化动作参数和校验。 |
| `--compile` | 复用现有 compile 开关和校验。 |
| `--compile-allow-graph-break` | 复用现有 compile graph-break 开关和校验；仅在 `--compile` 下有意义。 |
| `--world-size`、`--ulysses-size`、`--cfg-parallel` | 复用现有并行参数；公共层校验 `U × C` 拓扑，adapter 校验模型专属输入 sharding。 |
| `--remote-source` | 复用已支持的远端来源语义；远端 profile 必须使用模型 RFC 登记的 canonical source/id 对。 |
| `--chrome-trace` | 复用现有 Runtime trace 输出能力。 |

公共 CLI 不接受 `--seed`、`--prompt`、图片路径、自定义 sigma/timestep、VAE stride/packing、`zero_cond_t`、scheduler class、extra JSON 或 DiT cache 参数；这些值不能通过未声明的环境变量或额外字段绕过 profile contract。入口不提供 `--mode` 或 `image-edit`。

公共组合校验至少包括：

1. `--batch-size`、`--output-image-size`、序列长度、步数和 source 尺寸必须是正整数；`--output-image-size HEIGHT WIDTH` 必须恰好出现一次。
2. `--sample-step <= 0` 在 resolver、builder 和 Runtime 启动前失败；N 不映射为 Diffusers `num_inference_steps`。
3. 未传入 `--use-cfg` 时禁止 `--negative-text-seq-len`；传入 `--use-cfg` 时必须同时提供 `--negative-text-seq-len`。
4. embedded guidance 是否启用及是否构造 `[B]` meta/dummy guidance 只由 exact profile 的 `guidance_embeds` 决定；公共 CLI 不声明对应数值参数。
5. text-to-image profile 不接受 source image；editing profile 按模型 RFC 校验 source 数量上下限。
6. `--cfg-parallel` 只允许与 `--use-cfg` 同时使用，且不能自行启用 true CFG；其 `C=2` 要求由公共拓扑校验确认 `world_size == U * C`。
7. `world_size` 必须与 `U × C` 一致；`U` 和 `C` 都是正整数，具体允许的 Ulysses 值由 profile 声明并再次校验输入维度整除性。
8. 正向和负向文本长度可以不同；不得在公共层强制填充为相同长度来掩盖 adapter 的 mask/shape 约束。
9. 未登记 profile、pipeline class、Transformer class 或 fingerprint 不匹配时，必须在构造 Runtime 前失败；scheduler 不构造、不调用。

公共入口使用 `--sample-step=N`；它与 `video_generate` 只共享参数名和“重复 N 次 Transformer workload”的用户层概念。图像入口直接执行 N 次 Transformer workload，不映射为 Diffusers `num_inference_steps`，不提供其它公共步数名，也不修改 `video_generate`。

<a id="image-generation-profile-adapter-contract"></a>

### 3.4 静态 exact-profile 与 adapter contract

公共层定义窄的内建 `ImageModelProfile` 描述、静态 profile 组装 seam 和 adapter contract。profile descriptor 按模型放在互相隔离的模块中，由公共层确定性组装为不可变 collection；不得把一个需要所有模型适配 PR 共同编辑的可变 tuple/map 作为唯一注册入口。该组装方式属于构建时静态组合，不能演化为运行时插件注册或 provider discovery。一个 profile 至少包含：

- canonical `remote_source` 和完整 `model_id`；
- 允许的本地配置 fingerprint；
- pipeline class 和 Transformer class；官方 scheduler 声明只作为上游背景，不在运行时解析或校验；
- embedded guidance 是否由 exact profile 的 `guidance_embeds` 启用；启用时是否构造 `[B]` meta/dummy guidance 以保留 operator graph；禁用时不构造；
- 文本序列上限和 profile 能力（text-to-image/editing）；
- source image 数量限制；
- 是否允许 `U>1`、CFG parallel，以及 profile 所需的拓扑约束；
- 对应的 adapter 构造函数或静态 adapter 标识。

adapter 必须提供以下窄契约：

1. **配置确认**：验证已解析目录的 pipeline、Transformer 和关键配置 fingerprint，拒绝同名但未验证的模型；官方 scheduler 不在运行时解析或校验。
2. **输入元数据**：根据请求尺寸、batch、文本长度和 source 尺寸生成有效尺寸、latent shape、token shape、position ID/mask 所需的 metadata，以及不能放入设备的 shape-only 说明。
3. **条件构造**：只构造 meta embeddings、pooled states、image/source token 和其它模型要求的 meta 输入；不运行 tokenizer、text encoder、VAE 或图片 I/O。
4. **timestep 参数**：提供模型要求的固定 meta/dummy timestep shape 和 dtype；公共层负责 N 次 workload 循环，不提供真实 timetable、sigma、shift 或时间值。
5. **一步 forward**：接收公共层选定的 branch、固定 dummy timestep、meta inputs 和并行上下文，执行或描述一次模型 Transformer forward；返回包含 output shape/ownership 的结果。
6. **生成输出 span**：声明每次 output 中 generated token 的连续 `start,length` span；editing 模型必须声明 source span 不属于任何数值状态更新。
7. **模型校验**：对文本上限、尺寸整除、source 数量、并行输入整除和 profile 特有约束返回稳定的诊断错误。
8. **既有融合复用**：审计 Transformer forward 中与 TensorCast 既有融合算子相应的语义候选（例如 RMSNorm、RoPE），并按数学语义、graph topology、layout、dtype cast、epsilon、affine/bias、broadcast、输入/输出数量与 ownership、mutation，以及 RoPE 轴/交错/partial rotary dimension 规则逐项判断。确认适用时必须接入现有 wrapper、patch 或 compile graph replacement seam；适用但未适配则不能通过模型适配验收。仅名称相似不足以认定适用。采用 wrapper/patch 时，模型 adapter 必须在调用 `torch.compile` 前完成 source-faithful 替换；采用现有 pattern replacement 时，由既有 compile backend 在编译过程中执行，不新增通用 callback。编译后由模型适配 PR 通过 captured/meta FX graph、compiled graph 和 Runtime operator 事件验证 exact emitted overload、输入输出与 mutation contract；公共运行路径不新增融合状态字段或动态 hook。

当前基础设施审计只证明既有 pattern 与性能属性可覆盖其各自的精确 operator contract，不证明 FLUX/Qwen 上游图一定命中。RMSNorm residual 候选必须区分 `torch.ops.tensor_cast.add_rms_norm.default` 的单输出语义与 `add_rms_norm2.default` 的归一化输出加 updated-residual 双输出语义。`torch.ops.tensor_cast.fused_rope.default` 是独立的 partial/3D 显式算子；当前未发现通用 graph pattern、performance property 或 profiling `op_mapping`，因此不能仅因名称含 RoPE 就作为 FLUX/Qwen measured path 的适用融合。本轮不补该基础设施；若未来需要，必须另行证明独立价值和性能覆盖。

公共层拥有上述方法的调用顺序、调用次数、阶段标签和结果包装；adapter 不得自行创建 scheduler、隐藏的 scheduler loop、CFG loop 或 Runtime 统计口径。模型专属公式和具体 token 数计算留在 FLUX/Qwen 模型 RFC，不在此公共 RFC 中重复。既有融合算子的候选清单、逐项匹配结论、接入方式和 compile graph 证据由模型适配 PR 提供。

远端分派只接受模型 RFC 明确登记的 canonical `(remote_source, model_id)`，使用精确匹配，不做前缀匹配。对本地目录，必须由完整配置和 fingerprint 唯一匹配。仅因为 Diffusers 存在同名 Transformer class，不足以自动支持该目录。config-only 解析失败时不回退社区 mirror、不下载完整权重；错误信息应明确要求用户提供已获授权的本地 Diffusers 配置目录。

<a id="image-generation-workload-contract"></a>

### 3.5 固定 dummy timestep 与 N 次 workload 迭代

公共层统一使用 `--sample-step=N`，直接表示 N 次 Transformer workload 迭代，按如下语义执行：

1. N 必须为正整数；公共层不解析或验证 scheduler config，不实例化 scheduler，不生成 timetable、sigma、真实 timestep 或 FlowMatch shift。
2. 每次迭代向 adapter/Transformer 传入固定的 meta/dummy timestep tensor；其 shape 和 dtype 必须满足模型要求，但不代表真实时间值，也不随迭代变化。
3. Runtime 执行恰好 N 次循环，不存在隐藏 warmup，也不从 N 中扣除 compile、config 解析或 host setup。
4. 每次迭代执行一次正向逻辑 Transformer forward；启用 true CFG 时增加一次负向逻辑 forward，因而逻辑 forward 总数为 N 或 2N。
5. 不执行 scheduler.step，不推进生成 latent 或 source token 的数值状态；CFG combine、generated span 截取和 ownership 仅保留必要的依赖/shape 语义，不纳入设备性能时间。
6. N 次迭代中的 Transformer forward 与相关 Ulysses/CFG 通信全部进入测量，由 Runtime 记录；不存在 scheduler update critical-path 或 call count。

公共层不假设所有模型有相同 latent packing、timestep shape 或输入公式。adapter 必须提供模型要求的固定 dummy timestep shape/dtype 和 generated span，不能让公共 helper 根据模型名猜测 token 数或尺寸。官方 scheduler 仅作为上游 pipeline 背景，不属于本运行时仿真。

<a id="image-generation-guidance-cfg-contract"></a>

### 3.6 Guidance 与 CFG 分支语义

embedded guidance 和 true CFG 是独立维度：

- exact profile 的 `guidance_embeds=true` 时，adapter 为每次 forward 构造 `[B]` meta/dummy guidance，保留 embedded-guidance operator graph；每步仍只有 1 次 Transformer forward。`guidance_embeds=false` 时不构造该输入。
- `--use-cfg` 启用 true CFG，并要求提供 `--negative-text-seq-len`；每步有 2 次逻辑 Transformer forward，正向和负向序列长度可以不同。
- CFG 合并的数据依赖可写为 `negative + s_cfg * (positive - negative)`，其中 `s_cfg > 1` 只是描述官方合并顺序的非 CLI 符号。首版不接收、推断或报告 CFG 数值 scale，因为该数值不改变 Transformer shape、operator graph、forward 次数、通信量或 critical path，且合并算术属于 `shape_only`。
- `guidance_embeds=true` 与 true CFG 同时存在时，每个正/负 forward 都构造 `[B]` meta/dummy guidance，仍然只有 2 次逻辑 forward，而不是 4 次；`guidance_embeds=false` 时两个 forward 都不构造该输入。模型 adapter 只能追加其模型 RFC 已明确的后处理（例如 Qwen norm rescale），不得改变 forward 调用数、不得引入 scheduler update，并须保持 generated/source 所有权。
- 公共 CLI 不暴露 embedded-guidance 数值参数；embedded guidance 完全由 exact profile 的 `guidance_embeds` config 决定。
- 图像入口沿用 `video_generate` 的 `--use-cfg` 参数名，但默认采用正、负两次独立顺序调用，不继承视频入口的 batch-concat workload 近似，以保持两个分支的 mask、sequence shape 和输出所有权语义。
- `--cfg-parallel` 只改变两个已启用分支的执行拓扑，不能自行启用 CFG，也不改变 logical forward 数和 CFG combine 依赖；公共结果必须同时给出 logical work 与 critical path，并明确不包含 scheduler update。

<a id="image-generation-topology-contract"></a>

### 3.7 U × C 拓扑与 critical path

公共并行模型用两个正交维度表达：

- `U`：Ulysses/序列并行 **group size**，不是组数；
- `C`：CFG branch 并行度。顺序 CFG 为 `C=1`，CFG parallel 为 `C=2`。

有效 world size 必须满足 `world_size = U × C`。CFG parallel 时，positive 和 negative 各拥有一个大小为 U 的 Ulysses group；rank 映射固定为 `rank = branch_id * U + ulysses_id`，其中 `branch_id=0` 为 positive、`branch_id=1` 为 negative。对每个 Ulysses rank，`cfg group(u)` 由同一 `ulysses_id=u` 的 positive/negative rank 组成。顺序 CFG 时只有一个 branch group，两个分支在同一组内顺序执行。`U=1` 是合法的退化拓扑；adapter 可以进一步拒绝其不支持的 `U>1`。

Ulysses 和 CFG branch communication 属于 `measured` 阶段，必须由 Runtime 记录。两个 branch 的 CFG combine、generated span 截取保留依赖及所有权语义，但不执行 scheduler latent update，属于 `shape_only`，不进入首版设备时间。

公共层不得把 CFG branch 的 logical work 直接乘到 critical path：

- **logical forward count**：`N`（无 true CFG）或 `2N`（true CFG），无论顺序还是并行。
- **logical measured work**：所有 branch 的 Transformer operator 和 communication 工作总和，用于解释总工作量。
- **critical-path execution time**：按 Runtime stream 和 dependency 计算，每次 workload branch 的最长依赖路径之和；并行的两个 CFG branch 重叠时，路径不把完全重叠的时间简单相加，不包含 scheduler update。
- **step statistics**：每步 critical path、branch 数、通信和 bound breakdown；总结果必须说明是否顺序 CFG 或 CFG parallel。

Ulysses 的具体输入 sharding、attention collective、output gather 和整除要求属于模型 RFC；公共层只承认 profile 提供的拓扑 contract，并确保 group size、world size 和 CFG 维度不会被静默改变。

<a id="image-generation-output-ownership-contract"></a>

### 3.8 Generated output span 与所有权

每次一步 forward 的结果必须携带 output ownership metadata。公共 contract 至少要求：

- generated token span 的值级 `start` 和 `length`，且必须是连续区间；当前 FLUX 声明完整 image output span，Qwen 声明 `[0, N_gen)`；
- source token span（如存在）的连续起始位置和长度；
- output 的总 token shape 和 batch 对应关系；
- generated span 表示输出 ownership；本运行时不执行 scheduler update；
- source span 在所有 N 步中保持条件输入，不发生数值状态推进；
- text 条件、source 条件和 generated 条件的序列布局可在 trace/result 中审计。

FLUX text-to-image profile 声明整个 image output 为连续 generated span；Qwen editing profile 声明 generated prefix `[0, N_gen)`。公共层只接受当前两种模型所需的连续 `start,length` 值级 contract，不能依赖“最后若干 token”猜测，也不引入任意索引集合或未来结构化 span。生成 span 的具体长度和 packing 公式由模型 RFC 提供；公共层只执行 span contract 并在 shape 不一致时失败。

<a id="image-generation-result-trace-contract"></a>

### 3.9 结果与 trace contract

每次成功运行至少输出结构化结果（JSON 或等价结果对象），包含以下字段类别：

1. **运行标识**：profile 名称、canonical model ID/source、config fingerprint、Diffusers baseline、device、dtype、量化/compile 状态。
2. **请求与有效尺寸**：batch-size、requested/effective output image size、text/negative text sequence length、每张 source image 的 requested/effective size；output size 不从 source size 推断。
3. **输入 shape**：latent channels、generated token 数、每张 source token 数、总 token 数、文本宽度、position ID/mask shape、generated/source span。
4. **迭代与 timestep**：`--sample-step` 的 N、固定 dummy timestep 的 shape/dtype、`scheduler_stage=excluded`；明确 scheduler config 解析、scheduler 实例化、timetable/sigma 生成、shift、`scheduler.step` 和 latent update 均被排除。
5. **调用语义**：embedded-guidance 是否启用（由 exact profile 的 `guidance_embeds` 决定）、`--use-cfg`/true CFG 是否启用、顺序或 CFG-parallel、每次迭代和总 logical Transformer forward 数；明确不构造或调用 scheduler，不报告 guidance scale 或 CFG 数值 scale。
6. **并行语义**：world size、Ulysses size `U`、CFG parallel degree `C`、拓扑验证结果、branch stream/dependency 说明。
7. **测量结果**：Transformer 去噪阶段模拟时间（critical path）、每步平均/分位时间（若 Runtime 提供）、logical measured work、operator table/call count、通信 breakdown 和 bound breakdown。
8. **阶段覆盖**：每个阶段的 `measured`、`shape_only` 或 `excluded` 状态，以及未测阶段的明确说明。
9. **trace 引用**：用户请求 Chrome trace 时输出 trace 路径或结果标识；默认不虚构 trace。

`Runtime.total_execution_time_s()` 的结果代表 Transformer 去噪阶段模拟时间（critical path），而不是端到端图片延迟。结果不得输出或暗示 prompt/VAE/文件 I/O 时间、真实显存权重占用、真实图片、吞吐或质量结论。未支持模型和失败运行不生成看似成功的部分结果；失败信息必须包括 profile 分派、配置校验或参数校验的具体原因。

融合候选清单、captured/compiled graph 和 exact operator mapping 属于模型适配 PR 的验收证据，不扩张为稳定运行结果字段。`--compile` 关闭时只能说明未执行 compiled-graph 验证，不能声称某位置已经图融合；`--compile` 开启但 exact operator 缺少 performance property、当前 profiling database/version 的可解析 mapping 或声明 shape range 的覆盖时，必须将其诊断为 mapping/shape-data gap，并拒绝生成包含该缺口的成功 measured 结果，不能回填 `0 ms`。

<a id="image-generation-module-boundary"></a>

### 3.10 现有模块复用与修改边界

公共架构优先复用：

- 现有 CLI 注册和参数解析模式；
- Diffusers config resolver 与 config-only 远端来源解析；
- meta-device Transformer builder；
- device/dtype、量化和 compile 配置；
- `ParallelGroup` 或等价现有并行组能力；
- `Runtime` 的 operator、通信、bound、total execution time 和 Chrome trace 能力。

只在现有模块没有必要调用 seam 时做最小修改。图像生成公共架构实现 PR 的新增核心路径应集中在 `cli.inference.image_generate`、窄的 image-generation contract/编排模块和对应公共测试；不修改视频执行循环以获得代码复用。若两个入口未来确实出现稳定且有测试证明的重复，再另行提出重构。对于既有 TensorCast 融合算子，公共架构只提供并复用现有 fusion seam 和可审计的 compile 图验证方式，不新增通用融合类型、动态 registry 或 callback；模型 adapter 负责候选审计、语义匹配和具体接入。

## 4. 替代方案与未采用原因

### 4.1 复用 `video_generate` 作为图像入口

该方案表面上减少 CLI 文件，但会把视频的 frame/sequence 和历史默认值带入图像语义，且无法自然表达二维 image shape、source token span 和独立的 N-step contract。它还会增加视频回归风险，因此不采用。`video_generate` 保持完全兼容。

### 4.2 为图像和视频立即抽取通用 generation executor

两者都包含重复 Transformer workload 循环，但输入 packing、CFG 分支、output ownership 和 measured-stage 边界尚未证明相同。过早抽象会把未验证的差异隐藏在通用层，扩大改动面。首版采用独立图像入口和窄公共 contract，待重复经过真实实现和测试证明后再评估重构。

### 4.3 按 Transformer 类名或模型 ID 前缀自动匹配

这种方式省去 profile 登记，但无法证明 pipeline、Transformer 配置 fingerprint 和模型 variant 兼容，遇到未来 checkpoint 时可能错误执行。公共架构必须 exact-match，并在未注册时明确失败。

### 4.4 动态插件或通用模型 registry

插件发现和运行时注册会增加安全、加载顺序、版本兼容和审计成本，而当前只规划两个已知模型族。静态内建 collection 已能表达需求，故不引入动态扩展点。

### 4.5 运行真实 prompt、VAE 和图片以获得“完整”延迟

这会引入权重、网络、图片内容、tokenizer 版本和文件 I/O 的不稳定因素，也无法直接对应当前 TensorCast Runtime 的设备 operator 统计。目标是可解释的 Transformer 去噪性能仿真，因此首版只构造 shape-only meta inputs 并测量 measured 阶段。

### 4.6 用 batch-concat 近似 true CFG

将正负条件拼接到 batch 可以减少调度代码，但可能改变 mask、sequence length、并行分片、内存形状和模型分支语义。公共默认采用两次独立逻辑 forward；只有显式 CFG parallel 才允许通过 Runtime 拓扑并发，不用 batch-concat 改写模型 contract。

## 5. 安全、隐私与 DFX 设计

### 5.1 安全与隐私

1. 本地目录路径不触发网络访问；只有非目录且通过 exact remote profile 校验的输入才允许进入 config-only 远端解析。
2. 远端解析只下载配置文件，不下载权重、图片或其它不必要资产；不能在 allowlist 不可表达时退化为完整仓库下载。
3. 不接受图片路径、图片像素或真实 prompt，避免将用户图像和文本内容写入 trace、缓存或结果。
4. 结果只记录必要的尺寸、长度、shape、profile fingerprint 和性能摘要；不得记录 prompt 原文、图片内容或认证信息。
5. 远端 source/model ID 采用模型 RFC 登记的 canonical 精确匹配；不通过用户输入执行任意代码、动态 import 或远程插件加载。
6. profile、配置和输入校验在 Runtime 启动前完成；错误信息不泄露 token、凭证或完整本地缓存中的敏感内容。
7. compile、量化、trace 等现有能力继续遵循仓库已有安全边界；本 RFC 不新增认证参数或凭证存储。

### 5.2 兼容性

- `video_generate` 模块路径、命令名、参数、默认值、Web UI 绑定和既有结果行为不变。
- `image_generate` 不承诺与 `video_generate` 参数完全对称；两者只共享 `--sample-step` 参数名和“重复 N 次 Transformer workload”的用户层概念。图像入口不映射 Diffusers `num_inference_steps`，不修改视频参数或增加其它公共步数名。
- Diffusers 可执行基线固定为 `0.38.0`；依赖升级必须重新审计 pipeline、Transformer signature 和 adapter shape contract。
- 未支持 profile 必须失败，而不是以兼容模式运行；失败消息包含下一步可操作建议，例如使用已登记模型或提供已获授权的本地配置目录。

### 5.3 可维护性与可测试性

公共 contract、阶段枚举、结果 schema 和错误类型应集中定义，避免 CLI、adapter 和 Runtime 各自复制计数逻辑。模型 RFC 只能增加其 profile 和 adapter 数据，不应修改公共 CLI 语义来迁就模型特例。

测试必须能够在无权重、无联网、无真实图片和无生产 profile 的环境中运行。test-only fake adapter 仅用于验证公共 orchestration，不得被生产静态 profile collection 引用，也不能使 RFC 声称任何真实模型已经受支持。

### 5.4 可靠性与失败可诊断性

以下错误在执行前失败：非法正值、CFG 参数组合错误、source 数量错误、profile 不存在、pipeline/Transformer/config fingerprint 不匹配、文本长度超限、尺寸不整除、world size 与 `U × C` 不一致、profile 不支持的并行配置。错误必须指出输入字段、期望约束和可行替代方式。

测量期间若 Runtime 或 adapter 报错，不输出完整成功结果，不把缺失 stage、operator 或 trace 填成零。部分诊断信息可以保留，但必须标记为失败运行。

## 6. 编程与调用设计

### 6.1 编程模型基本设计

公共 API 面向 Python 3.10+ 和现有 TensorCast 运行环境。开发者可以通过 CLI 使用，也可以在已有推理编排中调用等价的公共 builder/adapter contract。公共层接收已校验的配置、请求 shape、workload 迭代次数、固定 dummy timestep、guidance/CFG 和并行设置，返回结构化 result；它不接收真实 prompt 或图片对象。

建议的内部职责边界如下：

- `image_generate`：CLI parser、help、参数组合校验和结果输出；
- 静态 profile collection：exact lookup 和 descriptor；
- image-generation contract/编排模块：adapter 调用、N 次 workload、CFG、拓扑和 stage/result contract；
- 现有 resolver/builder/runtime：配置解析、meta 模型、设备和性能统计；
- 模型 RFC adapter：fingerprint、shape、packing、模型专属 forward 和 generated span。

这些是职责边界而非要求公开为稳定 Python API 的文件名；实现应优先贴合仓库现有模块组织。

### 6.2 公共调用输入

公共调用必须能获得以下元素：

1. 精确的 local model directory 或 canonical remote source/model ID；
2. batch、requested size、text/negative sequence length 和可选 source sizes；
3. `--sample-step`（直接表示 N 次 Transformer workload 迭代，不映射为 Diffusers `num_inference_steps`）、profile 决定的 embedded guidance、`--use-cfg` 和 CFG parallel 参数；
4. device、dtype、量化、compile、world size、Ulysses size 和 trace 选项；
5. 当前 Diffusers baseline 和 profile fingerprint。

调用者不能通过未声明的 kwargs 绕过 profile 校验、注入 scheduler 或动态 adapter，或替换 generated span。

### 6.3 使用约束

- 本入口只测 Transformer 去噪阶段，不提供图片文件输出。
- 真实 prompt、图片路径和 source pixels 不可作为输入；必须改为有效条件序列长度和源图原始尺寸。
- embedded guidance 由 exact profile 的 `guidance_embeds` 决定；true 时构造 `[B]` meta/dummy guidance 保留 operator graph，false 时不构造，结果只报告 embedded-guidance 状态。`--use-cfg` 只表达 true CFG 是否启用。CFG 合并系数 `s_cfg` 仅是 `shape_only` 数据依赖中的符号，不是 CLI 或结果字段。
- source image 参数只能由允许 editing 的 profile 使用；source 数量和尺寸由 model RFC 进一步限制。
- `--cfg-parallel` 只允许与 `--use-cfg` 同时使用，并要求合法 `U × C` 拓扑；它不能自行启用 true CFG。
- 生产模型必须先加入模型 RFC 和静态 profile；公共架构不提供未登记模型的通用 fallback。

## 7. 测试设计

### 7.1 CLI 与 validation 单元测试

1. 验证 `image_generate --help` 只显示已声明的公共参数，包括恰好一次的 `--output-image-size HEIGHT WIDTH`、`--sample-step`、`--use-cfg` 和 stage/性能口径说明；不暴露其它尺寸、步数或 embedded-guidance 数值参数。
2. 验证 batch、output/source 尺寸、文本长度和 `--sample-step` 的零值、负值、非整数或格式错误输入在 Runtime 前失败，并验证 output size 缺失、重复或参数个数不为两个时失败。
3. 验证 `--use-cfg` 与 `--negative-text-seq-len` 的组合：未传 `--use-cfg` 时负向长度被拒绝，传入 `--use-cfg` 却缺少负向长度时失败，`--cfg-parallel` 不能自行启用 CFG。
4. 验证 `guidance_embeds=true` 的 fake profile 构造 `[B]` meta/dummy guidance 并保留每步 1 次 forward，false 的 profile 不构造该输入；结果只报告 embedded-guidance 状态。
5. 验证 text-to-image 拒绝 source image，editing fake profile 按声明的 source cardinality 接受/拒绝输入。
6. 验证不同正向/负向 text sequence length 可通过公共层，具体 shape 由 adapter 校验。
7. 验证 `world_size == U × C`、`C=2` 只在 CFG parallel 下成立，以及非法拓扑在执行前失败。

### 7.2 静态 profile 与 resolver 测试

1. 未注册生产 model ID 稳定失败，错误明确说明当前构建未注册 profile。
2. 远端只接受 exact `(remote_source, model_id)`；相似前缀、未来 variant 和错误 source 均失败。
3. 本地目录的 pipeline、Transformer 或关键 config fingerprint 任一不匹配时失败；scheduler 不参与 fingerprint 校验。
4. gated 或 config-only 远端解析失败时不回退社区 mirror、不下载完整权重，并给出本地配置目录替代方式。
5. test-only fake adapter 可被显式测试装配，但生产 profile collection 不包含 fake profile。

### 7.3 Fake adapter orchestration 测试

1. `N=1`、`N=3` 和边界值分别产生恰好 N 个 workload 迭代；每次迭代收到固定 shape/dtype 的 dummy timestep，且不产生 timetable 或 scheduler update。
2. fake adapter 记录 scheduler 未被构造、调用或推进；无 true CFG 时 measured logical forward 数为 N，true CFG 顺序和 CFG parallel 均为 2N。
3. `guidance_embeds=true` 时每步构造 `[B]` meta/dummy guidance 但仍为 1 次 forward；与 true CFG 同时启用时每个正/负 forward 都保留该 operator path，仍为 2 次而非 4 次；false 时不构造。
4. source span 在每一步保持不变；generated/source ownership 和 CFG combine 依赖可审计，但不存在 scheduler update；结果记录 generated/source token 数和所有权。
5. `U=1,C=1`、`U>1,C=1`、`U=1,C=2` 和合法 `U>1,C=2`（若 fake profile 声明支持）分别验证 group、stream 和 dependency。
6. 顺序 CFG 的 critical path 包含两个分支；CFG parallel 的 logical work 仍为两条分支之和，但 critical path 不把重叠时间简单相加。
7. Runtime operator/communication/bound breakdown、stage coverage 和可选 Chrome trace 字段全部可验证。
8. 失败的 adapter shape、span、整除和 topology contract 不生成成功结果。
9. 对每个模型 adapter，测试覆盖 `applicable / adapted / not_applicable` 融合清单；每个候选先以锁定源码和 captured/meta FX graph 证明 graph topology、shape/layout/dtype、输入输出 ownership 与 mutation 精确匹配已有 pattern。适用候选必须通过 `--compile` 图和 Runtime operator 事件确认 exact lowered operator overload、预期 TensorCast 高层融合节点存在，以及由该节点吸收的 primitive 不再独立出现或重复计量；RMS residual 候选必须明确验证单输出或双输出 ownership。
10. wrapper/patch 接入必须在 `torch.compile` 前完成，compile graph replacement 继续由现有 backend/pattern pipeline 执行；fake-adapter 顺序测试只验证这个既有调用顺序，不建立新 callback。若现有 pass 可能产生多个等价 form，模型适配 PR 必须事先列出 accepted equivalent，并证明语义、输出 ownership 和性能边界等价。不得新增未声明的 graph break；显式启用 `--compile-allow-graph-break` 时，必须列出预期 boundary 和原因。
11. 进入 `measured` 的 exact fused operator 必须具有当前 performance property，并能通过记录的 profiling database/version、operator mapping 和既有插值规则覆盖声明 shape range。将 mapping error、shape-data gap 和 semantic `not_applicable` 分开；不得以 `0 ms` 或新增稳定 Runtime 结果字段伪造融合证据。当前 `fused_rope.default` 缺少上述通用 pattern/性能映射证据，fake 或模型测试不得把它列为已支持 measured fusion。

### 7.4 回归与集成测试

1. `video_generate` 的模块路径、help、参数默认值和既有结果测试保持通过。
2. image CLI 与现有 resolver、meta builder、device/dtype、量化、compile 和 Runtime 的最小集成测试不需要下载权重。
3. 至少一个本地 config-only fake directory 验证解析到 adapter，再完成一条 meta forward 和 N-step loop。
4. 测试结果明确区分 measured/shape_only/excluded，不允许将 prompt、VAE、I/O 作为 0 ms measured stage。
5. 不将真实 Hub、真实图片或生产 checkpoint 下载作为 committed 回归测试；真实模型验证属于对应模型适配 PR 的交付期 config-only/meta 验证。

## 8. 缺点和风险

1. **结果不是端到端延迟**：用户可能把 Transformer critical path 误读为完整图片生成延迟。结果字段、CLI help 和文档必须持续显示阶段覆盖声明，并明确排除 prompt/VAE/I/O。
2. **meta/shape-only 不能反映真实数据依赖的全部成本**：其价值是统一形状和算子性能仿真，不是图片质量或真实 pipeline benchmark。模型专属 shape contract 必须有 config-only/meta 测试。
3. **静态 exact-profile 扩展成本较高**：每个新模型都需要模型 RFC、fingerprint 和独立测试，但这种成本换取了可审计的兼容性和稳定失败，优于错误自动匹配。
4. **公共 contract 可能暂时不覆盖未来模型差异**：避免加入未证明的扩展点；当至少两个模型真实证明 contract 不足时，再以独立 RFC 讨论公共变更。
5. **CFG parallel 的统计较复杂**：logical work 与 critical path 可能被用户混淆。结果必须同时列出 forward count、work、stream/dependency 和 critical path，不用单一时间字段承载所有含义。
6. **Diffusers 版本变更风险**：`0.38.0` 以后的 pipeline/Transformer 签名或模型输入 contract 可能改变。升级必须重新审计 adapter 和测试矩阵，而不是只更新依赖版本。
7. **远端配置依赖环境**：网络、权限和 gated checkpoint 可能导致 config-only 解析失败；错误信息提供已获授权本地目录作为回退，且本地路径流程不变。
8. **并行能力分布不均**：公共层支持 `U × C` 语义，但具体模型可能只支持 `U=1` 或顺序 CFG。adapter 必须显式声明限制，不能通过公共层放宽。
9. **公共和视频代码并行演进**：不立即抽取通用 executor 会保留少量重复，但避免未经证明的跨入口耦合；重复达到可验证阈值后再单独重构。
10. **既有融合语义漂移**：模块名相似并不保证数学、layout 或 dtype 兼容；错误复用可能改变模型结果，遗漏适用融合又会降低预期性能。模型适配 PR 必须保留逐项匹配和 fallback 证据，compile 图必须同时验证高层融合节点、primitive 未展开和 graph-break 不增加。

## 9. RFC 索引与实现边界

<a id="image-generation-public-contract-index"></a>

### 9.1 规范文档与实现交付索引

本 RFC 与两个模型 RFC 组成三份规范文档，不增加第四份公共基础设施 RFC。实现交付使用稳定的描述性名称，不以合入顺序编号：

| 实现交付 | 规范文档 | 前置依赖 | 独立性边界 |
| :--- | :--- | :--- | :--- |
| 图像生成公共架构实现 PR | [本 RFC：公共架构实现边界](#image-generation-public-implementation-boundary) | 无模型适配依赖 | 只实现公共 contract、静态 profile 组装 seam 和 fake-adapter 验证，不包含生产 profile。 |
| FLUX.1-dev 适配 PR | [FLUX.1-dev RFC：适配 PR 边界](./rfc_add_flux1_dev_support_zh.md#flux1-dev-adaptation-boundary) | 图像生成公共架构实现 PR | 不依赖 Qwen-Image-Edit profile、adapter、fixture 或测试。 |
| Qwen-Image-Edit 适配 PR | [Qwen-Image-Edit RFC：适配 PR 边界](./rfc_add_qwen_image_edit_support_zh.md#qwen-image-edit-adaptation-boundary) | 图像生成公共架构实现 PR | 不依赖 FLUX.1-dev profile、adapter、fixture 或测试。 |

模型适配实现继承的公共 contract 按以下章节索引：

| 公共 contract | 规范章节 |
| :--- | :--- |
| CLI 与参数校验 | [3.3 公共 CLI 与参数校验](#image-generation-public-cli-contract) |
| exact-profile、静态组装与 adapter | [3.4 静态 exact-profile 与 adapter contract](#image-generation-profile-adapter-contract) |
| 固定 timestep 与 N-step workload | [3.5 固定 dummy timestep 与 N 次 workload 迭代](#image-generation-workload-contract) |
| Guidance 与 true CFG | [3.6 Guidance 与 CFG 分支语义](#image-generation-guidance-cfg-contract) |
| `U × C` 拓扑与 critical path | [3.7 U × C 拓扑与 critical path](#image-generation-topology-contract) |
| generated/source output 所有权 | [3.8 Generated output span 与所有权](#image-generation-output-ownership-contract) |
| 结果与 Chrome trace | [3.9 结果与 trace contract](#image-generation-result-trace-contract) |
| 现有模块复用与修改边界 | [3.10 现有模块复用与修改边界](#image-generation-module-boundary) |

FLUX.1-dev RFC 只负责精确 `black-forest-labs/FLUX.1-dev` profile、pipeline/Transformer/config fingerprint、FLUX 专属 packing/IDs、embedded guidance、固定 dummy timestep shape/dtype、四类输入 sharding、Ulysses collective、output gather 和模型专属测试。Qwen-Image-Edit RFC 只负责登记的 Qwen Image Edit profile、original/Plus source cardinality、generated/source packing 和 prefix、`img_shapes`、固定 dummy timestep shape/dtype、true-CFG 特有行为、2511 config-driven 差异、U=1/CFG parallel 限制和模型专属测试。模型 RFC 可以提供各自公式和常量，但不得重复或改写上述公共 contract。

<a id="image-generation-public-implementation-boundary"></a>

### 9.2 图像生成公共架构实现 PR 边界

图像生成公共架构实现 PR 的建议范围：

```text
cli/main.py
cli/inference/image_generate.py
公共 image-generation contract/workload/CFG 编排模块
现有 resolver/builder/runtime/parallel 模块的最小必要 seam
公共 CLI、fake adapter、结果与回归测试
```

图像生成公共架构实现 PR 包含：

- `image-generate` 注册、module parser 和公共 validation；
- 静态 profile descriptor/lookup 与确定性组装 seam，但不含生产 profile；组装规则必须允许每个模型在模型独占模块中提供 descriptor，不得把共享可变 collection 留给多个模型适配 PR 分别修改；
- config-only selection 到现有 resolver/builder 的调用链；
- N 次 Transformer workload、固定 dummy timestep、true CFG、`U × C` groups、Runtime/dependencies；明确不包含 scheduler helper；
- generated span、stage coverage 和 result contract；
- test-only fake adapter 及视频不变回归；
- 既有 TensorCast 融合 seam 的最小公共调用顺序和测试约定：模型 wrapper/patch preparation 先于 `torch.compile`，compile replacement 继续复用现有 backend/pattern pipeline，编译后图证据由模型适配 PR 提供；公共架构实现 PR 只用 fake adapter 验证顺序，不实现具体模型融合 patch。

图像生成公共架构实现 PR 不包含 FLUX/Qwen 常量、模型专属 shape 公式、生产 model ID 或 Web UI 改动，也不新增 CLI 开关、稳定 Runtime 融合结果字段、动态 fusion registry、callback 或通用融合类型。真实模型 ID 在该 PR 中必须明确失败为“当前构建未注册该图像模型 profile”。该 PR 不新建动态 registry、provider discovery、scheduler hierarchy 或任意输出回调。任何公共组装规则或公共 contract 的后续修改都必须作为独立的公共架构变更交付，不得夹带在模型适配 PR 中。

### 9.3 FLUX.1-dev 适配 PR

[FLUX.1-dev 适配 PR](./rfc_add_flux1_dev_support_zh.md#flux1-dev-adaptation-boundary)新增 FLUX adapter/profile，并通过公共静态组装 seam 接入模型独占的 profile descriptor。它实现并测试 FLUX 专属 config fingerprint、meta inputs、packing、IDs、固定 dummy timestep shape/dtype、embedded/true guidance、四输入 Ulysses、attention collective、output gather 和 `U × C` topology；同时审计 FLUX forward 的既有 TensorCast 融合候选，逐项证明语义匹配，接入适用的现有 fusion seam，并提供 `--compile` 图证据确认高层融合节点未展开且未新增 graph break。它不修改公共组装规则、Qwen-Image-Edit 文件、公共 CLI、视频或 Web UI，也不因语义不匹配而强行融合。

### 9.4 Qwen-Image-Edit 适配 PR

[Qwen-Image-Edit 适配 PR](./rfc_add_qwen_image_edit_support_zh.md#qwen-image-edit-adaptation-boundary)新增 Qwen adapter/profile，并通过公共静态组装 seam 接入三个模型独占的精确 profile descriptor。它实现并测试 original/Plus source cardinality、effective-size metadata、generated/source packing、`img_shapes`、generated prefix、Qwen 固定 dummy timestep shape/dtype、true-CFG 语义和 2511 配置差异；同时审计共享 Transformer 路径的既有 TensorCast 融合候选，按数学/layout/dtype/epsilon/affine 与 RoPE 轴语义接入适用 fusion seam，并以 `--compile` 图验证高层融合节点、primitive 展开和 graph-break 结果。它默认只支持模型 RFC 声明的 U=1/CFG parallel，不修改公共组装规则、FLUX.1-dev 文件、公共 CLI、Runtime、视频或 Web UI；语义不适用的候选必须记录 fallback。

### 9.5 依赖关系与独立验收

依赖图只有一层：图像生成公共架构实现 PR 不依赖任何模型适配；FLUX.1-dev 适配 PR 与 Qwen-Image-Edit 适配 PR 分别只依赖已合入的公共架构 contract，二者互不依赖。两个模型适配 PR 应从公共架构实现的同一稳定 commit 建立，可并行开发和评审，并可在公共依赖满足后按任意顺序合入；各自 PR 描述应记录实际依赖的公共架构 commit。

独立验收要求如下：图像生成公共架构实现 PR 的测试在没有任何生产 profile 时通过；FLUX.1-dev 测试不得导入或要求 Qwen-Image-Edit profile、adapter、fixture 或测试存在；Qwen-Image-Edit 测试不得导入或要求 FLUX.1-dev profile、adapter、fixture 或测试存在。任一模型适配发现公共字段、组装规则或 contract 不足时，必须先提交独立的公共架构变更并更新本 RFC，再基于新的稳定 commit 更新模型适配；不得在模型适配 PR 中静默扩大公共范围。

图像生成公共架构实现 PR 的最小验收是：fake-adapter 测试全部通过；未知生产 profile 稳定失败；N 次 workload、N/2N forward、固定 dummy timestep、scheduler 未构造/未调用、CFG 顺序/并行 critical path、U×C 校验、Runtime breakdown、stage coverage 和视频回归全部通过；不下载任何模型权重。两个模型适配 PR 分别证明自身 exact profile、config-only/meta forward、专属 shape、parallel 和失败语义，并提交既有融合候选的 `applicable / adapted / not_applicable` 清单及 `--compile` 图证据：适用候选必须复用现有融合 seam，预期高层融合节点不得退化为 primitive 展开，也不得新增 graph break；不适用候选必须说明语义差异和 fallback。

## 10. 结论

本 RFC 采用独立图像 CLI、静态 exact-profile、窄 adapter contract 和明确的 measured/shape_only/excluded 阶段边界。它把可复用的 N 次 Transformer workload、固定 dummy timestep、CFG、并行和 Runtime 语义集中起来，同时把模型公式、packing、fingerprint 和真实 profile 留给后续模型 RFC。该设计在不改变 `video_generate` 的前提下，为图像生成 Transformer 去噪性能仿真提供可测试、可审计且不依赖权重和真实用户数据的公共基础。
