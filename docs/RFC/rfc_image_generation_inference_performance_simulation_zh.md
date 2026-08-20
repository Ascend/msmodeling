# RFC：图像生成推理性能仿真架构

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | 实现并合入 |
| **作者** | `minghang_c` |
| **创建日期** | 2026-08-06 |
| **更新日期** | 2026-08-13 |
| **相关 Issue/PR** | [Issue #314](https://gitcode.com/Ascend/msmodeling/issues/314) |

---

## 1. 摘要与冻结决策

本 RFC 冻结图像生成 Transformer 性能仿真的公共边界。入口是
`cli/inference/image_generate.py`，它直接拥有 CLI policy、配置与拓扑校验、模型选择、config-only 加载、输入准备、CFG 变换、Ulysses 分片/聚合、缓存生命周期、compile、Runtime、N-step workload 循环、计时、表格和 trace。可复用的 Diffusers/model technology 放在 `tensor_cast/diffusers/`，不建立新的图像公共 Python API。

首版只模拟 Transformer workload，不生成图片。它不执行 tokenizer、text encoder、真实图片读取、VAE encode/decode、scheduler、latent 数值更新、图片输出或质量评估。`--sample-step=N` 是 N 次使用固定 timestep kwargs 的相同 Transformer workload iteration，而不是 Diffusers `num_inference_steps`。

以下决策是规范性要求：

- 不新增 `tensor_cast/image_generation/`，未来删除旧的 `contract.py`、`profiles.py`、`orchestrator.py`、`runner.py`；不提供兼容 shim、re-export、稳定的 image Python API、通用 image engine 或动态 registry/plugin/provider discovery。
- 不定义 `ImageGenerationRequest`、`ImageExecutionConfig`、`ImageGenerationResult`、`ImageModelProfile`、`ImageProfileCollection`、`ImageGenerationAdapter`，也不定义 protocol、stage/result/ownership 对象。模型专属技术通过静态过程式分派函数完成。
- 核心 PR 不注册任何生产模型 kind；真实生产请求必须清晰失败。FLUX 和 Qwen 模型 PR 依赖核心 PR，并各自添加分支、模型 helper 和完整成功生命周期。
- `video_generate.py` 的路径、参数、默认值、缓存 fallback 和行为完全不变。

<a id="image-generation-public-contract-index"></a>

## 2. 公共实现边界与依赖

### 2.1 最终模块布局

```text
cli/inference/image_generate.py
  tensor_cast/diffusers/image_dispatch.py
  tensor_cast/diffusers/flux_image.py
  tensor_cast/diffusers/qwen_image_edit.py
  tensor_cast/diffusers/model_resolver.py
  tensor_cast/diffusers/diffusers_model.py
  tensor_cast/diffusers/dit_cache_registry.py
  tensor_cast/diffusers/cache_agent/
  tensor_cast/runtime.py
  tensor_cast/parallel_group.py
  tensor_cast/ops/internal.py
```

不新增 `tensor_cast/inference/`、CLI 子包或新的顶层 modality package。`image_generate.py` 是唯一的公共 policy/lifecycle 入口；`tensor_cast/diffusers` 只承载可复用的 Diffusers/model technology。模型 helper 的 source-faithful patch 或 wrapper 必须在 cache replacement 和 `torch.compile()` 之前完成；输入准备先于 model build 是有意的生命周期顺序。

### 2.2 公共与模型 PR

核心 PR 的交付范围：重写 image CLI，加入空的/unsupported 的静态 dispatch，加入 validated-config build seam 和 explicit cache-spec seam，删除旧 image package/tests，保留 generic Runtime/internal ops。核心 PR 不伪造生产模型、不提供成功的 fake lifecycle；核心测试覆盖 validation/failure、cache seam、video 与 Runtime regression。

FLUX PR 与 Qwen PR 只依赖核心 PR，可并行开发。它们分别添加模型 helper、dispatch branch、精确配置校验、完整成功 lifecycle、Chrome trace 与可工作的 cache wrapper；工作模型 cache wrapper 是完成门槛。

## 3. 公共 CLI 契约

<a id="image-generation-public-cli-contract"></a>

命令为：

```bash
python -m cli.inference.image_generate MODEL_ID [OPTIONS]
msmodeling inference image-generate MODEL_ID [OPTIONS]
```

保留的图像参数：

| 参数 | 规范语义 |
| :--- | :--- |
| `MODEL_ID` / `--model-id` | 本地 Diffusers 目录，或精确允许的 `(remote_source, model_id)`。位置参数与 `--model-id` 二选一；`--model_id` 为隐藏别名。 |
| `--batch-size B` | 正整数。普通 workload 的基础 batch，不是 prompt 数或源图数。 |
| `--output-image-size HEIGHT WIDTH` | 恰好一次，两个正整数；只用于推导 shape，不输出图片。 |
| `--text-seq-len N` | 正整数，实际进入 Transformer 的文本条件长度。 |
| `--source-image-size HEIGHT WIDTH` | 可重复；每项两个正整数，只接受尺寸，不接受路径或像素。仅 editing kind 可用。 |
| `--sample-step N` | 正整数，执行 N 次相同 Transformer workload iteration。 |
| `--use-cfg` | 启用 video-style CFG workload approximation。 |
| `--cfg-parallel` | 仅在 `--use-cfg` 下启用 CFG group 拓扑。 |
| `--num-devices`、`--ulysses-size` | 正式并行参数；`--world-size` 为隐藏兼容别名。 |
| `--chrome-trace-file` | 仅在 Runtime 成功后生成 Chrome trace；`--chrome-trace` 为隐藏兼容别名。 |
| `--device`、`--dtype` | 复用既有 device/dtype 语义。 |
| `--remote-source` | 远端来源；必须参与 exact pair 匹配。 |
| `--quantize-linear-action`、`--quantize-attention-action` | 复用既有量化参数。 |
| `--compile`、`--compile-allow-graph-break` | 复用既有 compile 参数。 |
| `--dit-cache` | 请求 DiT block cache。 |
| `--cache-step-range START,END` | genuine cache request 所需的 inclusive step range。 |
| `--cache-step-interval N` | cache interval；大于 1 才构成 genuine request。 |
| `--cache-block-range START,END` | block range，start-inclusive/end-exclusive。 |

完全删除 `--negative-text-seq-len`。负向分支使用与正向相同的 `--text-seq-len` shape；本模拟不暴露 CFG scale，不做数值 CFG combine。

公共校验必须在 Runtime 启动前完成：正整数与参数个数、editing source 限制、`cfg-parallel` 必须伴随 `use-cfg`、以及拓扑约束。`--cfg-parallel` 时 `world_size=2U`；非 cfg-parallel 时 `world_size=U`。FLUX 可支持 `U>1`，Qwen 首版仅 `U=1`，具体拒绝由 model-specific validation 完成。

<a id="image-generation-dispatch-contract"></a>

## 4. 静态 exact dispatch、模型选择与配置

### 4.1 支持的 model kind

规范中的 kind 只有：

- `flux1-dev`
- `qwen-image-edit`
- `qwen-image-edit-2509`
- `qwen-image-edit-2511`

核心 PR 不注册生产 kind，因此核心构建对这些请求也必须明确报告 unsupported，不能假成功。模型 PR 通过 `image_dispatch.py` 增加对应精确分支。

`image_dispatch.py` 只能使用静态、懒导入、过程式分派。kind-based 分派点固定使用以下 exact 形态；resolve 阶段先由 exact canonical identity/config 得到 kind，其余函数再按 kind 进入相同静态分支：

```python
if kind == "flux1-dev":
    from . import flux_image
    ...
elif kind == "qwen-image-edit":
    from . import qwen_image_edit
    ...
elif kind == "qwen-image-edit-2509":
    from . import qwen_image_edit
    ...
elif kind == "qwen-image-edit-2511":
    from . import qwen_image_edit
    ...
else:
    raise ValueError(...)
```

禁止 import-time registration、任意用户控制的 dynamic import、前缀 dispatch、模块-as-protocol、handler object、provider/plugin discovery 或可变 registry。

### 4.2 精确选择与配置校验

公共入口必须先完成 config-only 选择，再构造 Transformer。远端只接受精确 `(remote_source, model_id)`；不接受前缀、mirror、source fallback、未来 variant 或社区猜测。远端只下载配置，不下载权重，不执行 remote code。gated/auth/config-only 失败时，错误必须建议用户提供已获授权的本地 Diffusers 目录。

本地目录由 pipeline class、Transformer class 和显式配置字段 whitelist 唯一选择候选 family。候选数为零、多个或 mismatch 都失败；class name alone never proves support。scheduler config 从不参与接受/拒绝，也不构造 scheduler。每个配置错误必须包含 path、expected、actual。

必须有最小 build seam：

```python
def build_diffusers_transformer_from_config(
    model_id: str,
    model_config: DiffusersConfig,
) -> DiffusersTransformerModel:
    ...
```

该 seam 确保 `resolve_image_model_kind` 与严格配置校验先于 model construction。任何 source-faithful model patch/wrapper 都在 cache replacement、`torch.compile()` 之前运行。

精确函数契约如下：

```python
def resolve_image_model_kind(
    model_id: str,
    remote_source: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> str: ...

def validate_image_config(
    kind: str,
    model_selection: DiffusersModelSelection,
    model_config: DiffusersConfig,
) -> None: ...


def prepare_image_model(
    kind: str,
    model: DiffusersTransformerModel,
    model_config: DiffusersConfig,
) -> DiffusersTransformerModel: ...
```

## 5. 图像 workload 与 dispatch 函数

<a id="image-generation-workload-contract"></a>
<a id="image-generation-cfg-contract"></a>

可复用技术函数的精确契约为：

```python
def prepare_image_inputs(
    kind: str,
    model_config: DiffusersConfig,
    *,
    batch_size: int,
    output_image_size: tuple[int, int],
    text_seq_len: int,
    source_image_sizes: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], int]: ...

def apply_image_cfg(
    kind: str,
    inputs: dict[str, object],
    *,
    batch_size: int,
    use_cfg: bool,
    cfg_parallel: bool,
) -> dict[str, object]: ...

def shard_image_inputs(
    kind: str,
    model_config: DiffusersConfig,
    inputs: dict[str, object],
    *,
    ulysses_size: int,
) -> tuple[dict[str, object], int | None]: ...

def forward_image_model(
    kind: str,
    model: DiffusersTransformerModel,
    inputs: dict[str, object],
    *,
    generated_token_count: int,
) -> torch.Tensor: ...

def image_cache_spec(
    kind: str,
    model_config: DiffusersConfig,
) -> DiTBlockCacheSpec: ...
```

这些函数均由 `image_dispatch.py` 做 exact lazy dispatch；它们不是公共稳定 API。`image_generate.py` 负责调用顺序、生命周期、迭代和结果打印，模型 helper 只负责可复用的 Diffusers/model technology。

每一步使用已在 kwargs 中固定的 timestep。`--sample-step=N` 执行 N 次相同 Transformer forward workload；不创建 timetable、sigma、scheduler 或 latent update。每种模式每步只有一个代表性 forward：

- 无 CFG：普通 batch 为 `B`，每步一次 forward。
- 普通 CFG：普通 batch 为 `2B`，每步一次 forward；两个相同 `text_seq_len` 的 half 由模型专属 duplication 构造。不得用 generic `shape[0] == B` 推导。
- CFG parallel：这是 single-rank representative simulation；local `B` forward 代表两个相同 shape 的并发 branch critical path。模拟器不声称每个 physical rank 都执行 forward，也不得顺序执行两次 branch。

这里的 CFG 是 video-style CFG workload approximation；它明确拒绝历史上的 true CFG 语义：不做数值 combine/scale，不调用 scheduler，不推进 latent。`--sample-step=N` 的每一步在所有模式都只计一个 representative forward。

## 6. 拓扑、CFG 与 Ulysses

<a id="image-generation-topology-contract"></a>

令 `U=ulysses_size`。无 cfg-parallel 时 world 为 `U`；cfg-parallel 时 world 为 `2U`。CFG-parallel 的 rank 和 group 固定为：

```text
positive branch: ranks [0, ..., U-1]
negative branch: ranks [U, ..., 2U-1]
cfg group(u): [u, U+u]
```

Ulysses gather 必须先发生，CFG-group `all_gather` dim 0 后发生，以得到 `2B` workload shape。CFG-parallel 是 single-rank representative simulation：local `B` forward 代表两个相同 shape 的并发 branch，模拟器只保留 representative rank state，不声称每个 physical rank 都执行 forward，也不得把两个 branch 作为两个顺序 Transformer 调用。FLUX 的 Ulysses sharding/gather 属于 FLUX RFC；Qwen `U=1` only 属于 Qwen RFC。

## 7. Cache 契约

<a id="image-generation-cache-contract"></a>

缓存通过既有模型能力扩展，精确契约为：

```python
@dataclass(frozen=True)
class DiTBlockCacheSpec:
    class_name: str
    model_type: str
    get_blocks_with_setters: GetBlocksWithSetters
    make_wrapped_forward: MakeWrappedForward


def enable_dit_block_cache(
    self,
    cache_config: CacheConfig,
    spec: DiTBlockCacheSpec | None = None,
) -> CacheState | None:
    ...
```

`spec=None` 必须保留既有 video/global registry behavior 和 unsupported-video fallback。显式 image spec 必须严格校验 `class_name`、block discovery、block range 和 replacement；任何失败都不能 fallback。既有 video specs 获得 `class_name` identity，但不改变 lookup behavior。

只有同时满足 `--dit-cache` 且 `cache-step-interval > 1` 才是 genuine cache request：

1. interval `<=1` 直接禁用 cache，不解析/要求 ranges，不构造 spec，不构造第二模型。
2. genuine request 必须有 `--cache-step-range START,END`；缺失、负数或 `START > END` 都失败。范围 inclusive；仅当 END 大于 `sample_step-1` 时向下 clamp，合法的较早窗口保持原值；clamp 后为空失败，`start=end` 合法。
3. `--cache-block-range` 为 start-inclusive/end-exclusive；省略表示全部；START/END 必须非负且 `START <= END`；END 可 clamp；为空或 start=end 失败；replacement 为零失败。
4. 从同一份已校验配置构造 baseline model 与第二个 cache model；对两者应用同一 source-faithful patch/wrapper，只在 cache model 上替换 blocks，然后分别 compile。顺序固定为 patch/wrapper、cache replacement、`torch.compile()`。
5. 状态是 per-run。窗口内选择 cache model，窗口外选择 baseline model：

```text
in_window = start <= step <= end
active_model = cache_model if in_window else baseline_model
reuse = in_window and ((step - start) % interval != 0)
```

普通 CFG 只保留一个 `2B` state；cfg-parallel simulator 只保留 representative rank state。

## 8. 运行时输出与阶段范围

<a id="image-generation-output-contract"></a>

输出沿用 video style，而不是结构化 image result。成功运行必须打印计时和 Runtime 表；请求 trace 时才导出并打印路径：

```python
print(f"Model compilation execution time: {run_end - run_start}s")
print(runtime.table_averages(group_by_input_shapes=False))
if chrome_trace:
    runtime.export_chrome_trace(chrome_trace)
    print(f"Chrome trace written to: {chrome_trace}")
```

Chrome trace 仅在 Runtime 成功后按请求生成。不得输出 image JSON/profile/stage/ownership object、scheduler 字段、generated path、quality 或 end-to-end latency。

Runtime 时间只表示 Transformer workload 的模拟执行时间。以下内容永远排除：tokenizer/text encoder、实际 image input、VAE、scheduler、latent update、output image、quality、weights 和 end-to-end latency。shape-only 输入只服务于 shape/operator graph，不宣称已执行图像生成。

## 9. 模块边界与模型职责

<a id="image-generation-module-boundary"></a>

`image_generate.py` 必须拥有：parser、generic validation、topology validation、model selection、config-only load、kind resolution、model-specific validation coordination、quantization、parallel config、input preparation、CFG transformation、Ulysses sharding/gather、cache lifecycle、compile、Runtime、N-step loop、timing/table/trace。

`tensor_cast/diffusers` 只拥有 Diffusers/model technology：配置读取、validated-config build seam、FLUX/Qwen 输入布局与 wrapper、dispatch、cache spec。不得把 policy/lifecycle 重新放入 image technology module。

模型 kinds 的职责：

- `flux1-dev`：exact canonical FLUX model identity/config fingerprint、config whitelist、模型输入准备、`U>1` sharding/gather、forward 和 cache spec。
- 三个 Qwen kinds：各自 exact model identity/config fingerprint、source cardinality/config whitelist、输入准备、`U=1` 限制、forward 和 cache spec。三种 kind 不能由前缀推断。

## 10. 替代方案（明确拒绝）

拒绝通用 image engine、稳定 image Python API、compatibility shim、公共 request/result/profile/adapter objects、动态 registry、插件/provider discovery、任意 handler、类名或 ID 前缀自动支持、scheduler abstraction、真实图片/权重/文本端到端执行，以及复用 video_generate 改写其行为。这些方案会扩大 public surface、隐藏模型差异或改变既有 video contract。

## 11. 验收与测试

核心 PR 必须验证：

- 所有 CLI 参数，尤其 `--negative-text-seq-len` 不存在，cache flags 和 image names 存在。
- zero/multiple/mismatch 的 pipeline、Transformer、explicit config whitelist、exact remote pair、gated/auth 失败；错误含 path/expected/actual。
- 核心构建没有生产 kind，生产请求清晰 unsupported；没有成功 fake lifecycle。
- `--sample-step=N` 恰好 N 次 representative forward；没有 scheduler 构造、scheduler step、timetable 或 latent update。
- baseline 与独立 cache model 都通过 `prepare_image_model`，且 model-specific patch/wrapper 先于 cache replacement 与 `torch.compile()`。
- 无 CFG 为 B、普通 CFG 为 2B 单 forward；cfg-parallel 为代表 rank 单 forward，且不顺序执行两个 branch。
- `U×C` topology、group formulas、Ulysses gather then CFG all-gather 顺序；Qwen `U>1` 由 Qwen PR 负责，不属于核心 PR 的 model-specific test floor。
- 删除旧的 `test_image_generation*.py`；覆盖 local no-match/ambiguous、cache interval/range/block-range 的所有边界、baseline+second model、replacement-before-compile、per-run state 和 active model selection。
- 显式 cache spec 的 class/no-block/empty-range/zero-replacement failure，以及 `enable_dit_block_cache(spec=None)` 的 video fallback regression、video CFG/cache 和 Runtime multistream regression。
- `runtime.table_averages(group_by_input_shapes=False)`、`print(f"Model compilation execution time: {run_end - run_start}s")`、成功 Runtime 后 trace；失败不生成成功 image result。
- `video_generate.py` 的路径、参数、默认值、cache fallback 和行为回归不变。

FLUX 与 Qwen PR 另外必须证明各自 config-only/meta forward、完整成功 lifecycle、Chrome trace、model-specific sharding/validation 及可工作的 cache wrapper。真实模型验证不得下载权重或依赖实际图片输出。

## 12. 交付关系

<a id="image-generation-public-implementation-boundary"></a>

交付关系如下：

| 交付 | 依赖 | 内容 |
| :--- | :--- | :--- |
| 图像生成核心 PR | 无 | CLI、静态空/unsupported dispatch、validated-config build seam、cache seam、failure/cache/video/Runtime tests；不注册生产 kind。 |
| FLUX PR | 核心 PR | `flux1-dev` branch、helper、完整成功 lifecycle、Ulysses 与 cache wrapper。 |
| Qwen PR | 核心 PR | 三个 Qwen branches、helper、完整成功 lifecycle、`U=1`/CFG parallel 与 cache wrapper。 |

模型 PR 可并行，互不依赖；若发现公共 contract 不足，必须先修改核心 RFC/核心 PR，不得在模型 PR 中静默扩大公共范围。

## 13. 结论

本 RFC 冻结一个只面向 Transformer workload 的图像 CLI：policy/lifecycle 集中在 `image_generate.py`，Diffusers 技术集中在 `tensor_cast/diffusers`，模型通过 exact lazy procedural dispatch 接入。核心实现不提供生产模型或 image public API；FLUX 与 Qwen 以独立 PR 增加精确 kind 和完整成功路径。CFG、Ulysses、cache、config-only construction、Runtime 输出和 video regression 均有明确边界，且不把排除的端到端图像生成工作伪装成性能结果。
