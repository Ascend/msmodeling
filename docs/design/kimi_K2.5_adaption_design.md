# 特性设计：TensorCast 适配 Kimi K2.5 模型

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 | RFC 文档 |
| -- | -- | -- | -- | -- |
| 2026-05-21 | 1.0 | 初稿完成，归档 TensorCast 适配 Kimi K2.5 模型的设计与实现 | 王燊（30062558） | — |

---

## 1. 背景描述

Kimi K2.5（`moonshotai/Kimi-K2.5`）是 Moonshot AI 发布的多模态大语言模型，具有 Vision-Language（VL）能力、Mixture-of-Experts（MoE）路由和 Multi-head Latent Attention（MLA）机制。该模型并非 HuggingFace Transformers 官方维护，属于**社区远端模型**（需 `trust_remote_code=True` 加载），其架构与命名约定在多处与 TensorCast 既有路径存在差异。

如果直接尝试在 TensorCast 中加载并编译 Kimi K2.5，会遇到以下核心问题：

1. **Transformers 环境兼容**：Kimi K2.5 依赖 `is_torch_fx_available`（transformers v5.x 已移除）、`flash_attention_2`（未安装环境无法使用），且 Windows 平台缺少 `signal.SIGALRM` 导致 `trust_remote_code` 交互式弹窗阻塞。
2. **Vision Config 字段缺失**：Kimi K2.5 的 vision config 使用 `merge_kernel_size` 命名，但 `input_generator` 期望 `spatial_merge_size`、`temporal_patch_size`、`in_channels` 等标准字段，缺失导致 `AttributeError`。
3. **VL Forward 接口差异**：TensorCast 通过 `model_runner` 注入额外 kwargs（如 `attention_meta`），但 Kimi K2.5 原始 VL forward 仅接受标准 HF 参数名；`image_grid_thw` 与 `grid_thws` 参数名不一致。
4. **Meta Device 图捕获**：`torch.compile` 图追踪时 `input_ids` 位于 `meta` 设备上，原始 `_merge_input_ids_with_image_features` 内部调用 embedding 层会在 meta tensor 上失败。
5. **视觉编码器适配**：`MoonViT3dEncoder` 缺少 `use_deterministic_attn` 属性，且需要 `tensor_cast` attention backend 注册以在非纯 eager 模式下正确执行（避免长序列 O(n²) 计算）。
6. **MoE 路由不可追踪**：原始 `DeepseekV3MoE.forward` 包含动态 token 派发逻辑，`torch.compile` 无法追踪；`MoEGate.forward` 包含非确定性采样。
7. **MLA RoPE 解析**：Kimi K2.5 decoder 仅传递 `position_ids` 而非预计算 RoPE (cos, sin) 张量，TensorCast MLA 需要显式 `position_embeddings`。
8. **Decoder Layer 适配**：原始 decoder 从 `self_attn` 解包 3 个值，但 TensorCast MLA wrapper 返回 2 个值（不含 attention weights）。
9. **Patch Embed 维度适配**：仿真场景中视觉 token 可能以 2D 扁平 tensor 传入，原始 Conv2d projection 期望 4D 输入。
10. **MoE Expert 计数**：`n_routed_experts` / `n_shared_experts` 存储在 `text_config` 内而非根 config 上，`patch_moe` 读取根 config 会抛出 `AttributeError`。

本特性的目标是让 TensorCast 能够正确编译和仿真 Kimi K2.5 模型，覆盖 VL 多模态流水线、MLA attention、MoE 专家路由和视觉编码器的性能模拟。设计上遵循最小侵入原则：所有适配补丁集中在 `kimi_k25.py` 内，通过 `model_type != "kimi_k25"` 门控实现严格隔离，不影响任何已有模型的加载和仿真路径。

---

## 2. 方案设计

### 2.1 整体设计思路

本方案将 Kimi K2.5 适配拆分为三个层次：

1. **环境兼容层（跨模型全局修复）**：
   - Windows SIGALRM 修复位于 `tensor_cast/transformers/utils.py`，模块级自动执行
   - 因为这是所有远端模型在 Windows 上的共性问题，不宜放在单模型适配文件内

2. **配置修补 + 类 Monkey-Patch 层**：
   - 所有 Kimi K2.5 特有逻辑集中在 `tensor_cast/transformers/builtin_model/kimi_k25.py`
   - 分为两个阶段执行：Phase 1（config 层修补）→ Phase 2（类的 monkey-patch）
   - 通过 `hf_config_patch_method` 在模型加载前注入到 `ModelProfile` 注册表中

3. **ModelProfile 注册层**：
   - 在 `kimi_k25.py` 模块底部注册 `ModelProfile`
   - 声明 MoE 模块名、MLA 模块名、专家计数字段、视觉路径等元数据
   - 关键字段：`moe_route_after_dp_transform=True`（DP≠EP 时路由在 DP 切片后执行）

### 2.2 全局 Windows SIGALRM 修复

`tensor_cast/transformers/utils.py` 在模块加载时执行 `_ensure_windows_trust_remote_code_compat()`：

- 检测 `signal.SIGALRM` 是否存在
- 若不存在（Windows），monkey-patch `transformers.dynamic_module_utils.resolve_trust_remote_code`
- 当 `trust_remote_code is None` 时默认设为 `True`，跳过交互式弹窗
- 通过 `_tensor_cast_patched` 标记防重复 patch
- Unix 平台不受影响，直接 return

### 2.3 Phase 1：配置层修补（`_patch_hf_config_for_kimi_k25`）

在模型加载前修改 HuggingFace config 对象和全局 import 状态，不依赖 `model_id`。

| Patch | 名称 | 问题 | 修复方式 |
|-------|------|------|----------|
| P1 | `is_torch_fx_available` 恢复 | transformers v5.x 移除了此函数，Kimi K2.5 代码依赖它 | `importlib.util.find_spec("torch.fx")` 实现并注入 |
| P2 | `flash_attention_2` 降级 | flash_attn 未安装时无法使用 | 检测后将 text/vision config 的 `_attn_implementation` 改为 `"tensor_cast"` |
| P3 | Vision Config 属性桥接 | 字段命名/缺失不兼容 | `merge_kernel_size[0]` → `spatial_merge_size`；注入 `temporal_patch_size=1`、`in_channels=3` |

### 2.4 Phase 2：类 Monkey-Patch 层（`_patch_model_classes_for_kimi_k25`）

通过 `get_class_from_dynamic_module` 动态导入远端模型类，在类级别注入修补方法后，HF loader 再实例化模型对象。

| Patch | 目标类 | 问题 | 修复方式 |
|-------|--------|------|----------|
| P4 | `KimiK25ForConditionalGeneration.forward` | 不接受 TensorCast 注入的额外 kwargs；参数名 `image_grid_thw` vs `grid_thws` 不一致 | 过滤 kwargs 到标准 HF 键集合；自动完成 `image_grid_thw` → `grid_thws` 映射 |
| P5 | `_merge_input_ids_with_image_features` | meta 设备上调用 embedding 层失败 | 检测 `device.type == 'meta'` 时直接返回同 shape 的 meta tensor |
| P6 | `MoonViT3dEncoder` | 缺少 `use_deterministic_attn` 属性；无 `tensor_cast` backend | 注入属性为 `False`；注册 `tensor_cast` 视觉注意力适配函数（支持 meta 设备和超过 4096 长度的安全降级） |
| P7 | `DeepseekV3MoE.forward` / `moe_infer` | 动态 token 派发无法被 trace | 替换为 `zeros_like` stub，真实语义由 `patch_moe` 的 MoELayer wrapper 接管 |
| P8 | `MoEGate.forward` | 非确定性 top-k 采样 | 替换为等权重确定性路由（`topk_weight = 1/top_k`） |
| **P9** | `MultiheadLatentAttentionTensorCast._resolve_position_embeddings` | Kimi decoder 仅传 `position_ids`，TensorCast MLA 需要显式 (cos, sin) | Monkey-patch 在 MLA 类上添加从 `position_ids` 计算 RoPE 的方法（避免污染通用 `mla.py`） |
| **P10** | `DeepseekV3DecoderLayer.forward` | 原始解包 3 值但 MLA wrapper 返回 2 值；缺少 RoPE；TensorCast kwargs 被 P4 过滤丢失 | 兼容 2/3 值解包；lazy-init `_has_rotary_emb`；从 `_extra_forward_kwargs` 恢复被过滤的 kwargs |
| **P11** | `MoonVision3dPatchEmbed.forward` | 仿真时 2D 扁平输入无法被 Conv2d 处理 | 2D 输入时按 `grid_thws` 切分、reshape 为 4D patches、用 linear projection 替代 Conv2d |
| **P12** | 根 config 专家计数 | `n_routed_experts` / `n_shared_experts` 在 `text_config` 内 | 从 `text_config` 复制到根 config；提供缺省 fallback (384 / 1) |

### 2.5 ModelProfile 注册

```python
ModelProfile(
    model_type="kimi_k25",
    moe_module_name="DeepseekV3MoE",            # MoE 模块类名
    mla_module_name="DeepseekV3Attention",       # MLA 模块类名
    moe_num_experts_key="n_routed_experts",      # 专家计数配置键
    language_layers_path_str="language_model.model.layers",
    visual_module_path="vision_tower",
    language_module_path="language_model",
    visual_layers_module_path="vision_tower.encoder.blocks",
    visual_layers_path_str="vision_tower.encoder.blocks",
    custom_expert_module_type=None,               # 不自定义专家拆分
    mla_field_names_override={                    # MLA 字段名映射
        "q_proj": "q_a_proj",
        "qk_head_dim": "q_head_dim",
    },
    hf_config_patch_method=_hf_config_patch_for_kimi_k25,  # 前置补丁入口
    moe_route_after_dp_transform=True,            # DP≠EP 时路由在 DP 切片后执行
)
```

### 2.6 适配逻辑分层架构

```text
┌────────────────────────────────────────────────────────────┐
│                 tensor_cast/transformers/                   │
├────────────────────────────────────────────────────────────┤
│  utils.py                           ← 全局共性问题           │
│  └─ _ensure_windows_trust_remote_code_compat()  (SIGALRM)  │
├────────────────────────────────────────────────────────────┤
│  custom_model_registry.py            ← 框架通用注册表        │
│  └─ ModelProfile.hf_config_patch_method  (回调入口)          │
├────────────────────────────────────────────────────────────┤
│  builtin_model/kimi_k25.py           ← Kimi K2.5 特有适配   │
│  └─ Phase 1: _patch_hf_config_for_kimi_k25() (P1-P3)      │
│  └─ Phase 2: _patch_model_classes_for_kimi_k25() (P4-P12)  │
│  └─ ModelProfile 注册                                      │
└────────────────────────────────────────────────────────────┘
│  isolation gate:  if model_type != "kimi_k25": return      │
└────────────────────────────────────────────────────────────┘
```

### 2.7 关键设计决策

**1. P9（MLA RoPE 解析）从 `mla.py` 迁移到 `kimi_k25.py`**

最初 `_resolve_position_embeddings` 方法被添加在通用 `layers/mla.py` 的 `MultiheadLatentAttentionTensorCast` 类中，但该方法仅服务于 Kimi K2.5（因其 decoder 只传 `position_ids` 而不传显式 RoPE 张量）。为了保持通用 MLA 层不被模型特有逻辑污染，最终将该方法以 monkey-patch 形式移至 `kimi_k25.py` 的 P9，通过 `hasattr` 检查注入到类上，同时从 `mla.py` 中彻底移除了该方法。

**2. SIGALRM 修复从 `kimi_k25.py` 上提至 `utils.py`**

最初 SIGALRM 修复（原 P3）位于 `kimi_k25.py` 的 Phase 1 中，但它是所有 `trust_remote_code=True` 远端模型在 Windows 上的共性问题。将其上提至 `tensor_cast/transformers/utils.py` 模块级执行，避免每个远端模型适配文件都需要重复处理。

**3. `moe_route_after_dp_transform=True`**

当数据并行度（DP）不等于专家并行度（EP）时，Kimi K2.5 的路由操作需要在 DP 切片**之后**执行，否则会在路由阶段对全量 token 执行导致性能膨胀。该字段已在 `custom_model_registry.py` 中新增为 `ModelProfile` 标准字段。

---

## 3. 使用说明

### 3.1 运行环境

- Python ≥ 3.10（官方推荐 3.13）
- 需要 Hugging Face 连接（国内建议 `$env:HF_ENDPOINT = "https://hf-mirror.com"`）
- `PYTHONPATH` 需包含项目根目录
- Windows 用户建议 PyTorch ≤ 2.8（规避兼容性问题）

### 3.2 仿真命令

注：cli.inference.text_generate 中“设备数”参数名为 num-devices ，tensor_cast.scripts.text_generate 中“设备数”参数名为 world-size

**纯文本推理仿真**（W4A8 动态量化 + TP8/EP16/DP2）：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K2.5" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-queries 24 \
  --query-length 1 \
  --context-length 4250 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --num-devices 16 \
  --tp-size 8 \
  --dp-size 2 \
  --ep-size 16 \
  --enable-shared-expert-tp
```

**多模态推理仿真**（带图像输入-目前没有实测数据，无法保证精度，后续等实测数据到了同步更新文档）：

```bash
python -m cli.inference.text_generate "moonshotai/Kimi-K2.5" \
  --device ATLAS_800_A3_560T_128G_DIE \
  --num-queries 24 \
  --query-length 1 \
  --context-length 4250 \
  --image-batch-size 1 \
  --image-height 1080 \
  --image-width 1920 \
  --compile \
  --quantize-linear-action W4A8_DYNAMIC \
  --num-devices 16 \
  --tp-size 8 \
  --dp-size 2 \
  --ep-size 16 \
  --enable-shared-expert-tp
```

### 3.3 Trace 与性能分析结果

Kimi K2.5 的 trace 表应体现以下关键语义块：

- `tensor_cast.mlapo_quant.default` — MLA projection-out 融合 + 量化
- `tensor_cast.all_to_all.default` — EP 通信（ep_size=16）
- `tensor_cast.multihead_latent_attention_quant.default` — MLA attention 量化
- `aten.addmm.default` — 视觉编码器计算（VL 场景）

### 3.4 配置约束

1. Kimi K2.5 必须使用 `trust_remote_code=True`（Windows 上自动处理，Unix 需手动确认或传参）。
2. `flash_attention_2` 未安装时自动降级为 `tensor_cast` attention 实现。
3. 视觉编码器长序列（>4096）自动退回 `zeros` 占位，防止 OOM。
4. `n_routed_experts` 实际值从远端 config 读取，fallback 值为 384。
5. 所有 Monkey-Patch 通过 `_patched_*` / `getattr` 标记防重复，仅首次加载时执行。

---

## 4. 测试设计

### 4.1 单元测试

测试文件：`tests/test_tensor_cast/test_kimi_k25.py`，包含四个场景覆盖文本推理和 VL 推理：

| 用例 | 场景 | 验证要点 |
|------|------|----------|
| `test_kimi_k25_text_only_generation` | 文本推理（decode）+ 复杂并行策略 | 模型正常加载 + 仿真完成；trace 中包含 `mlapo_quant` + `all_to_all` |
| `test_kimi_k25_vision_language_generation` | VL 推理 + 图像输入 | `is_vl_model=True`；`pixel_values` 正常生成；权重含 vision tower |
| `test_kimi_k25_long_context_text_generation` | 长文本推理（4500 token） | KV cache 非零；`mlapo_quant` + `all_to_all` 正常 |
| `test_kimi_k25_long_context_vision_language_generation` | 长文本 VL 推理 | 综合 VL + long context 验证；权重含 vision tower；KV cache 正常分配 |

运行命令：

```bash
# 国内需先设置 HF 镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 跑全部 Kimi 测试
python -m pytest tests/test_tensor_cast/test_kimi_k25.py -v

# 单用例调试
python -m pytest tests/test_tensor_cast/test_kimi_k25.py::TestKimiK25::test_kimi_k25_text_only_generation -v
```

### 4.2 集成测试

除 Kimi 专用测试外，以下已有测试确保 Adapt 不影响其他模型：

```bash
python -m pytest tests/test_tensor_cast/test_deepseek_v32.py -v
python -m pytest tests/test_tensor_cast/test_vl_compile.py -v
python -m pytest tests/test_tensor_cast/test_mla.py -v
```

### 4.3 成功标准

1. 模型构建成功，HF `KimiK25Config` 可被 TensorCast 识别。
2. 文本推理场景 算子 中包含 `tensor_cast.mlapo_quant`、`tensor_cast.all_to_all`。
3. 仿真耗时结果的精度符合预期：prifill阶段误差<30%，decode阶段误差<20%，误差的比对对象为实测profilling数据
4. 已有模型（DeepSeek-V3.1/V3.2、Qwen3-VL、GLM-4.5V 等）测试不受影响。
