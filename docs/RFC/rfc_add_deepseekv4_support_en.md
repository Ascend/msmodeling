# RFC: DeepSeek-V4 Model Adaptation Support (Flash/Pro)

## Metadata

| Item | Content |
| :--- | :--- |
| **Status** | Approved |
| **Author** | — |
| **Creation Date** | 2026-06-03 |
| **Related Links** | [DeepSeek-V4 Adaptation Design](../design/deepseek_v4_adaptation_design.md) |

---

## 1. Overview

This proposal aims to add TensorCast support for DeepSeek-V4 (Flash/Pro) model, covering Head Compression (HC), layered KV compression, shared KV sparse attention, Lightning Indexer, and Hash Routing MoE, enabling proper model compilation and performance simulation.

---

## 2. Motivation

DeepSeek-V4 is the latest version of the DeepSeek series, introducing several key architectural upgrades compared to V3/V3.2:

1. **Head Compression**: Token-level information compression and restoration via Sinkhorn-based mixing, improving model expressiveness and inference efficiency.
2. **Layered KV Compression**: Per-layer configurable compression strategies (sliding window / indexed compression / heavy compression), balancing KV cache size and attention coverage.
3. **Shared KV Attention**: Uses single wkv projection to generate shared KV vectors, paired with grouped O projection to reduce computational complexity.
4. **Lightning Indexer**: Learned sparse attention selection, selecting critical KVs for attention computation in ratio=4 layers.
5. **Hash Routing MoE**: Token-id based hash routing instead of softmax top-k routing, achieving deterministic and efficient expert routing.

Without V4 adaptation, TensorCast cannot load and simulate this model, affecting the ability to evaluate performance of the latest DeepSeek architecture.

---

## 3. Goals

### 3.1 Goals

- [x] Support DeepSeek-V4 model loading and compilation
- [x] Support Head Compression (HC) performance modeling
- [x] Support layered KV compression strategies (ratio=0/4/128)
- [x] Support shared KV sparse attention
- [x] Support Lightning Indexer (ratio=4)
- [x] Support Hash Routing MoE
- [x] Support Clamped SwiGLU expert activation

### 3.2 Non-Goals

- No support for parallel combination testing with other models
- No support for KV cache block and indexer joint optimization
- No support for CP (Context Parallel) combined with V4 sparse attention

---

## 4. Use Case Analysis

### 4.1 DeepSeek-V4 Model Inference Simulation

**Scenario**: Simulate DeepSeek-V4-Flash/Pro model performance under different parallel configurations.

**Features**:

- Model loading: `AutoConfig` / `AutoModel` registration
- HC pre/post wrapping: Each decoder layer attention and FFN before/after
- Layered attention: Different attention paths for ratio=0/4/128
- MoE routing: Hash routing and non-hash routing

**Performance Indicators**:

- Prefill stage error < 30%
- Decode stage error < 20%

**DFX Requirements**:

- Compatibility: Support coexistence with V3.2 model profile
- Maintainability: V4-specific logic concentrated in isolated files
- Testability: Provide unit tests and integration tests coverage

### 4.2 V4-Specific Operator Performance Evaluation

**Scenario**: Evaluate computational and memory overhead of V4 new semantic operators.

**Features**:

- HC operators: Sinkhorn iterations, weighted reduction
- Compressor: Layered KV compression cost
- Lightning Indexer: Learned sparse selection cost
- Sparse Attention: Shared KV attention cost

---

## 5. Design

### 5.1 Overall Design

V4 adaptation is divided into five layers:

1. **Model Registration Layer**: `deepseek_v4` model profile registration
2. **HC Semantic Operator Layer**: 4 semantic operators for Head Compression
3. **KV Compression Operator Layer**: compressor and scatter_nd_update_mla
4. **Sparse Attention Operator Layer**: quant_lightning_indexer and sparse_attn_sharedkv
5. **MoE Routing Operator Layer**: moe_gating_top_k/hash and v4_clamped_swiglu

### 5.2 Key Design Decisions

#### 5.2.1 HC Semantic Operator Merge Strategy

`hc_pre_sinkhorn` merges Sinkhorn iterations and weighted reduction into one semantic operator, ensuring the cost model accounts for them together. Separating them would cause:

- Cost model unable to correlate Sinkhorn iterations and subsequent reduction dependency
- Performance estimation may underestimate combined kernel fusion benefits

#### 5.2.2 KV Write Functional Handle

`scatter_nd_update_mla` returns a functional handle instead of a data handle, ensuring torch.compile cannot perform DCE on the upstream KV producer chain. Design pattern is consistent with V3.2's `mlapo_quant` returning cache handle.

#### 5.2.3 Dynamic Topk Width

`quant_lightning_indexer` output width uses `min(topk_limit, active_seq_len)` instead of fixed `topk_limit`:

- Avoids decode stage topk exceeding actual available compressed sequence length
- Ensures performance estimation aligns with actual runtime behavior

#### 5.2.4 O Projection Independent TP Group

V4's O projection uses independent `o_proj_tp_group`, separable from attention's `tp_group`:

- Needs separate registration of `wo_a` and `o_proj` in TP plan
- `o_proj_tp_group.world_size` must be ≤ `o_groups`

### 5.3 Technology Selection

**Selected Approach**: New independent V4 semantic operators and performance models

**Alternative**: Reuse V3.2 operators with parameter control

Alternative not selected because:

- V4 HC mechanism is completely orthogonal to V3.2, cannot be parameterized
- V4 compressor and indexer semantics differ significantly from V3.2 DSA indexer
- Layered compression requires per-layer configuration support

### 5.4 Security, Privacy, and DFX Design

**Compatibility**:

- `deepseek_v4` as independent model_type, does not affect V3/V3.2 paths
- V4 Config subclasses `DeepseekV3Config`, reuses common fields

**Maintainability**:

- V4-specific logic concentrated in `deepseek_v4.py`
- Performance models concentrated in `builtin_model/deepseek_v4.py`
- Uses `_tensor_cast_patched` / `getattr` markers to prevent duplicate patches

**Testability**:

- Unit tests cover each semantic operator
- Integration tests cover end-to-end inference flow

### 5.5 Programming and Integration Design

#### 5.5.1 Model Loading Interface

```python
from tensor_cast.transformers.builtin_model.deepseek_v4 import DeepseekV4Config, DeepseekV4Model

# Load via HF AutoModel
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", config=config)
```

#### 5.5.2 V4-Specific Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compress_ratios` | List[int] | Required | Per-layer KV compression strategy |
| `layer_types` | List[str] | Optional | Per-layer attention type |
| `topk_limit` | int | Optional | Lightning indexer top-k limit |
| `num_hash_layers` | int | 0 | Hash routing MoE layer count |
| `hc_mult` | int | 4 | HC expansion factor |
| `hc_sinkhorn_iters` | int | 20 | Sinkhorn iteration count |
| `o_groups` | int | 1 | O projection group count |
| `o_lora_rank` | int | Optional | O projection lora rank |
| `score_func` | str | "sqrtsoftplus" | Routing score function |
| `swiglu_limit` | float | Optional | Clamped SwiGLU limit value |

#### 5.5.3 Semantic Operator Interfaces

**hc_pre_sinkhorn**

```python
torch.ops.tensor_cast.hc_pre_sinkhorn(
    x,                    # HC mix tensor [B,S,mix_hc]
    hidden_states,        # Original HC-expanded [B,S,Hc,D]
    hc_scale,             # Sinkhorn scale [3, Hc]
    hc_base,              # Sinkhorn base [3, Hc]
    hc_mult,              # HC expansion factor
    sinkhorn_iters,       # Sinkhorn iterations
    hc_eps,               # Epsilon for sinkhorn
) -> (reduced, post, comb)
```

**quant_lightning_indexer**

```python
torch.ops.tensor_cast.quant_lightning_indexer(
    q_states,             # RoPE-processed q [B,S,H,D]
    weights,              # weights_proj output [B,S,H]
    indexer_cache,        # Indexer KV cache
    topk_limit,           # Top-k limit
    tp_world_size,        # TP world size
    seq_lens,             # Per-request seq lengths
    query_lens,           # Per-request query lengths
) -> topk_indices
```

---

## 6. Test Design

### 6.1 Unit Tests

| Test Case | Verification Points |
|----------|---------------------|
| `test_v4_config_parsing` | compress_ratios validation, ratio range check |
| `test_hc_semantic_ops` | hc_pre_*/hc_post/hc_head appear in trace |
| `test_v4_attention_wrapper` | Different paths for ratio=0/4/128 |
| `test_lightning_indexer` | Dynamic topk width, indexer_cache layout |
| `test_moe_routing` | Hash routing and non-hash routing distinction |
| `test_clamped_swiglu` | v4_clamped_swiglu vs standard SiLU behavior |

### 6.2 Integration Tests

```bash
python -m pytest tests/test_tensor_cast/test_deepseek_v4.py -v
python -m pytest tests/test_tensor_cast/test_mla.py -v
python -m pytest tests/test_tensor_cast/test_moe_layer.py -v
```

### 6.3 End-to-End Verification

Use DeepSeek-V4 text generation compile command for end-to-end verification:

1. Model construction succeeds, HF `DeepseekV4Config` can be recognized by TensorCast
2. Compilation process no longer has HC-related attribute missing errors
3. HC semantic operators appear in trace
4. `quant_lightning_indexer` appears in ratio=4 layers
5. Existing model tests are unaffected

---

## 7. Drawbacks and Risks

### 7.1 Potential Risks

1. **Breaking Change**: V4's O projection uses independent TP group, may affect existing parallel configurations
2. **Performance Regression**: HC operators may introduce additional overhead at low hc_mult values
3. **Complexity Increase**: 11 new semantic operators and corresponding performance models

### 7.2 Mitigation Measures

1. Clearly declare `o_proj_tp_group` constraints in ModelProfile, provide clear error messages
2. Verify HC operator overhead at various hc_mult configurations through performance model
3. Unit tests cover all new operators, ensuring cost model accuracy

---

## 8. Existing Technology Reference

- **DeepSeek-V3.2 Adaptation**: GLM5 design doc contains detailed design of V3.2 DSA indexer; this proposal extends for V4-specific HC and layered compression
- **DeepSeek Official Implementation**: `deepseek-ai/DeepSeek-V4-Flash/inference/model.py` as reference implementation

---

## 9. Unresolved Questions

1. **KV Cache Block Joint Optimization**: Currently indexer returns topk_indices, but mechanism for loading KV cache blocks according to indexer results is not yet implemented
2. **CP + V4 Sparse Attention**: Context Parallel combined with V4 layered sparse attention is not yet supported

---

## Appendix

### References

- [TensorCast DeepSeek-V3.2 Adaptation](./rfc_add_deepseekv32_support_en.md)
- [GLM5 TensorCast Adaptation](../design/glm5-tensorcast-adaptation-design.md)

### Glossary

| Term | Description |
|------|-------------|
| HC | Head Compression, token-level compression via Sinkhorn mixing |
| Compressor | V4 KV compressor, generates coarse-grained KV stream in layers |
| Lightning Indexer | V4 learned sparse attention selection |
| Hash Routing | Token-id based MoE routing, replaces softmax top-k |
| Clamped SwiGLU | SwiGLU activation with gate clamping |
| Ratio | KV compression ratio, 0=sliding window, 4=indexed compression, 128=heavy compression |
