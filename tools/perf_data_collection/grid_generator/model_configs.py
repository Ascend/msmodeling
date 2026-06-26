"""Target model architecture configs for theory-guided shape grid generation.

Each entry describes the key architectural parameters needed to derive
all operator shapes. Used by `generate_shape_grid.py --target-models ...`
to prune the GEMM (N,K) cartesian product to only model-relevant pairs.

Reference: docs/RFC/rfc_performance_database_collection_tooling_zh.md.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Architectural parameters for one LLM."""

    name: str
    hidden_size: int
    intermediate_size: int  # FFN intermediate (or shared-expert for MoE)
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int = 128

    # MLA specific parameters (DeepSeek-V3/V2)
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 64

    # MoE parameters (0 means dense model)
    num_experts: int = 0
    num_experts_per_card: int = 0
    expert_intermediate_size: int = 0
    topk: int = 0

    # Parallel configs to enumerate
    tp_sizes: tuple[int, ...] = (1, 4, 8)
    ep_sizes: tuple[int, ...] = (1,)
    model_key: str = ""

    def is_mla(self) -> bool:
        """Check if this model uses MLA (Multi-head Latent Attention)."""
        return self.q_lora_rank > 0 or self.kv_lora_rank > 0

    def matmul_nk_pairs(self) -> set[tuple[int, int]]:
        """Return all (N, K) pairs this model's MatMul ops can produce."""
        pairs: set[tuple[int, int]] = set()
        h = self.hidden_size
        inter = self.intermediate_size

        for tp in self.tp_sizes:
            ht = max(1, h // tp)
            # Standard Attention Projections
            if not self.is_mla():
                # QKV projection: (h/tp, h)
                n_qkv = ht
                pairs.add((n_qkv, h))
                # Output projection: (h, h/tp)
                pairs.add((h, ht))
            else:
                # MLA Projections (DSv3 style)
                # 1. Q compression: (q_lora_rank, h)
                pairs.add((self.q_lora_rank, h))
                # 2. Q up-projection: (h_heads * qk_head_dim / tp, q_lora_rank)
                # Note: head_dim in MLA usually refers to latent dim or qk_head_dim
                q_up = max(1, (self.num_attention_heads * self.head_dim) // tp)
                pairs.add((q_up, self.q_lora_rank))
                # 3. KV compression: (kv_lora_rank + qk_rope_head_dim, h)
                pairs.add((self.kv_lora_rank + self.qk_rope_head_dim, h))
                # 4. KV up-projection: (h_heads * d_v_heads / tp, kv_lora_rank)
                # For DSv3, d_v_heads is also head_dim (128)
                pairs.add((q_up, self.kv_lora_rank))

            # Gate/Up: (inter/tp, h) — for dense FFN or shared expert
            if inter > 0:
                n_gate = max(1, inter // tp)
                pairs.add((n_gate, h))
                # Down: (h, inter/tp)
                pairs.add((h, n_gate))

        return pairs

    def expert_nk_pairs(self) -> set[tuple[int, int]]:
        """Return (N, K) for MoE expert GEMM (GroupedMatmul).

        Note: EP (Expert Parallelism) splits experts across cards but does NOT
        change weight dimensions (N, K), so ep_sizes is not iterated here.
        """
        if self.num_experts == 0:
            return set()
        pairs: set[tuple[int, int]] = set()
        h = self.hidden_size
        ei = self.expert_intermediate_size

        for tp in self.tp_sizes:
            ht = max(1, h // tp)
            eit = max(1, ei // tp)
            # Expert gate/up: (ei/tp, h/tp)
            pairs.add((eit, ht))
            # Expert down: (h/tp, ei/tp)
            pairs.add((ht, eit))

        return pairs


# ── Built-in model configs ────────────────────────────────────

GLM51_CONFIG = ModelConfig(
    name="GLM-5.1",
    hidden_size=6144,
    intermediate_size=12288,
    num_attention_heads=64,
    num_kv_heads=64,  # MLA latent heads
    head_dim=256,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    num_experts=256,
    num_experts_per_card=32,
    expert_intermediate_size=2048,
    topk=8,
    tp_sizes=(1, 2, 4, 8, 16),
    ep_sizes=(1, 2, 4, 8),
    model_key="glm51",
)


def _normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace("/", "")
        .replace(" ", "")
    )


DEEPSEEK_V3_CONFIG = ModelConfig(
    name="DeepSeek-V3",
    hidden_size=7168,
    intermediate_size=18432,
    num_attention_heads=128,
    num_kv_heads=1,  # MLA latent
    head_dim=128,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    num_experts=256,
    num_experts_per_card=32,
    expert_intermediate_size=2048,
    topk=8,
    tp_sizes=(1, 2, 4, 8, 16),
    ep_sizes=(1, 2, 4, 8),
    model_key="deepseekv3",
)

QWEN3_32B_CONFIG = ModelConfig(
    name="Qwen3-32B",
    hidden_size=5120,
    intermediate_size=25600,
    num_attention_heads=64,
    num_kv_heads=8,
    head_dim=128,
    tp_sizes=(1, 2, 4, 8, 16),
    model_key="qwen332b",
)

LLAMA_70B_CONFIG = ModelConfig(
    name="LLaMA-70B",
    hidden_size=8192,
    intermediate_size=28672,
    num_attention_heads=64,
    num_kv_heads=8,
    head_dim=128,
    tp_sizes=(1, 4, 8, 16),
    model_key="llama70b",
)

MODEL_IDS: dict[str, ModelConfig] = {
    "deepseek-ai/DeepSeek-V3": DEEPSEEK_V3_CONFIG,
    "Qwen/Qwen3-32B": QWEN3_32B_CONFIG,
    "meta-llama/Meta-Llama-3-70B": LLAMA_70B_CONFIG,
    "zai-org/GLM-5.1": GLM51_CONFIG,
}

# Keys must be lowercase and without punctuation for normalize_name()
MODELS: dict[str, ModelConfig] = {
    _normalize_name(model_id): config for model_id, config in MODEL_IDS.items()
}

MODELS_HF_PATHS: dict[str, str] = {
    _normalize_name(model_id): model_id for model_id in MODEL_IDS
}

LEGACY_MODEL_NAME_HINTS: dict[str, str] = {
    "deepseekv3": "deepseek-ai/DeepSeek-V3",
    "dsv3": "deepseek-ai/DeepSeek-V3",
    "metallama370b": "meta-llama/Meta-Llama-3-70B",
    "qwen332b": "Qwen/Qwen3-32B",
    "llama70b": "meta-llama/Meta-Llama-3-70B",
    "glm51": "zai-org/GLM-5.1",
}

_RESOLVED_CONFIGS: dict[str, ModelConfig] = {}


def _fetch_from_huggingface(model_name: str, model_id: str) -> ModelConfig:
    # Define repo root from tools/perf_data_collection/model_configs.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from tensor_cast.transformers.utils import AutoModelConfigLoader
    except ImportError:
        raise ImportError(f"Required 'tensor_cast.transformers.utils.AutoModelConfigLoader' to load config from {model_id}")

    logging.info(f"Fetching config for {model_name} from HuggingFace ({model_id}) via AutoModelConfigLoader...")
    try:
        loader = AutoModelConfigLoader()
        cfg = loader.load_config(model_id)
    except Exception as e:
        raise ValueError(f"Failed to fetch config for '{model_id}' from HuggingFace: {e}")

    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))
    intermediate_size = getattr(cfg, "intermediate_size", 0)
    num_attention_heads = getattr(cfg, "num_attention_heads", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "multi_query_group_num", num_attention_heads))
    # MLA specific (e.g. DeepSeek-V3)
    q_lora_rank = getattr(cfg, "q_lora_rank", 0)
    kv_lora_rank = getattr(cfg, "kv_lora_rank", 0)
    qk_nope_head_dim = getattr(cfg, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(cfg, "qk_rope_head_dim", getattr(cfg, "rotary_dim", 64))
    if q_lora_rank > 0 or kv_lora_rank > 0:
        head_dim = getattr(
            cfg,
            "v_head_dim",
            getattr(cfg, "qk_head_dim", getattr(cfg, "head_dim", 128)),
        )
    else:
        head_dim = getattr(
            cfg,
            "head_dim",
            getattr(cfg, "kv_channels", int(hidden_size / num_attention_heads) if num_attention_heads else 128),
        )

    # MoE specific
    num_experts = getattr(cfg, "n_routed_experts", getattr(cfg, "num_experts", 0))
    topk = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", getattr(cfg, "num_experts_per_token", 0)))
    expert_intermediate_size = getattr(cfg, "moe_intermediate_size", getattr(cfg, "expert_intermediate_size", 0))

    if num_experts > 0:
        num_experts_per_card = getattr(cfg, "n_experts_per_card", num_experts)
        valid_ep = [1] + [s for s in (2, 4, 8) if s <= num_experts]
        ep_sizes = tuple(sorted(set(valid_ep)))
    else:
        num_experts_per_card = 0
        ep_sizes = (1,)

    static_config = MODELS.get(_normalize_name(model_id))
    model_key = static_config.model_key if static_config is not None else _normalize_name(model_id)

    return ModelConfig(
        name=model_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_experts=num_experts,
        num_experts_per_card=num_experts_per_card,
        expert_intermediate_size=expert_intermediate_size,
        topk=topk,
        tp_sizes=(1, 2, 4, 8, 16),
        ep_sizes=ep_sizes,
        model_key=model_key,
    )


def resolve_configs(model_names: list[str] | None) -> list[ModelConfig]:
    """Resolve model names to ModelConfig objects."""
    if model_names is None:
        return list(dict.fromkeys(MODELS.values()))
    
    configs = []
    for name in model_names:
        norm_name = _normalize_name(name)
        if norm_name in LEGACY_MODEL_NAME_HINTS:
            raise ValueError(
                f"Unsupported legacy model name '{name}'. "
                f"Use '{LEGACY_MODEL_NAME_HINTS[norm_name]}' to match text_generate model_id naming."
            )
        model_id = MODELS_HF_PATHS.get(norm_name, name)
        
        if model_id not in _RESOLVED_CONFIGS:
            try:
                # Attempt to prioritize dynamic loading from HuggingFace
                _RESOLVED_CONFIGS[model_id] = _fetch_from_huggingface(name, model_id)
            except Exception as e:
                # Fallback to local static dict if network fails or config-class mismatch occurs
                if norm_name in MODELS:
                    # Specific hint for DSv3 vs V32 mismatch often seen in Ascend envs
                    hint = ""
                    if "DeepseekV32Config" in str(e):
                        hint = " (Note: DeepSeek-V3 vs V32 config class mismatch detected)"
                    logging.warning(
                        f"Failed to fetch '{name}' config from HuggingFace{hint}: {e}. "
                        f"Falling back to built-in static config."
                    )
                    _RESOLVED_CONFIGS[model_id] = MODELS[norm_name]
                else:
                    logging.error(f"Failed to fetch '{name}' from HuggingFace and no offline fallback is available.")
                    raise
        configs.append(_RESOLVED_CONFIGS[model_id])
    return configs


def get_matmul_nk_pairs(model_names: list[str] | None = None) -> set[tuple[int, int]]:
    """Collect all (N, K) pairs for specified models (or all if None)."""
    configs = resolve_configs(model_names)
    pairs: set[tuple[int, int]] = set()
    for cfg in configs:
        pairs |= cfg.matmul_nk_pairs()
    return pairs


def get_expert_nk_pairs(model_names: list[str] | None = None) -> set[tuple[int, int]]:
    """Collect all expert (N, K) pairs for specified MoE models."""
    configs = resolve_configs(model_names)
    pairs: set[tuple[int, int]] = set()
    for cfg in configs:
        pairs |= cfg.expert_nk_pairs()
    return pairs


def get_moe_configs(model_names: list[str] | None = None) -> list[ModelConfig]:
    """Return MoE model configs for specified models."""
    return [cfg for cfg in resolve_configs(model_names) if cfg.num_experts > 0]
