# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build constrained Theory evaluation environments from ModelRunContext."""

from __future__ import annotations

import math
from typing import Mapping

from tools.model_diagnostics.domain.models import (
    ExecutionPhase,
    ModelRunContext,
    validate_expert_parallel_features,
)
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.mtp_window import (
    parse_num_mtp_tokens,
    validate_mtp_decode_window,
)


def _require_positive(context: ModelRunContext, field_name: str) -> int:
    value = getattr(context, field_name)
    if value is None:
        raise SpecificationLoadError(f"ModelRunContext.{field_name} is required for Theory evaluation")
    return value


def _config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecificationLoadError(f"model_config.{key} must be a positive integer")
    return value


def _routed_expert_count(config: Mapping[str, object]) -> int:
    """Return the routed expert count used by Qwen or DeepSeek configs."""
    if "n_routed_experts" in config:
        return _config_int(config, "n_routed_experts")
    if "num_experts" in config:
        return _config_int(config, "num_experts")
    raise SpecificationLoadError("model_config requires num_experts or n_routed_experts")


def _optional_config_int(config: Mapping[str, object], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecificationLoadError(f"model_config.{key} must be an integer")
    if value <= 0:
        raise SpecificationLoadError(f"model_config.{key} must be positive")
    return value


def _dtype_binding(
    config: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SpecificationLoadError(f"quantization_config.{key} must be a non-empty string")
    return value


def _moe_combine_dtype(config: Mapping[str, object], activation_dtype: str) -> str | None:
    """Resolve the routed-expert weighted-reduction dtype from model semantics."""

    model_type = config.get("model_type")
    if model_type == "deepseek_v32":
        return "float32"
    if model_type in {"deepseek_v3", "glm_moe_dsa", "kimi_k2"}:
        return activation_dtype
    return None


def _assign_experts(num_experts: int, world_size: int, rank: int) -> tuple[int, int]:
    """Mirror ``tensor_cast.layers.moe_layer.assign_experts`` (no tensor_cast import)."""

    if world_size <= 0:
        raise SpecificationLoadError("expert parallel device count must be positive")
    if rank < 0 or rank >= world_size:
        raise SpecificationLoadError("MoE expert-assignment rank must be in [0, device_count)")
    num_experts_per_device = num_experts // world_size
    num_experts_rest = num_experts % world_size
    if rank < num_experts_rest:
        start = rank * (num_experts_per_device + 1)
        num_local_experts = num_experts_per_device + 1
    else:
        start = num_experts_rest * (num_experts_per_device + 1) + (rank - num_experts_rest) * num_experts_per_device
        num_local_experts = num_experts_per_device
    return start, num_local_experts


def _optional_config_bool(config: Mapping[str, object], key: str, *, default: bool = False) -> bool:
    value = config.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SpecificationLoadError(f"model_config.{key} must be a boolean")
    return value


def _derive_ep_expert_counts(
    *,
    ep: int,
    top_k: int,
    num_routing_experts: int,
    enable_external_shared_experts: bool,
    enable_redundant_experts: bool,
) -> tuple[int, int]:
    """Mirror ``shard_model_by_ep`` external/redundant counts (no tensor_cast import)."""

    try:
        validate_expert_parallel_features(
            ep,
            enable_external_shared_experts=enable_external_shared_experts,
            enable_redundant_experts=enable_redundant_experts,
        )
    except ValueError as error:
        raise SpecificationLoadError(str(error)) from error
    if ep <= 1:
        return 0, 0

    if enable_external_shared_experts:
        if top_k + 1 > ep:
            external = 1
        else:
            external = math.ceil(ep / (top_k + 1))
    else:
        external = 0

    if external >= ep:
        raise SpecificationLoadError(
            "num_external_shared_experts must be smaller than expert_parallel_size"
        )

    if enable_external_shared_experts:
        routing_devices = ep - external
        redundant = routing_devices - (num_routing_experts % routing_devices)
        if not enable_redundant_experts and redundant == routing_devices:
            redundant = 0
    elif enable_redundant_experts:
        redundant = ep
    else:
        redundant = 0

    if redundant < 0:
        raise SpecificationLoadError("derived redundant expert count must be non-negative")
    return external, redundant


def _validate_ep_token_args(
    *,
    routed_tokens: int,
    top_k: int,
    num_global_experts: int,
    ep: int,
    ep_rank: int,
    num_external_shared_experts: int = 0,
) -> None:
    """Validate the integer-domain inputs used by analytic EP token dispatch."""

    if routed_tokens <= 0:
        raise SpecificationLoadError("routed token count (T*Ktop) must be positive")
    if top_k <= 0:
        raise SpecificationLoadError("num_experts_per_tok must be positive")
    if num_global_experts <= 0:
        raise SpecificationLoadError("global expert count must be positive")
    if isinstance(num_external_shared_experts, bool) or not isinstance(num_external_shared_experts, int):
        raise SpecificationLoadError("external shared expert count must be an integer")
    if num_external_shared_experts < 0:
        raise SpecificationLoadError("external shared expert count must be non-negative")
    if num_external_shared_experts >= ep:
        raise SpecificationLoadError(
            "external shared expert count must be smaller than expert_parallel_size"
        )
    if ep_rank < 0 or ep_rank >= ep:
        raise SpecificationLoadError("analytic EP rank must be in [0, EP)")


def _simulated_ep_local_token_count(
    *,
    routed_tokens: int,
    top_k: int,
    num_global_experts: int,
    ep: int,
    ep_rank: int,
    num_external_shared_experts: int = 0,
) -> int:
    """EP-local Te under tensor_cast analytic MoE dispatch assumptions.

    Mirrors ``FusedMoETensorCast.get_split_sizes`` including optional external
    shared-expert ranks and redundant experts in ``num_global_experts``: spread
    ``routed_tokens`` (=T·Ktop) uniformly across global experts (and external
    ranks when enabled), then Te = this_rank_share · EP because each peer is
    assumed to send the same share. Never use ``T·Ktop/EP``.
    """

    _validate_ep_token_args(
        routed_tokens=routed_tokens,
        top_k=top_k,
        num_global_experts=num_global_experts,
        ep=ep,
        ep_rank=ep_rank,
        num_external_shared_experts=num_external_shared_experts,
    )

    per_expert = routed_tokens // num_global_experts
    remainder = routed_tokens % num_global_experts
    by_expert = [
        per_expert + (1 if expert_index < remainder else 0) for expert_index in range(num_global_experts)
    ]

    input_split_sizes_by_device: list[int] = []
    if num_external_shared_experts > 0:
        tokens_per_external = routed_tokens // top_k // num_external_shared_experts
        external_rest = routed_tokens // top_k % num_external_shared_experts
        for rank in range(num_external_shared_experts):
            input_split_sizes_by_device.append(tokens_per_external + (1 if rank < external_rest else 0))

    routing_devices = ep - num_external_shared_experts
    for rank in range(num_external_shared_experts, ep):
        start, num_local = _assign_experts(
            num_global_experts,
            routing_devices,
            rank - num_external_shared_experts,
        )
        input_split_sizes_by_device.append(sum(by_expert[start : start + num_local]))

    return input_split_sizes_by_device[ep_rank] * ep


def _moe_input_token_count(*, tokens: int, tp: int, dp: int, ep: int) -> int:
    """Mirror ``ParallelMoELayer._dp_transform_enter`` token-domain conversion."""

    has_ep = ep > 1
    transform_dp_group = dp != ep if has_ep else dp != 1
    if not transform_dp_group:
        return tokens
    if has_ep:
        return math.ceil(tokens / tp)
    return tokens * dp


def build_theory_env(context: ModelRunContext) -> dict[str, object]:
    """Derive B/Q/S/T/H/TP and related bindings for shape/dtype expressions."""

    batch = _require_positive(context, "batch_size")
    query = _require_positive(context, "query_length")
    context_length = context.context_length if context.context_length is not None else 0
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 0:
        raise SpecificationLoadError("context_length must be a non-negative integer")

    config = context.model_config
    hidden = _config_int(config, "hidden_size")
    vocab = _config_int(config, "vocab_size")
    intermediate = _config_int(config, "intermediate_size")
    num_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads")
    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % num_heads != 0:
            raise SpecificationLoadError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden // num_heads
    elif isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise SpecificationLoadError("model_config.head_dim must be a positive integer")

    tp = context.parallel.tensor_parallel_size
    dp = context.parallel.data_parallel_size
    ep = context.parallel.expert_parallel_size
    mdp = context.parallel.moe_data_parallel_size
    mlp_tp = _optional_config_int(config, "mlp_tp_size", default=tp)
    o_proj_tp = _optional_config_int(config, "o_proj_tp_size", default=tp)
    # ParallelConfig defaults lm-head TP to the model tensor-parallel degree.
    lmhead_tp = _optional_config_int(config, "lmhead_tp_size", default=tp)

    dtype = config.get("torch_dtype") or config.get("dtype") or "float16"
    if not isinstance(dtype, str) or not dtype.strip():
        raise SpecificationLoadError("model dtype must be a non-empty string")
    quantization = context.quantization_config
    activation_dtype = _dtype_binding(
        quantization,
        "activation_dtype",
        default=dtype,
    )
    linear_input_dtype = _dtype_binding(
        quantization,
        "linear_input_dtype",
        default=activation_dtype,
    )
    weight_dtype = _dtype_binding(
        quantization,
        "weight_dtype",
        default=dtype,
    )
    scale_dtype = _dtype_binding(
        quantization,
        "scale_dtype",
        default="float32",
    )
    accumulation_dtype = _dtype_binding(
        quantization,
        "accumulation_dtype",
        default=dtype,
    )
    output_dtype = _dtype_binding(
        quantization,
        "output_dtype",
        default=activation_dtype,
    )

    # Local rank batch after DP split, matching the alignment table B=ceil(num_queries/DP).
    local_batch = int(math.ceil(batch / dp))
    seq = context_length + query
    tokens = local_batch * query
    local_heads = num_heads // tp
    if num_heads % tp != 0:
        raise SpecificationLoadError("num_attention_heads must be divisible by TP")
    local_kv = max(num_kv_heads // tp, 1)
    ftp = intermediate // mlp_tp
    if intermediate % mlp_tp != 0:
        raise SpecificationLoadError("intermediate_size must be divisible by MLP_TP")
    vtp = vocab // lmhead_tp
    if vocab % lmhead_tp != 0:
        raise SpecificationLoadError("vocab_size must be divisible by lmhead_tp_size")

    embedding_tp_mode = config.get("word_embedding_tp")
    if embedding_tp_mode not in {None, "col", "row"}:
        raise SpecificationLoadError("model_config.word_embedding_tp must be 'col', 'row', or null")
    embedding_vocab = int(math.ceil(vocab / tp)) if embedding_tp_mode == "row" else vocab
    embedding_hidden = int(math.ceil(hidden / tp)) if embedding_tp_mode == "col" else hidden
    embedding_output_hidden = embedding_hidden if embedding_tp_mode == "col" else hidden

    validate_mtp_decode_window(context)
    mtp = parse_num_mtp_tokens(context)
    rtgt = local_batch * (mtp + 1)
    rprop = local_batch
    if context.phase is ExecutionPhase.PREFILL:
        rout = local_batch
    elif context.phase is ExecutionPhase.DECODE:
        rout = tokens
    else:
        rout = local_batch

    # Paged-attention pool sizing mirrors tensor_cast.core.input_generator.generate_inputs.
    block_size = _optional_config_int(config, "block_size", default=128)
    max_context_length = seq + mtp + 1
    num_blocks = (max_context_length * local_batch + block_size - 1) // block_size
    max_blocks_per_seq = (seq + block_size - 1) // block_size

    # MoE Theory symbols. Bound only when HF/config exposes MoE fields so Dense
    # Specs remain unchanged.
    # E: routed expert count (num_experts / n_routed_experts)
    # Ktop: experts per token (num_experts_per_tok)
    # Fmoe: MoE intermediate (moe_intermediate_size); never reuse dense F
    # MTPt: fixed at 1 (--moe-tp-size > 1 unsupported; Fe = Fmoe).
    # MDP: --moe-dp-size. It participates in parallel-layout validation, while
    #     Tmoe mirrors ParallelMoELayer._dp_transform_enter directly:
    #     EP>1 and DP!=EP -> ceil(T/TP); EP==1 and DP!=1 -> T*DP; otherwise T.
    # Te: EP-local tokens after dispatch. EP=1 => Tmoe*Ktop. EP>1 => tensor_cast
    #     get_split_sizes over routed_tokens=Tmoe*Ktop (uniform-over-global-experts,
    #     optional external shared ranks + redundant experts, symmetric all_to_all;
    #     not T*Ktop/EP). Counts mirror shard_model_by_ep from enable_* flags.
    moe_env: dict[str, object] = {}
    has_routed = "num_experts" in config or "n_routed_experts" in config
    if has_routed or "num_experts_per_tok" in config or "moe_intermediate_size" in config:
        experts = _routed_expert_count(config)
        ktop = _config_int(config, "num_experts_per_tok")
        fmoe = _config_int(config, "moe_intermediate_size")
        fe = fmoe  # MTPt is fixed at 1: Fe = Fmoe.
        tmoe = _moe_input_token_count(tokens=tokens, tp=tp, dp=dp, ep=ep)
        enable_external = _optional_config_bool(config, "enable_external_shared_experts")
        enable_redundant = _optional_config_bool(config, "enable_redundant_experts")
        if ep == 1:
            _derive_ep_expert_counts(
                ep=ep,
                top_k=ktop,
                num_routing_experts=experts,
                enable_external_shared_experts=enable_external,
                enable_redundant_experts=enable_redundant,
            )
            te = tmoe * ktop
        else:
            external, redundant = _derive_ep_expert_counts(
                ep=ep,
                top_k=ktop,
                num_routing_experts=experts,
                enable_external_shared_experts=enable_external,
                enable_redundant_experts=enable_redundant,
            )
            # Analytic Runtime starts at rank 0, or jumps to the first routing
            # rank when device-side external shared experts occupy lower ranks.
            ep_rank = external if external > 0 else 0
            te = _simulated_ep_local_token_count(
                routed_tokens=tmoe * ktop,
                top_k=ktop,
                num_global_experts=experts + redundant,
                ep=ep,
                ep_rank=ep_rank,
                num_external_shared_experts=external,
            )
        moe_env = {
            "E": experts,
            "Ktop": ktop,
            "Fmoe": fmoe,
            "MTPt": 1,  # Fixed: --moe-tp-size > 1 is unsupported by this module.
            "MDP": mdp,
            "Tmoe": tmoe,
            "Fe": fe,
            "Te": te,
        }
        # MOE_GATE_TOKENS: number of tokens the routed gate observes. Runtime
        # models with raw-logits gating (DeepSeek V3/V3.1, GLM-5/5.1,
        # Kimi-K2-Base) run the gate on the full sequence when EP>1 and only
        # then exchange tokens (ParallelMoELayer._forward_ep_raw_logits); all
        # other layouts/models gate on the post-transform domain Tmoe.
        model_type = config.get("model_type")
        if model_type in {"deepseek_v3", "glm_moe_dsa"} and ep > 1:
            moe_env["MOE_GATE_TOKENS"] = tokens
        else:
            moe_env["MOE_GATE_TOKENS"] = tmoe
        moe_combine_dtype = _moe_combine_dtype(config, activation_dtype)
        if moe_combine_dtype is not None:
            moe_env["MOE_COMBINE_DTYPE"] = moe_combine_dtype

    mla_env: dict[str, object] = {}
    if "q_lora_rank" in config or "kv_lora_rank" in config:
        qk_nope = _config_int(config, "qk_nope_head_dim")
        qk_rope = _config_int(config, "qk_rope_head_dim")
        mla_env = {
            "Qlora": _config_int(config, "q_lora_rank"),
            "KVlora": _config_int(config, "kv_lora_rank"),
            "QKnope": qk_nope,
            "QKrope": qk_rope,
            "Vh": _config_int(config, "v_head_dim"),
            "Hmla": qk_nope + qk_rope,
        }

    dsa_env: dict[str, object] = {}
    if "index_topk" in config:
        dsa_env = {"Dsa_k": min(_config_int(config, "index_topk"), seq)}

    shared_env: dict[str, object] = {}
    n_shared = config.get("n_shared_experts")
    if n_shared is not None:
        if isinstance(n_shared, bool) or not isinstance(n_shared, int) or n_shared < 0:
            raise SpecificationLoadError("model_config.n_shared_experts must be a non-negative integer")
        if n_shared > 0:
            shared_env = {
                "Nshared": n_shared,
                "Fshared": _config_int(config, "moe_intermediate_size") * n_shared,
            }

    return {
        "B": local_batch,
        "Q": query,
        "C": context_length,
        "S": seq,
        "T": tokens,
        "H": hidden,
        "V": vocab,
        "F": intermediate,
        "Nh": num_heads,
        "Nkv": num_kv_heads,
        "Dh": head_dim,
        "TP": tp,
        "DP": dp,
        "EP": ep,
        "MLP_TP": mlp_tp,
        "OTP": o_proj_tp,
        "LMTP": lmhead_tp,
        "MTP": mtp,
        "Rtgt": rtgt,
        "Rprop": rprop,
        "Lh": local_heads,
        "Lkv": local_kv,
        "Ftp": ftp,
        "Vtp": vtp,
        "EV": embedding_vocab,
        "EH": embedding_hidden,
        "EO": embedding_output_hidden,
        "Rout": rout,
        "Bs": block_size,
        "Nblk": num_blocks,
        "Mb": max_blocks_per_seq,
        "D": dtype,
        "ACT": activation_dtype,
        "LINEAR_IN": linear_input_dtype,
        "WEIGHT": weight_dtype,
        "SCALE": scale_dtype,
        "ACC": accumulation_dtype,
        "OUT": output_dtype,
        "int64": "int64",
        "float32": "float32",
        "unknown": None,
        **moe_env,
        **mla_env,
        **dsa_env,
        **shared_env,
    }
