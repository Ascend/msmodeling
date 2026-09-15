"""TensorCast adaptation patches for Moonshot AI Kimi K3.

Kimi K3 (``model_type="kimi_k3"``) is a 2.8T VL model with:
  - KDA (Kimi Delta Attention) linear attention on 69/93 text layers
  - Gated MLA full attention on 24/93 text layers
  - Latent MoE (896 experts, 16 active, down/norm/up projection wrappers)
  - SiTU activation (``beta * tanh(g/beta) * sigmoid(g) * up``)
  - AttnRes cross-layer residual (every 12 layers)
  - KimiDynamicCache (custom cache with conv_states / recurrent_states)

All patches are gated by ``model_type == "kimi_k3"`` and isolated to this
file. The adaptation follows the design document at
``docs/design/kimi_k3_adaptation_design.md`` and reuses patterns from
``kimi_k25.py`` (VL framework, MLA RoPE patch, MoE stub) and
``qwen3_next.py`` (KDA → linear_attention routing, meta mask patch).

Patch numbering follows the design doc  scheme:
  config-level:  ``_patch_hf_config_for_kimi_k3``
  class-level:  ``_patch_model_classes_for_kimi_k3``
"""

import contextlib
import contextvars
import importlib.machinery
import importlib.util
import logging
import sys
import threading
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Tuple

import torch

from ..custom_model_registry import ModelProfile, get_visual_layers, register_model_profile
from ...layers.internal import CopyLayerWrapper, RegionMarkerWrapper

logger = logging.getLogger(__name__)

# ============================================================
# Global patch tracking — stores original references for test cleanup
# ============================================================
# Every ``_install_*`` function stores the original (pre-patch) reference
# here with a unique key. ``_restore_k3_global_state()`` uses this dict to
# undo all patches, ensuring tests don't leak global state into other test
# suites (Issue 402: test infection caused Qwen3-MoE UT failures).
_K3_PATCH_ORIGINALS: dict = {}

# Module names injected into ``sys.modules`` by ``_install_fla_stub``.
# Stored so cleanup can remove only our stubs, not a real ``fla`` package.
_K3_FLA_STUB_NAMES = (
    "fla",
    "fla.modules",
    "fla.ops",
    "fla.ops.kda",
    "fla.ops.utils",
    "fla.ops.utils.index",
    "fla.utils",
)

# ============================================================
# fla-core stub modules
# ============================================================
# ``modeling_kimi_linear.py`` L46-53 hard-imports fla-core and raises
#       ``ImportError`` if missing. We inject stub modules into ``sys.modules``
#       so the import succeeds; the actual KDA computation is rerouted by the patch
#       to ``torch.ops.tensor_cast.linear_attention``.
# The stubs only need to provide correct shape inference — real computation
# never runs because P9 replaces ``KimiDeltaAttention.forward``.


class _ShortConvolutionStub(torch.nn.Module):
    """Stub for ``fla.modules.ShortConvolution``.

    Real impl: causal depthwise conv1d + activation on q/k/v.
    Stub returns the input unchanged (shape preserved) because P9 reroutes
    the entire KDA forward before any conv call.
    """

    def __init__(self, hidden_size: int, kernel_size: int, activation: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        # Real ShortConvolution has a learnable conv1d filter; provide a
        # parameter so weight-iteration code does not crash.
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, kernel_size))

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, cache


class _FusedRMSNormGatedStub(torch.nn.Module):
    """Stub for ``fla.modules.FusedRMSNormGated``.

    Real impl: ``RMSNorm(x) * sigmoid(g)``.  Stub returns ``x`` unchanged
    (shape preserved) because P9 reroutes the KDA forward.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


def _chunk_kda_stub(q, k, v, g, beta, **kwargs):
    """Stub for ``fla.ops.kda.chunk_kda`` / ``fused_recurrent_kda``.

    Returns ``(output, recurrent_state)`` with output shape derived from ``v``.
    Real computation is rerouted by the patch to ``tensor_cast.linear_attention``.
    """
    if v.dim() == 4:
        b, s, h, dv = v.shape
    else:
        # Fallback: derive shape from leading dims
        b = v.shape[0] if v.dim() >= 1 else 1
        s = v.shape[1] if v.dim() >= 2 else 1
        h = v.shape[-2] if v.dim() >= 3 else 1
        dv = v.shape[-1] if v.dim() >= 1 else 1
    out = torch.empty((b, s, h, dv), dtype=v.dtype, device=v.device)
    return out, None


_FLA_STUB_INSTALLED = False

# Submodules whose importability decides whether the real ``fla`` package
# is safe to use. ``modeling_kimi_linear.py`` L47–52 hard-imports all four;
# if any one fails (e.g. ``fla-core`` installed but ``triton`` missing on
# NPU/CPU), we fall back to stubs instead of trusting the broken package.
_FLA_REQUIRED_SUBMODULES = (
    "fla.modules",
    "fla.ops.kda",
    "fla.ops.utils.index",
    "fla.utils",
)


def _new_fla_stub_module(name: str, *, is_package: bool = False) -> ModuleType:
    """Create an importlib-compatible in-memory module for the fla stub."""
    spec = ModuleSpec(name, loader=None, is_package=is_package)
    return importlib.util.module_from_spec(spec)


def _fla_submodules_importable() -> bool:
    """Return ``True`` only if all ``fla`` submodules needed by K3 import cleanly.

    Used by :func:`_install_fla_stub` to detect a half-installed ``fla-core``
    (package present in ``sys.modules`` but a submodule import raises
    ``ImportError``, e.g. when ``triton`` is missing on NPU/CPU).
    """
    for name in _FLA_REQUIRED_SUBMODULES:
        try:
            __import__(name)
        except ImportError:
            return False
    return True


def _install_fla_stub() -> None:
    """Idempotently inject fla stub modules into ``sys.modules``.

    Skip injection only if the ``fla`` package is present in ``sys.modules``
    AND every required submodule imports without error. A half-installed
    ``fla-core`` (package present, submodules broken by missing ``triton``)
    falls through to stub injection so K3's ``modeling_kimi_linear`` can
    finish loading. Real KDA computation is rerouted by the patch to
    ``torch.ops.tensor_cast.linear_attention``; the stubs are never executed.
    """
    global _FLA_STUB_INSTALLED
    if _FLA_STUB_INSTALLED:
        return
    if "fla" in sys.modules and _fla_submodules_importable():
        # Real fla-core is fully usable — respect the real package
        _FLA_STUB_INSTALLED = True
        return

    fla = _new_fla_stub_module("fla", is_package=True)
    fla_modules = _new_fla_stub_module("fla.modules")
    fla_ops = _new_fla_stub_module("fla.ops", is_package=True)
    fla_ops_kda = _new_fla_stub_module("fla.ops.kda")
    fla_ops_utils = _new_fla_stub_module("fla.ops.utils", is_package=True)
    fla_ops_utils_index = _new_fla_stub_module("fla.ops.utils.index")
    fla_utils = _new_fla_stub_module("fla.utils")

    fla_modules.ShortConvolution = _ShortConvolutionStub
    fla_modules.FusedRMSNormGated = _FusedRMSNormGatedStub
    fla_ops_kda.chunk_kda = _chunk_kda_stub
    fla_ops_kda.fused_recurrent_kda = _chunk_kda_stub

    # fla.ops.utils.index: prepare_cu_seqlens_from_mask / prepare_lens_from_mask
    def _prepare_cu_seqlens_from_mask(mask):
        if mask is None:
            return None
        return torch.cumsum(mask.sum(dim=-1).to(torch.int32), dim=0)

    def _prepare_lens_from_mask(mask):
        if mask is None:
            return None
        return mask.sum(dim=-1).to(torch.int32)

    fla_ops_utils_index.prepare_cu_seqlens_from_mask = _prepare_cu_seqlens_from_mask
    fla_ops_utils_index.prepare_lens_from_mask = _prepare_lens_from_mask

    # fla.utils.tensor_cache: decorator that returns the function unchanged
    fla_utils.tensor_cache = lambda func: func

    # Wire up parent → child module attributes so `from fla.ops.utils.index import X` works
    fla.modules = fla_modules
    fla.ops = fla_ops
    fla_ops.utils = fla_ops_utils
    fla_ops_utils.index = fla_ops_utils_index
    fla.utils = fla_utils
    fla_ops.kda = fla_ops_kda

    # Save original sys.modules entries for cleanup (only if not already saved).
    for _name in _K3_FLA_STUB_NAMES:
        _key = f"sys.modules.{_name}"
        if _key not in _K3_PATCH_ORIGINALS:
            _K3_PATCH_ORIGINALS[_key] = sys.modules.get(_name)

    sys.modules.update(
        {
            "fla": fla,
            "fla.modules": fla_modules,
            "fla.ops": fla_ops,
            "fla.ops.kda": fla_ops_kda,
            "fla.ops.utils": fla_ops_utils,
            "fla.ops.utils.index": fla_ops_utils_index,
            "fla.utils": fla_utils,
        }
    )
    _FLA_STUB_INSTALLED = True
    logger.info("Installed fla-core stub for Kimi K3.")


# ============================================================
# Latent MoE projection support (K3-specific)
# ============================================================
# K3's ``KimiSparseMoeBlock`` wraps experts with Latent MoE projections:
#   - ``routed_expert_down_proj`` (7168→3584) BEFORE expert dispatch
#   - ``routed_expert_norm`` + ``routed_expert_up_proj`` (3584→7168) AFTER combine
# The standard ``MoELayer.__init__`` only reads ``gate``/``experts``/
# ``shared_experts``/``shared_experts_gate``/``top_k`` (defined by
# ``MoEFieldNames``) — the projection layers are silently discarded when
# ``patch_moe`` replaces ``KimiSparseMoeBlock`` with ``MoELayer``.  This causes
# experts to receive 7168-dim input while expecting 3584-dim, raising
# ``RuntimeError: shape '[1000, 7168]' is invalid for input of size 3584000``.
#
# Monkey-patch ``MoELayer.__init__``, ``ParallelMoELayer.__init__``, and
# ``FusedMoETensorCast.forward`` at runtime to capture and apply the
# projections.  All patches are guarded — non-K3 models have no projection
# attributes (``getattr`` returns ``None``), so behavior is unchanged.

_LATENT_MOE_PATCH_INSTALLED = False


def _install_latent_moe_patch() -> None:
    """Idempotently install runtime monkey-patches for Latent MoE projections.

    Installs three patches on classes from :mod:`tensor_cast.layers.moe_layer`:
      1. ``MoELayer.__init__`` — capture projections from the original module.
      2. ``ParallelMoELayer.__init__`` — propagate projections during EP rebuild.
      3. ``FusedMoETensorCast.forward`` — apply down_proj before expert dispatch
         and up_proj+norm after expert combine.

    For non-K3 models the projection attributes are ``None`` (or absent), so
    every patched method falls through to the original implementation with zero
    behavioral change.
    """
    global _LATENT_MOE_PATCH_INSTALLED
    if _LATENT_MOE_PATCH_INSTALLED:
        return

    from tensor_cast.layers.moe_layer import (
        FusedMoETensorCast,
        MoELayer,
        ParallelMoELayer,
    )

    # ------------------------------------------------------------------
    # Patch 1: MoELayer.__init__ — capture projections from original module
    # ------------------------------------------------------------------
    # After the original __init__ creates self.fused_moe (without projections),
    # read routed_expert_down_proj/up_proj/norm from the original
    # KimiSparseMoeBlock module and attach them to self.fused_moe.
    # Non-K3 modules: getattr returns None → no attributes set → no impact.
    _orig_moe_layer_init = MoELayer.__init__
    _K3_PATCH_ORIGINALS.setdefault("MoELayer.__init__", _orig_moe_layer_init)

    def _patched_moe_layer_init(self, moe_config, module):
        _orig_moe_layer_init(self, moe_config, module)
        down_proj = getattr(module, "routed_expert_down_proj", None)
        up_proj = getattr(module, "routed_expert_up_proj", None)
        norm = getattr(module, "routed_expert_norm", None)
        if down_proj is not None or up_proj is not None:
            self.fused_moe.routed_expert_down_proj = down_proj
            self.fused_moe.routed_expert_up_proj = up_proj
            self.fused_moe.routed_expert_norm = norm

    MoELayer.__init__ = _patched_moe_layer_init

    # ------------------------------------------------------------------
    # Patch 2: ParallelMoELayer.__init__ — propagate projections during EP
    # ------------------------------------------------------------------
    # shard_model_by_ep rebuilds fused_moe with new EP parameters.  The
    # projections (shared Linear, not per-expert) must be carried from the old
    # fused_moe to the new one.
    #
    # CRITICAL ORDERING: ``ParallelMoELayer.__init__`` calls
    # ``super().__init__(module)`` (``ModelWrapperBase``), which sets
    # ``self._inner = module`` (a direct reference, NOT a copy).  It then
    # REASSIGNS ``self._inner.fused_moe = fused_moe_cls(...)`` — i.e. it
    # replaces ``module.fused_moe`` with a brand-new ``FusedMoETensorCast``
    # that has no projection attributes.  Reading ``module.fused_moe`` AFTER
    # ``_orig_parallel_moe_init`` therefore returns the new (projection-less)
    # instance, and the projections captured by Patch 1 are lost.
    #
    # capture projections from the ORIGINAL ``module.fused_moe`` BEFORE
    # ``_orig_parallel_moe_init`` runs, then attach them to the NEW
    # ``self._inner.fused_moe`` AFTER it has been created.
    _orig_parallel_moe_init = ParallelMoELayer.__init__
    _K3_PATCH_ORIGINALS.setdefault("ParallelMoELayer.__init__", _orig_parallel_moe_init)

    def _patched_parallel_moe_init(self, module, *args, **kwargs):
        # Capture projections from the ORIGINAL fused_moe BEFORE _orig init
        # replaces module.fused_moe with a new FusedMoETensorCast instance.
        old_fused_moe = getattr(module, "fused_moe", None)
        down_proj = getattr(old_fused_moe, "routed_expert_down_proj", None)
        up_proj = getattr(old_fused_moe, "routed_expert_up_proj", None)
        norm = getattr(old_fused_moe, "routed_expert_norm", None)

        _orig_parallel_moe_init(self, module, *args, **kwargs)

        # Attach projections to the NEW fused_moe created by _orig init.
        # self._inner IS module (ModelWrapperBase uses a direct reference),
        # and self._inner.fused_moe is now the new FusedMoETensorCast.
        if down_proj is not None or up_proj is not None:
            new_fused_moe = getattr(self._inner, "fused_moe", None)
            if new_fused_moe is not None:
                new_fused_moe.routed_expert_down_proj = down_proj
                new_fused_moe.routed_expert_up_proj = up_proj
                new_fused_moe.routed_expert_norm = norm

    ParallelMoELayer.__init__ = _patched_parallel_moe_init

    # ------------------------------------------------------------------
    # Patch 3: FusedMoETensorCast.forward — apply projections
    # ------------------------------------------------------------------
    # Latent MoE flow (matching K3's KimiSparseMoeBlock.forward order):
    #   ① down_proj: 7168→3584 (only expert path; shared experts stay 7168)
    #   ② original forward with skip_shared_experts=True (expert dispatch/combine)
    #   ③ norm → up_proj: 3584→7168 (after combine, before shared experts)
    #   ④ shared experts on original 7168-dim hidden_states
    _orig_fused_moe_forward = FusedMoETensorCast.forward
    _K3_PATCH_ORIGINALS.setdefault("FusedMoETensorCast.forward", _orig_fused_moe_forward)

    def _patched_fused_moe_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        skip_shared_experts: bool = False,
    ) -> torch.Tensor:
        down_proj = getattr(self, "routed_expert_down_proj", None)
        up_proj = getattr(self, "routed_expert_up_proj", None)
        norm = getattr(self, "routed_expert_norm", None)

        # Non-K3 models or no projections → original path (zero impact)
        if down_proj is None and up_proj is None:
            return _orig_fused_moe_forward(self, hidden_states, topk_indices, topk_weights, skip_shared_experts)

        # External shared experts mode not yet supported with Latent MoE
        # projections (shared-expert ranks would receive wrong dim).  K3 does
        # not use this mode, so fall back to original forward.
        if self.num_external_shared_experts > 0:
            return _orig_fused_moe_forward(self, hidden_states, topk_indices, topk_weights, skip_shared_experts)

        # ① Project down to latent dim for expert dispatch
        expert_hidden = down_proj(hidden_states) if down_proj is not None else hidden_states

        # ② Expert dispatch/combine without shared experts (they run on full dim)
        expert_out = _orig_fused_moe_forward(self, expert_hidden, topk_indices, topk_weights, skip_shared_experts=True)

        # ③ Project back up to full hidden dim (norm → up_proj, matching K3 order)
        if norm is not None:
            expert_out = norm(expert_out)
        if up_proj is not None:
            expert_out = up_proj(expert_out)

        # ④ Shared experts on original (full-dim) hidden_states
        if self.shared_experts and self.num_external_shared_experts == 0 and not skip_shared_experts:
            expert_out = expert_out + self._run_shared_experts(hidden_states)

        return expert_out.to(hidden_states.dtype)

    FusedMoETensorCast.forward = _patched_fused_moe_forward

    _LATENT_MOE_PATCH_INSTALLED = True
    logger.info("Installed Latent MoE projection patch for Kimi K3.")


# ============================================================
# CopyLayerWrapper attribute passthrough + block_residual (K3-specific)
# ============================================================
# K3's ``KimiDecoderLayer`` sets ``is_linear_attn`` (bool) to distinguish
#      KDA (linear attention) layers from MLA layers.  The model forward
#      (modeling_kimi_linear.py:1195) reads ``decoder_layer.is_linear_attn`` to
#      pick the correct attention mask for each layer.
#
#      When ``enable_repetition`` is on (the default), ``maybe_reuse_layers``
#      wraps repeated layers with ``CopyLayerWrapper``.  ``CopyLayerWrapper``
#      extends ``torch.nn.Module`` directly (NOT ``ModelWrapperBase``) and only
#      copies ``attention_type`` / ``layer_type`` — ``is_linear_attn`` is lost,
#      causing ``AttributeError: 'CopyLayerWrapper' object has no attribute
#      'is_linear_attn'``.
#
#      Additionally, K3 uses AttnRes cross-layer residuals: the model forward
#      calls ``decoder_layer(..., block_residual=block_residual)`` and expects
#      the return tuple ``(hidden_states, block_residual)``.  But
#      ``CopyLayerWrapper.forward`` replays a marked region and fills all
#      extra tuple positions with ``None``, so ``block_residual`` becomes
#      ``None``.  This causes ``TypeError: expected Tensor as element 0 in
#      argument 0, but got NoneType`` in ``_apply_output_attn_res``.
#
#      P14 already skips AttnRes accumulation on meta device (passing
#      ``block_residual`` through unchanged), so ``CopyLayerWrapper`` just
#      needs to propagate the input ``block_residual`` in its output tuple.
#
# Monkey-patch ``CopyLayerWrapper.__init__`` to also copy
#      ``is_linear_attn``, and patch ``CopyLayerWrapper.forward`` to pass
#      through ``block_residual`` from kwargs.  Non-K3 layers don't have
#      these attributes / don't pass ``block_residual``, so behavior is
#      unchanged.

_COPY_LAYER_ATTR_PATCH_INSTALLED = False


def _install_copy_layer_attr_patch() -> None:
    """Idempotently patch ``CopyLayerWrapper`` for K3-specific needs.

    Two patches are installed:
      1. ``__init__`` — copy ``is_linear_attn`` from the original layer so the
         model forward can select the correct attention mask per layer type.
      2. ``forward`` — propagate ``block_residual`` from kwargs into the output
         tuple instead of ``None``, so K3's AttnRes post-loop
         ``_apply_output_attn_res`` receives a tensor.

    Non-K3 models are unaffected: they lack ``is_linear_attn`` and don't pass
    ``block_residual`` to decoder layers.
    """
    global _COPY_LAYER_ATTR_PATCH_INSTALLED
    if _COPY_LAYER_ATTR_PATCH_INSTALLED:
        return

    from tensor_cast.layers.internal import CopyLayerWrapper

    # --- Patch 1: __init__ — copy is_linear_attn -----------------------
    _orig_copy_layer_init = CopyLayerWrapper.__init__
    _K3_PATCH_ORIGINALS.setdefault("CopyLayerWrapper.__init__", _orig_copy_layer_init)

    def _patched_copy_layer_init(self, region_id, layer, representative):
        _orig_copy_layer_init(self, region_id, layer, representative)
        # Copy K3-specific attributes that the model forward reads on each
        # decoder layer.  Non-K3 layers lack these attributes → skipped.
        for attr_name in ("is_linear_attn",):
            if hasattr(layer, attr_name):
                setattr(self, attr_name, getattr(layer, attr_name))

    CopyLayerWrapper.__init__ = _patched_copy_layer_init

    # --- Patch 2: forward — propagate block_residual -------------------
    # CopyLayerWrapper.forward constructs the output tuple as
    # ``(hidden_states, None, ...)``  — filling extra positions with None.
    # K3's AttnRes expects ``(hidden_states, block_residual)`` where
    # ``block_residual`` is the input kwarg (passed through unchanged by the patch's
    # meta path).  Without this fix, ``block_residual`` becomes ``None``,
    # causing ``TypeError`` in ``_apply_output_attn_res``.
    _orig_copy_layer_forward = CopyLayerWrapper.forward
    _K3_PATCH_ORIGINALS.setdefault("CopyLayerWrapper.forward", _orig_copy_layer_forward)

    def _patched_copy_layer_forward(self, *args, **kwargs):
        result = _orig_copy_layer_forward(self, *args, **kwargs)
        # Only applies when block_residual is passed (K3 AttnRes path).
        # Non-K3 models never pass block_residual → no change.
        if isinstance(result, tuple) and "block_residual" in kwargs and len(result) >= 2:
            result = (result[0], kwargs["block_residual"]) + result[2:]
        return result

    CopyLayerWrapper.forward = _patched_copy_layer_forward

    _COPY_LAYER_ATTR_PATCH_INSTALLED = True
    logger.info("Installed CopyLayerWrapper attribute passthrough + block_residual propagation for Kimi K3.")


# ============================================================
# KDA TP plan extension (K3-specific, P0 fix + profiling alignment)
# ============================================================
# K3's KDA layers (KimiDeltaAttention, 69 layers) use standard q_proj/
#      k_proj/v_proj naming, but transformations.py's mla_config branch only
#      shards q_proj/q_b_proj/kv_b_proj, not k_proj/v_proj. This causes:
#        1. KDA's k_proj/v_proj weights are unsharded (each rank holds the
#           full 96-head weights), inflating model weights by ~12 GB
#           (TP=16: 69 layers x 2 projections x 88M params/layer).
#        2. _patched_kda_forward passes the full 96 heads, so the performance
#           model computes FLOPs for 96 heads, making linear_attention
#           account for 71.6% of total time (5.515s / 7.703s).
#        3. f_b_proj/b_proj unsharded makes g_delta/beta use the full
#           96 heads, inconsistent with NPU profiling's TP-local 12 heads
#           (8x redundant FLOPs).
#
# Monkey-patch MultiheadLatentAttentionTensorCast.build_tp_plan_extras
#      (the MLA TP plan extension hook, called from transformations.py:591)
#      to add COLWISE sharding for k_proj/v_proj/f_b_proj/b_proj on K3's KDA
#      layers. Detection via config_info.linear_attn_config identifies K3 and
#      leaves other MLA models (K2.5/V4 etc., which lack this field) unaffected.
#
#      q_proj is already sharded by the mla_config branch, o_proj by the
#      generic o_proj rule; this patch adds k_proj/v_proj/f_b_proj/b_proj
#      and g_proj (when mla_use_output_gate is set).

_KDA_TP_PLAN_PATCH_INSTALLED = False


def _install_kda_tp_plan_patch() -> None:
    """Idempotently patch ``build_tp_plan_extras`` to add KDA TP sharding for K3.

    Adds COLWISE TP sharding for ``KimiDeltaAttention``'s ``k_proj`` / ``v_proj``
    so each TP rank only holds ``96 / tp_size`` heads of KDA weights. Non-K3
    models (no ``linear_attn_config`` on ``config_info``) are unaffected — the
    original ``build_tp_plan_extras`` (empty dict) is returned unchanged.
    """
    global _KDA_TP_PLAN_PATCH_INSTALLED
    if _KDA_TP_PLAN_PATCH_INSTALLED:
        return

    from tensor_cast.layers import COLWISE_LINEAR
    from tensor_cast.layers.mla import (
        MultiheadLatentAttentionTensorCast,
        tp_plan_module_path,
    )

    _orig_build_tp_plan_extras = MultiheadLatentAttentionTensorCast.build_tp_plan_extras.__func__
    _K3_PATCH_ORIGINALS.setdefault(
        "MultiheadLatentAttentionTensorCast.build_tp_plan_extras", _orig_build_tp_plan_extras
    )

    def _k3_build_tp_plan_extras(cls, prefix, params, config_info):
        extras = dict(_orig_build_tp_plan_extras(cls, prefix, params, config_info))
        # K3 detection: text_config carries linear_attn_config (KDA config).
        # Other MLA models (K2.5/V4) lack this field → unchanged behavior.
        linear_attn_cfg = getattr(config_info, "linear_attn_config", None)
        if linear_attn_cfg is not None:
            # params already carries head_num=96 (set by mla_config branch at
            # transformations.py:577). k_proj/v_proj out_features = 96*128,
            # so head_num=96 yields 6 heads/rank at TP=16.
            k3_params = dict(params)
            extras.update(
                {
                    tp_plan_module_path(prefix, "self_attn.k_proj"): (
                        COLWISE_LINEAR,
                        k3_params,
                    ),
                    tp_plan_module_path(prefix, "self_attn.v_proj"): (
                        COLWISE_LINEAR,
                        k3_params,
                    ),
                    # alignment: TP-shard f_b_proj (128→12288 ⇒ 128→1536/rank)
                    # and b_proj (7168→96 ⇒ 7168→12/rank) so g_delta/beta become
                    # TP-local (12 heads), matching NPU profiling where
                    # RecurrentKda consumes per-rank 12-head inputs directly
                    # (no internal head-slicing). Without this each rank
                    # materialises the full 96-head g_delta (8× redundant FLOPs)
                    # and the perf model bills an extra GEMM between conv and
                    # delta-rule that is absent in profiling.
                    tp_plan_module_path(prefix, "self_attn.f_b_proj"): (
                        COLWISE_LINEAR,
                        k3_params,
                    ),
                    tp_plan_module_path(prefix, "self_attn.b_proj"): (
                        COLWISE_LINEAR,
                        k3_params,
                    ),
                }
            )
        # g_proj COLWISE sharding when MLA uses output gate.
        # g_proj maps hidden_size → num_heads * v_head_dim (7168 → 12288).
        # COLWISE shards by head (head_num=96 in params), so each rank holds
        # num_heads_per_rank * v_head_dim (768 at TP=16) — matching the
        # TP-sharded attention output for the element-wise gate multiply.
        # P13b detects tp_size > 1 and skips its runtime slice.
        if getattr(config_info, "mla_use_output_gate", False):
            gate_params = dict(params)
            extras[tp_plan_module_path(prefix, "self_attn.g_proj")] = (
                COLWISE_LINEAR,
                gate_params,
            )
        return extras

    MultiheadLatentAttentionTensorCast.build_tp_plan_extras = classmethod(_k3_build_tp_plan_extras)

    _KDA_TP_PLAN_PATCH_INSTALLED = True
    logger.info("Installed KDA TP plan extension (k_proj/v_proj/f_b_proj/b_proj COLWISE) for Kimi K3.")


# ============================================================
# SiTU activation pattern + GMM fusion
# ============================================================
# K3 MoE experts use ``SituAndMul`` (beta * tanh(g/beta) * sigmoid(g) * up).
#       Tracing decomposes SiTU into thousands of elementwise aten ops
#       (div/tanh/sigmoid/mul). Two patches install the SiTU fusion chain:
#   (a) Pattern-match the decomposed ops → ``tensor_cast.situ`` op.
#   (b) Add ``situ`` to ``SinkSplitPass.binary_ops`` (consistency with swiglu).
#   (c) GroupedMatmulSituPass (pre-fused GEMM1+SiTU) was deleted:
#       profiling uses SituMxQuant (SiTU+Quant post-fusion), so SituMxQuantPass
# fuses situ→dynamic_quantize instead. The
#       grouped_matmul_*_situ ops and their performance models were also
#       removed as dead code.
#
# The ``situ`` op signature is ``situ(gate, up, beta, linear_beta)`` — two
# separate tensors (NOT a concatenated gate_up), so the pattern starts from
# already-chunked gate/up and avoids shape-dependent slice ops.

# ============================================================
# SiTU activation pattern
# ============================================================
# The SiTU pattern registration lives in ``compilation/patterns/situ.py``
# and is enabled via the ``enable_situ`` config flag.  ``situ`` is also
# registered in SinkSplitPass's binary_ops directly in sink_split_pass.py.
# Mx quantization fusion passes (situ+quant, rms_norm+quant) were removed
# because the simulation tool does not support the MX quantization format
# used in the profiling data; operators stay in their decomposed form.

# ============================================================
# VL nested lm_head / embed_tokens TP sharding
# ============================================================
_LM_HEAD_TP_PATCH_INSTALLED = False


# K3 is a VL model with ``language_module_path="language_model"``.  The
#      actual module paths are ``language_model.lm_head`` and
#      ``language_model.model.embed_tokens``, but the TP plan in
#      ``transformations.get_tp_plan`` hardcodes ``"lm_head"`` and
#      ``"embed_tokens"`` without the VL prefix.  ``fnmatch`` requires a
#      full-string match, so these patterns never hit the nested paths and
#      lm_head / embed_tokens stay unsharded.
#
#      This also affects K2.5 (same VL structure), but per project constraint
#      we only fix K3 here to avoid altering existing model baselines.
#
# Monkey-patch ``shard_model_by_tp`` so that after the original function
#      finishes, we manually replace ``language_model.lm_head`` (and
#      ``language_model.model.embed_tokens`` if embedding_parallel is set)
#      with their Parallel counterparts.  Non-K3 models are unaffected because
#      the patched function checks for the ``language_model`` prefix attribute
#      before acting.
def _install_lm_head_tp_patch() -> None:
    """Idempotently patch ``shard_model_by_tp`` to TP-shard nested VL lm_head/embed_tokens."""
    global _LM_HEAD_TP_PATCH_INSTALLED
    if _LM_HEAD_TP_PATCH_INSTALLED:
        return

    import torch

    from tensor_cast.layers import COLWISE_LINEAR, PARALLEL_EMBEDDING, PARALLEL_MODULE_CLS
    from tensor_cast.layers.quant_linear import QuantLinearBase
    from tensor_cast.transformers import transformations as _tfm

    _orig_shard_model_by_tp = _tfm.shard_model_by_tp
    _K3_PATCH_ORIGINALS.setdefault("transformations.shard_model_by_tp", _orig_shard_model_by_tp)

    def _patched_shard_model_by_tp(model, report=None):
        model = _orig_shard_model_by_tp(model, report)

        # ---- Guard: only act on VL models with language_module_path ----
        language_prefix = "language_model"
        try:
            lang_mod = model._inner.get_submodule(language_prefix)
        except (AttributeError, ModuleNotFoundError):
            return model
        if lang_mod is None:
            return model

        # ---- Helper: shard a single module if not already parallel ----
        def _maybe_shard(name, parallel_type, params):
            try:
                module = model._inner.get_submodule(name)
            except (AttributeError, ModuleNotFoundError):
                return False
            if module is None:
                return False
            # Skip if already TP-sharded
            if type(module).__name__ in ("ColumnParallelLinear", "RowParallelLinear", "ParallelEmbedding"):
                return False
            if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding, QuantLinearBase)):
                return False
            parallel_module = PARALLEL_MODULE_CLS[parallel_type](module, **params)
            model._replace_module(name, parallel_module)
            # Update patch report: remove the bare pattern from unmatched,
            # record the actual module name as replaced.
            for r in getattr(model, "patch_reports", []):
                if r.pass_name == "Shard":  # nosec B105  # pass name, not a password
                    bare = name.rsplit(".", 1)[-1]  # "lm_head" or "embed_tokens"
                    if bare in r.unmatched_patterns:
                        r.unmatched_patterns.remove(bare)
                    if name not in r.replaced_modules:
                        r.replaced_modules.append(name)
                    break
            return True

        # ---- 1. TP-shard language_model.lm_head (COLWISE) ----
        lmhead_params = {
            "tp_group": model.parallel_group_manager.lmhead_tp_group,
            "global_tp_group": model.parallel_group_manager.tp_group,
            "gather_output": True,
        }
        _maybe_shard(f"{language_prefix}.lm_head", COLWISE_LINEAR, lmhead_params)

        # ---- 2. TP-shard language_model.model.embed_tokens (if parallel) ----
        embedding_parallel = model.model_config.parallel_config.embedding_parallel
        if embedding_parallel:
            embed_params = {
                "tp_group": model.parallel_group_manager.tp_group,
                "shard_mode": embedding_parallel,
            }
            _maybe_shard(f"{language_prefix}.model.embed_tokens", PARALLEL_EMBEDDING, embed_params)

        return model

    _tfm.shard_model_by_tp = _patched_shard_model_by_tp
    _LM_HEAD_TP_PATCH_INSTALLED = True
    logger.info("Installed VL lm_head/embed_tokens TP sharding patch for Kimi K3.")


_VISION_RMS_NORM_PATCH_INSTALLED = False


def _install_vision_rms_norm_patch() -> None:
    """Idempotently patch ``shard_model_by_tp`` to fuse vision RMSNorm.

    K3's MoonViT vision tower (27 layers × 2 norms + 1 final = 55 RMSNorm)
    and mm_projector.post_norm (1 RMSNorm) use ``nn.RMSNorm``. Because the
    vision forward is wrapped with ``torch._dynamo.disable`` (see
    ``model_builder._wrap_visual_forward``), the FX pattern match pass
    (``TorchRMSNormDecomposedPattern``) never sees these norms, so they
    decompose into 7 aten ops each (pow/mean/add_/rsqrt/mul×2/add) instead
    of fusing into ``tensor_cast.rms_norm``.

    This hook runs after ``shard_model_by_tp`` (model instance is built) and
    replaces every ``nn.RMSNorm`` under ``vision_tower`` / ``mm_projector``
    with ``RMSNormFusedWrapper``, which calls ``torch.ops.tensor_cast.rms_norm``
    directly. Text-side ``KimiRMSNorm`` is already fused by the pattern pass
    and is left untouched (it is not an ``nn.RMSNorm`` instance).
    """
    global _VISION_RMS_NORM_PATCH_INSTALLED
    if _VISION_RMS_NORM_PATCH_INSTALLED:
        return

    import torch

    from tensor_cast.layers.minimax_m3_attention import RMSNormFusedWrapper
    from tensor_cast.transformers import transformations as _tfm

    _orig_shard_model_by_tp = _tfm.shard_model_by_tp
    _K3_PATCH_ORIGINALS.setdefault("transformations.shard_model_by_tp", _orig_shard_model_by_tp)

    def _patched_shard_model_by_tp(model, report=None):
        model = _orig_shard_model_by_tp(model, report)

        # Guard: only act on VL models with a vision_tower submodule.
        targets = []
        for prefix in ("vision_tower", "mm_projector"):
            try:
                sub = model._inner.get_submodule(prefix)
            except (AttributeError, ModuleNotFoundError):
                sub = None
            if sub is not None:
                targets.append((prefix, sub))
        if not targets:
            return model

        replaced = 0
        for prefix, sub in targets:
            for name, module in sub.named_modules():
                if not isinstance(module, torch.nn.RMSNorm):
                    continue
                # Skip if already wrapped.
                if isinstance(module, RMSNormFusedWrapper):
                    continue
                wrapper = RMSNormFusedWrapper(module, is_gemma=False)
                # ``nn.RMSNorm`` defaults ``eps`` to ``None`` (PyTorch resolves
                # it to 1e-5 inside ``F.rms_norm``), but ``_get_eps`` returns
                # ``None`` verbatim and ``tensor_cast::rms_norm`` requires a
                # float. Normalize here so the op dispatch succeeds.
                if wrapper.eps is None:
                    wrapper.eps = 1e-5
                if name:
                    full_name = f"{prefix}.{name}"
                else:
                    full_name = prefix
                model._replace_module(full_name, wrapper)
                replaced += 1

        if replaced:
            logger.info(
                "Fused %d vision/mm_projector RMSNorm modules into tensor_cast.rms_norm.",
                replaced,
            )

        return model

    _tfm.shard_model_by_tp = _patched_shard_model_by_tp
    _VISION_RMS_NORM_PATCH_INSTALLED = True
    logger.info("Installed vision RMSNorm fusion patch for Kimi K3.")


# ============================================================
# Non-expert quant exclusion when DISABLED
# ============================================================
# When the user passes ``--quantize-non-expert-linear-action DISABLED``,
#      they expect non-expert layers (attention projections, dense MLP, shared
#      experts) to stay BF16 while only routed experts get the broad
#      ``--quantize-linear-action`` quant type.  However, ``create_quant_config``
#      treats DISABLED as "skip override registration", so non-expert layers
#      fall through to the broad ``layers.*`` pattern and are STILL quantized
#      (a silent no-op rather than an exclusion).
#
#      monkey-patch ``create_quant_config`` so that when non-expert is
#      DISABLED but linear is enabled, the non-expert patterns are added to
#      ``modules_to_not_convert`` — excluding them from quantization entirely.
#      This makes DISABLED behave intuitively ("don't quantize these layers").
#
#      Needed for K3 native quantization alignment (xuqiu.txt): routed experts
#      W4A8, other parts (MLA/KDA/shared experts) BF16.

_NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED = False

# K3-specific non-expert patterns NOT covered by the generic
# ``_NON_EXPERT_LINEAR_PATTERNS``.  K3 uses ``block_sparse_moe`` (not ``mlp``)
# as the MoE container attribute, and MSModeling's ``MoELayer`` wrapping adds
# a ``.fused_moe.`` level in the module path.  These broader patterns match
# shared experts and Latent MoE projections regardless of wrapping depth.
_K3_NON_EXPERT_PATTERNS = (
    # Shared experts — match any path ending in shared_experts.{gate,up,down}_proj.
    # Covers both pre-wrap (block_sparse_moe.shared_experts.*) and post-wrap
    # (block_sparse_moe.fused_moe.shared_experts.*) module paths.
    "*.shared_experts.gate_proj",
    "*.shared_experts.up_proj",
    "*.shared_experts.down_proj",
    # Latent MoE projections — shared infrastructure, NOT routed experts.
    # These compress (down_proj: 7168→3584) and decompress (up_proj: 3584→7168)
    # hidden states around the expert dispatch.
    # When --quantize-non-expert-linear-action DISABLED: stay BF16 (excluded).
    # When --quantize-non-expert-linear-action W4A8_DYNAMIC: quantized to W4A8
    #   (aligns with profiling: QuantBatchMatmulV3 for routed_expert_down/up_proj).
    "*.routed_expert_down_proj",
    "*.routed_expert_up_proj",
)

# Attention-only exclusion patterns — used when non-expert quantization is
# ENABLED (W4A8_DYNAMIC) AND attention quantization is DISABLED, to keep all
# attention BF16 while allowing shared experts, dense MLP, and Latent
# projections to be W4A8.
# Profiling confirms: MLA/KDA attention QKVO projections use MatMulV3 (BF16),
# NOT QuantBatchMatmul. So attention must stay BF16 even when non-expert W4A8.
# NOTE: The broad ``*.self_attn.*`` wildcard here also excludes ``kv_b_proj``,
# which is only safe when ``quantize_attention_action == DISABLED`` (attention
# quantization off → ``mla.py`` never calls ``_quantize_kv_b_decomposition``).
_K3_ATTENTION_ONLY_EXCLUSION_PATTERNS = (
    "*.self_attn.*",
    "*.attn.qkv",
    "*.attn.proj",
)

# Attention exclusion patterns that KEEP ``kv_b_proj`` quantizable — used when
# ``quantize_attention_action != DISABLED``.  In this case ``mla.py``'s
# ``_quantize_kv_b_decomposition`` IS invoked and requires ``kv_b_proj`` to be
# a ``TensorCastQuantLinear`` (any linear quant type: W4A8/W8A16/FP8/MXFP4/...).
# So we must NOT exclude ``kv_b_proj`` from linear quantization; instead we
# precisely list every *other* MLA/KDA attention submodule to keep them BF16
# (profiling: these projections use MatMulV3 BF16, NOT QuantBatchMatmul).
#
# MLA submodules (K3 native): q_a_proj, q_b_proj, kv_a_proj_with_mqa, o_proj,
#   g_proj (output gate, use_full_rank_gate=true branch),
#   g_a_proj + g_b_proj (use_full_rank_gate=false branch — absent in K3 but
#   listed for forward-compat with models using the low-rank gate path).
# KDA submodules (K3 native): q_proj, k_proj, v_proj, f_a_proj, f_b_proj, b_proj.
# ``kv_b_proj`` is intentionally NOT listed so it falls through to the broad
# ``layers.*`` linear pattern and gets quantized by whatever
# ``--quantize-*-linear-action`` the user chose.
_K3_ATTN_EXCLUDE_KEEP_KV_B_PATTERNS = (
    # MLA projections (all except kv_b_proj)
    "*.self_attn.q_a_proj",
    "*.self_attn.q_b_proj",
    "*.self_attn.kv_a_proj_with_mqa",
    "*.self_attn.o_proj",
    "*.self_attn.g_proj",
    "*.self_attn.g_a_proj",
    "*.self_attn.g_b_proj",
    # KDA projections
    "*.self_attn.q_proj",
    "*.self_attn.k_proj",
    "*.self_attn.v_proj",
    "*.self_attn.f_a_proj",
    "*.self_attn.f_b_proj",
    "*.self_attn.b_proj",
    # Generic fallback (non-K3 attention layouts)
    "*.attn.qkv",
    "*.attn.proj",
)

# Broad attention wildcards embedded in the generic
# ``_NON_EXPERT_LINEAR_PATTERNS`` (config.py).  ``*.self_attn.*`` over-excludes
# ``kv_b_proj`` (matches ``self_attn.kv_b_proj``); ``*.attn.qkv``/``*.attn.proj``
# are listed for completeness.  When attention quantization is ENABLED, these
# MUST be filtered out of the non-expert exclusion set and replaced with
# ``_K3_ATTN_EXCLUDE_KEEP_KV_B_PATTERNS`` (precise, keeps kv_b_proj quantizable),
# otherwise ``mla.py:_quantize_kv_b_decomposition`` crashes at the
# ``isinstance(kv_b_proj, TensorCastQuantLinear)`` check.
_K3_ATTN_BROAD_PATTERNS_IN_NON_EXPERT = (
    "*.self_attn.*",
    "*.attn.qkv",
    "*.attn.proj",
)


def _install_non_expert_quant_exclusion_patch():
    global _NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED
    if _NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED:
        return

    from tensor_cast.core.quantization import config as _quant_config_module

    # ── K3 quantization exclusion via transformations.quantize_linear ──
    # The quant_config is created in ``ConfigResolver.__init__`` (line 71)
    # BEFORE K3 patches are installed (patches run in
    # ``TransformerModel.__init__`` via ``_apply_hf_config_patches``).
    # Patching ``create_quant_config`` globally is therefore ineffective
    # during normal simulation AND pollutes non-K3 models when the patch
    # leaks into other test suites (Issue 402: Qwen3-MoE's
    # modules_to_not_convert grew from 1→21 entries).
    #
    # Instead, patch ``transformations.quantize_linear`` (the function that
    # actually calls ``quantize_linear_modules``) to inject K3 patterns into
    # ``model.model_config.quant_config.modules_to_not_convert`` right before
    # quantization happens.  This is guaranteed to run AFTER K3 patches are
    # installed, and is guarded by ``model_type == "kimi_k3"`` so non-K3
    # models are completely unaffected.
    from tensor_cast.transformers import transformations as _transformations_module

    _orig_quantize_linear = _transformations_module.quantize_linear
    _K3_PATCH_ORIGINALS["transformations.quantize_linear"] = _orig_quantize_linear

    def _patched_quantize_linear(model, report=None):
        try:
            model_config = model.model_config
            hf_config = getattr(model_config, "hf_config", None)
            model_type = getattr(hf_config, "model_type", None) if hf_config else None
            # Only inject patterns for K3 models
            if model_type == "kimi_k3":
                quant_cfg = model_config.quant_config
                if quant_cfg is not None:
                    # Check if user explicitly enabled non-expert quantization
                    # (e.g., --quantize-non-expert-linear-action W4A8_DYNAMIC).
                    # If non-expert patterns are in linear_configs, the user wants
                    # them quantized — do NOT inject exclusion patterns.
                    non_expert_quantized = any(
                        pattern in quant_cfg.linear_configs
                        for pattern in _quant_config_module._NON_EXPERT_LINEAR_PATTERNS
                    )
                    if not non_expert_quantized:
                        # DISABLED: exclude ALL non-expert layers (attention +
                        # shared experts + dense MLP + Latent projections) → BF16,
                        # UNLESS attention quant is enabled (then keep kv_b_proj
                        # quantizable — mla.py requires it).
                        _attn_enabled = quant_cfg.attention_configs.get(-1) is not None
                        if quant_cfg.modules_to_not_convert is None:
                            quant_cfg.modules_to_not_convert = []
                        # Add generic non-expert patterns — filter broad attention
                        # wildcards when attention enabled so kv_b_proj stays quantizable.
                        _non_expert_added = 0
                        for pattern in _quant_config_module._NON_EXPERT_LINEAR_PATTERNS:
                            if _attn_enabled and pattern in _K3_ATTN_BROAD_PATTERNS_IN_NON_EXPERT:
                                continue
                            if pattern not in quant_cfg.modules_to_not_convert:
                                quant_cfg.modules_to_not_convert.append(pattern)
                                _non_expert_added += 1
                        # Precise attention patterns (keep kv_b_proj) when attention enabled.
                        _attn_precise_added = 0
                        if _attn_enabled:
                            for pattern in _K3_ATTN_EXCLUDE_KEEP_KV_B_PATTERNS:
                                if pattern not in quant_cfg.modules_to_not_convert:
                                    quant_cfg.modules_to_not_convert.append(pattern)
                                    _attn_precise_added += 1
                        # Add K3-specific patterns
                        _k3_added = 0
                        for pattern in _K3_NON_EXPERT_PATTERNS:
                            if pattern not in quant_cfg.modules_to_not_convert:
                                quant_cfg.modules_to_not_convert.append(pattern)
                                _k3_added += 1
                        if _attn_enabled:
                            _msg = (
                                "Non-expert quantization DISABLED but attention "
                                "quantization ENABLED. Filtered broad attention wildcards "
                                "(kept kv_b_proj quantizable, mla.py requires it); other "
                                "MLA/KDA submodules, shared experts, dense MLP, and Latent "
                                "projections stay BF16. Added %d non-expert + %d precise-"
                                "attn + %d K3 patterns (total: %d)."
                            )
                            logger.warning(
                                _msg,
                                _non_expert_added,
                                _attn_precise_added,
                                _k3_added,
                                len(quant_cfg.modules_to_not_convert),
                            )
                        else:
                            logger.warning(
                                "Injected %d non-expert exclusion patterns into "
                                "quant_config.modules_to_not_convert before quantization "
                                "(total: %d patterns). Non-expert quantization is DISABLED.",
                                _non_expert_added + _k3_added,
                                len(quant_cfg.modules_to_not_convert),
                            )
                    else:
                        # Non-expert linear quant ENABLED. Choose attention
                        # exclusion patterns based on the attention quant switch.
                        # QuantConfig has no ``attention_action`` field; the
                        # canonical detection (transformations.py:1058-1059) is
                        # ``attention_configs.get(-1) is not None`` ⇔ enabled.
                        #   - attention DISABLED → exclude ALL attention (incl.
                        #     kv_b_proj) BF16; mla.py won't call the kv_b
                        #     decomposition hook.
                        #   - attention ENABLED  → KEEP kv_b_proj quantizable (mla.py
                        #     requires it to be a TensorCastQuantLinear); exclude
                        #     only the other MLA/KDA submodules so they stay BF16.
                        _attn_enabled = quant_cfg.attention_configs.get(-1) is not None
                        if not _attn_enabled:
                            _attn_patterns = _K3_ATTENTION_ONLY_EXCLUSION_PATTERNS
                            _log_tag = "attention DISABLED → all attention BF16"
                        else:
                            _attn_patterns = _K3_ATTN_EXCLUDE_KEEP_KV_B_PATTERNS
                            _log_tag = (
                                "attention ENABLED → kv_b_proj kept quantizable "
                                "(required by mla.py), other MLA/KDA BF16"
                            )
                        if quant_cfg.modules_to_not_convert is None:
                            quant_cfg.modules_to_not_convert = []
                        for pattern in _attn_patterns:
                            if pattern not in quant_cfg.modules_to_not_convert:
                                quant_cfg.modules_to_not_convert.append(pattern)
                        logger.warning(
                            "Non-expert linear quantization is ENABLED. "
                            "Injected %d attention exclusion patterns (%s). "
                            "Shared experts, dense MLP, and Latent projections "
                            "remain linear-quantized (aligns with profiling).",
                            len(_attn_patterns),
                            _log_tag,
                        )
        except Exception as e:
            logger.warning("Failed to inject K3 exclusion patterns: %s", e)
        return _orig_quantize_linear(model, report)

    _patched_quantize_linear.__wrapped__ = _orig_quantize_linear
    _transformations_module.quantize_linear = _patched_quantize_linear

    _NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED = True
    logger.info(
        "Installed non-expert quant exclusion patch for Kimi K3: "
        "non-expert DISABLED → excludes ALL non-expert layers (attention + shared "
        "experts + dense MLP + Latent projections) → BF16. "
        "non-expert ENABLED → attention exclusion depends on quantize_attention_action: "
        "DISABLED → exclude ALL attention (incl. kv_b_proj) BF16 (safe: mla.py hook "
        "never called); "
        "ENABLED → exclude other MLA/KDA submodules to keep BF16 BUT keep kv_b_proj "
        "quantizable (mla.py _quantize_kv_b_decomposition requires it to be a "
        "TensorCastQuantLinear, supports any linear quant type: W4A8/W8A16/FP8/MXFP4). "
        "Patches transformations.quantize_linear only (model_type-guarded); "
        "create_quant_config is NOT patched (it runs before K3 patches install "
        "and would pollute non-K3 models — Issue 402)."
    )


# ============================================================
# apply_attn_res fused op + performance model
# ============================================================
# K3's AttnRes (cross-layer residual, every 12 layers) uses
#      `_apply_attn_res` which decomposes into many aten ops (cat, pow,
#      mean, rsqrt, mul, sum, softmax, matmul). On NPU, this is a single
#      fused kernel (`_apply_attn_res_kernel`, ~6.5us per call, 2 calls
#      per layer). The previous patch skipped AttnRes entirely on meta
#      device, causing the op to be completely missing from the simulation
#      trace.
#
# Register a `tensor_cast.apply_attn_res` fused op that models the
#      entire computation (RMSNorm + linear projection + softmax + weighted
#      sum) as a single op, matching the NPU's fused kernel granularity.
#      The patch is then modified to use this op instead of skipping AttnRes,
#      and The patch monkey-patches the module-level `_apply_attn_res` function
#      so the model-level `_apply_output_attn_res` also routes to the op.
#
# The op is called twice per layer:
#   ① Before attention: combine current hidden_states with block_residual
#      using self_attention_res_proj / self_attention_res_norm
#   ② After attention, before MLP: combine prefix_sum with block_residual
#      using mlp_res_proj / mlp_res_norm
# Plus once at model level (after all layers): output_attn_res_proj / norm

_ATTN_RES_OP_REGISTERED = False


def _restore_k3_global_state() -> None:
    """Undo all global patches installed by K3 adaptation.

    Resets every ``_*_INSTALLED`` flag, removes fla stubs from
    ``sys.modules``, and restores **both** module-level and class-level
    patched callables from ``_K3_PATCH_ORIGINALS``.

    Restoring class-level patches (MoELayer.__init__, FusedMoETensorCast.forward,
    CopyLayerWrapper, build_tp_plan_extras) is critical: if the install flag
    is reset to ``False`` but the method is NOT restored, the next install
    saves the *already-wrapped* method as the "original" and installs a new
    wrapper on top, forming a wrapper chain.  For ``FusedMoETensorCast.forward``
    this could double-execute ``routed_expert_down_proj`` and cause dimension
    errors (PR review feedback on Issue 402 fix).

    Call this in test ``tearDown`` to prevent test infection (Issue 402).
    """
    global _FLA_STUB_INSTALLED, _LATENT_MOE_PATCH_INSTALLED
    global _COPY_LAYER_ATTR_PATCH_INSTALLED, _KDA_TP_PLAN_PATCH_INSTALLED
    global _LM_HEAD_TP_PATCH_INSTALLED, _VISION_RMS_NORM_PATCH_INSTALLED
    global _NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED

    # 1. Reset install flags so patches can be re-applied on next K3 run.
    _FLA_STUB_INSTALLED = False
    _LATENT_MOE_PATCH_INSTALLED = False
    _COPY_LAYER_ATTR_PATCH_INSTALLED = False
    _KDA_TP_PLAN_PATCH_INSTALLED = False
    _LM_HEAD_TP_PATCH_INSTALLED = False
    _VISION_RMS_NORM_PATCH_INSTALLED = False
    _NON_EXPERT_QUANT_EXCLUSION_PATCH_INSTALLED = False

    # 2. Remove fla stubs from sys.modules (only our injected ones).
    for _name in _K3_FLA_STUB_NAMES:
        _key = f"sys.modules.{_name}"
        _orig = _K3_PATCH_ORIGINALS.get(_key, ...)
        if _orig is ... or _orig is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _orig

    # 3. Restore module-level patched functions.
    from tensor_cast.transformers import transformations as _tfm

    if "transformations.quantize_linear" in _K3_PATCH_ORIGINALS:
        _tfm.quantize_linear = _K3_PATCH_ORIGINALS["transformations.quantize_linear"]
    if "transformations.shard_model_by_tp" in _K3_PATCH_ORIGINALS:
        _tfm.shard_model_by_tp = _K3_PATCH_ORIGINALS["transformations.shard_model_by_tp"]

    import transformers.masking_utils as _mu

    if "masking_utils.create_causal_mask" in _K3_PATCH_ORIGINALS:
        _mu.create_causal_mask = _K3_PATCH_ORIGINALS["masking_utils.create_causal_mask"]

    from transformers.utils import generic as _generic

    if "generic.OutputRecorder" in _K3_PATCH_ORIGINALS:
        _orig_or = _K3_PATCH_ORIGINALS["generic.OutputRecorder"]
        if _orig_or is None:
            if hasattr(_generic, "OutputRecorder"):
                delattr(_generic, "OutputRecorder")
        else:
            _generic.OutputRecorder = _orig_or

    import transformers.utils.import_utils as _iu

    if "import_utils.is_torch_fx_available" in _K3_PATCH_ORIGINALS:
        _orig_fx = _K3_PATCH_ORIGINALS["import_utils.is_torch_fx_available"]
        if _orig_fx is None:
            if hasattr(_iu, "is_torch_fx_available"):
                delattr(_iu, "is_torch_fx_available")
        else:
            _iu.is_torch_fx_available = _orig_fx

    # 4. Restore class-level patched methods.
    #    These MUST be restored when the corresponding install flag is reset,
    #    otherwise the next install wraps the already-wrapped method,
    #    forming a callable chain that can double-execute side effects.
    from tensor_cast.layers.moe_layer import (
        FusedMoETensorCast,
        MoELayer,
        ParallelMoELayer,
    )

    if "MoELayer.__init__" in _K3_PATCH_ORIGINALS:
        MoELayer.__init__ = _K3_PATCH_ORIGINALS["MoELayer.__init__"]
    if "ParallelMoELayer.__init__" in _K3_PATCH_ORIGINALS:
        ParallelMoELayer.__init__ = _K3_PATCH_ORIGINALS["ParallelMoELayer.__init__"]
    if "FusedMoETensorCast.forward" in _K3_PATCH_ORIGINALS:
        FusedMoETensorCast.forward = _K3_PATCH_ORIGINALS["FusedMoETensorCast.forward"]

    from tensor_cast.layers.internal import CopyLayerWrapper

    if "CopyLayerWrapper.__init__" in _K3_PATCH_ORIGINALS:
        CopyLayerWrapper.__init__ = _K3_PATCH_ORIGINALS["CopyLayerWrapper.__init__"]
    if "CopyLayerWrapper.forward" in _K3_PATCH_ORIGINALS:
        CopyLayerWrapper.forward = _K3_PATCH_ORIGINALS["CopyLayerWrapper.forward"]

    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

    if "MultiheadLatentAttentionTensorCast.build_tp_plan_extras" in _K3_PATCH_ORIGINALS:
        # The original was saved as the bare function (via __func__);
        # restore it as a classmethod to match the original descriptor type.
        _orig_tp_plan = _K3_PATCH_ORIGINALS["MultiheadLatentAttentionTensorCast.build_tp_plan_extras"]
        MultiheadLatentAttentionTensorCast.build_tp_plan_extras = classmethod(_orig_tp_plan)

    # 5. Clear the originals dict so the next install cycle starts fresh.
    _K3_PATCH_ORIGINALS.clear()

    logger.info("Restored all K3 global patches to pre-installation state.")


def _install_attn_res_op() -> None:
    """Register the ``tensor_cast.apply_attn_res`` fused op and its performance model.

    Idempotent — safe to call multiple times. The op is registered via
    ``register_tensor_cast_op`` (same mechanism as all other TC ops) and the
    performance model via ``OpInvokeInfo.register_op_properties``.

    Op signature:
        prefix_sum:     (num_tokens, hidden_size)
        block_residual: (num_tokens, num_blocks, hidden_size)
        norm_weight:    (hidden_size,)           — RMSNorm weight
        proj_weight:    (hidden_size,)           — squeezed from Linear(1, hidden)
        eps:            float                    — RMSNorm epsilon
    Returns:
        (num_tokens, hidden_size)
    """
    global _ATTN_RES_OP_REGISTERED
    if _ATTN_RES_OP_REGISTERED:
        return

    from ...utils import register_tensor_cast_op
    from ...performance_model.op_invoke_info import OpInvokeInfo
    from ...performance_model import _accumulate_compute_ops

    @register_tensor_cast_op("apply_attn_res")
    def _apply_attn_res_op(
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        norm_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Fused AttnRes: RMSNorm + linear-projection scoring + softmax + weighted sum.

        Models K3's ``_apply_attn_res`` (modeling_kimi_linear.py:1075-1088) as a
        single fused op, matching NPU's ``_apply_attn_res_kernel`` granularity.

        Computation (all in fp32, matching original upcast):
            v = cat(block_residual, prefix_sum.unsqueeze(1))   # (tokens, blocks+1, hidden)
            k = v * rsqrt(mean(v^2) + eps) * norm_weight       # RMSNorm + scale
            scores = (k * proj_weight).sum(-1)                 # (tokens, blocks+1)
            probs = softmax(scores, dim=-1)                    # (tokens, blocks+1)
            output = matmul(probs.unsqueeze(1), v).squeeze(1)  # (tokens, hidden)
        """
        num_tokens = prefix_sum.size(0)
        hidden_size = prefix_sum.size(1)
        return torch.empty(num_tokens, hidden_size, dtype=prefix_sum.dtype, device=prefix_sum.device)

    # ------------------------------------------------------------------
    # Performance model: roofline cost for the fused AttnRes kernel.
    # ------------------------------------------------------------------
    # The kernel is memory-bound for small num_blocks (≤8 for 93 layers /
    # block_size=12). Compute is modelled in fp32 (original upcast) with:
    #   - RMSNorm:  5 * N_elements GP ops (pow + mean + rsqrt + mul)
    #   - Score:    2 * N_elements GP ops (element-wise mul + reduction sum)
    #   - Softmax:  5 * N_vectors GP ops (max + sub + exp + sum + div)
    #   - MatMul:   N_elements MMA ops (probs @ v)
    # where N_elements = num_tokens * (num_blocks+1) * hidden_size
    # and   N_vectors  = num_tokens * (num_blocks+1)
    @OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.apply_attn_res.default)
    def _attn_res_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
        prefix_sum = op_invoke_info.args[0]
        block_residual = op_invoke_info.args[1]

        num_tokens = prefix_sum.size(0)
        hidden_size = prefix_sum.size(1)
        # ``block_residual`` is ``None`` in the MTP path: each MTP block is a
        # standalone ``KimiDecoderLayer`` invoked by ``MultiTokenPredictorLayer``
        # without an accumulated residual, so the per-layer AttnRes records
        # the fused op with ``block_residual=None`` for MTP layers whose
        # ``layer_idx % attn_res_block_size != 0``. Treat None as 0 blocks
        # (only the prefix_sum vector participates) — matching the degenerate
        # ``new_zeros(tokens, 0, hidden)`` case of the main model's first block.
        num_blocks = block_residual.size(1) if (block_residual is not None and block_residual.dim() >= 2) else 0
        num_vectors = num_blocks + 1  # block_residual rows + prefix_sum

        # Memory: automatically derived from op inputs (prefix_sum,
        # block_residual, norm_weight, proj_weight) and output.
        properties = op_invoke_info.get_memory_access_properties()

        n_elements = num_tokens * num_vectors * hidden_size
        n_vectors_total = num_tokens * num_vectors

        # RMSNorm (fp32): pow + mean + add_eps + rsqrt + mul ≈ 5N
        _accumulate_compute_ops(properties, torch.float32, gp_ops=n_elements * 5)
        # Score: (k * proj_weight).sum(-1) ≈ 2N (mul + add)
        _accumulate_compute_ops(properties, torch.float32, gp_ops=n_elements * 2)
        # Softmax: max + sub + exp + sum + div ≈ 5 per vector
        _accumulate_compute_ops(properties, torch.float32, gp_ops=n_vectors_total * 5)
        # Weighted sum: probs @ v → MMA ops
        _accumulate_compute_ops(properties, prefix_sum.dtype, mma_ops=n_elements)

        return properties

    _ATTN_RES_OP_REGISTERED = True
    logger.info("Registered tensor_cast.apply_attn_res fused op + performance model.")


# ============================================================================
# mla_prolog fused op registration (mlapo over-fusion split)
# ----------------------------------------------------------------------------
# TC's ``mlapo`` op fuses q_a_proj + q_a_norm + q_b_proj + kv_a_proj +
#      kv_a_norm + RoPE into a single graph node. NPU profiling shows
#      ``q_a_proj`` as an independent ``MatMulV3`` kernel (16us) and the
#      remaining operations fused into ``MlaPrologV3`` (23us). This op
#      models the ``MlaPrologV3`` portion (everything except q_a_proj) so
#      that the K3 MLA forward patch can split ``mlapo`` into
#      ``aten.mm`` (q_a_proj) + ``mla_prolog`` (rest), aligning trace
#      granularity with profiling.
# ============================================================================
_MLA_PROLOG_OP_REGISTERED = False


def _install_mla_prolog_op() -> None:
    """Register the ``tensor_cast.mla_prolog`` fused op and its performance model.

    Idempotent — safe to call multiple times. The op is registered via
    ``register_tensor_cast_op`` (same mechanism as all other TC ops) and the
    performance model via ``OpInvokeInfo.register_op_properties``.

    Op signature:
        hidden_states:        (num_tokens, hidden_size) — used for kv_a_proj
        qa:                   (num_tokens, q_lora_rank) — q_a_proj output (from aten.mm)
        cos/sin:              rotary embedding caches (1, seq_len, qk_rope_head_dim)
        q_a_layernorm_weight: (q_lora_rank,)
        q_b_proj_weight:      (num_heads * qk_head_dim, q_lora_rank)
        kv_a_proj_weight:     (kv_lora_rank + qk_rope_head_dim, hidden_size)
        kv_a_layernorm_weight:(kv_lora_rank,)
        num_heads/qk_* dims/kv_lora_rank/q_lora_rank: structural scalars

    Returns:
        q_states:    (num_tokens, num_heads, qk_head_dim)
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot:       (num_tokens, qk_rope_head_dim)
        qa_normed:   (num_tokens, q_lora_rank)
    """
    global _MLA_PROLOG_OP_REGISTERED
    if _MLA_PROLOG_OP_REGISTERED:
        return

    from ...utils import register_tensor_cast_op
    from ...performance_model.op_invoke_info import OpInvokeInfo
    from ...performance_model import _quantized_weight_compute_dtype

    @register_tensor_cast_op("mla_prolog")
    def _mla_prolog_op(
        hidden_states: torch.Tensor,
        qa: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_a_layernorm_weight: torch.Tensor,
        q_b_proj_weight: torch.Tensor,
        kv_a_proj_weight: torch.Tensor,
        kv_a_layernorm_weight: torch.Tensor,
        num_heads: int,
        qk_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        *,
        is_decode_values: Optional[list[bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused MLA prolog (excluding q_a_proj).

        Models NPU profiling's ``MlaPrologV3`` kernel, which fuses
        q_a_norm + q_b_proj + kv_a_proj + kv_a_norm + RoPE.
        ``q_a_proj`` is split out as an independent ``aten.mm``,
        matching profiling's independent ``MatMulV3`` kernel.
        ``is_decode_values`` preserves per-request phase metadata for the
        profiling lookup without changing the fused op's output shapes.
        """
        num_tokens = hidden_states.size(0)
        device = hidden_states.device
        dtype = hidden_states.dtype
        qa_normed_dim = q_lora_rank or 0
        return (
            torch.empty((num_tokens, num_heads, qk_head_dim), dtype=dtype, device=device),
            torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
            torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
            torch.empty((num_tokens, qa_normed_dim), dtype=dtype, device=device),
        )

    # ------------------------------------------------------------------
    # Performance model: roofline cost for the fused MLA prolog kernel.
    # ------------------------------------------------------------------
    # FLOPs are derived from _mlapo_properties_helper (op2-op7), with op1
    # (q_a_proj) removed since it is now an independent aten.mm.
    #   - op2: q_a_layernorm   (GP): 5 * num_tokens * q_lora_rank
    #   - op3: q_b_proj        (MMA): 2 * num_tokens * q_lora_rank * num_heads * qk_head_dim
    #   - op4: q_RoPE          (GP): 3 * num_tokens * num_heads * qk_rope_head_dim
    #   - op5: kv_a_proj       (MMA): 2 * num_tokens * hidden_size * (kv_lora_rank + qk_rope_head_dim)
    #   - op6: kv_a_layernorm  (GP): 5 * num_tokens * kv_lora_rank
    #   - op7: k_RoPE          (GP): 3 * num_tokens * qk_rope_head_dim
    @OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mla_prolog.default)
    def _mla_prolog_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
        hidden_states = op_invoke_info.args[0]  # (num_tokens, hidden_size)
        # args[1] = qa (q_a_proj output, num_tokens × q_lora_rank)
        kv_a_proj_weight = op_invoke_info.args[6]
        num_heads = op_invoke_info.args[8]
        qk_head_dim = op_invoke_info.args[9]
        qk_rope_head_dim = op_invoke_info.args[11]
        kv_lora_rank = op_invoke_info.args[12]
        q_lora_rank = op_invoke_info.args[13]

        num_tokens = hidden_states.size(0)
        hidden_size = hidden_states.size(1)

        # op1 (q_a_proj) has been split out to aten.mm — not counted here.
        # op2: q_a_layernorm (GP)
        op2_ops = num_tokens * q_lora_rank * 5
        # op3: q_b_proj (MMA)
        op3_ops = num_tokens * q_lora_rank * num_heads * qk_head_dim * 2
        # op4: q_RoPE (GP)
        op4_ops = num_tokens * num_heads * qk_rope_head_dim * 3
        # op5: kv_a_proj_with_mqa (MMA)
        op5_ops = num_tokens * hidden_size * (kv_lora_rank + qk_rope_head_dim) * 2
        # op6: kv_a_layernorm (GP)
        op6_ops = num_tokens * kv_lora_rank * 5
        # op7: k_RoPE (GP)
        op7_ops = num_tokens * qk_rope_head_dim * 3

        total_mma_ops = op3_ops + op5_ops  # op1 removed (now aten.mm)
        total_gp_ops = op2_ops + op4_ops + op6_ops + op7_ops

        properties = op_invoke_info.get_memory_access_properties()
        activation_bytes = hidden_states.element_size()
        q_a_bytes = num_tokens * q_lora_rank * activation_bytes
        qa_normed_read_bytes = q_a_bytes
        compressed_kv_bytes = num_tokens * (kv_lora_rank + qk_rope_head_dim) * activation_bytes
        properties.memory_readwrite_bytes += 2 * (q_a_bytes + compressed_kv_bytes)
        properties.memory_read_bytes += qa_normed_read_bytes
        properties.extra_static_cost_count += 15  # keep parity with original mlapo

        mma_dtype = _quantized_weight_compute_dtype(kv_a_proj_weight.dtype)
        compute_ops = properties.compute_ops.setdefault(mma_dtype, OpInvokeInfo.ComputeOps())
        compute_ops.mma_ops += total_mma_ops
        compute_ops = properties.compute_ops.setdefault(hidden_states.dtype, OpInvokeInfo.ComputeOps())
        compute_ops.gp_ops += total_gp_ops
        return properties

    _MLA_PROLOG_OP_REGISTERED = True
    logger.info("Registered tensor_cast.mla_prolog fused op + performance model.")


def _patch_hf_config_for_kimi_k3(config) -> bool:
    """Apply HuggingFace config and import-environment patches for Kimi K3.

    These patches modify the Transformers *environment* (not model classes)
    and must run BEFORE the model is loaded. Does not require ``model_id``.

    Returns ``True`` if any patch was applied.
    """
    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k3":
        return False

    patched = False

    # ----------------------------------------------------------------
    # fla-core import stub
    # ----------------------------------------------------------------
    # modeling_kimi_linear.py raises ImportError if fla-core missing.
    #      Inject stub modules so the import succeeds.
    _install_fla_stub()
    patched = True  # always mark patched — stub installation is required

    # ----------------------------------------------------------------
    # Patch transformers 5.x OutputRecorder API gap
    # ----------------------------------------------------------------
    # modeling_kimi_linear.py L44 imports ``OutputRecorder`` from
    #      ``transformers.utils.generic``, but transformers 5.13+ removed it
    #      (probe verified on this venv: ``hasattr(OutputRecorder) == False``).
    #      K3's modeling code only references it for decorator metadata
    #      that the simulation path never reaches. Stub it as a no-op class
    #      so the import succeeds; the actual forward flow is rerouted by the patch.
    from transformers.utils import generic as _tc_generic

    if not hasattr(_tc_generic, "OutputRecorder"):
        _K3_PATCH_ORIGINALS.setdefault("generic.OutputRecorder", getattr(_tc_generic, "OutputRecorder", None))

        class OutputRecorder:  # noqa: N801 — matches upstream symbol name
            """Stub for ``transformers.utils.generic.OutputRecorder``.

            Real impl (transformers ≤ 5.12) records module outputs by layer
            index for selective capture. The simulation path never reaches
            that code, so a no-op is sufficient.
            """

            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

            def __repr__(self) -> str:
                return "<OutputRecorder stub>"

        _tc_generic.OutputRecorder = OutputRecorder
        patched = True
        logger.info("Patched transformers.utils.generic.OutputRecorder for Kimi K3.")

    # ----------------------------------------------------------------
    # Bridge create_causal_mask() K3 kwargs → transformers 5.13 signature
    # ----------------------------------------------------------------
    # modeling_kimi_linear.py L1172-1179 calls ``create_causal_mask()``
    #      with two kwargs that transformers 5.13+ no longer accepts:
    #        (a) ``input_embeds=inputs_embeds`` — typo (missing trailing ``s``).
    #            Older transformers tolerated it via ``**kwargs`` absorption;
    #            5.13 declares the param strictly as ``inputs_embeds``.
    #        (b) ``cache_position=cache_position`` — fully removed in 5.13.
    #            The function's own docstring marks it "Deprecated and unused",
    #            so dropping it is semantically safe.
    #      Both raise ``TypeError: ... got an unexpected keyword argument``.
    #      We wrap the function to (a) alias ``input_embeds`` → ``inputs_embeds``
    #      and (b) pop the deprecated ``cache_position`` before forwarding.
    # NOTE: Must run before ``modeling_kimi_linear.py`` is imported (class-level),
    #       so its ``from transformers.masking_utils import create_causal_mask``
    #       binds the wrapped version.
    import transformers.masking_utils as _tc_masking_utils

    _orig_create_causal_mask = _tc_masking_utils.create_causal_mask
    _K3_PATCH_ORIGINALS.setdefault("masking_utils.create_causal_mask", _orig_create_causal_mask)
    if not getattr(_orig_create_causal_mask, "_tensor_cast_k3_causal_mask_bridge", False):

        def _create_causal_mask_k3_bridge(*args, **kwargs):
            # (a) alias typo'd param name
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            # (b) drop deprecated & unused param (per upstream docstring)
            kwargs.pop("cache_position", None)
            return _orig_create_causal_mask(*args, **kwargs)

        _create_causal_mask_k3_bridge._tensor_cast_k3_causal_mask_bridge = True
        _create_causal_mask_k3_bridge.__wrapped__ = _orig_create_causal_mask
        _tc_masking_utils.create_causal_mask = _create_causal_mask_k3_bridge
        patched = True
        logger.info(
            "Wrapped transformers.masking_utils.create_causal_mask for Kimi K3 "
            ": alias 'input_embeds' → 'inputs_embeds', drop 'cache_position'."
        )

    # ----------------------------------------------------------------
    # Latent MoE projection support (K3-specific)
    # ----------------------------------------------------------------
    # KimiSparseMoeBlock wraps experts with routed_expert_down_proj
    #      (7168→3584) / routed_expert_norm / routed_expert_up_proj (3584→7168).
    #      patch_moe replaces KimiSparseMoeBlock with MoELayer, discarding the
    #      projections.  Install runtime monkey-patches on MoELayer,
    #      ParallelMoELayer, and FusedMoETensorCast to capture and apply them.
    #      Must run before model loading so patches are in place before
    #      patch_moe and shard_model_by_ep execute.
    #      Non-K3 models: projections default to None → original behavior.
    _install_latent_moe_patch()
    patched = True

    # ----------------------------------------------------------------
    # CopyLayerWrapper attribute passthrough (K3-specific)
    # ----------------------------------------------------------------
    # K3's model forward reads ``decoder_layer.is_linear_attn`` to choose
    #      the attention mask.  ``maybe_reuse_layers`` wraps repeated layers
    #      with ``CopyLayerWrapper`` which doesn't copy this attribute.
    #      Must run before model loading so the patch is in place before
    #      ``maybe_reuse_layers`` executes.
    _install_copy_layer_attr_patch()
    patched = True

    # ----------------------------------------------------------------
    # KDA TP plan extension — k_proj/v_proj COLWISE sharding
    # ----------------------------------------------------------------
    # K3's KDA layers (KimiDeltaAttention, 69 layers) use standard
    # q_proj/k_proj/v_proj naming, but the mla_config TP branch only
    # shards q_proj/q_b_proj/kv_b_proj. k_proj/v_proj stay replicated,
    # inflating model weight by ~12 GB (TP=16) and causing the perf
    # model to compute KDA FLOPs for the full 96 heads per rank.
    # Must run before TP plan generation (transformations.get_shard_plan)
    # so the hook is in place when build_tp_plan_extras is called.
    _install_kda_tp_plan_patch()
    patched = True

    # SiTU activation pattern is enabled by default in config.py
    # (``enable_situ = True``).  The pattern lives in
    # ``compilation/patterns/situ.py`` and is registered by ``lazy_init``
    # during compilation; no manual registration is needed here.

    # ----------------------------------------------------------------
    # VL nested lm_head / embed_tokens TP sharding fix
    # ----------------------------------------------------------------
    # K3 is a VL model — lm_head and embed_tokens are nested under
    #      ``language_model.`` but the TP plan hardcodes bare ``"lm_head"``
    #      / ``"embed_tokens"`` patterns that never match the nested path.
    #      Must run before ``shard_model`` so the monkey-patch on
    #      ``shard_model_by_tp`` is in place.
    _install_lm_head_tp_patch()
    patched = True

    # ----------------------------------------------------------------
    # Vision RMSNorm fusion (vision_tower + mm_projector)
    # ----------------------------------------------------------------
    # MoonViT uses nn.RMSNorm (55 instances) but vision forward is
    #      torch._dynamo.disable-wrapped, so the FX pattern match pass cannot
    #      fuse them. Replace with RMSNormFusedWrapper at TP-shard time so
    #      they record as tensor_cast.rms_norm in the trace.
    _install_vision_rms_norm_patch()
    patched = True

    # ----------------------------------------------------------------
    # Non-expert quant exclusion when DISABLED
    # ----------------------------------------------------------------
    # --quantize-non-expert-linear-action DISABLED should keep non-expert
    #      layers (attention/KDA/shared experts) BF16, but currently falls
    #      through to the broad pattern and quantizes them anyway. This patch
    #      adds non-expert patterns to modules_to_not_convert when DISABLED,
    #      aligning with K3 native quantization (xuqiu.txt): routed experts
    #      W4A8, other parts BF16.
    _install_non_expert_quant_exclusion_patch()
    patched = True

    # ----------------------------------------------------------------
    # apply_attn_res fused op registration
    # ----------------------------------------------------------------
    # K3's AttnRes cross-layer residual was previously skipped on meta
    #      device, causing `_apply_attn_res_kernel` to be completely
    #      missing from the simulation trace. Registering a fused op here
    #      allows P14 to model AttnRes as a single op matching the NPU's
    #      fused kernel granularity.
    #      Must run before model loading / tracing so the op is available
    #      when the patch's patched forward calls it.
    _install_attn_res_op()
    patched = True

    # ----------------------------------------------------------------
    # Restore is_torch_fx_available
    # ----------------------------------------------------------------
    # transformers v5.x removed is_torch_fx_available; K3 remote code
    #      may reference it during config validation.
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        _K3_PATCH_ORIGINALS.setdefault(
            "import_utils.is_torch_fx_available", getattr(import_utils, "is_torch_fx_available", None)
        )

        def is_torch_fx_available():
            return importlib.util.find_spec("torch.fx") is not None

        import_utils.is_torch_fx_available = is_torch_fx_available
        patched = True

    # ----------------------------------------------------------------
    # Downgrade flash_attention_2 → tensor_cast
    # ----------------------------------------------------------------
    # K3 config.json specifies "_attn_implementation": "flash_attention_2".
    #      transformers enforces flash_attn availability during
    #      PreTrainedModel.__init__() — if flash_attn is not installed,
    #      ImportError is raised BEFORE the model instance is returned.
    #      We downgrade to "tensor_cast" so the HF loader skips the check.
    def _downgrade_attn_implementation(cfg) -> bool:
        if getattr(cfg, "_attn_implementation", None) == "flash_attention_2":
            if importlib.util.find_spec("flash_attn") is None:
                logger.warning(
                    "Flash Attention 2 is requested but not installed. "
                    "Falling back to 'tensor_cast' attention implementation for Kimi K3 simulation."
                )
                cfg._attn_implementation = "tensor_cast"
                return True
        return False

    text_downgraded = _downgrade_attn_implementation(config)
    if hasattr(config, "vision_config") and config.vision_config is not None:
        vision_downgraded = _downgrade_attn_implementation(config.vision_config)
        if vision_downgraded:
            text_downgraded = True

    # Also downgrade text_config if present (K3 nests text config)
    if hasattr(config, "text_config") and config.text_config is not None:
        tc_downgraded = _downgrade_attn_implementation(config.text_config)
        if tc_downgraded:
            text_downgraded = True

    if text_downgraded:
        patched = True

    # ----------------------------------------------------------------
    # Bridge vision config attributes
    # ----------------------------------------------------------------
    # K3 vision config uses ``merge_kernel_size`` (list) instead of
    #      ``spatial_merge_size`` and may omit ``temporal_patch_size`` /
    #      ``in_channels``. The generic image-input generator expects these.
    if hasattr(config, "vision_config") and config.vision_config is not None:
        vc = config.vision_config

        if hasattr(vc, "merge_kernel_size"):
            mk = vc.merge_kernel_size
            vc.spatial_merge_size = mk[0] if isinstance(mk, (list, tuple)) else mk
            patched = True

        if not hasattr(vc, "temporal_patch_size"):
            vc.temporal_patch_size = 1
            patched = True

        if not hasattr(vc, "in_channels"):
            vc.in_channels = 3
            patched = True

    # ----------------------------------------------------------------
    # Copy expert counts from text_config to root config
    # ----------------------------------------------------------------
    # K3 stores ``num_experts`` / ``num_shared_experts`` inside
    #      ``text_config``. Downstream MoE patching (transformations.patch_moe)
    #      reads them from the root config object.
    if hasattr(config, "text_config") and config.text_config is not None:
        tc = config.text_config

        if hasattr(tc, "num_experts") and not hasattr(config, "num_experts"):
            setattr(config, "num_experts", tc.num_experts)
            patched = True

        if hasattr(tc, "num_shared_experts") and not hasattr(config, "num_shared_experts"):
            setattr(config, "num_shared_experts", tc.num_shared_experts)
            patched = True

    if not hasattr(config, "num_experts"):
        setattr(config, "num_experts", 896)
        logger.warning("num_experts not found in config or text_config; falling back to K3 default 896.")
        patched = True

    if not hasattr(config, "num_shared_experts"):
        setattr(config, "num_shared_experts", 2)
        logger.warning("num_shared_experts not found in config or text_config; falling back to K3 default 2.")
        patched = True

    return patched


# ============================================================
# class-level monkey-patches
# ============================================================


def _get_k3_class_from_source(class_ref: str, model_id: str, remote_source: str = "huggingface"):
    """Resolve a K3 class object from HF Hub or ModelScope.

    HF mode reuses transformers' ``get_class_from_dynamic_module`` which reads
    from the HF dynamic module cache. ModelScope mode imports the modeling
    module directly from the ModelScope snapshot directory so the patched
    class object is the same one that ``modelscope.AutoModel.from_config``
    will instantiate later.

    This unifies class loading across remote sources so  monkey-patches
    always target the class object that the model loader will actually use,
    avoiding the silent "patched but not effective" failure mode that occurs
    when HF-cache classes are patched but ModelScope loads a different copy.

    When ``remote_source`` is ``"auto"``, the function first attempts HF load
    (which also covers local-path mode since transformers' dynamic module
    loader resolves local model_id directly), and falls back to ModelScope
    if HF lookup fails. This auto mode avoids requiring callers to thread
    ``remote_source`` through the public patch API.

    IMPORTANT — same-package class resolution for ``modeling_kimi_linear``:
    ``modeling_kimi_k3.py`` does ``from .modeling_kimi_linear import ...``.
    When ``get_class_from_dynamic_module("modeling_kimi_k3.KimiK3...", ...)``
    runs first, it registers ``modeling_kimi_linear`` into
    ``sys.modules`` under the **same package prefix** (e.g.
    ``transformers_modules.master.bc28055d10f4f1b1.modeling_kimi_linear``).
    But a subsequent ``get_class_from_dynamic_module("modeling_kimi_linear.X",
    ...)`` creates a **different** package prefix (e.g. ``...89ac47a0...``),
    yielding a different class object. Patching that second copy has no effect
    on the model because the model references the first copy.

    To fix this, for any sub-module of the K3 package (i.e. class_ref whose
    module name is not ``modeling_kimi_k3`` itself), we first scan
    ``sys.modules`` for a module that shares the same package prefix as the
    already-loaded ``modeling_kimi_k3`` module, and return the class from
    that module. Only if no such module exists do we fall back to
    ``get_class_from_dynamic_module``.
    """
    module_name = class_ref.split(".", 1)[0]
    class_name = class_ref.split(".", 1)[1]

    # Same-package class resolution: prefer the modeling_kimi_linear module
    # that was loaded as part of the modeling_kimi_k3 package (via
    # ``from .modeling_kimi_linear import ...``), not a separately-loaded
    # copy with a different package prefix.
    if module_name != "modeling_kimi_k3":
        resolved_cls = _resolve_class_from_k3_package(module_name, class_name)
        if resolved_cls is not None:
            return resolved_cls

    if remote_source == "auto":
        # Try HF first (covers both HF cache and local path mode)
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            return get_class_from_dynamic_module(class_ref, model_id, force_download=False)
        except Exception as hf_error:
            # Fall back to ModelScope: sync the repo (config + code only) and
            # load the module from the snapshot directory. The module is
            # registered into sys.modules under the plain module name so that
            # modelscope.AutoModel.from_config, which also resolves trust_remote_code
            # via the local snapshot, hits the same cached module object.
            try:
                return _load_k3_class_from_modelscope(class_ref, model_id)
            except Exception:
                # ModelScope fallback also failed — surface the original HF
                # error so the caller sees the most relevant failure cause.
                raise hf_error

    if remote_source == "modelscope":
        return _load_k3_class_from_modelscope(class_ref, model_id)

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(class_ref, model_id, force_download=False)


def _load_k3_class_from_modelscope(class_ref: str, model_id: str):
    """Load a K3 class from the ModelScope snapshot directory.

    The snapshot is materialized via ``_modelscope_snapshot_config_only`` which
    downloads only config + Python code (no weight tensors). The module is
    imported under the plain module name (e.g. ``modeling_kimi_k3``) and cached
    in ``sys.modules`` so that ``modelscope.AutoModel.from_config`` reuses the
    same module object when it later instantiates the model.
    """
    import importlib
    import os
    from tensor_cast.transformers.utils import _modelscope_snapshot_config_only

    resolved_root = _modelscope_snapshot_config_only(model_id)
    module_name = class_ref.split(".", 1)[0]
    module_path = os.path.join(resolved_root, f"{module_name}.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"ModelScope K3 modeling file not found: {module_path}. "
            f"Ensure the ModelScope snapshot for {model_id!r} contains {module_name}.py."
        )
    # Reuse an already-cached module if present (idempotent across ).
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    class_name = class_ref.split(".", 1)[1]
    return getattr(module, class_name)


def _resolve_class_from_k3_package(module_name: str, class_name: str):
    """Return ``module_name.class_name`` from the K3 package in ``sys.modules``.

    ``modeling_kimi_k3.py`` imports sibling modules via ``from .X import ...``.
    Once P6 has loaded ``modeling_kimi_k3`` (e.g. under
    ``transformers_modules.master.<hash>.modeling_kimi_k3``), its sibling
    ``modeling_kimi_linear`` is registered under the **same package prefix**.

    A subsequent ``get_class_from_dynamic_module("modeling_kimi_linear.X",
    ...)`` would create a **different** package prefix and return a different
    class object, so patching it has no effect on the model. This helper scans
    ``sys.modules`` for a sibling module that shares the same package prefix
    as the already-loaded ``modeling_kimi_k3`` module and returns the class
    from there.

    Returns ``None`` if no matching module is found (caller falls back to
    ``get_class_from_dynamic_module``).
    """
    # Find the modeling_kimi_k3 module already loaded into sys.modules.
    # P6 runs before , so by the time this is called for
    # modeling_kimi_linear.* classes, modeling_kimi_k3 should be present.
    k3_module_key = None
    for mod_key in list(sys.modules.keys()):
        if mod_key.endswith(".modeling_kimi_k3") and sys.modules[mod_key] is not None:
            k3_module_key = mod_key
            break

    if k3_module_key is None:
        # modeling_kimi_k3 not loaded yet — cannot resolve sibling.
        return None

    # Derive the package prefix (e.g. "transformers_modules.master.<hash>").
    package_prefix = k3_module_key.rsplit(".", 1)[0]
    target_module_key = f"{package_prefix}.{module_name}"

    # Best case: sibling module already loaded as part of the package.
    target_module = sys.modules.get(target_module_key)
    if target_module is not None and hasattr(target_module, class_name):
        return getattr(target_module, class_name)

    # Sibling not yet imported — try to import it from the same package so
    # it lands under the same package prefix.
    try:
        import importlib

        target_module = importlib.import_module(target_module_key)
        if hasattr(target_module, class_name):
            return getattr(target_module, class_name)
    except ImportError:
        pass

    return None


class K3IdentityRotaryEmb(torch.nn.Module):
    """Identity RoPE shim for K3 MTP support.

    K3's MLA has ``mla_use_nope=True`` (``rotary_emb=None`` on attention
    modules). ``MtpWrapper.__init__`` calls ``_find_text_rotary_emb`` which
    requires a proper ``nn.Module`` named ``rotary_emb`` to compute
    ``position_embeddings``. Without this shim, MTP enablement fails with
    ``ValueError: Unable to find rotary embedding module from {model}``.

    The shim is injected into ``KimiLinearModel.__init__`` so it
    appears at path ``language_model.model.rotary_emb`` — the path that
    ``_find_text_rotary_emb`` prefers. It returns identity (cos=1, sin=0),
    which is correct for K3's no-RoPE MLA.
    """

    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[1] if hidden_states.ndim >= 3 else hidden_states.shape[0]
        target_dtype = dtype or hidden_states.dtype
        cos = torch.ones(seq_len, self.head_dim, device=hidden_states.device, dtype=target_dtype)
        sin = torch.zeros(seq_len, self.head_dim, device=hidden_states.device, dtype=target_dtype)
        return cos, sin


def _k3_kda_state_size_gb(model) -> float:
    """Per-rank KDA state (``recurrent_states`` + ``conv_states``) memory in GB.

    K3 hybrid attention uses KDA (``KimiDeltaAttention``, ~69/93 layers) with
    a *fixed-size* recurrent state plus a short-conv tail state. Unlike MLA's
    paged KV cache, KDA state does NOT grow with context length T, but it
    DOES occupy real GPU memory and must NOT be treated as zero memory.

    MSModeling's ``kv_cache_excluded_layer_indices`` correctly excludes KDA
    layers from per-token KV cache accounting (they are not paged), but this
    also drops KDA's fixed state from the memory budget entirely. This
    helper computes the per-rank persistent footprint so the ModelRunner patch can fold it
    back into the memory figures.

    The persistent state buffers we account for are:

    * ``recurrent_states`` — shape
      ``[batch, num_heads_per_rank, head_dim, head_dim]``, fp32 for
      numerical stability of the delta-rule recurrence.
    * ``conv_states`` — 3 tensors (q, k, v), each shaped
      ``[batch, short_conv_kernel_size - 1, num_heads_per_rank * head_dim]``,
      bf16 (matches K3's ``dtype: bfloat16``).

    KDA attention is TP-sharded by heads (same policy as MLA — see P5
    ``_k3_build_tp_plan_extras``), so each rank maintains state for
    ``num_heads_per_rank`` heads. Steady-state decode keeps ``batch_size=1``
    per rank for the *persistent* state buffers; prefill allocates larger
    transient buffers, but only the persistent footprint survives across
    tokens, matching MLA's per-token KV accounting philosophy.

    Returns 0.0 for non-K3 models or when ``linear_attn_config`` is missing,
    so callers can invoke unconditionally.
    """
    try:
        model_config = getattr(model, "model_config", None)
        if model_config is None:
            return 0.0
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None or getattr(hf_config, "model_type", None) != "kimi_k3":
            return 0.0
        # K3 is a VL model; ``linear_attn_config`` lives on ``text_config``.
        # ModelWrapper exposes ``text_config`` on the wrapped model; fall
        # back to the HF config's nested ``text_config`` if the wrapper
        # does not.
        text_config = getattr(model, "text_config", None) or getattr(hf_config, "text_config", None)
        if text_config is None:
            return 0.0
        linear_attn_cfg = getattr(text_config, "linear_attn_config", None)
        if not isinstance(linear_attn_cfg, dict):
            return 0.0
        kda_layers = linear_attn_cfg.get("kda_layers") or []
        if not kda_layers:
            return 0.0
        num_kda_layers = len(kda_layers)
        head_dim = int(linear_attn_cfg.get("head_dim", 128) or 128)
        num_heads = int(linear_attn_cfg.get("num_heads", 96) or 96)
        short_conv_kernel_size = int(linear_attn_cfg.get("short_conv_kernel_size", 4) or 4)

        parallel_config = getattr(model_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        if tp_size <= 0:
            tp_size = 1
        num_heads_per_rank = max(1, num_heads // tp_size)

        batch_size = 1

        recurrent_state_bytes_per_layer = (
            batch_size * num_heads_per_rank * head_dim * head_dim * 4  # fp32
        )
        conv_state_bytes_per_layer = (
            3 * batch_size * (short_conv_kernel_size - 1) * (num_heads_per_rank * head_dim) * 2  # bf16
        )

        total_bytes = num_kda_layers * (recurrent_state_bytes_per_layer + conv_state_bytes_per_layer)
        return total_bytes / 1024**3
    except Exception as e:
        logger.warning("Failed to compute K3 KDA state size: %s", e)
        return 0.0


def _make_k3_mla_prolog_hook(original_compute_mla_prolog):
    """Build the K3 MLA prolog hook while keeping it independently testable."""

    def _patched_compute_mla_prolog(self, hidden_states_view, cos, sin, *, is_decode_values=None):
        if not bool(getattr(self._inner, "use_output_gate", False)):
            return original_compute_mla_prolog(
                self,
                hidden_states_view,
                cos,
                sin,
                is_decode_values=is_decode_values,
            )

        self.q_a_proj_weight, self.q_a_proj_scale, self.q_a_proj_offset = self.extract_qparams(self.q_a_proj)
        self.q_b_proj_weight, self.q_b_proj_scale, self.q_b_proj_offset = self.extract_qparams(self.q_b_proj)
        self.kv_a_proj_weight, self.kv_a_proj_scale, self.kv_a_proj_offset = self.extract_qparams(
            self.kv_a_proj_with_mqa
        )
        self.q_a_layernorm_weight = self.q_a_layernorm.weight.data
        self.kv_a_layernorm_weight = self.kv_a_layernorm.weight.data
        linear_quant_enabled = (
            self.q_a_proj_scale is not None and self.q_b_proj_scale is not None and self.kv_a_proj_scale is not None
        )
        if linear_quant_enabled:
            return original_compute_mla_prolog(
                self,
                hidden_states_view,
                cos,
                sin,
                is_decode_values=is_decode_values,
            )

        qa = torch.mm(hidden_states_view, self.q_a_proj_weight.t())
        return torch.ops.tensor_cast.mla_prolog(
            hidden_states_view,
            qa,
            cos,
            sin,
            self.q_a_layernorm_weight,
            self.q_b_proj_weight,
            self.kv_a_proj_weight,
            self.kv_a_layernorm_weight,
            self._num_heads_per_rank,
            self.qk_head_dim,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
            self.q_lora_rank,
            is_decode_values=is_decode_values,
        )

    return _patched_compute_mla_prolog


def _patch_model_classes_for_kimi_k3(config, model_id) -> bool:
    """Monkey-patch remote model classes before model instantiation.

    These patches modify **class-level methods**, so they MUST run before the
    HF loader constructs model objects. Requires ``model_id`` to locate and
    import the remote modeling files via ``get_class_from_dynamic_module``
    (HF mode / local path mode) or via the ModelScope snapshot directory
    (ModelScope mode, auto-detected by ``_get_k3_class_from_source``).

    When ``config.name_or_path`` is set (e.g. by ModelScope SDK to the local
    snapshot path), it takes precedence over ``model_id`` for class loading,
    because ``modelscope.AutoModel.from_config`` will later use
    ``config.name_or_path`` as the repo_id for
    ``get_class_from_dynamic_module``. Using the same path here ensures the
    patched class object is the one actually instantiated.
    """
    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k3" or model_id is None:
        return False

    # Prefer config.name_or_path (resolved local snapshot path) over the raw
    # model_id when loading classes. Under ModelScope mode the two differ:
    #   model_id = "moonshotai/Kimi-K3"             (HF Hub id, hits HF cache)
    #   config.name_or_path = "C:\...\modelscope\...\snapshots\master"  (local)
    # modelscope.AutoModel.from_config uses config.name_or_path as repo_id,
    # so we must patch the class loaded from the same path.
    effective_model_id = getattr(config, "name_or_path", None) or model_id
    if effective_model_id != model_id:
        logger.info(
            "K3 class patch: using config.name_or_path=%s instead of model_id=%s "
            "to match the loader's class resolution path",
            effective_model_id,
            model_id,
        )

    import signal as _signal

    patched = False

    # ----------------------------------------------------------------
    # Pre-Patch: Windows SIGALRM — resolve trust_remote_code without alarm
    #
    # ----------------------------------------------------------------
    if not hasattr(_signal, "SIGALRM"):
        import transformers.dynamic_module_utils

        _orig_resolve = transformers.dynamic_module_utils.resolve_trust_remote_code
        if not getattr(_orig_resolve, "_tensor_cast_patched", False):

            def _patched_resolve(trust_remote_code, *args, **kwargs):
                if trust_remote_code is None:
                    trust_remote_code = True
                return _orig_resolve(trust_remote_code, *args, **kwargs)

            _patched_resolve._tensor_cast_patched = True
            transformers.dynamic_module_utils.resolve_trust_remote_code = _patched_resolve

    # =================================================================
    # KimiK3ForConditionalGeneration.forward — filter TC kwargs
    #
    # ----------------------------------------------------------------
    # TensorCast injects extra kwargs (attention_meta, kv_cache_by_layers,
    #      etc.) via model_runner, but K3's VL forward only accepts standard
    #      HF keys. Also maps image_grid_thw → grid_thws (generic → K3 name).
    # =================================================================
    try:
        class_ref_vl = "modeling_kimi_k3.KimiK3ForConditionalGeneration"
        vl_cls = _get_k3_class_from_source(class_ref_vl, effective_model_id, "auto")

        if not hasattr(vl_cls, "_original_vl_forward"):
            vl_cls._original_vl_forward = vl_cls.forward

        _STANDARD_K3_VL_FORWARD_KEYS = frozenset(
            {
                "input_ids",
                "pixel_values",
                "grid_thws",
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
                "labels",
                "use_cache",
                "output_attentions",
                "output_hidden_states",
                "return_dict",
            }
        )

        def patched_vl_forward(self, *args, **kwargs):
            # Inject TC kwargs into attention layers BEFORE calling the
            # original forward (which filters them out). The decoder
            # (patched by the patch) reads them back from _extra_forward_kwargs.
            from tensor_cast.transformers.model import _EXTRA_TC_KWARGS_KEYS

            _tc_extra = {k: kwargs[k] for k in _EXTRA_TC_KWARGS_KEYS if k in kwargs and kwargs[k] is not None}
            if _tc_extra:
                try:
                    for layer in self.language_model.model.layers:
                        if hasattr(layer, "self_attn"):
                            layer.self_attn._extra_forward_kwargs = _tc_extra
                except AttributeError as e:
                    logger.warning(
                        "Failed to inject TC kwargs into K3 attention layers: %s. "
                        "This may affect tensor casting for Kimi K3.",
                        e,
                    )

            hf_kwargs = {k: v for k, v in kwargs.items() if k in _STANDARD_K3_VL_FORWARD_KEYS}
            # Map generic image_grid_thw → K3's grid_thws
            if "grid_thws" not in hf_kwargs and "image_grid_thw" in kwargs:
                hf_kwargs["grid_thws"] = kwargs["image_grid_thw"]
            return vl_cls._original_vl_forward(self, *args, **hf_kwargs)

        vl_cls.forward = patched_vl_forward
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 VL forward: {e}")

    # =================================================================
    # _merge_input_ids_with_image_features — meta device stub
    #
    # ----------------------------------------------------------------
    # During torch.compile graph capture, input_ids live on 'meta'
    #      device. The original merge function calls embedding layers which
    #      raise on meta tensors. Return correctly-shaped meta embedding.
    # NOTE: K3 signature differs from K2.5:
    #   K3: (image_features, inputs_embeds, input_ids, attention_mask, labels)
    #   K2.5: (image_features, feature_lens, input_ids, attention_mask, position_ids, labels)
    # =================================================================
    try:
        if not hasattr(vl_cls, "_original_merge_input_ids_with_image_features"):
            vl_cls._original_merge_input_ids_with_image_features = vl_cls._merge_input_ids_with_image_features

        def patched_merge_input_ids_with_image_features(
            self,
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask=None,
            labels=None,
        ):
            batch_size, sequence_length = input_ids.shape
            if input_ids.device.type == "meta":
                embed_dim = (
                    image_features[0].shape[-1] if len(image_features) > 0 else self.config.text_config.hidden_size
                )
                # K3 returns (final_embedding, final_attention_mask, final_labels, position_ids)
                position_ids = torch.zeros(
                    batch_size,
                    sequence_length,
                    dtype=torch.long,
                    device="meta",
                )
                return (
                    torch.empty(
                        batch_size,
                        sequence_length,
                        embed_dim,
                        device="meta",
                        dtype=inputs_embeds.dtype,
                    ),
                    attention_mask,
                    labels,
                    position_ids,
                )

            return vl_cls._original_merge_input_ids_with_image_features(
                self,
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )

        vl_cls._merge_input_ids_with_image_features = patched_merge_input_ids_with_image_features
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 merge_input_ids: {e}")

    # =================================================================
    # MoonViT3dEncoder — register tensor_cast attention backend
    #
    # ----------------------------------------------------------------
    # K3 vision encoder checks ``attn_implementation`` and dispatches
    #      via ``VL_VISION_ATTENTION_FUNCTIONS`` (remote modeling_kimi_k3.py
    #      L532). The remote dict only ships ``flash_attention_2`` and
    #      ``eager`` backends, so any other value raises ``KeyError``.
    #
    #      The vision tower's ``attn_implementation`` is set from
    #      ``VisionTowerConfig._attn_implementation`` (remote L869), which
    #      copies the **root** ``KimiK3Config._attn_implementation`` — NOT
    #      ``vision_config._attn_implementation``. The root config has no
    #      explicit ``_attn_implementation`` (config.json omits it), so
    #      transformers auto-resolves it. Because
    #      ``MoonViT3dPretrainedModel._supports_sdpa = True`` (remote L654)
    #      and flash_attn is unavailable, transformers selects ``"sdpa"``.
    #      the patch's downgrade of ``config.vision_config._attn_implementation`` to
    #      ``"tensor_cast"`` therefore never reaches the vision tower.
    #
    # Register ``visual_tc_adapter`` under every backend transformers
    #      may auto-select for the vision tower: ``tensor_cast``,
    #      ``eager``, and ``sdpa`` (the actual auto-resolved value). The
    #      adapter handles meta tensors (calls ``tensor_cast.attention`` op
    #      for tracing/perf modeling) and real tensors (O(n²) fallback with a
    #      seq_length > 4096 OOM guard). Non-K3 models are unaffected — the
    #      registration only mutates the K3 remote module's dict.
    # =================================================================
    try:
        class_ref_enc = "modeling_kimi_k3.MoonViT3dEncoder"
        _get_k3_class_from_source(class_ref_enc, effective_model_id, "auto")

        # Find the remote module to register the attention backend
        import sys as _sys

        for name, module in list(_sys.modules.items()):
            if "kimi_k3" in name and "modeling_kimi_k3" in name:
                if hasattr(module, "VL_VISION_ATTENTION_FUNCTIONS"):

                    def visual_tc_adapter(
                        q,
                        k,
                        v,
                        q_cu_seqlens,
                        k_cu_seqlens,
                        max_seqlen_q,
                        max_seqlen_k,
                        deterministic=False,
                    ):
                        import math

                        seq_length = q.shape[0]
                        num_heads = q.shape[1]
                        head_dim = q.shape[-1]

                        if q.device.type == "meta":
                            # Call the fused tensor_cast.attention op so it
                            # appears in the chrome trace for performance
                            # modeling. Shape mapping (varlen → TC convention):
                            #   q: (seq_len, num_heads, head_dim)
                            #      → query: (seq_len, num_heads * head_dim)
                            query = q.reshape(seq_length, num_heads * head_dim)
                            return torch.ops.tensor_cast.attention(
                                query,
                                k,
                                v,
                                None,  # attention_mask
                                None,  # block_table
                                None,  # query_start_loc
                                None,  # seq_lens
                                None,  # query_lens
                            )

                        if seq_length > 4096:
                            logger.warning(
                                "K3 visual attention sequence length %d exceeds "
                                "safe threshold. Skipping O(n²) attention to avoid OOM.",
                                seq_length,
                            )
                            return torch.zeros(
                                seq_length,
                                num_heads * head_dim,
                                device=q.device,
                                dtype=q.dtype,
                            )

                        # Build block-diagonal causal mask
                        attention_mask = torch.full(
                            [1, seq_length, seq_length],
                            float("-inf"),
                            device=q.device,
                            dtype=q.dtype,
                        )
                        q_cu_seqlens_list = q_cu_seqlens.tolist()
                        for i in range(1, len(q_cu_seqlens_list)):
                            start = q_cu_seqlens_list[i - 1]
                            end = q_cu_seqlens_list[i]
                            attention_mask[..., start:end, start:end] = 0.0

                        q = q.transpose(0, 1)
                        k = k.transpose(0, 1)
                        v = v.transpose(0, 1)
                        attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
                        attn_weight += attention_mask
                        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
                        attn_output = attn_weight @ v
                        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
                        return attn_output

                    module.VL_VISION_ATTENTION_FUNCTIONS["tensor_cast"] = visual_tc_adapter
                    module.VL_VISION_ATTENTION_FUNCTIONS["eager"] = visual_tc_adapter
                    # ``sdpa`` is what transformers actually auto-resolves the
                    # vision tower to (root config has no explicit
                    # _attn_implementation and _supports_sdpa=True). Without
                    # this entry the vision forward raises ``KeyError: 'sdpa'``.
                    module.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = visual_tc_adapter
                    break
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 visual encoder: {e}")

    # =================================================================
    # KimiDeltaAttention.forward → decomposed linear_attn_* sub-ops
    # ----------------------------------------------------------------
    # KDA uses fla-core's chunk_kda / fused_recurrent_kda which are
    #      untraceable and stubbed by the patch. Previously the entire forward was
    #      routed to a single ``tensor_cast.linear_attention`` op (monolithic
    #      fusion). Now decomposed into granular sub-ops to match NPU
    #      profiling granularity and allow per-GEMM quantization visibility.
    #
    # Decomposition (8 GEMMs → aten.mm, 3 sub-ops for conv/delta/norm):
    #   1. q_proj, k_proj, v_proj           → aten.mm (3 GEMMs)
    #   2. causal conv on merged qkv        → linear_attn_causal_conv[_update]
    #   3. f_a_proj, f_b_proj (delta-rule g)→ aten.mm (2 GEMMs)
    #   4. b_proj (beta)                    → aten.mm (1 GEMM)
    #   5. chunk/recurrent gated delta rule → linear_attn_[chunk|recurrent]_gated_delta_rule
    #   6. g_proj (output gate)             → aten.mm (1 GEMM)
    #   7. gated RMSNorm                    → linear_attn_gated_rmsnorm
    #   8. o_proj                            → aten.mm (1 GEMM)
    #
    # K3 vs Qwen3.5 key differences:
    #   - q/k/v: 3 separate GEMMs (not fused in_proj_qkv)
    #   - delta-rule g: vector (b,t,h,d) from f_a_proj+f_b_proj GEMMs
    #     → skip linear_attn_fused_gdn_gating (expects scalar g from
    #       A_log/dt_bias, incompatible with K3's vector g)
    #   - output gate: g_proj (use_full_rank_gate=true), sigmoid activation
    #     (vs Qwen3.5's in_proj_z + silu — FLOPs approximated by op model)
    #   - beta: b_proj raw output (sigmoid inside delta-rule kernel)
    #
    # K3 attribute mapping (from KimiDeltaAttention.__init__):
    #   self.num_k_heads  (config linear_attn_config["num_heads"] = 96)
    #   self.num_heads    (same, used as num_v_heads)
    #   self.head_k_dim   (= self.head_dim = 128)
    #   self.head_dim     (head_v_dim = 128)
    #   self.conv_size    (short_conv_kernel_size = 4)
    #   self.use_full_rank_gate (true in K3 config)
    #   self.o_norm       (FusedRMSNormGated stub with .weight and .eps)
    # =================================================================
    try:
        class_ref_kda = "modeling_kimi_linear.KimiDeltaAttention"
        kda_cls = _get_k3_class_from_source(class_ref_kda, effective_model_id, "auto")

        def _k3_kda_has_previous_state(cache_position) -> bool:
            """Determine if KDA has previous KV state for decode vs prefill."""
            if cache_position is None or not hasattr(cache_position, "numel"):
                return False
            if cache_position.numel() == 0:
                return False
            # TensorCast may inject metadata for compile/trace scenarios
            tagged = getattr(cache_position, "tensor_cast_has_previous_state", None)
            if tagged is not None:
                return bool(tagged)
            # Meta tensors (symbolic tracing) cannot be queried for values
            if hasattr(cache_position, "is_meta") and cache_position.is_meta:
                return False
            try:
                return cache_position[0].item() > 0
            except RuntimeError:
                return False

        def _k3_is_decode_batch(attention_meta) -> bool:
            """Check if all requests in the batch are in decode mode.

            input_generator sets ``attention_meta.is_decode_values`` (a list of
            bools, one per request) from the ``--decode`` CLI flag. K3's
            input_generator does not inject ``cache_position`` with decode
            metadata (unlike Qwen3.5), so this is the primary decode signal.
            Returns False when the metadata is absent (e.g. symbolic tracing).
            """
            if attention_meta is None:
                return False
            is_decode_values = getattr(attention_meta, "is_decode_values", None)
            if is_decode_values is None:
                return False
            try:
                return all(bool(v) for v in is_decode_values)
            except TypeError:
                return False

        def _k3_infer_decode_from_request_lengths(attention_meta) -> bool:
            """Recover decode intent for runs launched without ``--decode``.

            Requests are packed into ``[1, num_tokens, d]``, so with more than
            one request per DP rank the packed ``seq_len`` (= batch of decode
            steps) exceeds 1 and the ``seq_len == 1`` fallback misroutes KDA to
            the prefill (chunk) kernel. When every request contributes exactly
            one new token on top of a non-empty context, the batch is
            unambiguously decode regardless of the ``--decode`` flag.
            """
            if attention_meta is None:
                return False
            query_lens = getattr(attention_meta, "query_lens_values", None) or []
            seq_lens = getattr(attention_meta, "seq_lens_values", None) or []
            if not query_lens or not seq_lens or len(query_lens) != len(seq_lens):
                return False
            try:
                return all(int(q) == 1 for q in query_lens) and any(int(s) > 1 for s in seq_lens)
            except (TypeError, ValueError):
                return False

        def _patched_kda_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cache_params=None,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            attention_meta = kwargs.pop("attention_meta", None)
            del kwargs, cache_params, attention_mask

            # TP-local head counts: q_proj is COLWISE-sharded by the mla_config
            # TP branch (head_num=96), carrying a ``tp_size`` attribute after
            # TP sharding. Defaults to 1 for single-device runs.
            _tp_size = getattr(self.q_proj, "tp_size", 1)
            local_num_k_heads = self.num_k_heads // _tp_size
            local_num_v_heads = self.num_heads // _tp_size
            head_k_dim = self.head_k_dim
            head_v_dim = self.head_dim
            conv_kernel_size = self.conv_size

            batch_size, seq_len, _ = hidden_states.shape

            # Determine decode (recurrent) vs prefill (chunk) path.
            # input_generator does not inject cache_position for kimi_k3, so
            # cache_position is normally None. Decode is detected from
            # attention_meta.is_decode_values (set by --decode). When that is
            # also unavailable (symbolic tracing), fall back to seq_len==1.
            # ``--decode``-less decode runs: requests are packed into
            # [1, num_tokens, d] so seq_len == num_tokens != 1 and the
            # seq_len fallback misses; recover the decode intent from
            # per-request lengths (every request contributes exactly 1 new
            # token on top of a non-empty context).
            has_previous_state = _k3_kda_has_previous_state(cache_position)
            _cp_absent = cache_position is None
            _cp_is_meta = bool(getattr(cache_position, "is_meta", False))
            _is_decode = _k3_is_decode_batch(attention_meta)
            _inferred_decode = not _is_decode and _k3_infer_decode_from_request_lengths(attention_meta)
            use_recurrent = (
                _is_decode or _inferred_decode or (seq_len == 1 and (has_previous_state or _cp_absent or _cp_is_meta))
            )
            # When decoding with MTP (seq_len > 1), flatten batch*seq_len into
            # the batch dim so the recurrent kernel sees seq_len==1 per item.
            flatten_decode_batch = use_recurrent and seq_len != 1
            core_batch_size = batch_size * seq_len if flatten_decode_batch else batch_size
            core_seq_len = 1 if flatten_decode_batch else seq_len

            # --- 1. Q/K/V projections (→ aten.mm) ---
            q_proj_states = self.q_proj(hidden_states)
            k_proj_states = self.k_proj(hidden_states)
            v_proj_states = self.v_proj(hidden_states)

            # --- 2. Short convolution (→ linear_attn_causal_conv[_update]) ---
            # Merge q/k/v into mixed_qkv (matches Qwen3.5 in_proj_qkv output layout)
            mixed_qkv = torch.cat([q_proj_states, k_proj_states, v_proj_states], dim=-1)
            # causal_conv performance model expects (b, dim, t) layout
            mixed_qkv = mixed_qkv.transpose(1, 2)
            conv_op = (
                torch.ops.tensor_cast.linear_attn_causal_conv_update
                if use_recurrent
                else torch.ops.tensor_cast.linear_attn_causal_conv
            )
            mixed_qkv = conv_op(mixed_qkv, conv_kernel_size)
            mixed_qkv = mixed_qkv.transpose(1, 2)

            # Split back and reshape to (b, t, h, d)
            key_dim = local_num_k_heads * head_k_dim
            value_dim = local_num_v_heads * head_v_dim
            query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
            query = query.reshape(core_batch_size, core_seq_len, local_num_k_heads, head_k_dim)
            key = key.reshape(core_batch_size, core_seq_len, local_num_k_heads, head_k_dim)
            value = value.reshape(core_batch_size, core_seq_len, local_num_v_heads, head_v_dim)

            # --- 3. Delta-rule gate g (vector per head) — 2 GEMMs (→ aten.mm) ---
            # K3: g = f_b_proj(f_a_proj(hidden_states)), shape (b,t,num_heads*head_dim)
            # This replaces linear_attn_fused_gdn_gating which expects scalar g
            # from A_log/dt_bias (Qwen3.5-style, incompatible with K3's vector g).
            # f_b_proj is now COLWISE TP-sharded, so output
            # is TP-local (12 heads × 128 = 1536 at TP=8). The -1 auto-infers
            # the local head count from the sharded output size, matching
            # query's local_num_v_heads. Profiling alignment: NPU runs 128→1536.
            g_delta = self.f_a_proj(hidden_states)
            g_delta = self.f_b_proj(g_delta)
            g_delta = g_delta.reshape(core_batch_size, core_seq_len, -1, head_v_dim)

            # --- 4. beta — 1 GEMM (→ aten.mm) ---
            # K3: beta = b_proj(hidden_states); sigmoid applied inside delta kernel
            # b_proj is now COLWISE TP-sharded, so beta is
            # TP-local (12 at TP=8), matching query's local_num_v_heads.
            # Profiling alignment: NPU runs 7168→12/rank.
            beta = self.b_proj(hidden_states)

            # --- 5. Delta rule (→ linear_attn_[chunk|recurrent]_gated_delta_rule) ---
            if use_recurrent:
                core_attn_out = torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule(
                    query, key, value, beta, g_delta, 1, 1
                )
            else:
                chunk_size = 64
                state_read_passes = 1 if has_previous_state else 0
                state_write_passes = 1
                core_attn_out = torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    beta,
                    g_delta,
                    chunk_size,
                    state_read_passes,
                    state_write_passes,
                )

            # --- 6. Output gate g (→ aten.mm) + Gated RMSNorm ---
            # K3: use_full_rank_gate=true → g_out = g_proj(hidden_states)
            # Uses sigmoid (vs Qwen3.5's silu); FLOPs approximated by op model.
            # g_proj is COLWISE TP-sharded,
            # so output is TP-local (12 heads × 128 = 1536 at TP=8). The -1
            # auto-infers the local head count, matching query's heads.
            if self.use_full_rank_gate:
                g_out = self.g_proj(hidden_states)
            else:
                g_out = self.g_b_proj(self.g_a_proj(hidden_states))
            g_out = g_out.reshape(core_batch_size, core_seq_len, -1, head_v_dim)

            norm_weight = getattr(self.o_norm, "weight", None)
            norm_eps = getattr(self.o_norm, "eps", 1e-6)
            core_attn_out = torch.ops.tensor_cast.linear_attn_gated_rmsnorm(core_attn_out, g_out, norm_weight, norm_eps)

            # --- 7. Output projection (→ aten.mm) ---
            core_attn_out = core_attn_out.reshape(core_batch_size * core_seq_len, -1)
            output = self.o_proj(core_attn_out)
            return output.reshape(batch_size, seq_len, -1)

        kda_cls.forward = _patched_kda_forward
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 KimiDeltaAttention: {e}")

    # =================================================================
    # KimiLinearModel._update_linear_attn_mask — meta tensor fix
    # (ported from Q3N)
    # ----------------------------------------------------------------
    # ``cache_position[0] > 0`` and ``torch.all(attention_mask == 1)``
    #      call ``.item()`` on meta tensors during symbolic tracing, which
    #      raises ``Tensor.item() cannot be called on meta tensors``.
    # =================================================================
    try:
        class_ref_linear_model = "modeling_kimi_linear.KimiLinearModel"
        linear_model_cls = _get_k3_class_from_source(class_ref_linear_model, effective_model_id, "auto")

        def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
            is_meta = (hasattr(cache_position, "is_meta") and cache_position.is_meta) or (
                attention_mask is not None and hasattr(attention_mask, "is_meta") and attention_mask.is_meta
            )
            if is_meta:
                return attention_mask

            try:
                if cache_position is None:
                    cache_condition = False
                else:
                    cache_condition = cache_position[0] > 0 if cache_position.numel() > 0 else False
                mask_condition = (
                    torch.all(attention_mask == 1).item()
                    if attention_mask is not None and attention_mask.numel() > 0
                    else False
                )
                if cache_condition or mask_condition:
                    return None
            except RuntimeError:
                logger.warning(
                    "K3 _update_linear_attn_mask fallback due to runtime error",
                    exc_info=True,
                )
            return attention_mask

        linear_model_cls._update_linear_attn_mask = _patched_update_linear_attn_mask
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 _update_linear_attn_mask: {e}")

    # =================================================================
    # KimiSparseMoeBlock.forward/moe_infer — stub for graph tracing
    #
    # ----------------------------------------------------------------
    # The real MoE forward contains dynamic dispatch logic (expert
    #      selection + token routing + expert combine) that torch.compile
    #      cannot trace. Stub returns correct shapes without executing
    #      experts. Actual performance modeling is handled by
    #      transformations.patch_moe() which wraps with fused MoELayer.
    # =================================================================
    try:
        class_ref_moe = "modeling_kimi_linear.KimiSparseMoeBlock"
        moe_cls = _get_k3_class_from_source(class_ref_moe, effective_model_id, "auto")

        def patched_moe_forward(_self, hidden_states):
            return torch.zeros_like(hidden_states)

        def patched_moe_infer(_self, x, _topk_ids, _topk_weight):
            return torch.zeros_like(x)

        if not hasattr(moe_cls, "_original_forward"):
            moe_cls._original_forward = moe_cls.forward
        moe_cls.forward = patched_moe_forward

        if not hasattr(moe_cls, "_original_moe_infer"):
            moe_cls._original_moe_infer = moe_cls.moe_infer
        moe_cls.moe_infer = patched_moe_infer
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 KimiSparseMoeBlock: {e}")

    # =================================================================
    # KimiMoEGate.forward — deterministic routing for simulation
    #
    # ----------------------------------------------------------------
    # The real gate performs top-k softmax + random sampling which is
    #      non-deterministic and untraceable. Replace with equal-weight
    #      routing to produce deterministic shapes during graph capture.
    # =================================================================
    try:
        class_ref_gate = "modeling_kimi_linear.KimiMoEGate"
        gate_cls = _get_k3_class_from_source(class_ref_gate, effective_model_id, "auto")

        def patched_gate_forward(self, hidden_states, **kwargs):
            if hidden_states.dim() == 3:
                bsz, seq_len, _ = hidden_states.shape
            else:
                bsz = hidden_states.shape[0]
                seq_len = 1
            device = hidden_states.device
            dtype = hidden_states.dtype
            top_k = self.top_k
            topk_idx = torch.zeros(bsz * seq_len, top_k, dtype=torch.long, device=device)
            topk_weight = torch.ones(bsz * seq_len, top_k, dtype=dtype, device=device) / top_k
            return topk_idx, topk_weight

        gate_cls.forward = patched_gate_forward
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 KimiMoEGate: {e}")

    # =================================================================
    # KimiMLAAttention — position_embeddings resolution + output gate
    #
    # ----------------------------------------------------------------
    # (a) K3's decoder only passes ``position_ids``, not pre-computed
    #         RoPE (cos, sin) tensors. TC MLA needs explicit
    #         position_embeddings. This method computes them from
    #         position_ids via the rotary_emb cache.
    #      (b) K3's MLA has ``mla_use_output_gate=True``: the attention
    #         output is gated by ``sigmoid(g_proj(hidden_states))`` before
    #         ``o_proj``. TC's MLA wrapper doesn't have this gate.
    # K3-specific: ``mla_use_nope=True`` means RoPE is NOT applied (identity
    # cos=1, sin=0). The rotary_emb is None on K3's MLA module, so the
    # position_embeddings resolver returns identity RoPE — correct for K3.
    # =================================================================
    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

    # Register mla_prolog before the K3 prolog hook can call it.
    _install_mla_prolog_op()

    # position_embeddings resolver
    if not hasattr(MultiheadLatentAttentionTensorCast, "_patched_rope_resolve"):

        def _patched_resolve_position_embeddings(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if position_embeddings is not None:
                return position_embeddings

            position_ids = kwargs.get("position_ids", None)

            if position_ids is not None and self._has_rotary_emb and hidden_states.device.type != "meta":
                max_pos = position_ids.max().item() + 1
                if hasattr(self.rotary_emb, "cos_cached"):
                    if self.rotary_emb.cos_cached.shape[0] < max_pos:
                        self.rotary_emb._update_cos_sin_tables(max_pos, hidden_states.device, hidden_states.dtype)
                cos = self.rotary_emb.cos_cached[position_ids].to(hidden_states.dtype)
                sin = self.rotary_emb.sin_cached[position_ids].to(hidden_states.dtype)
                return (cos, sin)

            # K3 path: rotary_emb is None (mla_use_nope=True), so return
            # identity RoPE (cos=1, sin=0). This is correct for K3.
            if self._has_rotary_emb:
                import warnings

                warnings.warn(
                    "position_embeddings was not provided and position_ids is "
                    "unavailable; RoPE will be disabled (cos=1, sin=0). If this "
                    "model uses RoPE-based attention, simulation results may be "
                    "inaccurate.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            seq_len = hidden_states.shape[1]
            dim = self.qk_rope_head_dim
            cos = torch.ones(seq_len, dim, device=hidden_states.device, dtype=hidden_states.dtype)
            sin = torch.zeros(seq_len, dim, device=hidden_states.device, dtype=hidden_states.dtype)
            return (cos, sin)

        MultiheadLatentAttentionTensorCast._resolve_position_embeddings = _patched_resolve_position_embeddings
        MultiheadLatentAttentionTensorCast._patched_rope_resolve = True
        patched = True

    # K3 customizes only the prolog and post-merge gate. The shared MLA
    # forward remains the single owner of projection, attention, DCP, and
    # phase-merge semantics so future core signature changes cannot drift here.
    if not hasattr(MultiheadLatentAttentionTensorCast, "_patched_k3_mla_hooks"):
        _original_mla_forward = MultiheadLatentAttentionTensorCast.forward
        _original_compute_mla_prolog = MultiheadLatentAttentionTensorCast._compute_mla_prolog
        _original_postprocess_attention_output = MultiheadLatentAttentionTensorCast._postprocess_attention_output

        def _is_k3_mla(self) -> bool:
            return bool(getattr(self._inner, "use_output_gate", False))

        _patched_compute_mla_prolog = _make_k3_mla_prolog_hook(_original_compute_mla_prolog)

        def _patched_postprocess_attention_output(self, attn_output, hidden_states):
            if not _is_k3_mla(self) or not hasattr(self._inner, "g_proj"):
                return _original_postprocess_attention_output(self, attn_output, hidden_states)

            g_proj = self._inner.g_proj
            if getattr(g_proj, "tp_size", 1) > 1:
                local_gate = torch.sigmoid(g_proj(hidden_states))
            else:
                full_gate = torch.sigmoid(g_proj(hidden_states))
                gate_slice = self._num_heads_per_rank * self.v_head_dim
                gate_offset = self.tp_group.rank_in_group * gate_slice
                local_gate = full_gate[..., gate_offset : gate_offset + gate_slice]
            return attn_output * local_gate

        def _patched_mla_forward_with_k3_position_embeddings(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache_unused: Optional[torch.Tensor] = None,
            attention_meta=None,
            **kwargs,
        ):
            if not _is_k3_mla(self):
                return _original_mla_forward(
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    kv_cache_unused,
                    attention_meta,
                    **kwargs,
                )
            if not hasattr(self, "_has_rotary_emb"):
                self._has_rotary_emb = hasattr(self._inner, "rotary_emb") and self._inner.rotary_emb is not None
            if position_embeddings is None:
                position_embeddings = self._resolve_position_embeddings(hidden_states, None, **kwargs)
            return _original_mla_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                kv_cache_unused,
                attention_meta,
                **kwargs,
            )

        MultiheadLatentAttentionTensorCast._compute_mla_prolog = _patched_compute_mla_prolog
        MultiheadLatentAttentionTensorCast._postprocess_attention_output = _patched_postprocess_attention_output
        MultiheadLatentAttentionTensorCast.forward = _patched_mla_forward_with_k3_position_embeddings
        MultiheadLatentAttentionTensorCast._patched_k3_mla_hooks = True
        patched = True
    # =================================================================
    # KimiDecoderLayer — AttnRes cross-layer residual stub
    # (K3-specific)
    # ----------------------------------------------------------------
    # K3's ``_forward_attn_residual`` accumulates ``block_residual``
    #      across layers (every 12 layers) and calls ``_apply_attn_res``
    #      twice per layer. On meta device (symbolic tracing), the original
    #      implementation decomposes into many aten ops (cat, pow, mean,
    #      rsqrt, mul, sum, softmax, matmul), but on NPU this is a single
    #      fused kernel (``_apply_attn_res_kernel``, ~6.5us). This patch
    #      re-implements the AttnRes flow on meta device using the
    #      ``tensor_cast.apply_attn_res`` fused op,
    #      matching the NPU's fused kernel granularity.
    #      The cross-layer ``block_residual`` state is properly accumulated
    #      (cat every 12 layers) and passed between layers, mirroring the
    #      original ``_forward_attn_residual`` semantics.
    # =================================================================
    try:
        class_ref_decoder = "modeling_kimi_linear.KimiDecoderLayer"
        decoder_cls = _get_k3_class_from_source(class_ref_decoder, effective_model_id, "auto")

        if not hasattr(decoder_cls, "_original_kda_decoder_forward"):
            decoder_cls._original_kda_decoder_forward = decoder_cls.forward

        def _patched_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            block_residual=None,
            **kwargs,
        ):
            # On meta device, implement AttnRes using the fused op
            if self.use_attn_residuals and hidden_states.is_meta:
                batch_size, seq_len, hidden_size = hidden_states.shape
                prefix_sum = hidden_states

                # ① Apply AttnRes BEFORE attention (if block_residual has
                #    accumulated blocks from previous 12-layer groups).
                #    Uses self_attention_res_proj / self_attention_res_norm.
                if block_residual is not None and block_residual.shape[1] > 0:
                    hidden_states = torch.ops.tensor_cast.apply_attn_res(
                        prefix_sum.view(-1, hidden_size),
                        block_residual,
                        self.self_attention_res_norm.weight,
                        self.self_attention_res_proj.weight.squeeze(0),
                        self.self_attention_res_norm.variance_epsilon,
                    ).view(batch_size, seq_len, hidden_size)

                # ② Accumulate block_residual every attn_res_block_size
                #    layers (default 12). The current prefix_sum is snapshotted
                #    into block_residual and prefix_sum is nulled so that the
                #    attention residual starts fresh from the attn output.
                if self.layer_idx % self.attn_res_block_size == 0:
                    new_block = prefix_sum.view(-1, hidden_size).unsqueeze(1)
                    if block_residual is None:
                        block_residual = new_block
                    else:
                        block_residual = torch.cat([block_residual, new_block], dim=1)
                    prefix_sum = None

                # ③ Input layernorm + Self Attention
                hidden_states = self.input_layernorm(hidden_states)

                # Recover tensor_cast-specific kwargs filtered by the VL
                # forward. P6 stores attention_meta / kv_cache_by_layers
                # / etc. on self.self_attn._extra_forward_kwargs because K3's
                # VL forward only accepts standard HF keys. Without this
                # recovery the MLA wrapper (MultiheadLatentAttentionTensorCast)
                # runs with kv_cache=None and attention_meta=None, which
                # records the multihead_latent_attention op with null args and
                # breaks the performance model at
                # _multihead_latent_attention_properties_helper
                # (kv_cache.size(-1) on None).
                if "attention_meta" not in kwargs:
                    extra_kwargs = getattr(self.self_attn, "_extra_forward_kwargs", None)
                    if extra_kwargs is not None and extra_kwargs.get("attention_meta") is not None:
                        for k, v in extra_kwargs.items():
                            if k not in kwargs and v is not None:
                                kwargs[k] = v

                if self.is_linear_attn is False:
                    attn_out = self.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )
                else:
                    attn_out = self.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        cache_params=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )
                # K3 MLA/KDA forward returns a single tensor (not a tuple)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]

                # ④ Update prefix_sum (residual add with attention output)
                if prefix_sum is not None:
                    prefix_sum = prefix_sum + attn_out
                else:
                    prefix_sum = attn_out

                # ⑤ Apply AttnRes AFTER attention, BEFORE MLP.
                #    Uses mlp_res_proj / mlp_res_norm.
                hidden_states = torch.ops.tensor_cast.apply_attn_res(
                    prefix_sum.view(-1, hidden_size),
                    block_residual,
                    self.mlp_res_norm.weight,
                    self.mlp_res_proj.weight.squeeze(0),
                    self.mlp_res_norm.variance_epsilon,
                ).view(batch_size, seq_len, hidden_size)

                # ⑥ Post attention layernorm + Fully Connected (MoE or MLP)
                hidden_states = self.post_attention_layernorm(hidden_states)
                if hasattr(self, "block_sparse_moe"):
                    hidden_states = self.block_sparse_moe(hidden_states)
                else:
                    hidden_states = self.mlp(hidden_states)

                # ⑦ Update prefix_sum (residual add with MLP/MoE output)
                if prefix_sum is None:
                    prefix_sum = hidden_states
                else:
                    prefix_sum = prefix_sum + hidden_states

                # Return (prefix_sum, block_residual) to match
                # KimiLinearModel.forward's AttnRes unpacking convention.
                return prefix_sum, block_residual

            # Non-meta path: use original forward (preserves AttnRes semantics)
            return decoder_cls._original_kda_decoder_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                block_residual=block_residual,
                **kwargs,
            )

        decoder_cls.forward = _patched_decoder_forward
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 KimiDecoderLayer AttnRes: {e}")

    # =================================================================
    # KimiDynamicCache — compatibility stub
    # (K3-specific)
    # ----------------------------------------------------------------
    # K3's ``KimiLinearModel.forward`` asserts
    #      ``isinstance(past_key_values, KimiDynamicCache)`` and constructs
    #      a ``KimiDynamicCache`` when ``use_cache=True``. For simulation
    #      (``use_cache=False``), the cache is never created and the assert
    #      is skipped. This patch makes KimiDynamicCache tolerant of meta
    #      device construction in case use_cache is enabled.
    # The KimiDynamicCache is constructed with ``config=self.config`` and
    # initializes conv_states/recurrent_states lists. On meta device, this
    # is safe (lists of None). No patch needed here.
    # =================================================================
    # No-op here — documented for completeness. If use_cache=True
    # causes issues later, patch KimiDynamicCache.__init__ here.

    # =================================================================
    # MoonVision3dPatchEmbed — support 2D (flattened) input
    #
    # ----------------------------------------------------------------
    # During simulation, vision tokens may arrive as a flat 2D tensor
    #      (total_tokens, channels) rather than 3D patches. The original
    #      Conv2d projection expects 4D input. Reshape 2D input back to
    #      4D chunks and use linear projection instead.
    # =================================================================
    try:
        class_ref_patch_embed = "modeling_kimi_k3.MoonVision3dPatchEmbed"
        patch_embed_cls = _get_k3_class_from_source(class_ref_patch_embed, effective_model_id, "auto")

        if not hasattr(patch_embed_cls, "_original_patch_embed_forward"):
            patch_embed_cls._original_patch_embed_forward = patch_embed_cls.forward

        def patched_patch_embed_forward(
            self,
            x: torch.Tensor,
            grid_thws: torch.Tensor,
        ) -> torch.Tensor:
            if x.dim() == 2:
                hidden_dim = x.shape[1]
                total_tokens = 0
                reshaped_parts = []
                out_dim, in_channels, kH, kW = self.proj.weight.shape
                expected_hidden_dim = in_channels * kH * kW
                if hidden_dim != expected_hidden_dim:
                    raise ValueError(
                        f"Hidden dim mismatch: input has {hidden_dim}, "
                        f"but proj expects {expected_hidden_dim} "
                        f"(in_channels={in_channels}, kernel_size=({kH}, {kW}))"
                    )

                for t, h, w in grid_thws.tolist():
                    num_tokens = t * h * w
                    part = x[total_tokens : total_tokens + num_tokens]
                    part = part.view(num_tokens, in_channels, kH, kW)
                    linear_weight = self.proj.weight.view(out_dim, in_channels * kH * kW)
                    projected = torch.nn.functional.linear(
                        part.reshape(num_tokens, -1),
                        linear_weight,
                        self.proj.bias,
                    )
                    reshaped_parts.append(projected)
                    total_tokens += num_tokens

                x = torch.cat(reshaped_parts, dim=0)
            else:
                x = self.proj(x).view(x.size(0), -1)
            x = self.pos_emb(x, grid_thws)
            return x

        patch_embed_cls.forward = patched_patch_embed_forward
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch K3 MoonVision3dPatchEmbed: {e}")

    # =================================================================
    # SituAndMul direct-mode + MLP forward patch
    #     (eliminate redundant cat→slice in FX graph)
    # ----------------------------------------------------------------
    # K3's KimiBlockSparseMLP.forward does:
    #   gate_up = torch.cat([w1(x), w3(x)], dim=-1)
    #   act_fn(gate_up)  # SituAndMul slices gate_up back into gate/up
    # When MSModeling fuses w1/w3 into one GMM, this creates a deep
    # split→getitem→cat→slice chain that the SiTU pattern match cannot
    # penetrate, blocking the SiTU pattern match.
    #
    # Patch SituAndMul.forward to accept (gate, up) as two separate
    # args (direct mode), and patch KimiBlockSparseMLP/KimiMLP.forward
    # to call act_fn(gate, up) instead of act_fn(cat([gate, up])).
    # This removes the cat→slice from the FX graph entirely, letting
    # Pattern A (split+getitem) in the SiTU pattern match directly.
    # =================================================================
    try:
        situ_cls = _get_k3_class_from_source("modeling_kimi_linear.SituAndMul", effective_model_id, "auto")

        if not getattr(situ_cls, "_tensor_cast_situ_direct", False):
            _orig_situ_forward = situ_cls.forward

            def _situ_forward_direct(self, x_or_gate, up=None):
                if up is None:
                    return _orig_situ_forward(self, x_or_gate)
                gate_fp32 = x_or_gate.to(torch.float32)
                up_fp32 = up.to(torch.float32)
                situ_a = self.beta * torch.tanh(gate_fp32 / self.beta) * torch.sigmoid(gate_fp32)
                if self.linear_beta is not None:
                    up_fp32 = self.linear_beta * torch.tanh(up_fp32 / self.linear_beta)
                return (situ_a * up_fp32).to(x_or_gate.dtype)

            _situ_forward_direct._tensor_cast_situ_direct = True
            situ_cls.forward = _situ_forward_direct
            patched = True
            logger.info("Patched SituAndMul.forward for direct mode.")

        block_sparse_mlp_cls = _get_k3_class_from_source(
            "modeling_kimi_linear.KimiBlockSparseMLP",
            effective_model_id,
            "auto",
        )

        if not getattr(block_sparse_mlp_cls, "_tensor_cast_mlp_direct", False):

            def _block_sparse_mlp_forward(self, hidden_states):
                if self.config.hidden_act == "situ":
                    gate = self.w1(hidden_states)
                    up = self.w3(hidden_states)
                    current = self.act_fn(gate, up)
                else:
                    current = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
                return self.w2(current)

            _block_sparse_mlp_forward._tensor_cast_mlp_direct = True
            block_sparse_mlp_cls.forward = _block_sparse_mlp_forward
            patched = True
            logger.info("Patched KimiBlockSparseMLP.forward for direct SiTU.")

        mlp_cls = _get_k3_class_from_source("modeling_kimi_linear.KimiMLP", effective_model_id, "auto")

        if not getattr(mlp_cls, "_tensor_cast_mlp_direct", False):

            def _mlp_forward(self, x):
                if self.config.hidden_act == "situ":
                    gate = self.gate_proj(x)
                    up = self.up_proj(x)
                    return self.down_proj(self.act_fn(gate, up))
                else:
                    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            _mlp_forward._tensor_cast_mlp_direct = True
            mlp_cls.forward = _mlp_forward
            patched = True
            logger.info("Patched KimiMLP.forward for direct SiTU.")

    except Exception as e:
        logger.warning(f"Could not patch K3 SituAndMul/MLP forward: {e}")

    # =================================================================
    # Monkey-patch module-level _apply_attn_res
    # (K3-specific)
    # ----------------------------------------------------------------
    # ``KimiLinearModel._apply_output_attn_res`` (called after all 93
    #      layers) invokes the module-level ``_apply_attn_res`` function
    #      directly (not via self.method). P14 only patches per-layer
    #      AttnRes (KimiDecoderLayer.forward); the model-level call still
    #      goes through the original function, which decomposes into many
    #      aten ops on meta device — mismatching the NPU's single fused
    #      ``_apply_attn_res_kernel``.
    #      This patch routes the module-level function to the
    #      ``tensor_cast.apply_attn_res`` fused op on meta device (tracing),
    #      while preserving the original computation for real (non-meta)
    #      execution.
    # =================================================================
    try:
        import sys as _sys

        _model_cls = _get_k3_class_from_source("modeling_kimi_linear.KimiLinearModel", effective_model_id, "auto")
        _kimi_linear_module = _sys.modules.get(_model_cls.__module__)

        if _kimi_linear_module is not None and hasattr(_kimi_linear_module, "_apply_attn_res"):
            if not getattr(_kimi_linear_module._apply_attn_res, "_tensor_cast_attn_res_patched", False):
                _orig_apply_attn_res = _kimi_linear_module._apply_attn_res

                def _patched_apply_attn_res(prefix_sum, block_residual, proj, norm):
                    """Route to tensor_cast.apply_attn_res on meta device.

                    On real tensors, delegates to the original implementation
                    so that non-tracing execution (e.g. eager / non-sim runs)
                    still produces correct numerical results.
                    """
                    if not prefix_sum.is_meta:
                        return _orig_apply_attn_res(prefix_sum, block_residual, proj, norm)
                    norm_weight = norm.weight
                    proj_weight = proj.weight.squeeze(0)  # (1, hidden) → (hidden,)
                    eps = norm.variance_epsilon
                    return torch.ops.tensor_cast.apply_attn_res(
                        prefix_sum, block_residual, norm_weight, proj_weight, eps
                    )

                _patched_apply_attn_res._tensor_cast_attn_res_patched = True
                _patched_apply_attn_res.__wrapped__ = _orig_apply_attn_res
                _kimi_linear_module._apply_attn_res = _patched_apply_attn_res
                patched = True
                logger.info("Patched module-level _apply_attn_res for fused op routing on meta device.")

    except Exception as e:
        logger.warning(f"Could not patch K3 module-level _apply_attn_res: {e}")

    # =================================================================
    # KimiLinearModel.__init__ — inject identity rotary_emb shim
    # ----------------------------------------------------------------
    # ``maybe_enable_mtp`` (model.py line 227) runs BEFORE
    #      ``patch_model`` (line 229). At that point K3's MLA attention
    #      still has ``rotary_emb=None`` (the attribute exists but is not
    #      a module), so ``_find_text_rotary_emb`` returns None and
    #      ``MtpWrapper.__init__`` raises
    #      ``ValueError: Unable to find rotary embedding module``.
    #
    #      K2.5 avoids this because its DeepSeek-V3 attention ships with a
    #      proper ``DeepseekV3RotaryEmbedding`` module. K3 uses
    #      ``mla_use_nope=True`` (identity RoPE), so there is no real
    #      rotary module to find.
    #
    # Patch ``KimiLinearModel.__init__`` to attach a
    #      ``K3IdentityRotaryEmb`` shim at the model level. After
    #      instantiation the module appears at path
    #      ``language_model.model.rotary_emb``, which
    #      ``_find_text_rotary_emb`` discovers and prefers. The shim
    #      returns ``cos=1, sin=0`` (identity RoPE), matching K3's
    #      ``mla_use_nope=True`` design.
    #
    #      The shim is harmless for non-MTP operation: only
    #      ``MtpWrapper`` calls ``_find_text_rotary_emb``; the main
    #      model's MLA resolver uses the attention-level
    #      ``rotary_emb=None`` and returns identity independently.
    # =================================================================
    try:
        class_ref_linear_model = "modeling_kimi_linear.KimiLinearModel"
        linear_model_cls = _get_k3_class_from_source(class_ref_linear_model, effective_model_id, "auto")

        if linear_model_cls is not None and not getattr(linear_model_cls, "_tensor_cast_mtp_rotary_shim", False):
            _original_linear_init = linear_model_cls.__init__

            def _patched_linear_init(self, config, *args, **kwargs):
                _original_linear_init(self, config, *args, **kwargs)
                qk_rope_dim = getattr(config, "qk_rope_head_dim", 64) or 64
                self.rotary_emb = K3IdentityRotaryEmb(qk_rope_dim)

            _patched_linear_init._tensor_cast_original = _original_linear_init
            linear_model_cls.__init__ = _patched_linear_init
            linear_model_cls._tensor_cast_mtp_rotary_shim = True
            patched = True
            logger.info("Patched KimiLinearModel.__init__ for MTP rotary_emb shim.")
    except Exception as e:
        logger.warning(f"Could not patch K3 KimiLinearModel for MTP shim: {e}")

    # =================================================================
    # ModelWrapper.forward — add output_intermediate_hidden_states for MTP
    #      (K3-specific)
    # ----------------------------------------------------------------
    # The generic ``ModelWrapper.forward`` (tensor_cast/transformers/
    #        model.py:321) delegates to the HF VL model and returns a single
    #        logits tensor. But ``MtpWrapper.forward`` (mtp.py:227) calls
    #        ``self._inner(..., output_intermediate_hidden_states=True)`` and
    #        expects a ``(logits, hidden_states)`` tuple.
    #
    #        Without this patch the VL forward (patched by the patch) filters the
    #        ``output_intermediate_hidden_states`` kwarg out of its allow-list
    #        and returns a single tensor, so torch.compile raises:
    #        ``AssertionError: Can't unpack a tensor of 1 rows into a tuple
    #        of 2 elements``.
    #
    #        The patch is additive and backwards-compatible: ONLY the MTP
    #        branch (``output_intermediate_hidden_states=True``) is
    #        intercepted; every other call delegates to the original
    #        ``ModelWrapper.forward`` unchanged, preserving K3's existing
    #        (non-MTP) simulation flow.
    #
    #        For the MTP text branch we bypass the full VL forward and
    #        directly call ``language_model.model(...)`` to obtain the full
    #        ``last_hidden_state`` (kept for rotary/proposal selection),
    #        then prune to target rows via ``select_lm_head_hidden_states``
    #        before applying ``lm_head`` — The
    #        TC kwargs (attention_meta, kv_cache_by_layers, …) are injected
    #        into each attention layer's ``_extra_forward_kwargs`` side
    #        channel so the patch's ``_patched_decoder_forward`` can recover them
    #        (otherwise MLA runs with ``kv_cache=None`` and the performance
    #        model crashes — see project memory).
    # =================================================================
    try:
        from tensor_cast.transformers.model import ModelWrapper
        from tensor_cast.layers.sampler import select_lm_head_hidden_states

        if not getattr(ModelWrapper, "_patched_for_k3_mtp", False):
            _original_mw_forward_k3 = ModelWrapper.forward

            def patched_mw_forward_k3(
                self,
                input_ids: Optional[torch.Tensor],
                position_ids: torch.Tensor,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_intermediate_hidden_states: bool = False,
                **kwargs: object,
            ):
                if not output_intermediate_hidden_states:
                    # Non-MTP path: delegate to the original forward unchanged.
                    return _original_mw_forward_k3(self, input_ids, position_ids, inputs_embeds, **kwargs)

                sampling_metadata = kwargs.get("sampling_metadata")
                has_image_input = kwargs.get("pixel_values") is not None or kwargs.get("image_grid_thw") is not None

                if not has_image_input and inputs_embeds is None and hasattr(self._inner, "language_model"):
                    # MTP text path: run the transformer body directly, keep
                    # full intermediate hidden states for rotary/proposal
                    # selection, but prune target rows before lm_head.
                    from tensor_cast.transformers.model import _EXTRA_TC_KWARGS_KEYS

                    lm = self._inner.language_model
                    _tc_extra = {k: kwargs[k] for k in _EXTRA_TC_KWARGS_KEYS if k in kwargs and kwargs[k] is not None}
                    if _tc_extra:
                        try:
                            for layer in lm.model.layers:
                                if hasattr(layer, "self_attn"):
                                    layer.self_attn._extra_forward_kwargs = _tc_extra
                        except AttributeError as exc:
                            logger.warning(
                                "Failed to inject TC kwargs into K3 attention layers (MTP, P20): %s",
                                exc,
                            )

                    body_outputs = lm.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    intermediate_hidden_states = body_outputs.last_hidden_state
                    hidden_states = select_lm_head_hidden_states(
                        intermediate_hidden_states,
                        sampling_metadata,
                        mode="target",
                    )
                    logits = lm.lm_head(hidden_states)
                    return logits, intermediate_hidden_states

                # Fallback for image / embedding MTP paths: run the full VL
                # forward with output_hidden_states and extract the last
                # hidden state from the returned output object.
                outputs = _original_mw_forward_k3(
                    self,
                    input_ids,
                    position_ids,
                    inputs_embeds,
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs,
                )
                logits = outputs.logits
                intermediate_hidden_states = (
                    outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
                )
                logits = select_lm_head_hidden_states(logits, sampling_metadata, mode="target")
                return logits, intermediate_hidden_states

            patched_mw_forward_k3._tensor_cast_original = _original_mw_forward_k3
            ModelWrapper.forward = patched_mw_forward_k3
            ModelWrapper._patched_for_k3_mtp = True
            patched = True
            logger.info("Patched ModelWrapper.forward for K3 MTP.")
    except Exception as e:
        logger.warning(f"Could not patch ModelWrapper.forward for K3 MTP: {e}")

    # =================================================================
    # KimiLinearForCausalLM.forward — PP side-channel + bypass lm_head
    # ----------------------------------------------------------------
    # Under pipeline parallelism, each stage's model is narrowed to
    #      ``CausalLmWrapper`` wrapping ``KimiLinearForCausalLM`` (K3's
    #      ``language_module_path="language_model"``; see
    #      ``_narrow_pipeline_vl_stage_to_language_model`` in
    #      ``model_builder.py``). The forward chain is:
    #        StageRunner → PipelineStageModel.forward
    #          → TransformerModel.forward (extracts & passes tc_kwargs)
    #          → CausalLmWrapper.forward       (forwards **kwargs)
    #          → KimiLinearForCausalLM.forward ← TWO bugs here
    #          → KimiLinearModel.forward        (transformer body)
    #          → decoder_layer.forward → MLA.forward
    #
    #      Bug 1 (kv_cache=None crash): ``KimiLinearForCausalLM.forward``
    #      calls ``self.model(...)`` WITHOUT forwarding ``**kwargs``, so
    #      ``kv_cache_by_layers`` / ``attention_meta`` never reach the
    #      decoder layers. The PP path also bypasses the VL forward
    #      which normally injects the ``_extra_forward_kwargs``
    #      side-channel. As a result P14 reads an empty side-channel,
    #      the MLA wrapper runs with ``kv_cache=None``, and the
    #      ``multihead_latent_attention_quant`` op is recorded with null
    #      args → the analytic performance model crashes at
    #      ``_multihead_latent_attention_properties_helper``
    #      (``kv_cache.size(-1)`` on None) during runtime replay.
    #
    #      Bug 2 (lm_head double-application): ``CausalLmWrapper.forward``
    #      (model.py:79-90) treats ``self._inner(...)[0]`` as
    #      ``hidden_states`` and then applies its own ``self.lm_head``
    #      (shared weight with ``KimiLinearForCausalLM.lm_head``, see
    #      model_builder.py:101-102). But ``KimiLinearForCausalLM.forward``
    #      returns logits (vocab_size dim) via its own ``self.lm_head``
    #      call (modeling_kimi_linear.py:1301), so the second lm_head in
    #      ``CausalLmWrapper`` receives a (batch, seq, vocab_size) tensor
    #      and crashes at ``linear(input=(1,1,163840), weight=(163840,7168))``
    #      with ``a and b must have same reduction dim``.
    #
    # Patch ``KimiLinearForCausalLM.forward`` so that on the PP path
    #      (detected by presence of TC kwargs in ``kwargs``):
    #      (a) inject TC kwargs into
    #          ``self.model.layers[i].self_attn._extra_forward_kwargs``
    #          BEFORE calling ``self.model(...)``, mirroring P6/P20;
    #      (b) call the transformer body (``self.model``) directly and
    #          return its ``last_hidden_state`` as a tuple — BYPASSING
    #          ``KimiLinearForCausalLM.lm_head`` — so that
    #          ``CausalLmWrapper.lm_head`` is the single lm_head applied.
    #      ``KimiLinearModel.forward`` already runs ``self.norm`` on the
    #      hidden states (modeling_kimi_linear.py:1219), so the returned
    #      ``last_hidden_state`` is normed and ready for lm_head.
    #      Non-PP / non-PP-quant paths are unaffected: when no TC kwargs
    #      are present in ``kwargs`` (e.g. P6 already filtered them on
    #      the VL path) ``_tc_extra`` is empty and the original forward
    #      runs unchanged.
    # =================================================================
    try:
        class_ref_lm = "modeling_kimi_linear.KimiLinearForCausalLM"
        linear_model_cls = _get_k3_class_from_source(class_ref_lm, effective_model_id, "auto")

        if linear_model_cls is not None and not getattr(linear_model_cls, "_tensor_cast_pp_side_channel", False):
            _original_lm_forward = linear_model_cls.forward

            def _patched_lm_forward_pp_side_channel(self, *args, **kwargs):
                from tensor_cast.transformers.model import _EXTRA_TC_KWARGS_KEYS

                _tc_extra = {k: kwargs[k] for k in _EXTRA_TC_KWARGS_KEYS if k in kwargs and kwargs[k] is not None}
                if _tc_extra:
                    # (a) Inject TC kwargs side-channel so the patch's decoder
                    # forward recovers attention_meta / kv_cache_by_layers.
                    try:
                        for layer in self.model.layers:
                            if hasattr(layer, "self_attn"):
                                layer.self_attn._extra_forward_kwargs = _tc_extra
                    except AttributeError as exc:
                        logger.warning(
                            "Failed to inject TC kwargs into K3 attention layers (PP, P21): %s",
                            exc,
                        )

                    # (b) Bypass self.lm_head: call transformer body
                    # directly and return normed hidden_states as a
                    # CausalLMOutputWithPast-shaped tuple so that
                    # CausalLmWrapper.forward picks [0]=hidden_states and
                    # applies its own (single) lm_head.  Mirror the
                    # self.model(...) call in KimiLinearForCausalLM.forward
                    # (modeling_kimi_linear.py:1285-1296) but drop the
                    # CausalLM-only fields (labels / generation_mode /
                    # return_dict) that the transformer body does not
                    # consume; TC kwargs stay out (recovered via
                    # side-channel at P14).
                    body_outputs = self.model(
                        input_ids=kwargs.get("input_ids"),
                        attention_mask=kwargs.get("attention_mask"),
                        position_ids=kwargs.get("position_ids"),
                        past_key_values=kwargs.get("past_key_values"),
                        inputs_embeds=kwargs.get("inputs_embeds"),
                        use_cache=kwargs.get("use_cache", False),
                        output_attentions=kwargs.get("output_attentions"),
                        output_hidden_states=kwargs.get("output_hidden_states"),
                        cache_position=kwargs.get("cache_position"),
                    )
                    hidden_states = (
                        body_outputs[0] if isinstance(body_outputs, (tuple, list)) else body_outputs.last_hidden_state
                    )
                    # CausalLMOutputWithPast field order: (loss, logits,
                    # past_key_values, hidden_states, attentions).  We
                    # place hidden_states in [0] (the slot
                    # CausalLmWrapper.forward reads) and leave the rest
                    # None.
                    return (hidden_states, None, None, None, None)

                # Non-PP path (no TC kwargs): delegate to original forward
                return _original_lm_forward(self, *args, **kwargs)

            _patched_lm_forward_pp_side_channel._tensor_cast_original = _original_lm_forward
            linear_model_cls.forward = _patched_lm_forward_pp_side_channel
            linear_model_cls._tensor_cast_pp_side_channel = True
            patched = True
            logger.info("Patched KimiLinearForCausalLM.forward for K3 PP side-channel.")
    except Exception as e:
        logger.warning(f"Could not patch K3 KimiLinearForCausalLM.forward for PP: {e}")

    # =================================================================
    # KDA state memory accounting (recurrent_states + conv_states)
    # ----------------------------------------------------------------
    # K3 hybrid attention uses KDA (``KimiDeltaAttention``, ~69/93
    #      layers) with a *fixed-size* recurrent state plus a short-conv
    #      tail state. Unlike MLA's paged KV cache, KDA state does NOT
    #      grow with context length T, but it DOES occupy real GPU memory
    #      and must NOT be treated as zero memory.
    #
    #      MSModeling's ``kv_cache_excluded_layer_indices`` correctly
    #      excludes KDA layers from per-token KV cache accounting (they
    #      are not paged), but this exclusion also drops KDA's fixed state
    #      from the memory budget entirely. The simulation then reports
    #      ``KV cache: 0.221 GB`` (MLA only) and
    #      ``Model activation size: 0.000 GB``, giving the false
    #      impression that KDA has zero memory footprint.
    #
    # Monkey-patch ``ModelRunner.run_inference`` and
    #      ``ModelRunner._build_pipeline_metrics`` to inject KDA state
    #      bytes into the returned ``ModelRunnerMetrics``:
    #        - ``kv_cache_size_gb``       += kda_state_gb  (KV-like memory)
    #        - ``peak_memory_usage_gb``  += kda_state_gb  (real GPU memory)
    #        - ``device_memory_available_gb`` -= kda_state_gb
    #        - ``kda_state_size_gb``      = kda_state_gb  (new attribute
    #          carrying the breakdown value for printing)
    #      Also patch ``ModelRunnerMetrics.print_info`` to print a
    #      breakdown line clarifying that KDA state is included in the
    #      KV cache figure above.
    #
    #      ``_k3_kda_state_size_gb(model)`` returns 0.0 for non-K3 models
    #      or when ``linear_attn_config`` is missing, so the patch is a
    #      pure no-op for every other model — no behavior change, no
    #      attribute injection, no extra print line.
    # =================================================================
    try:
        from tensor_cast.core.model_runner import (
            ModelRunner as _K3_ModelRunner,
            ModelRunnerMetrics as _K3_ModelRunnerMetrics,
        )

        if not getattr(_K3_ModelRunner, "_tensor_cast_k3_kda_accounting", False):
            _orig_run_inference_k3 = _K3_ModelRunner.run_inference
            _orig_build_pipeline_metrics_k3 = _K3_ModelRunner._build_pipeline_metrics
            _orig_print_info_k3 = _K3_ModelRunnerMetrics.print_info

            def _k3_inject_kda_state(metrics, kda_gb):
                """Mutate ``metrics`` in place to fold KDA state into the
                KV-cache/peak/available figures and attach the breakdown
                attribute. No-op when ``kda_gb`` is 0.
                """
                if kda_gb <= 0:
                    return
                metrics.kda_state_size_gb = kda_gb
                metrics.kv_cache_size_gb += kda_gb
                metrics.peak_memory_usage_gb += kda_gb
                metrics.device_memory_available_gb -= kda_gb

            def _patched_run_inference_k3(self, *args, **kwargs):
                # Scope the MetaConstantFolder guard to this run so dynamo
                # compilation (which happens inside run_inference) never
                # folds K3's region markers / torch.empty stub allocations,
                # while other execution contexts keep the default policy.
                with _k3_constant_folder_guard(getattr(self, "model", None)):
                    metrics = _orig_run_inference_k3(self, *args, **kwargs)
                _k3_inject_kda_state(metrics, _k3_kda_state_size_gb(self.model))
                return metrics

            def _patched_build_pipeline_metrics_k3(self, pipeline_result, *, batch_size, run_time_s):
                metrics = _orig_build_pipeline_metrics_k3(
                    self, pipeline_result, batch_size=batch_size, run_time_s=run_time_s
                )
                _k3_inject_kda_state(metrics, _k3_kda_state_size_gb(self.model))
                return metrics

            def _patched_print_info_k3(self):
                _orig_print_info_k3(self)
                kda_gb = getattr(self, "kda_state_size_gb", 0.0)
                if kda_gb > 0:
                    # KDA state has already been folded into the
                    # ``KV cache`` and ``Memory available`` figures above;
                    # surface it as a breakdown line so users can see
                    # that KDA's fixed recurrent+conv state is not zero
                    # memory.
                    print(f"    KDA state (recurrent+conv, included in KV cache above): {kda_gb:.6f} GB")

            _K3_ModelRunner.run_inference = _patched_run_inference_k3
            _K3_ModelRunner._build_pipeline_metrics = _patched_build_pipeline_metrics_k3
            _K3_ModelRunnerMetrics.print_info = _patched_print_info_k3
            _K3_ModelRunner._tensor_cast_k3_kda_accounting = True
            patched = True
            logger.info("Patched ModelRunner/ModelRunnerMetrics for K3 KDA state memory accounting.")
    except Exception as e:
        logger.warning(f"Could not patch ModelRunner for K3 KDA state accounting: {e}")

    return patched


# ============================================================
# Orchestrator: hf_config_patch_method entry point
# ============================================================

_patched_kimi_k3 = False


def _hf_config_patch_for_kimi_k3(config, model_id=None):
    """Pre-load entry point: apply HF config fixes, then model class patches.

    Called by ``AutoModelConfigLoader._apply_hf_config_patches`` BEFORE
    the HuggingFace model is instantiated.

    Config-level patches run for every new config instance.
    Class-level patches run once per process, guarded by
    ``_patched_kimi_k3`` to avoid redundant work.

    Class-level patches use ``_get_k3_class_from_source`` with ``"auto"``
    mode so they work under both HF and ModelScope remote sources without
    requiring ``remote_source`` to be threaded through the public API.
    """
    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k3":
        return

    # Config-level patches (always run for every new config)
    config_patched = _patch_hf_config_for_kimi_k3(config)

    # Class-level patches (run once per process, requires model_id)
    global _patched_kimi_k3
    if _patched_kimi_k3:
        return

    classes_patched = _patch_model_classes_for_kimi_k3(config, model_id)

    if config_patched or classes_patched:
        _patched_kimi_k3 = True
        logger.info("Patched transformers environment for Kimi-K3")


# ============================================================
# ModelProfile registration
# ============================================================


def _k3_is_k3_model(model) -> bool:
    """Return whether ``model`` is a Kimi K3 model."""
    try:
        model_config = getattr(model, "model_config", None)
        if model_config is None:
            return False
        hf_config = getattr(model_config, "hf_config", None)
        return getattr(hf_config, "model_type", None) == "kimi_k3"
    except Exception:
        return False


# The wrapper is installed once, while this ContextVar makes the changed
# folding policy local to the current K3 execution context. Concurrent non-K3
# threads keep the original behavior.
_K3_CONSTANT_FOLDER_GUARD_ACTIVE = contextvars.ContextVar("tensor_cast_k3_constant_folder_guard_active", default=False)
_K3_CONSTANT_FOLDER_INSTALL_LOCK = threading.Lock()
_K3_CONSTANT_FOLDER_ORIG_RUN_NODE = None


def _guarded_run_node(self, node):
    """Skip K3-only folding targets in the active K3 execution context."""
    if _K3_CONSTANT_FOLDER_GUARD_ACTIVE.get() and node.op == "call_function":
        target_str = str(node.target)
        if "_internal_" in target_str or (target_str.startswith("aten.") and "empty" in target_str):
            return self.unknown_value
    return _K3_CONSTANT_FOLDER_ORIG_RUN_NODE(self, node)


def _install_k3_constant_folder_wrapper() -> None:
    """Install the context-aware wrapper once without per-run global swapping."""
    global _K3_CONSTANT_FOLDER_ORIG_RUN_NODE

    from ...compilation.constant_folding import MetaConstantFolder

    if _K3_CONSTANT_FOLDER_ORIG_RUN_NODE is not None:
        return
    with _K3_CONSTANT_FOLDER_INSTALL_LOCK:
        if _K3_CONSTANT_FOLDER_ORIG_RUN_NODE is None:
            _K3_CONSTANT_FOLDER_ORIG_RUN_NODE = MetaConstantFolder.run_node
            MetaConstantFolder.run_node = _guarded_run_node


@contextlib.contextmanager
def _k3_constant_folder_guard(model):
    """Disable unsafe marker/empty folding only in this K3 run context."""
    if not _k3_is_k3_model(model):
        yield
        return

    _install_k3_constant_folder_wrapper()
    token = _K3_CONSTANT_FOLDER_GUARD_ACTIVE.set(True)
    try:
        yield
    finally:
        _K3_CONSTANT_FOLDER_GUARD_ACTIVE.reset(token)


def _patch_model_for_kimi_k3(model) -> None:
    """Resolve region-marker pairing failures in multimodal compile mode.

    This method is registered as ``ModelProfile.patch_method`` and runs after
    ``maybe_reuse_layers`` (see ``TransformerModel.__init__`` call order:
    ``maybe_reuse_layers`` → ``patch_model``).

    Three issues cause ``Region end with id ... not paired with a region begin``
    in multimodal ``--compile`` mode:

    1. **Vision tower region markers** — ``maybe_reuse_layers`` wraps vision
       encoder blocks with ``RegionMarkerWrapper``/``CopyLayerWrapper``. The
       ``_prepare_vl_compile`` graph break (``torch._dynamo.disable``) splits
       vision ``region_begin``/``region_end`` across FX sub-graphs.

    2. **Singleton language-layer region markers** — Layer 0 has a unique
       structure (``repeat_count=1``). Dynamo's FX DCE drops its
       ``region_begin`` (output appears dead because the inner forward
       reassigns ``hidden_states``) while keeping ``region_end``, producing
       an unpaired end.

    3. **Graph break propagating to language layers** — The
       ``torch._dynamo.disable`` wrapping forces a graph break that isolates
       language-layer forward into a separate sub-graph where the patch's
       ``torch.where`` anchoring (tuned for single-graph) doesn't fully
       protect region markers.

    4. **MetaConstantFolder folding region markers** — The meta-device stub
       builds ``inputs_embeds`` via ``torch.empty``; ``fold_meta_constants``
       folds that allocation and propagates constants through the layer,
       causing DCE to drop ``region_begin`` while ``region_end`` survives.

    Fixes applied:
      - Restore all vision tower wrappers to original layers.
      - Monkey-patch ``_prepare_vl_compile`` to skip
        ``torch._dynamo.disable`` for K3, allowing single-graph compile.
      - Unwrap singleton (``repeat_count==1``) language-layer
        ``RegionMarkerWrapper`` instances — no replay copies reference them, so this
        is safe and only loses region grouping for one layer out of 93.
      - Scope a K3-context-local ``MetaConstantFolder`` guard (never folds
        ``tensor_cast._internal_*`` marker ops or ``aten.empty*`` allocations)
        to K3's ``run_inference`` window. The installed wrapper is inert in
        concurrent non-K3 execution contexts.
    """
    import operator as _operator

    from ..custom_model_registry import get_language_layers as _get_lang_layers

    # The context-aware wrapper is activated only by K3 run_inference; other
    # execution contexts keep the default constant-folding policy.

    # ----------------------------------------------------------------
    # Restore vision tower layers from region wrappers
    # ----------------------------------------------------------------
    visual_layers = get_visual_layers(model)
    if visual_layers is not None:
        restored_count = 0
        for i, layer in enumerate(visual_layers):
            if isinstance(layer, RegionMarkerWrapper):
                visual_layers[i] = layer._inner
                restored_count += 1
            elif isinstance(layer, CopyLayerWrapper):
                # CopyLayerWrapper does not retain its original layer reference.
                # Use the representative's original layer (computationally
                # equivalent on meta device).
                representative = getattr(layer, "representative", None)
                if representative is not None and hasattr(representative, "_inner"):
                    visual_layers[i] = representative._inner
                    restored_count += 1
        if restored_count > 0:
            logger.info(
                "Restored %d/%d vision tower layers from region wrappers",
                restored_count,
                len(visual_layers),
            )

    # ----------------------------------------------------------------
    # Skip _prepare_vl_compile's torch._dynamo.disable for K3
    # ----------------------------------------------------------------
    import tensor_cast.core.model_builder as _model_builder

    _orig_prepare_vl_compile = getattr(_model_builder, "_prepare_vl_compile", None)
    if _orig_prepare_vl_compile is not None and not getattr(_orig_prepare_vl_compile, "_k3_patched", False):

        def _k3_prepare_vl_compile(m):
            model_type = getattr(getattr(m, "hf_config", None), "model_type", None)
            if model_type == "kimi_k3":
                logger.info(
                    "Skipping _prepare_vl_compile torch._dynamo.disable for K3 "
                    "— allowing single-graph compile to preserve region-marker pairing"
                )
                return False
            return _orig_prepare_vl_compile(m)

        _k3_prepare_vl_compile._k3_patched = True
        _model_builder._prepare_vl_compile = _k3_prepare_vl_compile
        logger.info("Patched model_builder._prepare_vl_compile for K3 (skip dynamo.disable)")

    # ----------------------------------------------------------------
    # Unwrap singleton language-layer (and MTP-layer)
    #         RegionMarkerWrappers
    # ----------------------------------------------------------------
    # NOTE: K3's ModelProfile sets ``language_layers_path_str =
    # "language_model.model.layers"`` (the VL model path).  However in
    # pipeline-parallel mode the stage model is narrowed via
    # ``_narrow_pipeline_vl_stage_to_language_model`` to a plain
    # ``KimiLinearForCausalLM`` wrapped inside a ``CausalLmWrapper``
    # (optionally inside an ``MtpWrapper``), so the
    # ``language_model.model.layers`` path no longer exists.  In that
    # narrow path the real decoder layers live under one of:
    #   (a) ``model.layers``           (KimiLinearForCausalLM unwrapped)
    #   (b) ``_inner.model.layers``    (CausalLmWrapper._inner)
    #   (c) ``_inner._inner.model.layers`` (MtpWrapper→CausalLmWrapper)
    #   (d) ``_inner.mtp.layers``      (MTP proposal layers)
    # We therefore probe multiple candidate paths and unwrap singletons
    # from every discovered layer list.  Any layer not found is silently
    # skipped; this is safe because ``repeat_count == 1`` singletons have
    # no CopyLayerWrapper references, so unwrapping only loses a tiny bit
    # of region-grouping granularity (at most one layer per list).
    def _k3_unwrap_singleton_wrappers_from(
        candidate_layers,
        tag: str,
        accumulator: dict,
    ) -> None:
        if not isinstance(candidate_layers, (list, torch.nn.ModuleList)):
            return
        restored = 0
        for i, lyr in enumerate(candidate_layers):
            if isinstance(lyr, RegionMarkerWrapper) and lyr.repeat_count == 1:
                candidate_layers[i] = lyr._inner
                restored += 1
            elif isinstance(lyr, CopyLayerWrapper):
                repr_wr = getattr(lyr, "representative", None)
                if (
                    isinstance(repr_wr, RegionMarkerWrapper)
                    and repr_wr.repeat_count == 1
                    and hasattr(repr_wr, "_inner")
                ):
                    # The representative itself is a singleton — keep the
                    # CopyLayerWrapper because it is referenced by replay;
                    # do nothing here.
                    pass
        if restored > 0:
            accumulator[tag] = accumulator.get(tag, 0) + restored

    _singleton_stats: dict = {}
    _candidates_to_try: list[tuple[str, str]] = []
    # (a) Canonical VL path from ModelProfile
    _vl_path = _get_lang_layers(model.hf_config.model_type)
    if _vl_path:
        _candidates_to_try.append((_vl_path, "vl/" + _vl_path))
    # (b) Plain text-model paths on PP-narrowed stages
    for _plain_path in (
        "layers",
        "model.layers",
        "_inner.layers",
        "_inner.model.layers",
        "_inner._inner.layers",
        "_inner._inner.model.layers",
    ):
        _candidates_to_try.append((_plain_path, "plain/" + _plain_path))
    # (c) MTP proposal layers
    for _mtp_path in (
        "_inner.mtp.layers",
        "mtp.layers",
    ):
        _candidates_to_try.append((_mtp_path, "mtp/" + _mtp_path))

    _total_unwrapped = 0
    try:
        _unwrapped_root = model.unwrap()
        # Also try ``model._inner`` directly because unwrap() skips past
        # MtpWrapper/CausalLmWrapper when the inner model has no .layers
        # attribute; we want to examine the wrapper stack too.
        _roots_to_search = [_unwrapped_root]
        _candidates_inner = getattr(model, "_inner", None)
        if _candidates_inner is not None and id(_candidates_inner) != id(_unwrapped_root):
            _roots_to_search.insert(0, _candidates_inner)
            _candidates_inner2 = getattr(_candidates_inner, "_inner", None)
            if _candidates_inner2 is not None:
                _roots_to_search.insert(0, _candidates_inner2)

        _visited_paths = set()
        for _root in _roots_to_search:
            for _attr_path, _tag in _candidates_to_try:
                try:
                    _layers_ref = _operator.attrgetter(_attr_path)(_root)
                except AttributeError:
                    continue
                _key = (id(_root), _attr_path)
                if _key in _visited_paths:
                    continue
                _visited_paths.add(_key)
                _before = _total_unwrapped
                _k3_unwrap_singleton_wrappers_from(_layers_ref, _tag, _singleton_stats)
                _total_unwrapped = sum(_singleton_stats.values())
    except Exception as _e:
        logger.warning("Could not inspect layer lists for singleton unwrap: %s", _e)

    if _singleton_stats:
        logger.info(
            "Restored %d singleton layer(s) from RegionMarkerWrapper: %s",
            _total_unwrapped,
            ", ".join(f"{k}={v}" for k, v in _singleton_stats.items()),
        )

    # ----------------------------------------------------------------
    # CopyLayerWrapper strip — DISABLED
    # ----------------------------------------------------------------
    # Stripping CopyLayerWrapper breaks layer reuse (maybe_reuse_layers)
    # in both PP and non-PP scenes, causing compile time explosion
    # (12.9s → 480.9s, 37x) and weight-accounting corruption (55 GB → 3.8 GB).
    # Singleton RegionMarkerWrapper unwrap (above) is sufficient for
    # region-marker pairing in both PP and non-PP modes.
    #
    # If a region-marker AssertionError reappears in PP, do NOT blindly
    # re-enable this strip.  Instead, apply a targeted fix: only strip the
    # specific wrappers whose ``region_begin`` is dropped by DCE, and verify
    # that layer reuse (repeat_count) is not broken.
    # The implementation was removed as dead code; see git history if a
    # targeted re-implementation is needed.
    return


register_model_profile(
    ModelProfile(
        model_type="kimi_k3",
        # MoE: K3 uses KimiSparseMoeBlock (Latent MoE with down/norm/up wrappers)
        moe_module_name="KimiSparseMoeBlock",
        moe_num_experts_key=["text_config", "num_experts"],
        # K3 uses sigmoid + noaux_tc top-k routing; gate returns processed weights
        moe_gate_returns_raw_logits=False,
        # When DP≠EP, route after DP slicing
        moe_route_after_dp_transform=True,
        # K3 experts are nn.ModuleList of KimiBlockSparseMLP (w1/w2/w3 naming),
        # not 3D-tensor stacked weights. The default MoeExpertMLP adapter expects
        # gate_up_proj/down_proj + expert_idx and only works for 3D-tensor experts.
        # Set to None so _patch_moe_expert_helper skips replacement and MoELayer
        # wraps the original ModuleList directly.
        custom_expert_module_type=None,
        # MLA: K3 uses KimiMLAAttention (Gated MLA with output gate)
        mla_module_name="KimiMLAAttention",
        mla_field_names_override={
            "q_proj": "q_a_proj",  # K3 uses q_a_proj (q_lora_rank=1536)
            "qk_head_dim": "q_head_dim",
        },
        # MTP: K3's config has num_nextn_predict_layers=0 (no native MTP),
        # but TensorCast can simulate MTP overhead by replicating decoder
        # layers. The MTP shim injects an identity rotary_emb so MtpWrapper
        # can find a rotary_emb module (K3's MLA has rotary_emb=None).
        # The shim returns cos=1, sin=0 (identity RoPE), matching K3's
        # mla_use_nope=True design. Use --num-mtp-tokens N to enable.
        mtp_block_module_name="KimiDecoderLayer",
        # Visual / language module paths (K3 VL structure)
        visual_module_path="vision_tower",
        language_module_path="language_model",
        visual_layers_module_path="vision_tower.encoder.blocks",
        visual_layers_path_str="vision_tower.encoder.blocks",
        language_layers_path_str="language_model.model.layers",
        # Entry point for all K3 patches
        hf_config_patch_method=_hf_config_patch_for_kimi_k3,
        # restore vision tower layers from region wrappers to
        # avoid graph-break region-marker pairing failures in multimodal
        # compile mode (runs after maybe_reuse_layers via patch_model).
        patch_method=_patch_model_for_kimi_k3,
    )
)
