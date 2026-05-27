import torch

from ..custom_model_registry import (
    ModelProfile,
    register_model_profile,
)


def _patch_hf_config_for_kimi_k25(config):
    """Fix HuggingFace config and import environment for Kimi K2.5.

    These patches modify the Transformers *environment* (not model classes)
    and must run BEFORE the model is loaded so the downstream loading code
    picks up the corrected settings.

    This function does NOT require ``model_id`` — it operates purely on the
    HF config object and global import state.
    """
    import transformers.utils.import_utils as import_utils
    import logging
    import importlib.util

    logger = logging.getLogger(__name__)

    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k25":
        return False

    patched = False

    # ----------------------------------------------------------------
    # Patch 1: Restore is_torch_fx_available
    # ----------------------------------------------------------------
    if not hasattr(import_utils, "is_torch_fx_available"):

        def is_torch_fx_available():
            return importlib.util.find_spec("torch.fx") is not None

        import_utils.is_torch_fx_available = is_torch_fx_available
        patched = True

    # ----------------------------------------------------------------
    # Patch 2: Downgrade flash_attention_2 → tensor_cast (PRE-LOAD)
    # ----------------------------------------------------------------
    # WHY:   Kimi K2.5's config.json specifies "_attn_implementation": "flash_attention_2".
    # transformers 5.x enforces flash_attn availability during PreTrainedModel.__init__()
    # via _flash_attn_2_can_dispatch() — if flash_attn is not installed, ImportError
    # is raised BEFORE the model instance is returned.
    #
    # The _attn_implementation reassignment in model.py L206 runs AFTER load_model()
    # returns, so it cannot prevent that early failure.  This patch intercepts the
    # config BEFORE loading and downgrades to "tensor_cast", letting the HF loader
    # skip the flash_attn check.
    # Only downgrades when flash_attn is genuinely absent to respect environments
    # that do have it installed.
    # ----------------------------------------------------------------
    def _downgrade_attn_implementation(cfg):
        if getattr(cfg, "_attn_implementation", None) == "flash_attention_2":
            if importlib.util.find_spec("flash_attn") is None:
                logger.warning(
                    "Flash Attention 2 is requested but not installed. "
                    "Falling back to 'tensor_cast' attention implementation for simulation."
                )
                cfg._attn_implementation = "tensor_cast"
                return True
        return False

    text_downgraded = _downgrade_attn_implementation(config)
    if hasattr(config, "vision_config"):
        vision_downgraded = _downgrade_attn_implementation(config.vision_config)
        if vision_downgraded:
            text_downgraded = True

    if text_downgraded:
        patched = True

    # ----------------------------------------------------------------
    # Patch 3: Bridge vision config attributes for input generator
    # ----------------------------------------------------------------
    # WHY:   Kimi K2.5 vision config uses different attribute names
    #        (``merge_kernel_size``) or omits attributes altogether
    #        (``temporal_patch_size``, ``in_channels``).  The generic
    #        image-input generator expects these attributes and fails
    #        with AttributeError inside transformers v5.x due to
    #        hasattr/__getattribute__ mismatch.
    # WITHOUT: AttributeError on spatial_merge_size /
    #          temporal_patch_size / in_channels.
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

    return patched


def _patch_model_classes_for_kimi_k25(config, model_id):
    """Monkey-patch *remote* model classes before model instantiation.

    These patches modify **class-level methods** (not instances), so they
    MUST run before the HF loader constructs model objects from the dynamic
    module.  Once the model is loaded, class monkey-patches have no effect
    on already-instantiated objects.

    Requires ``model_id`` to locate and import the remote modeling files.
    """
    import logging
    import sys
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from typing import Optional, Tuple

    logger = logging.getLogger(__name__)

    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k25" or model_id is None:
        return False

    patched = False

    try:
        # ----------------------------------------------------------------
        # Patch 4: Filter KimiK25ForConditionalGeneration.forward kwargs
        # ----------------------------------------------------------------
        # WHY:   TensorCast injects extra kwargs (attention_meta,
        #        kv_cache_by_layers, etc.) via model_runner, but the
        #        original VL forward only accepts standard HF keys.
        #        Passing unexpected kwargs causes TypeError.
        # WITHOUT: TypeError from unexpected keyword arguments.
        class_ref_vl = "modeling_kimi_k25.KimiK25ForConditionalGeneration"
        vl_cls = get_class_from_dynamic_module(class_ref_vl, model_id, force_download=False)

        if not hasattr(vl_cls, "_original_vl_forward"):
            vl_cls._original_vl_forward = vl_cls.forward

        _STANDARD_VL_FORWARD_KEYS = frozenset(
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
            # original forward (which filters them out).  The decoder
            # (patched by P10) reads them back from _extra_forward_kwargs.
            from tensor_cast.transformers.model import _EXTRA_TC_KWARGS_KEYS

            _tc_extra = {k: kwargs[k] for k in _EXTRA_TC_KWARGS_KEYS if k in kwargs and kwargs[k] is not None}
            _injected = []
            if _tc_extra:
                try:
                    for layer in self.language_model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            layer.self_attn._extra_forward_kwargs = _tc_extra
                            _injected.append(layer.self_attn)
                except AttributeError as e:
                    logger.warning(
                        "Failed to inject TC kwargs into attention layers: %s. "
                        "This may affect tensor casting for Kimi K2.5 vision-language model.",
                        e,
                    )

            hf_kwargs = {k: v for k, v in kwargs.items() if k in _STANDARD_VL_FORWARD_KEYS}
            # The generic input generator uses "image_grid_thw" but
            # Kimi K2.5's forward expects "grid_thws".
            if "grid_thws" not in hf_kwargs and "image_grid_thw" in kwargs:
                hf_kwargs["grid_thws"] = kwargs["image_grid_thw"]
            return vl_cls._original_vl_forward(self, *args, **hf_kwargs)

        vl_cls.forward = patched_vl_forward

        # ----------------------------------------------------------------
        # Patch 5: _merge_input_ids_with_image_features (meta device)
        # ----------------------------------------------------------------
        # WHY:   During torch.compile graph capture, input_ids live on
        #        the 'meta' device.  The original merge function calls
        #        embedding layers which raise on meta tensors.  This
        #        patch returns a correctly-shaped meta embedding to
        #        keep the graph tracer happy.
        # WITHOUT: RuntimeError from operations on meta tensors.
        if not hasattr(vl_cls, "_original_merge_input_ids_with_image_features"):
            vl_cls._original_merge_input_ids_with_image_features = vl_cls._merge_input_ids_with_image_features

        def patched_merge_input_ids_with_image_features(
            self,
            image_features,
            feature_lens,
            input_ids,
            attention_mask=None,
            position_ids=None,
            labels=None,
        ):
            batch_size, sequence_length = input_ids.shape
            if input_ids.device.type == 'meta':
                embed_dim = (
                    image_features[0].shape[-1] if len(image_features) > 0 else self.config.text_config.hidden_size
                )
                return (
                    torch.empty(batch_size, sequence_length, embed_dim, device='meta', dtype=self.dtype),
                    attention_mask,
                    labels,
                    position_ids,
                )

            return vl_cls._original_merge_input_ids_with_image_features(
                self,
                image_features,
                feature_lens,
                input_ids,
                attention_mask,
                position_ids,
                labels,
            )

        vl_cls._merge_input_ids_with_image_features = patched_merge_input_ids_with_image_features
        patched = True

    except Exception as e:
        logger.warning(f"Could not patch remote VL / attention class attributes: {e}")

    try:
        # ----------------------------------------------------------------
        # Patch 6: MoonViT3dEncoder — add deterministic attn flag & adapter
        # ----------------------------------------------------------------
        # WHY:   The remote encoder checks ``self.use_deterministic_attn``
        #        but never defines it.  We must add the attribute so the
        #        check doesn't fail.  Additionally, we register a
        #        'tensor_cast' attention adapter that handles meta tensors
        #        and avoids O(n²) computation for very long sequences.
        # WITHOUT: AttributeError for missing use_deterministic_attn;
        #          KeyError for missing "tensor_cast" attention backend.
        class_ref_enc = "modeling_kimi_k25.MoonViT3dEncoder"
        encoder_cls = get_class_from_dynamic_module(class_ref_enc, model_id, force_download=False)
        if not hasattr(encoder_cls, "use_deterministic_attn"):
            setattr(encoder_cls, "use_deterministic_attn", False)
            patched = True

            for name, module in sys.modules.items():
                if "moonshotai" in name and "modeling_kimi_k25" in name:
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
                            if q.device.type == 'meta':
                                head_dim = q.shape[-1]
                                num_heads = q.shape[1]
                                return torch.empty(
                                    seq_length,
                                    num_heads * head_dim,
                                    device='meta',
                                    dtype=q.dtype,
                                )

                            if seq_length > 4096:
                                logger.warning(
                                    "Visual attention sequence length %d exceeds safe "
                                    "threshold. Skipping O(n²) attention to avoid OOM.",
                                    seq_length,
                                )
                                head_dim = q.shape[-1]
                                num_heads = q.shape[1]
                                return torch.zeros(
                                    seq_length,
                                    num_heads * head_dim,
                                    device=q.device,
                                    dtype=q.dtype,
                                )

                            attention_mask = torch.zeros(
                                [1, seq_length, seq_length],
                                device=q.device,
                                dtype=torch.bool,
                            )

                            q_cu_seqlens_list = q_cu_seqlens.tolist()
                            for i in range(1, len(q_cu_seqlens_list)):
                                start = q_cu_seqlens_list[i - 1]
                                end = q_cu_seqlens_list[i]
                                attention_mask[..., start:end, start:end] = True

                            q = q.transpose(0, 1)
                            k = k.transpose(0, 1)
                            v = v.transpose(0, 1)

                            attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
                            attn_weight += attention_mask
                            attn_weight = torch.softmax(
                                attn_weight,
                                dim=-1,
                                dtype=torch.float32,
                            ).to(q.dtype)

                            attn_output = attn_weight @ v
                            attn_output = attn_output.transpose(0, 1)
                            attn_output = attn_output.reshape(seq_length, -1)
                            return attn_output

                        module.VL_VISION_ATTENTION_FUNCTIONS["tensor_cast"] = visual_tc_adapter
                        module.VL_VISION_ATTENTION_FUNCTIONS["eager"] = visual_tc_adapter
                        break

        # ----------------------------------------------------------------
        # Patch 7: DeepseekV3MoE — stub forward for graph tracing
        # ----------------------------------------------------------------
        # WHY:   The real MoE forward contains dynamic dispatch logic
        #        (expert selection + token routing + expert combine)
        #        that torch.compile cannot trace — the control flow
        #        depends on runtime token-to-expert assignments.
        #        Additionally the real computations (matmul across all
        #        experts) would OOM during graph capture.  This stub
        #        returns correct shapes without executing any experts,
        #        allowing compile to proceed.  The actual performance
        #        modeling is handled later by transformations.patch_moe()
        #        which wraps these modules with fused MoELayer wrappers.
        # WITHOUT: torch.compile failure (untraceable dynamic dispatch)
        #          or OOM during graph capture.
        class_ref_moe = "modeling_deepseek.DeepseekV3MoE"
        moe_cls = get_class_from_dynamic_module(class_ref_moe, model_id, force_download=False)

        def patched_forward(_self, hidden_states):
            return torch.zeros_like(hidden_states)

        def patched_moe_infer(_self, x, _topk_ids, _topk_weight):
            return torch.zeros_like(x)

        if not hasattr(moe_cls, "_original_forward"):
            moe_cls._original_forward = moe_cls.forward
        moe_cls.forward = patched_forward

        if not hasattr(moe_cls, "_original_moe_infer"):
            moe_cls._original_moe_infer = moe_cls.moe_infer
        moe_cls.moe_infer = patched_moe_infer

        # ----------------------------------------------------------------
        # Patch 8: MoEGate — deterministic routing for simulation
        # ----------------------------------------------------------------
        # WHY:   The real gate performs top-k softmax + random sampling
        #        which is non-deterministic and untraceable.  We replace
        #        it with equal-weight routing to produce deterministic
        #        shapes during graph capture.
        # WITHOUT: Non-deterministic / un-traceable routing logic during
        #          torch.compile; shape mismatches downstream.
        class_ref_gate = "modeling_deepseek.MoEGate"
        gate_cls = get_class_from_dynamic_module(class_ref_gate, model_id, force_download=False)

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

        # ----------------------------------------------------------------
        # Patch 9: Monkey-patch _resolve_position_embeddings onto MLA
        # ----------------------------------------------------------------
        # WHY:   Kimi K2.5's decoder only passes ``position_ids``, not
        #        pre-computed RoPE (cos, sin) tensors.  TensorCast MLA
        #        needs explicit position_embeddings.  This method
        #        computes them from position_ids via the rotary_emb
        #        cache.  Moved here from layers/mla.py to keep the
        #        generic MLA layer free of model-specific logic.
        # WITHOUT: Missing RoPE → simulation results inaccurate
        #          (cos=1, sin=0 fallback).
        from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

        if not hasattr(MultiheadLatentAttentionTensorCast, "_patched_rope_resolve"):

            def _patched_resolve_position_embeddings(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
                **kwargs,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Compute position_embeddings from position_ids when not explicitly provided.

                This provides compatibility when the caller (e.g. patched decoder forward)
                only passes ``position_ids`` instead of pre-computed RoPE tensors.
                The resolved (cos, sin) tuple is always returned.
                """
                if position_embeddings is not None:
                    return position_embeddings

                position_ids = kwargs.get("position_ids", None)

                if position_ids is not None and self._has_rotary_emb and hidden_states.device.type != 'meta':
                    max_pos = position_ids.max().item() + 1
                    if hasattr(self.rotary_emb, "cos_cached"):
                        if self.rotary_emb.cos_cached.shape[0] < max_pos:
                            self.rotary_emb._update_cos_sin_tables(max_pos, hidden_states.device, hidden_states.dtype)
                    cos = self.rotary_emb.cos_cached[position_ids].to(hidden_states.dtype)
                    sin = self.rotary_emb.sin_cached[position_ids].to(hidden_states.dtype)
                    return (cos, sin)

                # No position info available → neutral RoPE (identity rotation).
                if self._has_rotary_emb:
                    import warnings

                    warnings.warn(
                        "position_embeddings was not provided and position_ids is unavailable; "
                        "RoPE will be disabled (cos=1, sin=0). If this model uses RoPE-based "
                        "attention, simulation results may be inaccurate.",
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

        # ----------------------------------------------------------------
        # Patch 10: DeepseekV3DecoderLayer — bridge HF ↔ TensorCast MLA
        # ----------------------------------------------------------------
        # WHY:   (a) The original decoder unpacks 3 values from self_attn
        #            but the TensorCast MLA wrapper returns 2 (no attn
        #            weights).  This patch handles both return conventions.
        #        (b) Kimi K2.5 computes RoPE internally but TensorCast MLA
        #            needs explicit (cos, sin) position_embeddings.
        #        (c) The patched VL forward filters out tensor_cast-
        #            specific kwargs; we recover them from
        #            _extra_forward_kwargs (injected by model_runner).
        # WITHOUT: ValueError from tuple unpacking; missing RoPE;
        #          missing attention_meta leading to broken KV cache ops.
        class_ref_decoder = "modeling_deepseek.DeepseekV3DecoderLayer"
        decoder_cls = get_class_from_dynamic_module(class_ref_decoder, model_id, force_download=False)

        if not hasattr(decoder_cls, "_original_decoder_forward"):
            decoder_cls._original_decoder_forward = decoder_cls.forward

        def patched_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Resolve position_embeddings (cos, sin) for TensorCast MLA.
            position_embeddings = kwargs.pop("position_embeddings", None)
            if position_embeddings is None:
                # Lazy-initialize _has_rotary_emb (moved from mla.py __init__
                # to avoid polluting the generic MLA layer).
                if not hasattr(self.self_attn, '_has_rotary_emb'):
                    self.self_attn._has_rotary_emb = hasattr(self.self_attn._inner, "rotary_emb")
                    if not self.self_attn._has_rotary_emb:
                        import warnings

                        warnings.warn(
                            f"MLA module '{type(self.self_attn._inner).__name__}' "
                            "lacks 'rotary_emb'.  If position_embeddings is not "
                            "provided at forward time, RoPE will be disabled "
                            "(cos=1, sin=0), producing incorrect results for "
                            "RoPE-dependent models.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                position_embeddings = self.self_attn._resolve_position_embeddings(
                    hidden_states, None, position_ids=position_ids, **kwargs
                )

            # Recover tensor_cast-specific kwargs filtered by the VL forward.
            if "attention_meta" not in kwargs:
                extra_kwargs = getattr(self.self_attn, '_extra_forward_kwargs', None)
                if extra_kwargs is not None and extra_kwargs.get('attention_meta') is not None:
                    for k, v in extra_kwargs.items():
                        if k not in kwargs and v is not None:
                            kwargs[k] = v

            attn_result = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            if isinstance(attn_result, tuple) and len(attn_result) == 2:
                hidden_states, present_key_value = attn_result
                self_attn_weights = None
            else:
                hidden_states, self_attn_weights, present_key_value = attn_result

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        decoder_cls.forward = patched_decoder_forward
        patched = True

        # ----------------------------------------------------------------
        # Patch 11: MoonVision3dPatchEmbed — support 2D (flattened) input
        # ----------------------------------------------------------------
        # WHY:   During simulation, vision tokens may arrive as a flat 2D
        #        tensor (total_tokens, channels) rather than the original
        #        3D patches.  The original Conv2d projection expects 4D
        #        input.  This patch reshapes 2D input back to 4D chunks
        #        and uses linear projection instead.
        # WITHOUT: RuntimeError from Conv2d receiving 2D input.
        class_ref_patch_embed = "modeling_kimi_k25.MoonVision3dPatchEmbed"
        patch_embed_cls = get_class_from_dynamic_module(
            class_ref_patch_embed,
            model_id,
            force_download=False,
        )

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
                    linear_weight = self.proj.weight.view(
                        out_dim,
                        in_channels * kH * kW,
                    )
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

        # ----------------------------------------------------------------
        # Patch 12: Fix expert counts on root config
        # ----------------------------------------------------------------
        # WHY:   Kimi K2.5 stores ``n_routed_experts`` and
        #        ``n_shared_experts`` inside ``text_config`` rather than
        #        at the root level.  The downstream MoE patching logic
        #        (transformations.patch_moe) reads them from the root
        #        config object.
        # WITHOUT: AttributeError when patch_moe tries to read
        #          ``config.n_routed_experts`` / ``config.n_shared_experts``.
        if config is not None:
            if hasattr(config, "text_config") and hasattr(config.text_config, "n_routed_experts"):
                setattr(
                    config,
                    "n_routed_experts",
                    config.text_config.n_routed_experts,
                )

            if not hasattr(config, "n_routed_experts"):
                setattr(config, "n_routed_experts", 384)
                logger.warning(
                    "n_routed_experts not found in config or text_config; "
                    "falling back to default value 384. "
                    "Verify that the model's expert count matches this default."
                )

            if not hasattr(config, "n_shared_experts"):
                if hasattr(config, "text_config") and hasattr(config.text_config, "n_shared_experts"):
                    setattr(
                        config,
                        "n_shared_experts",
                        config.text_config.n_shared_experts,
                    )
                else:
                    setattr(config, "n_shared_experts", 1)
                    logger.warning(
                        "n_shared_experts not found in config or text_config; "
                        "falling back to default value 1. "
                        "Verify that the model's shared expert count matches this default."
                    )

    except Exception as e:
        logger.warning(f"Could not patch remote modules: {e}")

    return patched


_patched_kimi_k25 = False


def _hf_config_patch_for_kimi_k25(config, model_id=None):
    """Pre-load entry point: apply HF config fixes, then model class patches.

    Called by :func:`AutoModelConfigLoader._apply_hf_config_patches` BEFORE
    the HuggingFace model is instantiated.
    """
    import logging

    logger = logging.getLogger(__name__)

    global _patched_kimi_k25
    if _patched_kimi_k25:
        return

    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k25":
        return

    # Phase 1 – environment / config-level patches (no model_id needed).
    config_patched = _patch_hf_config_for_kimi_k25(config)

    # Phase 2 – model class monkey-patches (requires model_id).
    classes_patched = _patch_model_classes_for_kimi_k25(config, model_id)

    if config_patched or classes_patched:
        _patched_kimi_k25 = True
        logger.info("Patched transformers environment for Kimi-K2.5")


register_model_profile(
    ModelProfile(
        model_type="kimi_k25",
        moe_module_name="DeepseekV3MoE",
        mla_module_name="DeepseekV3Attention",
        moe_num_experts_key="n_routed_experts",
        language_layers_path_str="language_model.model.layers",
        visual_module_path="vision_tower",
        language_module_path="language_model",
        visual_layers_module_path="vision_tower.encoder.blocks",
        visual_layers_path_str="vision_tower.encoder.blocks",
        custom_expert_module_type=None,
        mla_field_names_override={
            "q_proj": "q_a_proj",
            "qk_head_dim": "q_head_dim",
        },
        hf_config_patch_method=_hf_config_patch_for_kimi_k25,
        # When DP≠EP, route is executed after DP slicing to avoid performance bloat caused by routing all tokens
        moe_route_after_dp_transform=True,
    )
)
