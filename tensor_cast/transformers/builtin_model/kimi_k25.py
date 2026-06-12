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

    # ----------------------------------------------------------------
    # Patch 4a: Windows SIGALRM — resolve trust_remote_code without alarm
    # ----------------------------------------------------------------
    # WHY:   Windows lacks signal.SIGALRM, which breaks transformers'
    #        trust_remote_code interactive prompt.  Default
    #        trust_remote_code=True on platforms without SIGALRM so
    #        that headless simulation never blocks on stdin.
    #        This was previously a global monkey-patch in utils.py;
    #        moved here to limit its scope to Kimi K2.5 only.
    # WITHOUT: Blocking on stdin / AttributeError from signal.SIGALRM
    #          when loading remote model code.
    import signal as _signal

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
                            num_heads = q.shape[1]
                            head_dim = q.shape[-1]

                            if q.device.type == 'meta':
                                # -------------------------------------------------------
                                # Call the fused tensor_cast.attention op so that
                                # `tensor_cast.attention.default` appears in the chrome
                                # trace, enabling accurate analytic performance modeling.
                                #
                                # Shape mapping (varlen → tensor_cast convention):
                                #   q: (seq_len, num_heads, head_dim)
                                #      → query: (seq_len, num_heads * head_dim)
                                #   k: (seq_len, num_heads, head_dim)
                                #      → key:   (seq_len, num_heads, head_dim)
                                #   v: (seq_len, num_heads, head_dim)
                                #      → value: (seq_len, num_heads, head_dim)
                                #
                                # Metadata is passed as None — matching the standard
                                # visual attention path in flash_attention_forward()
                                # (attention.py L80: attention_meta = None).  The
                                # performance model falls back to deriving seq_lens
                                # and query_lens from query.shape, avoiding
                                # .item() on meta tensors.
                                # -------------------------------------------------------
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
                                    "Visual attention sequence length %d exceeds safe "
                                    "threshold. Skipping O(n²) attention to avoid OOM.",
                                    seq_length,
                                )
                                return torch.zeros(
                                    seq_length,
                                    num_heads * head_dim,
                                    device=q.device,
                                    dtype=q.dtype,
                                )

                            # Build causal-like attention mask: allow attention
                            # only within each image chunk (diagonal blocks).
                            # Using -inf for masked positions (correct additive mask)
                            # instead of boolean True/False (which would add 1.0/0.0).
                            attention_mask = torch.full(
                                [1, seq_length, seq_length],
                                float('-inf'),
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

        # ----------------------------------------------------------------
        # Patch 10a: Register 'tensor_cast' in ATTENTION_CLASSES
        # ----------------------------------------------------------------
        # WHY:   Patch 2 downgrades config._attn_implementation from
        #        'flash_attention_2' to 'tensor_cast'.  Later, MTP block
        #        creation calls ATTENTION_CLASSES[config._attn_implementation]
        #        which only knows 'eager' / 'sdpa' / 'flash_attention_2'.
        # WITHOUT: KeyError: 'tensor_cast' during MTP block construction.
        import sys

        remote_module = sys.modules.get(decoder_cls.__module__)
        if remote_module is not None and hasattr(remote_module, 'ATTENTION_CLASSES'):
            if 'tensor_cast' not in remote_module.ATTENTION_CLASSES:
                fallback = remote_module.ATTENTION_CLASSES.get('sdpa') or remote_module.ATTENTION_CLASSES.get('eager')
                if fallback is None:
                    raise ValueError(
                        f"ATTENTION_CLASSES lacks 'sdpa' or 'eager' fallback. "
                        f"Available: {list(remote_module.ATTENTION_CLASSES.keys())}"
                    )
                remote_module.ATTENTION_CLASSES['tensor_cast'] = fallback

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

        # ----------------------------------------------------------------
        # Patch 13: ModelWrapper — add output_intermediate_hidden_states for MTP
        #           and apply selected_token_indices for prefill token pruning
        # ----------------------------------------------------------------
        # WHY:   (a) The generic ``ModelWrapper`` only returns a single tensor
        #        from ``forward()``, but ``MtpWrapper`` expects
        #        ``(logits, hidden_states)`` when MTP is enabled.
        #        (b) ``ModelWrapper`` delegates directly to the HF model
        #        which has its own internal ``lm_head`` — it cannot apply
        #        ``selected_token_indices`` *before* the lm_head like
        #        ``CausalLmWrapper`` can.  Instead we apply it *after* the
        #        HF model's forward to select only the desired logit rows.
        #        (c) For the MTP branch we also prune the intermediate
        #        hidden_states so MTP layers only process the selected tokens.
        #
        #        The patch is additive and backwards-compatible: the default
        #        path (no MTP, no selected_indices) is identical to the
        #        original behaviour.
        # WITHOUT: (a) ``AssertionError: Can't unpack a tensor of 1 rows
        #          into a tuple of 2 elements`` in ``MtpWrapper.forward()``.
        #          (b) 42000×7168×163840 lm_head matmul instead of
        #          12×7168×163840 during prefill, inflating compute cost
        #          ~3500×.
        from tensor_cast.transformers.model import ModelWrapper

        if not hasattr(ModelWrapper, "_patched_for_mtp"):
            _original_mw_forward = ModelWrapper.forward

            def patched_mw_forward(
                self,
                input_ids: Optional[torch.Tensor],
                position_ids: torch.Tensor,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_intermediate_hidden_states: bool = False,
                **kwargs: object,
            ):
                # Extract selected_token_indices from sampling_metadata
                # (injected by generate_inputs() for prefill token pruning).
                sampling_metadata = kwargs.get("sampling_metadata")
                selected_indices = sampling_metadata.selected_token_indices if sampling_metadata is not None else None

                if output_intermediate_hidden_states:
                    # MTP path: delegate to HF model, prune logits only.
                    # intermediate_hidden_states must NOT be pruned here
                    # because MtpWrapper needs the full seq_len for
                    # rotary_emb and the MTP layers will apply
                    # selected_token_indices themselves (see
                    # MultiTokenPredictor.forward in mtp.py).
                    kwargs_with_hidden = {**kwargs, "output_hidden_states": True}
                    outputs = self._inner(
                        input_ids=input_ids,
                        use_cache=False,
                        position_ids=position_ids,
                        inputs_embeds=inputs_embeds,
                        return_dict=False,
                        **kwargs_with_hidden,
                    )
                    logits = outputs[0]
                    intermediate_hidden_states = outputs[1][-1]
                    if selected_indices is not None:
                        logits = logits.index_select(1, selected_indices)
                    return logits, intermediate_hidden_states

                # Non-MTP path
                if selected_indices is not None and inputs_embeds is None:
                    # ------------------------------------------------------------
                    # Fix: Check whether image inputs are present.  If the user
                    # supplied pixel_values / image_grid_thw, we must route
                    # through the full VL forward (KimiK25ForConditionalGeneration)
                    # so that the visual encoder is executed and image features
                    # are merged with text embeddings.  Bypassing the VL forward
                    # would silently drop the image, producing wrong results
                    # and a misleading trace (no visual ops).
                    # ------------------------------------------------------------
                    has_image_input = kwargs.get("pixel_values") is not None or kwargs.get("image_grid_thw") is not None
                    if not has_image_input:
                        # Optimization: prune hidden_states BEFORE lm_head.
                        # Bypass the HF VL forward (KimiK25ForConditionalGeneration)
                        # and directly call the language model's transformer body,
                        # then apply lm_head on only the selected tokens.
                        # This avoids computing lm_head on all tokens.
                        #
                        # We must inject tensor_cast kwargs (attention_meta,
                        # kv_cache_by_layers, etc.) into each attention layer's
                        # _extra_forward_kwargs side-channel, replicating what
                        # Patch 4 does for the normal VL forward path.  Without
                        # this the MLA layers see None kv_cache and the
                        # performance estimator crashes.
                        from tensor_cast.transformers.model import _EXTRA_TC_KWARGS_KEYS

                        lm = self._inner.language_model
                        _tc_extra = {
                            k: kwargs[k] for k in _EXTRA_TC_KWARGS_KEYS if k in kwargs and kwargs[k] is not None
                        }
                        if _tc_extra:
                            for layer in lm.model.layers:
                                if hasattr(layer, "self_attn"):
                                    layer.self_attn._extra_forward_kwargs = _tc_extra

                        body_outputs = lm.model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=False,
                            return_dict=True,
                        )
                        hidden_states = body_outputs.last_hidden_state
                        hidden_states = hidden_states.index_select(1, selected_indices)
                        logits = lm.lm_head(hidden_states)
                        return logits

                # Default / fallback path
                logits = _original_mw_forward(self, input_ids, position_ids, inputs_embeds, **kwargs)
                if selected_indices is not None:
                    logits = logits.index_select(1, selected_indices)
                return logits

            ModelWrapper.forward = patched_mw_forward
            ModelWrapper._patched_for_mtp = True
            patched = True

        # ----------------------------------------------------------------
        # Patch 14: DeepseekV3RotaryEmbedding — handle position_ids as seq_len
        # ----------------------------------------------------------------
        # WHY:   ``maybe_enable_mtp`` (line 228) runs BEFORE
        #        ``patch_rotary_emb`` (line 231), so ``MtpWrapper.__init__``
        #        captures the inner ``DeepseekV3RotaryEmbedding`` (not the
        #        ``CachingRotaryEmb`` wrapper that is applied later).
        #        The inner ``forward(x, seq_len)`` expects an integer
        #        ``seq_len``, but ``MtpWrapper`` passes ``position_ids``
        #        (a tensor).  This patch makes the inner rotary embedding
        #        tolerate a tensor ``seq_len`` by extracting its maximum
        #        value.
        # WITHOUT: ``TypeError: arange() received an invalid combination
        #          of arguments - got (Tensor, ...)`` at the
        #          ``rotary_emb`` call in ``MtpWrapper.forward()``.
        class_ref_rotary = "modeling_deepseek.DeepseekV3RotaryEmbedding"
        rotary_cls = get_class_from_dynamic_module(class_ref_rotary, model_id, force_download=False)

        if not hasattr(rotary_cls, "_patched_for_kimi_k25"):
            _original_rotary_forward = rotary_cls.forward

            def patched_rotary_forward(self, x, seq_len=None):
                if isinstance(seq_len, torch.Tensor):
                    # MtpWrapper passes position_ids (tensor) as seq_len.
                    # Determine the sequence-length integer for arange/slicing.
                    if seq_len.device.type == "meta":
                        # TorchDynamo tracing on meta: use config value as a
                        # safe upper bound. The cache will be rebuilt with the
                        # real max position at runtime.
                        max_pos = self.max_position_embeddings
                    else:
                        # Runtime (eager, after graph-break resume).
                        # +1 because position_ids are 0-based (e.g. [0..N-1]).
                        max_pos = int(seq_len.max().item()) + 1
                    if self.max_seq_len_cached is None or max_pos > self.max_seq_len_cached:
                        self._set_cos_sin_cache(
                            seq_len=max_pos,
                            device=x.device,
                            dtype=x.dtype,
                        )
                    return (
                        self.cos_cached[:max_pos].to(dtype=x.dtype),
                        self.sin_cached[:max_pos].to(dtype=x.dtype),
                    )
                return _original_rotary_forward(self, x, seq_len)

            rotary_cls.forward = patched_rotary_forward
            rotary_cls._patched_for_kimi_k25 = True
            patched = True

        # ----------------------------------------------------------------
        # Patch 15: MultiTokenPredictorLayer — unpack tuple from decoder
        # ----------------------------------------------------------------
        # WHY:   Both the original and patched
        #        ``DeepseekV3DecoderLayer.forward`` return a tuple
        #        ``(hidden_states, ...)``.  ``MultiTokenPredictorLayer``
        #        passes this tuple through to ``MultiTokenPredictor``,
        #        which tries to call ``.index_select()`` on it — tuples
        #        don't have that method.  Unpack the first tensor element.
        # WITHOUT: ``AttributeError: 'tuple' object has no attribute
        #          'index_select'`` at ``MultiTokenPredictor.forward()``.
        from tensor_cast.layers.mtp import MultiTokenPredictorLayer

        if not hasattr(MultiTokenPredictorLayer, "_patched_for_kimi_k25"):
            _original_mtp_layer_forward = MultiTokenPredictorLayer.forward

            def patched_mtp_layer_forward(
                self,
                inputs_embeds: torch.Tensor,
                position_ids: torch.Tensor,
                previous_hidden_states: torch.Tensor,
                position_embeddings: Optional[torch.Tensor] = None,
                **kwargs,
            ):
                hidden_states = _original_mtp_layer_forward(
                    self,
                    inputs_embeds,
                    position_ids,
                    previous_hidden_states,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                # The decoder layer (e.g. DeepseekV3DecoderLayer) returns
                # a tuple (hidden_states, ...).  Unpack the first tensor.
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                return hidden_states

            MultiTokenPredictorLayer.forward = patched_mtp_layer_forward
            MultiTokenPredictorLayer._patched_for_kimi_k25 = True
            patched = True

    except Exception as e:
        logger.warning(f"Could not patch remote modules: {e}")

    return patched


def _shard_lm_head_for_kimi_vl(model):
    """Manually apply ``ColumnParallelLinear`` to the nested lm_head.

    Kimi K2.5 is a VL model where ``lm_head`` lives inside the
    ``language_model`` submodule (``_inner.language_model.lm_head``),
    not at the top level.  The standard ``shard_model_by_tp`` uses a
    fnmatch pattern ``"lm_head"`` which only matches a top-level
    (unprefixed) name.  After ``strip_module_name``, the nested path
    becomes ``"language_model.lm_head"`` → no match → lm_head stays as
    a raw ``nn.Linear`` and escapes TP sharding.

    This function is called AFTER ``shard_model`` in the custom
    pipeline and replaces the still-unsharded lm_head with a
    ``ColumnParallelLinear`` that gathers output across the TP group.

    Args:
        model: A ``TransformerModel`` whose ``_inner`` is a ``ModelWrapper``
               wrapping the Kimi HF model.
    """
    from tensor_cast.layers.parallel_linear import ColumnParallelLinear

    pgm = model.parallel_group_manager
    lmhead_tp_group = pgm.lmhead_tp_group
    tp_group = pgm.tp_group

    if lmhead_tp_group.world_size <= 1:
        return  # No TP configured — nothing to do.

    # Two nested lm_head instances escape the standard
    # ``shard_model_by_tp`` fnmatch pattern ``"lm_head"``:
    # 1. VL model:  ``*language_model.lm_head``
    # 2. MTP block: ``*mtp.lm_head``
    #
    # Iterate all modules and shard every still-raw nn.Linear
    # whose path ends with one of those suffixes.
    _LMIHEAD_SUFFIXES = ("language_model.lm_head", "mtp.lm_head")
    for name, module in model._inner.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(_LMIHEAD_SUFFIXES):
            params = {
                "tp_group": lmhead_tp_group,
                "global_tp_group": tp_group,
                "gather_output": True,
            }
            parallel_module = ColumnParallelLinear(module, **params)
            model._replace_module(name, parallel_module)


_patched_kimi_k25 = False
_shard_model_patched = False


def _patch_shard_model_for_kimi_vl():
    """Monkey-patch ``shard_model`` to automatically shard nested lm_head.

    Kimi K2.5's ``lm_head`` lives at ``language_model.lm_head`` (not top-level),
    so the standard ``shard_model_by_tp`` fnmatch pattern ``"lm_head"``
    misses it (``strip_module_name`` yields ``"language_model.lm_head"``).

    This patch wraps ``shard_model`` to call ``_shard_lm_head_for_kimi_vl``
    after the standard sharding.

    IMPORTANT:  Two references must be patched because ``model.py`` imports
    ``shard_model`` via ``from ... import shard_model``, creating a local
    binding that bypasses a module-attribute monkey-patch.
    """
    global _shard_model_patched
    if _shard_model_patched:
        return

    from tensor_cast.transformers import transformations as _t
    from tensor_cast.transformers import model as _model

    _original_shard_model = _t.shard_model

    def _patched_shard_model(model):
        result = _original_shard_model(model)
        _shard_lm_head_for_kimi_vl(result)
        return result

    # Patch both references:
    # 1. transformations.shard_model — for callers that use the module attribute
    # 2. model.shard_model         — for model.py's ``from ... import shard_model``
    _t.shard_model = _patched_shard_model
    _model.shard_model = _patched_shard_model
    _shard_model_patched = True


# ----------------------------------------------------------------
# Patch 16: resize_image — Kimi K2.5 specific image resize logic
# ----------------------------------------------------------------
_resize_image_patched = False


def _patch_resize_image_for_kimi_k25(model_id):
    """Monkey-patch ``resize_image`` to use Kimi K2.5's resize logic.

    WHY:   The generic ``resize_image`` in ``input_generator.py`` delegates
           to Qwen2-VL's ``smart_resize``, which relies on the image
           processor's ``size`` attribute for min/max pixel limits.
           Kimi K2.5's ``KimiK25VisionProcessor`` (from remote code) does
           NOT expose a standard ``size`` attribute — it uses
           ``media_proc_cfg["in_patch_limit"]`` instead.  Without this
           patch, ``smart_resize`` falls back to its hardcoded defaults
           (``max_pixels=1_003_520``), which are too restrictive for
           Kimi K2.5's larger images (e.g. 1080×1920 = 2 073 600 pixels),
           causing the image to be incorrectly downscaled.

    HOW:   When ``model_id`` contains "kimi" (case-insensitive), the
           patched ``resize_image`` bypasses ``smart_resize`` entirely and
           computes resized dimensions directly by rounding the original
           image dimensions to multiples of ``patch_size * merge_size``.
           This preserves the full resolution (limited only by
           ``in_patch_limit``, which is generous enough for common
           resolutions).

    WITHOUT: Vision token count mismatch — e.g. 4888 tokens instead of
             the expected 10764 for a 1080×1920 image.
    """
    global _resize_image_patched
    if _resize_image_patched:
        return

    import logging

    logger = logging.getLogger(__name__)

    from tensor_cast.core import input_generator as _ig

    _original_resize_image = _ig.resize_image

    def _kimi_resize_image(
        mid,
        mtype,
        image_height,
        image_width,
        patch_size,
        merge_size,
        temporal_patch_size,
    ):
        # Only intercept Kimi K2.5 (model_id check is case-insensitive).
        if "kimi" not in mid.lower():
            return _original_resize_image(
                mid,
                mtype,
                image_height,
                image_width,
                patch_size,
                merge_size,
                temporal_patch_size,
            )

        # Kimi K2.5 does NOT use Qwen2-VL's smart_resize.
        # MoonViT processes images at (near) full resolution: dimensions are
        # simply rounded to multiples of ``patch_size * merge_size``.
        #
        # The processor's ``media_proc_cfg["in_patch_limit"]`` defines the
        # maximum number of patches (typically 16384), which translates to a
        # generous pixel budget (16384 * 14 * 14 = 3 211 264 px for
        # patch_size=14).  Common resolutions like 1080×1920 (2 073 600 px)
        # fall well within this limit, so no downscaling is needed.
        factor = patch_size * merge_size
        resized_height = ((image_height + factor - 1) // factor) * factor
        resized_width = ((image_width + factor - 1) // factor) * factor
        logger.info(
            "Kimi K2.5 image resize: %dx%d -> %dx%d (factor=%d, bypassed Qwen2-VL smart_resize)",
            image_height,
            image_width,
            resized_height,
            resized_width,
            factor,
        )
        return resized_height, resized_width

    _ig.resize_image = _kimi_resize_image
    _resize_image_patched = True


def _hf_config_patch_for_kimi_k25(config, model_id=None):
    """Pre-load entry point: apply HF config fixes, then model class patches.

    Called by :func:`AutoModelConfigLoader._apply_hf_config_patches` BEFORE
    the HuggingFace model is instantiated.

    The patching is split into two tiers:

    * **Per-config patches** (always run):  ``_attn_implementation``
      downgrade, vision-config attribute bridging, environment checks
      (e.g. ``is_torch_fx_available``).  These operate on the config
      *object* and MUST execute for every new config instance, even
      when class-level monkey-patches have already been applied.

    * **Class-level patches** (run once):  model-class monkey-patching,
      ``shard_model`` wrapping, ``resize_image`` patching.  These
      modify global state (module attributes / function references)
      and are guarded by ``_patched_kimi_k25`` to avoid redundant work.
    """
    import logging

    logger = logging.getLogger(__name__)

    model_type = getattr(config, "model_type", None)
    if model_type != "kimi_k25":
        return

    # ----------------------------------------------------------------
    # Phase 1 – config-level patches (always run for every new config)
    # ----------------------------------------------------------------
    config_patched = _patch_hf_config_for_kimi_k25(config)

    # ----------------------------------------------------------------
    # Phases 2-4 – class-level / global patches (run once per process)
    # ----------------------------------------------------------------
    # These modify module-level state (monkey-patches, function
    # references, etc.).  They are idempotent but expensive, so we
    # guard them with a global flag.
    global _patched_kimi_k25
    if _patched_kimi_k25:
        return

    # Phase 2 – model class monkey-patches (requires model_id).
    classes_patched = _patch_model_classes_for_kimi_k25(config, model_id)

    # Phase 3 – wrap shard_model to handle nested lm_head.
    _patch_shard_model_for_kimi_vl()

    # Phase 4 – patch resize_image for Kimi K2.5's image resize logic.
    _patch_resize_image_for_kimi_k25(model_id)

    if config_patched or classes_patched:
        _patched_kimi_k25 = True
        logger.info("Patched transformers environment for Kimi-K2.5")


register_model_profile(
    ModelProfile(
        model_type="kimi_k25",
        moe_module_name="DeepseekV3MoE",
        mla_module_name="DeepseekV3Attention",
        mtp_block_module_name="DeepseekV3DecoderLayer",
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
