"""Native DeepSeek V4.1-Flash DSpark draft model.

The checkpoint's ``mtp.*`` modules are DSpark stages, not generic MTP layers
and not the Qwen3 draft backbone used by TensorCast's generic DSpark adapter.
This module keeps the reusable proposal/Markov/Confidence machinery while
modeling the V4.1-specific draft blocks, target taps, mHC stream and MoE shape.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from .dspark import ConfidenceHead, DsparkWrapper, MarkovHead
from .sampler import Sampler, SamplingMetadata
from .utils import ModelWrapperBase
from ..model_config import DsparkConfig
from ..transformers.builtin_model.deepseek_v41 import (
    DeepseekV41MoE,
    DeepseekV41RMSNorm,
)


def _identity_pre_mix(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
    shape = (*hidden_states.shape[:-2], int(hc_mult))
    result = hidden_states.new_zeros(shape, dtype=torch.float32)
    result[..., 0] = 1.0
    return result


class DeepseekV41DsparkAttention(nn.Module):
    """Window-only DSpark attention from the reference ``DSparkAttention``.

    Draft attention owns its KV stream.  It intentionally does not read or
    publish ``DeepseekV41SharedAttentionState`` used by backbone CSA2 layers.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.n_groups = int(config.o_groups)
        self.o_lora_rank = int(config.o_lora_rank)
        self.window_size = int(config.sliding_window)
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV41RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV41RMSNorm(self.head_dim)
        per_group = self.num_heads * self.head_dim // self.n_groups
        self.wo_a = nn.Linear(per_group, self.n_groups * self.o_lora_rank, bias=False)
        self.o_proj = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.softmax_scale = self.head_dim**-0.5

    def _project_main_kv(self, main_hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        main_kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(main_hidden))
        cos = torch.empty(main_hidden.shape[1], self.rope_head_dim, dtype=main_kv.dtype, device=main_kv.device)
        sin = torch.empty_like(cos)
        torch.ops.tensor_cast.apply_rope_inplace(main_kv, cos, sin, True, False, self.rope_head_dim)
        main_kv, _ = torch.ops.tensor_cast.dynamic_quantize_symmetric(
            main_kv,
            [-1],
            scale_dtype=torch.float32,
            out_dtype=torch.float8_e4m3fn,
        )
        return main_kv

    def prefill(self, main_hidden: torch.Tensor, start_pos: int = 0, window_kv=None) -> torch.Tensor:
        """Seed this draft layer's FP8 sliding-window KV cache."""
        main_kv = self._project_main_kv(main_hidden, start_pos)
        if window_kv is None:
            window_kv = torch.zeros(
                main_kv.shape[0], self.window_size, self.head_dim, dtype=main_kv.dtype, device=main_kv.device
            )
        tail = main_kv[:, -self.window_size :, :]
        first_position = int(start_pos) + main_kv.shape[1] - tail.shape[1]
        positions = torch.arange(first_position, first_position + tail.shape[1], device=main_kv.device)
        slots = torch.arange(main_kv.shape[0], device=main_kv.device)[:, None] * self.window_size
        slots = slots + positions[None, :] % self.window_size
        return torch.ops.tensor_cast.csa2_cache_update(window_kv, tail, slots)

    def forward(
        self,
        hidden_states: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int,
        kv_state: dict[int, torch.Tensor],
        layer_id: int = 0,
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        local_heads = q.shape[-1] // self.head_dim
        q = q.view(batch, seq, local_heads, self.head_dim)
        cos = torch.empty(seq, self.rope_head_dim, dtype=q.dtype, device=q.device)
        sin = torch.empty_like(cos)
        torch.ops.tensor_cast.apply_rope_inplace(q, cos, sin, True, False, self.rope_head_dim)
        main_kv = self._project_main_kv(main_hidden, start_pos)
        draft_kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        torch.ops.tensor_cast.apply_rope_inplace(draft_kv, cos, sin, True, False, self.rope_head_dim)
        draft_kv, _ = torch.ops.tensor_cast.dynamic_quantize_symmetric(
            draft_kv,
            [-1],
            scale_dtype=torch.float32,
            out_dtype=torch.float8_e4m3fn,
        )
        # Native decode updates one ring slot from main_x, then attends over
        # the full 128-token draft window plus the current proposal block.
        window_kv = kv_state.get(layer_id)
        if window_kv is None:
            raise ValueError("Native DSpark decode requires a seeded request KV window")
        if window_kv.shape != (batch, self.window_size, self.head_dim):
            raise ValueError("Native DSpark KV window does not match the request batch")
        ring_slot = int(start_pos) % self.window_size
        slots = torch.arange(batch, device=main_kv.device, dtype=torch.long) * self.window_size + ring_slot
        window_kv = torch.ops.tensor_cast.csa2_cache_update(window_kv, main_kv, slots)
        kv_state[layer_id] = window_kv
        kv = torch.ops.tensor_cast.cat([window_kv, draft_kv], 1)
        topk = torch.empty(batch, seq, self.window_size + seq, dtype=torch.int32, device=hidden_states.device)
        sink = self.attn_sink[:local_heads]
        output = torch.ops.tensor_cast.sparse_attn_sharedkv(q, kv, sink, topk, self.softmax_scale, self.head_dim)
        torch.ops.tensor_cast.apply_rope_inplace(output, cos, sin, True, True, self.rope_head_dim)
        output = output.to(hidden_states.dtype)
        local_groups = max(self.n_groups * local_heads // self.num_heads, 1)
        output = output.view(batch, seq, local_groups, -1)
        weight = self.wo_a.weight.view(local_groups, self.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, weight)
        return self.o_proj(output.flatten(2))


class DeepseekV41DsparkBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hc_mult = int(config.hc_mult)
        self.hc_eps = float(config.hc_eps)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        hidden = int(config.hidden_size)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * hidden
        self.attn = DeepseekV41DsparkAttention(config)
        self.mlp = DeepseekV41MoE(config)
        self.input_layernorm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(self, hidden_states, fn, scale, base, pre_mix):
        return torch.ops.tensor_cast.hc_pre_v3(
            hidden_states,
            fn,
            scale,
            base,
            pre_mix,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.config.rms_norm_eps,
            self.hc_eps,
            "incoming",
        )

    def forward(self, hidden_states, pre_mix, main_hidden, start_pos, kv_state, layer_id):
        residual = hidden_states
        reduced, attn_post, attn_comb, attn_pre = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix
        )
        reduced = torch.ops.tensor_cast.rms_norm(reduced, self.input_layernorm.weight.data, self.config.rms_norm_eps)
        reduced = self.attn(reduced, main_hidden, start_pos, kv_state, layer_id)
        hidden_states = torch.ops.tensor_cast.hc_post(reduced, residual, attn_post, attn_comb, self.hc_mult)
        residual = hidden_states
        reduced, ffn_post, ffn_comb, ffn_pre = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre
        )
        reduced = torch.ops.tensor_cast.rms_norm(
            reduced, self.post_attention_layernorm.weight.data, self.config.rms_norm_eps
        )
        reduced = self.mlp(reduced)
        hidden_states = torch.ops.tensor_cast.hc_post(reduced, residual, ffn_post, ffn_comb, self.hc_mult)
        return hidden_states, ffn_pre


class DeepseekV41NativeDsparkDraft(nn.Module):
    def __init__(self, config, scfg: DsparkConfig):
        super().__init__()
        config = copy.deepcopy(config)
        config.n_routed_experts = int(getattr(config, "dspark_n_routed_experts", 128))
        config.num_experts_per_tok = int(getattr(config, "dspark_num_experts_per_tok", 3))
        config.engram_layer_ids = []
        self.native_dspark = True
        self.config = config
        self.target_layer_ids = list(
            getattr(config, "dspark_target_layer_ids", None) or getattr(config, "excluded_dspark_layer_ids", ())
        )
        self.verify_window_size = int(scfg.dspark_block_size)
        self.proposal_count = self.verify_window_size - 1
        if self.proposal_count <= 0:
            raise ValueError("DeepSeek V4.1 native DSpark requires at least one speculative token")
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", 0))
        hidden = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.main_proj = nn.Linear(hidden * len(self.target_layer_ids), hidden, bias=False)
        self.main_norm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(DeepseekV41DsparkBlock(config) for _ in range(scfg.num_draft_layers))
        self.norm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(
            int(config.vocab_size), int(config.vocab_size), int(scfg.markov_rank), hidden_size=hidden
        )
        self.confidence_head = ConfidenceHead(hidden + int(scfg.markov_rank))
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, hidden)
        self.lm_head: nn.Module = nn.Linear(hidden, config.vocab_size, bias=False)

    def set_shared(self, embed_tokens: nn.Module, lm_head: nn.Module):
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def _main_hidden(self, aux_hiddens):
        return self.main_norm(self.main_proj(torch.ops.tensor_cast.cat(aux_hiddens, -1)))

    def prefill(self, aux_hiddens, start_pos: int = 0, kv_state=None):
        main_hidden = self._main_hidden(aux_hiddens)
        return tuple(
            layer.attn.prefill(main_hidden, start_pos, kv_state.get(i) if kv_state is not None else None)
            for i, layer in enumerate(self.layers)
        )

    def forward(self, aux_hiddens, anchor_ids, start_pos: int, kv_state):
        main_hidden = self._main_hidden(aux_hiddens)
        ids = anchor_ids.new_full((anchor_ids.shape[0], self.proposal_count), self.noise_token_id)
        ids[:, 0] = anchor_ids
        hidden = self.embed_tokens(ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = _identity_pre_mix(hidden, self.hc_mult)
        for layer_id, layer in enumerate(self.layers):
            hidden, pre_mix = layer(hidden, pre_mix, main_hidden, start_pos, kv_state, layer_id)
        hidden = torch.ops.tensor_cast.hc_pre_reduce(hidden, pre_mix)
        hidden = torch.ops.tensor_cast.rms_norm(hidden, self.norm.weight.data, self.config.rms_norm_eps)
        return hidden, self.lm_head(hidden)


class DeepseekV41NativeDsparkWrapper(DsparkWrapper):
    def __init__(self, scfg, hf_config, model, draft):
        ModelWrapperBase.__init__(self, model)
        self.dspark_config = scfg
        self.dflash_config = scfg.to_dflash_config()
        self.hf_config = hf_config
        self.draft_hf_config = None
        self.dflash_block_size = int(scfg.dspark_block_size)
        self.draft = draft
        # Native DSpark attention uses the V4.1 partial-RoPE semantics inside
        # its model-specific attention and has no generic draft rotary module.
        self.rotary_emb = None
        self.sampler = Sampler()
        self._aux_layer_ids = list(draft.target_layer_ids)

    def _sample_sequential(self, draft_hidden, base_logits, batch_size, block, next_tokens):
        """Sample native DSpark proposals over the full vocabulary."""
        hidden_b = draft_hidden.reshape(batch_size, block, -1)
        logits_b = base_logits.reshape(batch_size, block, -1)
        prev = next_tokens[:, -1]
        tokens = []
        confidence = []
        markov_bias_module = self.draft.markov_head.markov_bias
        tp_group = getattr(markov_bias_module, "tp_group", None)
        for idx in range(block):
            local_logits = logits_b[:, idx]
            if tp_group is not None and local_logits.shape[-1] != self.hf_config.vocab_size:
                logits = tp_group.all_gather(local_logits, dim=-1)[..., : self.hf_config.vocab_size]
            else:
                logits = local_logits
            markov_embed, bias, _ = self.draft.markov_head(prev, hidden_b[:, idx], None)
            if bias.shape[-1] != logits.shape[-1]:
                if tp_group is None:
                    raise ValueError("Native DSpark requires full-vocabulary Markov bias")
                bias = tp_group.all_gather(bias, dim=-1)[..., : logits.shape[-1]]
            logits = logits + bias
            confidence_input = torch.cat([hidden_b[:, idx], markov_embed], dim=-1)
            confidence.append(self.draft.confidence_head(confidence_input).squeeze(-1))
            prev = torch.argmax(logits, dim=-1)
            tokens.append(prev)
        return torch.stack(tokens, dim=1), torch.stack(confidence, dim=1)

    def forward(self, input_ids, position_ids, inputs_embeds=None, **kwargs):
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        if sampling_metadata is None:
            raise ValueError("DeepSeek V4.1 native DSpark requires sampling_metadata")
        result = self._inner(
            input_ids,
            position_ids,
            inputs_embeds,
            output_intermediate_hidden_states=True,
            output_aux_hidden_state_layer_ids=self._aux_layer_ids,
            **kwargs,
        )
        target_logits, _, aux_hiddens = result
        next_tokens = self.sampler(target_logits, sampling_metadata)
        batch = next_tokens.shape[0]
        is_decode = self._is_decode_step(kwargs)
        attention_meta = kwargs.get("attention_meta")
        start_pos = int(getattr(attention_meta, "start_pos", 0) or 0)
        normalized = [h.reshape(batch, -1, h.shape[-1]) for h in aux_hiddens]
        request_state = kwargs.get("csa2_state")
        if not is_decode:
            draft_kv_state = self.draft.prefill(
                normalized, start_pos, request_state.draft_kv_by_layer if request_state is not None else None
            )
            if request_state is not None:
                request_state.draft_kv_by_layer.update(enumerate(draft_kv_state))
            return target_logits, draft_kv_state
        if request_state is None:
            raise ValueError("Native DSpark decode requires request-owned csa2_state")
        anchor = self._per_request_anchor_tokens(next_tokens, batch, sampling_metadata=sampling_metadata)
        # Official native DSpark decode consumes only the current target token
        # from layers 37/38/39, while the target model verifies n+1 rows.
        normalized = [hidden[:, -1:, :] for hidden in normalized]
        draft_hidden, draft_logits = self.draft(normalized, anchor, start_pos, request_state.draft_kv_by_layer)
        return self._propose_draft_tokens(draft_hidden, draft_logits, batch, self.draft.proposal_count, next_tokens)


def build_deepseek_v41_native_dspark(model, scfg: DsparkConfig, hf_config):
    expected_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
    target_ids = list(
        getattr(hf_config, "dspark_target_layer_ids", None) or getattr(hf_config, "excluded_dspark_layer_ids", ())
    )
    if expected_layers <= 0 or not target_ids:
        raise ValueError("DeepSeek V4.1 config does not declare a native DSpark draft model")
    if int(scfg.num_draft_layers) != expected_layers:
        raise ValueError(
            f"DeepSeek V4.1 native DSpark requires num_draft_layers={expected_layers}, got {scfg.num_draft_layers}"
        )
    scfg.sliding_window = int(getattr(hf_config, "sliding_window", 128) or 128)
    native_config = copy.deepcopy(hf_config)
    native_config.n_routed_experts = int(getattr(hf_config, "dspark_n_routed_experts", 128))
    native_config.num_experts_per_tok = int(getattr(hf_config, "dspark_num_experts_per_tok", 3))
    native_config.engram_layer_ids = []
    native_config.num_nextn_predict_layers = expected_layers
    native_config.dspark_target_layer_ids = target_ids
    draft = DeepseekV41NativeDsparkDraft(native_config, scfg)
    from .dflash import resolve_target_embed_and_lm_head

    embed, lm_head = resolve_target_embed_and_lm_head(model)
    draft.set_shared(embed, lm_head)
    return DeepseekV41NativeDsparkWrapper(scfg, hf_config, model._inner, draft)
