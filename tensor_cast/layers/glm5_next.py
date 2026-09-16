"""GLM-5.3-Flash language-only wrappers.

The upstream GLM5Next implementation has the same MLA tensor layout as the
existing GLM5 DSA path, but its decoder passes ``None`` for RoPE because this
model uses NoPE on MLA layers.  TensorCast's MLA wrapper still needs a
shape-carrying pair of tensors, including when the rotary dimension is zero.
"""

import torch
import torch.nn.functional as F

from .glm5 import Glm5SparseAttention
from .mla import tp_plan_nested_module_path
from . import COLWISE_LINEAR
from ..utils import exact_division


class Glm5NextSparseAttention(Glm5SparseAttention):
    """Reuse GLM5 sparse-MLA semantics for GLM5Next DSA layers.

    GLM5Next currently configures all indexer layers as ``full``.  The parent
    class retains the established full/shared IndexShare behaviour should a
    compatible GLM5Next configuration expose it in the future.
    """

    @property
    def indexer_cache_width(self) -> int:
        """Packed k-pool state: key, compression logits, and one valid flag."""
        return 2 * int(self.indexer.head_dim) + 1

    @property
    def indexer_cache_dtype(self) -> torch.dtype:
        """HF stores the k-pool state in the indexer projection dtype."""
        return self.indexer.wq_b.weight.dtype

    @classmethod
    def build_tp_plan_extras(cls, prefix, params, config_info):
        local_params = {**params, "head_num": config_info.linear_num_heads}
        return {
            tp_plan_nested_module_path(prefix, projection): (COLWISE_LINEAR, dict(local_params))
            # q_proj is already covered by the generic MLA plan. Repeating it
            # through this broad nested glob would wrap and shard it twice.
            for projection in ("k_proj", "v_proj", "b_proj", "forget_gate.f_b_proj", "g_b_proj")
        }

    def _run_sparse_attention_indexer(
        self, hidden_states, qa_normed, position_embeddings, attention_meta=None, **kwargs
    ):
        if attention_meta is None:
            raise ValueError("GLM5Next k-pool requires per-request attention metadata")
        indexer = self.indexer
        q = indexer.wq_b(qa_normed).view(*hidden_states.shape[:2], indexer.num_heads, indexer.head_dim)
        k = torch.ops.tensor_cast.layer_norm(
            indexer.wk(hidden_states), indexer.k_norm.weight, indexer.k_norm.bias, indexer.k_norm.eps
        )
        gates = F.linear(hidden_states, indexer.index_kpool_compress_gate)
        weights = indexer.weights_proj(hidden_states)
        cache = kwargs["indexer_cache_by_layers"][self.layer_idx]
        return torch.ops.tensor_cast.glm5_next_kpool_indexer(
            q,
            k,
            gates,
            weights,
            indexer.index_kpool_compress_ape,
            cache,
            attention_meta.query_lens,
            attention_meta.seq_lens,
            indexer.index_kpool,
            indexer.topk_limit,
            indexer.index_kpool_always_select_tail,
        )

    def _get_backend_kwargs(self, pre_attn_out):
        return {"topk_limit": pre_attn_out.shape[-1], "topk_indices": pre_attn_out}


def glm5_next_kda_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Keep real projection/TP calls around a KDA-specific semantic core."""
    extra = getattr(self, "_extra_forward_kwargs", {})
    metadata = kwargs.get("attention_meta", extra.get("attention_meta"))
    caches = kwargs.get("kv_cache_by_layers", extra.get("kv_cache_by_layers"))
    if metadata is None or caches is None:
        raise ValueError("GLM5Next KDA requires attention_meta and per-layer recurrent state")
    if attention_mask is not None:
        hidden_states = torch.ops.tensor_cast.linear_attn_apply_padding_mask(hidden_states, attention_mask)
    heads = exact_division(self.num_heads, getattr(self, "tensor_cast_tp_size", 1))
    dim = self.head_dim
    rank = getattr(self, "tensor_cast_tp_rank", 0)
    shape = (*hidden_states.shape[:2], heads, dim)
    q, k, v = (projection(hidden_states).view(shape) for projection in (self.q_proj, self.k_proj, self.v_proj))
    forget = self.forget_gate.f_b_proj(self.forget_gate.f_a_proj(hidden_states)).view(shape)
    beta = self.b_proj(hidden_states)
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
    # Conv and scalar parameters are not Linear modules: explicitly select each
    # local Q/K/V head partition without applying a global-group Conv1d.
    start, end = rank * heads * dim, (rank + 1) * heads * dim
    conv = self.conv1d.weight.view(3, self.num_heads * dim, self.conv_kernel_size)[:, start:end]
    output = torch.ops.tensor_cast.glm5_next_kda(
        q,
        k,
        v,
        forget,
        beta,
        gate,
        conv.contiguous(),
        self.forget_gate.A_log[rank * heads : (rank + 1) * heads],
        self.forget_gate.dt_bias[start:end],
        self.o_norm.weight,
        caches[self.layer_idx],
        metadata.query_lens,
        metadata.seq_lens,
        metadata.is_decode_values,
        self.forget_gate.safe_gate_lower_bound,
        self.o_norm.variance_epsilon,
    )
    return self.o_proj(output.flatten(-2))


def glm5_next_mhc_forward(self, hidden_streams):
    hc_mult = self.hc_mult
    flat = hidden_streams.float().flatten(start_dim=2)
    inv_rms = torch.ops.tensor_cast.hc_pre_inv_rms(hidden_streams, hc_mult)
    mixes = torch.matmul(flat, self.fn.float().transpose(0, 1)) * inv_rms
    collapsed, post, comb = torch.ops.tensor_cast.hc_pre_sinkhorn(
        mixes,
        hidden_streams,
        self.scale,
        self.base,
        hc_mult,
        self.hc_sinkhorn_iters,
        self.hc_eps,
    )
    return post, comb, collapsed


def glm5_next_mhc_head_forward(self, hidden_streams):
    return torch.ops.tensor_cast.mhc_head(hidden_streams)


def glm5_next_decoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    prev_topk_indices=None,
    **kwargs,
):
    """Preserve the upstream collapse/sublayer/expand ordering at both sites."""
    residual = hidden_states
    post, comb, collapsed = self.attn_hc(hidden_states)
    collapsed = self.input_layernorm(collapsed)
    topk_indices = None
    if self.block_type == "linear_attention":
        output = self.self_attn(collapsed, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
    else:
        if position_embeddings is None:
            position_embeddings = glm5_next_nope_position_embeddings(collapsed)
        output, _, topk_indices = self.self_attn(
            hidden_states=collapsed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
    hidden_states = torch.ops.tensor_cast.hc_post(output, residual, post, comb, self.attn_hc.hc_mult)
    residual = hidden_states
    post, comb, collapsed = self.ffn_hc(hidden_states)
    output = self.mlp(self.post_attention_layernorm(collapsed))
    return torch.ops.tensor_cast.hc_post(output, residual, post, comb, self.ffn_hc.hc_mult), topk_indices


def glm5_next_nope_position_embeddings(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-width RoPE tensors for a GLM5Next NoPE MLA invocation."""
    sequence_length = hidden_states.shape[1]
    empty = hidden_states.new_empty((sequence_length, 0))
    return empty, empty
