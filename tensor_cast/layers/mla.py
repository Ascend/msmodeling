from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional

import torch

from .. import ops  # noqa: F401
from ..model_config import MlaConfig, MultiheadLatentAttentionQuantConfig
from ..parallel_group import ParallelGroup
from ..utils import exact_division
from .attention import AttentionMetadataBase
from .quant_linear import TensorCastQuantLinear
from .utils import get_partial_sharded, ModelWrapperBase


class MultiheadLatentAttentionBase(torch.nn.Module, ABC):
    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        decode_only: bool = False,
    ) -> None:
        super().__init__()
        self.mla_config = mla_config
        self._inner = mla_module
        self.decode_only = decode_only
        self.quant_config: Optional[MultiheadLatentAttentionQuantConfig] = None

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        pass

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.mla_config.field_names, name):
            return getattr(self._inner, getattr(self.mla_config.field_names, name))
        return super().__getattr__(name)

    def quantize_params(self):
        """
        Called during the initialization phase after the inner module is quantized.
        This allows quantization for extra parameters in this module.
        """


# rotary embedding functions copied from DeepSeek-v3 model in Transformers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])

    s, n, d = q.shape
    q = q.view(s, n, d // 2, 2).transpose(-1, -2).reshape(s, n, d)

    s, d = k.shape
    k = k.view(s, d // 2, 2).transpose(-1, -2).reshape(s, d)

    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiheadLatentAttentionTensorCast(MultiheadLatentAttentionBase):
    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
    ):
        super().__init__(mla_config, mla_module, decode_only)
        sharded_weight = get_partial_sharded(
            self.kv_b_proj.weight.data,
            tp_group.world_size,
            tp_group.rank_in_group,
            unit_num=self.num_heads,
        )
        self.kv_b_proj_weight_t = sharded_weight.transpose(0, 1)
        kv_b_proj_view = self.kv_b_proj_weight_t.view(
            self.kv_lora_rank,
            exact_division(self.num_heads, tp_group.world_size),
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_view.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        self.W_UV = W_UV.transpose(
            0, 1
        )  # (num_heads_per_rank, kv_lora_rank, v_head_dim)
        self.W_UK_T = W_UK.permute(
            1, 2, 0
        )  # (num_heads_per_rank, qk_nope_head_dim, kv_lora_rank)
        self._num_heads_per_rank = self.W_UV.size(0)
        self.tp_group = tp_group

    @staticmethod
    def extract_qparams(
        module: torch.nn.Module,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        target = module
        if hasattr(module, "_inner"):
            target = module._inner
        if isinstance(target, TensorCastQuantLinear):
            return target.qweight, target.weight_scale, target.weight_offset
        weight = getattr(target, "weight", None)
        if weight is None:
            raise AttributeError(
                f"Module {module.__class__.__name__} does not expose a weight tensor. "
            )
        return weight.data, None, None

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ):
        """
        Pre-attention processing hook that runs before core attention computation.
        This hook is INTENDED FOR IN-PLACE CACHE PREPARATION (e.g., writing precomputed key
        features or index data into pre-allocated cache tensors such as indexer_cache).
        """
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache_unused: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        kv_cache_by_layers = kwargs.pop("kv_cache_by_layers", None)
        kv_cache = kv_cache_by_layers[self.layer_idx] if kv_cache_by_layers else None
        batch_size, seq_length = hidden_states.shape[:-1]
        num_tokens = batch_size * seq_length
        hidden_states_view = hidden_states.view(num_tokens, -1)
        cos, sin = position_embeddings
        self.q_a_proj_weight, self.q_a_proj_scale, self.q_a_proj_offset = (
            self.extract_qparams(self.q_a_proj)
        )
        self.q_b_proj_weight, self.q_b_proj_scale, self.q_b_proj_offset = (
            self.extract_qparams(self.q_b_proj)
        )
        self.kv_a_proj_weight, self.kv_a_proj_scale, self.kv_a_proj_offset = (
            self.extract_qparams(self.kv_a_proj_with_mqa)
        )
        self.q_a_layernorm_weight = self.q_a_layernorm.weight.data
        self.kv_a_layernorm_weight = self.kv_a_layernorm.weight.data
        self.q_b_proj_weight = get_partial_sharded(
            self.q_b_proj_weight,
            self.tp_group.world_size,
            self.tp_group.rank_in_group,
            unit_num=self.num_heads,
        )
        linear_quant_enabled = (
            getattr(self, "q_a_proj_scale", None) is not None
            and getattr(self, "q_b_proj_scale", None) is not None
            and getattr(self, "kv_a_proj_scale", None) is not None
        )
        if linear_quant_enabled:
            q_states, kv_c_normed, k_rot = torch.ops.tensor_cast.mlapo_quant(
                hidden_states_view,
                cos,
                sin,
                self.q_a_proj_weight,
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
                self.q_a_proj_scale,
                self.q_a_proj_offset,
                self.q_b_proj_scale,
                self.q_b_proj_offset,
                self.kv_a_proj_scale,
                self.kv_a_proj_offset,
            )
        else:
            q_states, kv_c_normed, k_rot = torch.ops.tensor_cast.mlapo(
                hidden_states_view,
                cos,
                sin,
                self.q_a_proj_weight,
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
            )

        pre_attn_out = self._pre_attention_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_meta=attention_meta,
            **kwargs,
        )

        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None
        if self.quant_config is not None:
            quant_config = self.quant_config
            out_dtype = self.quant_config.get_quant_dtype()
            q_states = torch.ops.tensor_cast.quantize(
                q_states,
                quant_config.query_scale,
                quant_config.query_offset,
                out_dtype,
            )
            kv_c_normed = torch.ops.tensor_cast.quantize(
                kv_c_normed,
                quant_config.kv_scale,
                quant_config.kv_offset,
                out_dtype,
            )
            k_rot = torch.ops.tensor_cast.quantize(
                k_rot,
                quant_config.kv_scale,
                quant_config.kv_offset,
                out_dtype,
            )

            if attention_meta is not None:
                torch.ops.tensor_cast.concat_and_cache_mla(
                    kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping
                )
        else:
            if attention_meta is not None:
                torch.ops.tensor_cast.concat_and_cache_mla(
                    kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping
                )

        extra_backend_kwargs = self._get_backend_kwargs(pre_attn_out)
        if self.quant_config is not None:
            attention_backend = partial(
                torch.ops.tensor_cast.multihead_latent_attention_quant,
                W_UK_T=self.W_UK_T,
                W_UV=self.W_UV,
                kv_b_proj=self.kv_b_proj_weight_t,
                v_head_dim=self.v_head_dim,
                query_scale=self.quant_config.query_scale,
                query_offset=self.quant_config.query_offset,
                kv_scale=self.quant_config.kv_scale,
                kv_offset=self.quant_config.kv_offset,
                kv_projected_scale=self.quant_config.kv_projected_scale,
                kv_projected_offset=self.quant_config.kv_projected_offset,
                qk_scale=self.quant_config.qk_scale,
                qk_offset=self.quant_config.qk_offset,
                v_scale=self.quant_config.v_scale,
                v_offset=self.quant_config.v_offset,
                attention_prob_scale=self.quant_config.attention_prob_scale,
                attention_prob_offset=self.quant_config.attention_prob_offset,
                kv_b_proj_scale=self.kv_b_proj_scale,
                kv_b_proj_offset=self.kv_b_proj_offset,
                out_scale=self.quant_config.out_scale,
                out_offset=self.quant_config.out_offset,
                out_dtype=hidden_states.dtype,
                **extra_backend_kwargs,
            )
        else:
            attention_backend = partial(
                torch.ops.tensor_cast.multihead_latent_attention,
                W_UK_T=self.W_UK_T,
                W_UV=self.W_UV,
                kv_b_proj=self.kv_b_proj_weight_t,
                v_head_dim=self.v_head_dim,
                **extra_backend_kwargs,
            )

        attn_output = attention_backend(
            q=q_states,
            kv_cache=kv_cache,
            block_table=attention_meta.block_table_tensor
            if attention_meta is not None
            else None,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_lens=query_lens,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _get_backend_kwargs(self, pre_attn_out) -> dict:
        """
        Hook for subclasses to inject additional arguments into the attention backend.
        Default implementation returns an empty dict (standard dense attention).
        Subclasses can override this to pass specific parameters (e.g., top-k, window size).
        """
        return {}

    def quantize_params(self):
        assert self.quant_config is not None, (
            "quant_config must be set before quantization"
        )
        out_dtype = self.quant_config.get_quant_dtype()
        kv_b_proj = self.kv_b_proj
        if not isinstance(kv_b_proj, TensorCastQuantLinear):
            raise ValueError("MLA quantization requires kv_b_proj to be quantized")
        self.kv_b_proj_scale = kv_b_proj.weight_scale
        self.kv_b_proj_offset = kv_b_proj.weight_offset
        self.kv_b_proj_weight_t = torch.ops.tensor_cast.quantize(
            self.kv_b_proj_weight_t,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UK_T = torch.ops.tensor_cast.quantize(
            self.W_UK_T,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UV = torch.ops.tensor_cast.quantize(
            self.W_UV,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )


class DeepseekSparseAttention(MultiheadLatentAttentionTensorCast):
    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
    ):
        super().__init__(mla_config, mla_module, tp_group, decode_only)
        self.indexer = DeepseekSparseAttentionIndexer(self._inner.indexer)

    def _get_backend_kwargs(self, pre_attn_out) -> dict:
        """
        Sparse Attention Args:
        - index_topk: int, number of selected sparse tokens
        - cached_topk_indices: Tensor, precomputed sparse position indices
        """
        return {
            "index_topk": self.indexer.index_topk,
            "cached_topk_indices": pre_attn_out,
        }

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ):
        # Cache the top-k indices to prevent the DSA operator from being optimized away by torch.compile
        return self._run_sparse_attention_indexer(
            hidden_states, position_embeddings, attention_meta, **kwargs
        )

    def _run_sparse_attention_indexer(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ):
        if getattr(self, "q_lora_rank", None) is None:
            return None
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_states_view = hidden_states.view(batch_size * seq_length, -1)
        # Use module directly: seamlessly supports both quant (TensorCastQuantLinear) and unquant paths
        temp_q_a_out = self.q_a_proj(hidden_states_view)
        qa_normed = self.q_a_layernorm(temp_q_a_out)

        indexer_cache_by_layers = kwargs.pop("indexer_cache_by_layers", None)
        indexer_cache = (
            indexer_cache_by_layers[self.layer_idx] if indexer_cache_by_layers else None
        )
        return self.indexer(
            hidden_states, qa_normed, position_embeddings, indexer_cache, attention_meta
        )


class DeepseekSparseAttentionIndexer(ModelWrapperBase):
    def __init__(self, indexer):
        super().__init__(indexer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        qa_normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        indexer_cache: torch.Tensor,
        attention_meta: Optional[AttentionMetadataBase] = None,
    ):
        bsz, seq_len, _ = hidden_states.size()
        q = self.wq_b(qa_normed)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        q_rot, q_pass = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_rot, k_pass = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        cos, sin = position_embeddings

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k_rot = k_rot.squeeze(2)
        k = torch.cat([k_rot, k_pass], dim=-1)

        torch.ops.tensor_cast.dsa_index_cache(
            k,
            indexer_cache,
            slot_mapping=attention_meta.slot_mapping if attention_meta else None,
            block_tables=attention_meta.block_table_tensor if attention_meta else None,
        )
        weights = self.weights_proj(hidden_states) * self.num_heads**-0.5

        # the second indexer_cache is a place holder
        index_score = torch.ops.tensor_cast.dsa_index(
            q.contiguous(), weights, indexer_cache, indexer_cache
        )
        topk_indices = index_score.topk(min(self.index_topk, seq_len), dim=-1)[1]
        return topk_indices
