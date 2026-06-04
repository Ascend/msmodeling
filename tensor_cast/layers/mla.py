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
    # Set to True in subclasses that accept parallel_group_manager in __init__
    supports_parallel_group_manager: bool = False

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
        self.tp_group = tp_group
        self._num_heads_per_rank = exact_division(self.num_heads, tp_group.world_size)
        self._setup_kv_b_decomposition(tp_group)

    def _setup_kv_b_decomposition(self, tp_group: ParallelGroup) -> None:
        """
        Hook: shard ``kv_b_proj`` across TP ranks and split it into ``W_UK``/``W_UV``.

        Default behaviour matches the legacy MLA path used by V3 / V3.2: it requires
        ``self.kv_b_proj`` to be present on the inner module. Subclasses whose
        attention does not have a ``kv_b_proj`` (e.g. DeepSeek V4 with shared KV) can
        override this hook with a no-op.
        """
        sharded_weight = get_partial_sharded(
            self.kv_b_proj.weight.data,
            tp_group.world_size,
            tp_group.rank_in_group,
            unit_num=self.num_heads,
        )
        self.kv_b_proj_weight_t = sharded_weight.transpose(0, 1)
        kv_b_proj_view = self.kv_b_proj_weight_t.view(
            self.kv_lora_rank,
            self._num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_view.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UV = W_UV.transpose(0, 1)  # (num_heads_per_rank, kv_lora_rank, v_head_dim)
        self.W_UK_T = W_UK.permute(1, 2, 0)  # (num_heads_per_rank, qk_nope_head_dim, kv_lora_rank)

    @classmethod
    def requires_indexer_cache(cls) -> bool:
        """
        Hook for subclasses that require the auxiliary sparse-attention indexer
        cache allocated by input_generator.

        The default MLA path does not use an indexer cache.
        """
        return False

    @classmethod
    def build_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        """
        Hook for subclasses to inject extra TP sharding rules into the generic MLA
        q/kv shard plan without adding model-specific branches to transformations.py.

        The default MLA path has no extra modules beyond q/kv_b projections.
        Subclasses can override this to register additional learned linears that
        belong to the attention block (e.g. V4 indexer projections).
        """
        return {}

    @classmethod
    def build_o_proj_tp_plan_extras(cls, prefix: str, params: dict, config_info) -> dict[str, tuple[str, dict]]:
        """
        Hook for subclasses to inject extra O-projection-related TP sharding rules
        into the generic MLA shard plan.
        """
        return {}

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
            raise AttributeError(f"Module {module.__class__.__name__} does not expose a weight tensor. ")
        return weight.data, None, None

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        qa_normed: Optional[torch.Tensor] = None,
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
        self.q_a_proj_weight, self.q_a_proj_scale, self.q_a_proj_offset = self.extract_qparams(self.q_a_proj)
        self.q_b_proj_weight, self.q_b_proj_scale, self.q_b_proj_offset = self.extract_qparams(self.q_b_proj)
        self.kv_a_proj_weight, self.kv_a_proj_scale, self.kv_a_proj_offset = self.extract_qparams(
            self.kv_a_proj_with_mqa
        )
        self.q_a_layernorm_weight = self.q_a_layernorm.weight.data
        self.kv_a_layernorm_weight = self.kv_a_layernorm.weight.data
        linear_quant_enabled = (
            getattr(self, "q_a_proj_scale", None) is not None
            and getattr(self, "q_b_proj_scale", None) is not None
            and getattr(self, "kv_a_proj_scale", None) is not None
        )
        if linear_quant_enabled:
            q_states, kv_c_normed, k_rot, qa_normed = torch.ops.tensor_cast.mlapo_quant(
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
            q_states, kv_c_normed, k_rot, qa_normed = torch.ops.tensor_cast.mlapo(
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

        if self.q_lora_rank is not None:
            qa_normed = qa_normed.view(batch_size, seq_length, -1)
        else:
            qa_normed = None
        pre_attn_out = self._pre_attention_forward(
            hidden_states=hidden_states,
            qa_normed=qa_normed,
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
                torch.ops.tensor_cast.concat_and_cache_mla(kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping)
        else:
            if attention_meta is not None:
                torch.ops.tensor_cast.concat_and_cache_mla(kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping)

        extra_backend_kwargs = {
            "topk_limit": None,
            "topk_indices": None,
            **self._get_backend_kwargs(pre_attn_out),
        }
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
            block_table=attention_meta.block_table_tensor if attention_meta is not None else None,
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
        assert self.quant_config is not None, "quant_config must be set before quantization"
        self._quantize_kv_b_decomposition()

    def _quantize_kv_b_decomposition(self) -> None:
        """
        Hook: quantize the ``kv_b_proj`` decomposition tensors set up by
        :meth:`_setup_kv_b_decomposition`. Subclasses that override the setup hook
        should typically override this one as well (see DeepSeek V4 wrapper).
        """
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


def _resolve_sparse_topk_limit(
    indexer: torch.nn.Module,
    config: Optional[torch.nn.Module] = None,
    topk_limit: Optional[int] = None,
) -> int:
    if topk_limit is not None:
        return topk_limit

    inner_topk_limit = getattr(indexer, "topk_limit", None)
    if inner_topk_limit is not None:
        return inner_topk_limit

    for candidate in (config, getattr(indexer, "config", None)):
        candidate_topk_limit = getattr(candidate, "topk_limit", None)
        if candidate_topk_limit is not None:
            return candidate_topk_limit

        candidate_index_topk = getattr(candidate, "index_topk", None)
        if candidate_index_topk is not None:
            return candidate_index_topk

    raise AttributeError("topk_limit")


class DeepseekSparseAttention(MultiheadLatentAttentionTensorCast):
    @classmethod
    def requires_indexer_cache(cls) -> bool:
        return True

    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
    ):
        super().__init__(mla_config, mla_module, tp_group, decode_only)
        self.indexer = DeepseekSparseAttentionIndexer(
            self._inner.indexer,
            topk_limit=_resolve_sparse_topk_limit(
                self._inner.indexer,
                config=getattr(self._inner, "config", None),
            ),
        )

    def _get_backend_kwargs(self, pre_attn_out) -> dict:
        """
        Sparse Attention Args:
        - topk_limit: int, number of selected sparse tokens
        - topk_indices: Tensor, precomputed sparse position indices
        """
        return {
            "topk_limit": self.indexer.topk_limit,
            "topk_indices": pre_attn_out,
        }

    def _pre_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        qa_normed: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self._run_sparse_attention_indexer(
            hidden_states, qa_normed, position_embeddings, attention_meta, **kwargs
        )

    def _run_sparse_attention_indexer(
        self,
        hidden_states: torch.Tensor,
        qa_normed: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ):
        if qa_normed is None:
            return None

        indexer_cache_by_layers = kwargs.pop("indexer_cache_by_layers", None)
        indexer_cache = indexer_cache_by_layers[self.layer_idx] if indexer_cache_by_layers else None
        return self.indexer(hidden_states, qa_normed, position_embeddings, indexer_cache, attention_meta)


class DeepseekSparseAttentionIndexer(ModelWrapperBase):
    def __init__(self, indexer, topk_limit: Optional[int] = None):
        super().__init__(indexer)
        self._topk_limit = _resolve_sparse_topk_limit(indexer, topk_limit=topk_limit)

    @property
    def num_heads(self) -> int:
        if hasattr(self._inner, "num_heads"):
            return self._inner.num_heads
        return self._inner.n_heads

    @property
    def head_dim(self) -> int:
        if hasattr(self._inner, "head_dim"):
            return self._inner.head_dim
        return self._inner.index_head_dim

    @property
    def topk_limit(self) -> int:
        return self._topk_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        qa_normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        indexer_cache: torch.Tensor,
        attention_meta: Optional[AttentionMetadataBase] = None,
    ):
        cos, sin = position_embeddings
        wq_b_weight, _, _ = MultiheadLatentAttentionTensorCast.extract_qparams(self.wq_b)
        wk_weight, _, _ = MultiheadLatentAttentionTensorCast.extract_qparams(self.wk)
        weights_proj_weight, _, _ = MultiheadLatentAttentionTensorCast.extract_qparams(self.weights_proj)
        # The performance model infers fp8-vs-bf16 behavior from the cache dtype.
        # The semantic op itself stays shape-only and does not encode the cost model.
        return torch.ops.tensor_cast.dsa_indexer(
            hidden_states,
            qa_normed,
            cos,
            sin,
            indexer_cache,
            attention_meta.slot_mapping if attention_meta else None,
            attention_meta.block_table_tensor if attention_meta else None,
            attention_meta.seq_lens if attention_meta is not None else None,
            wq_b_weight,
            wk_weight,
            weights_proj_weight,
            self.k_norm.weight,
            self.num_heads,
            self.head_dim,
            self.qk_rope_head_dim,
            self.topk_limit,
        )
