# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
"""DSpark draft modeling for TensorCast (G5: isolated from dflash.py).

Reuses DFlash backbone via import/composition; adds Markov/Confidence sequential
sampling. See docs/RFC/rfc_dspark_tensorcast_modeling_zh.md.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import Qwen3Config

from ..model_config import DflashConfig, DsparkConfig
from .dflash import (
    DflashDraftModel,
    DflashWrapper,
    apply_cli_overrides_to_source_and_dcfg,
    build_draft_hf_config,
    resolve_target_embed_and_lm_head,
)


class MarkovHead(torch.nn.Module):
    """Low-rank Markov logit-bias head (vanilla / gated / rnn).

    Aligns with upstream ``speculators.models.dspark.MarkovHead``:

    - ``vanilla``: ``bias = W2(W1(prev))``
    - ``gated``: ``gate = σ(Linear([h_i, W1(prev)]))``; ``bias = W2(gate * W1(prev))``
    - ``rnn``: block-local state; ``z=[s, W1(prev), h_i]`` → gated update → ``bias = W2(tanh(out))``

    Attribute names ``markov_embed`` / ``markov_bias`` map checkpoint ``markov_w1`` /
    ``markov_w2``. ``markov_bias`` must use the **same** ColumnParallel TP policy as
    shared ``lm_head`` (same ``tp_group`` / ``gather_output``): either both full V
    after gather, or both local ``V/TP``. Divergent layouts break ``logits + bias``.
    ``gate_proj`` / ``joint_proj`` stay unsharded (rank-sized, not vocab-sized).
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        *,
        hidden_size: int = 0,
        head_type: str = "vanilla",
    ):
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        if head_type in ("gated", "rnn") and hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0 for markov_head_type={head_type!r}")

        self.head_type = head_type
        self.markov_rank = markov_rank
        self.hidden_size = int(hidden_size)
        self.markov_embed = torch.nn.Embedding(vocab_size, markov_rank)
        # Built at full vocab; shard_model replaces this with ColumnParallelLinear.
        self.markov_bias = torch.nn.Linear(markov_rank, draft_vocab_size, bias=False)
        self.gate_proj: Optional[torch.nn.Linear] = None
        self.joint_proj: Optional[torch.nn.Linear] = None
        if head_type == "gated":
            self.gate_proj = torch.nn.Linear(hidden_size + markov_rank, markov_rank, bias=True)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] over [state; prev_emb; hidden].
            self.joint_proj = torch.nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank, bias=True)

    def forward(
        self,
        prev_token_ids: torch.Tensor,
        hidden_i: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """One serial step.

        Returns ``(markov_embed, bias, new_state)``. ``markov_embed`` is always
        ``W1(prev)`` (Confidence may concatenate it). ``new_state`` is only set
        for ``rnn``; otherwise ``None``.
        """
        markov_embed = self.markov_embed(prev_token_ids)

        if self.head_type == "vanilla":
            return markov_embed, self.markov_bias(markov_embed), None

        if hidden_i is None:
            raise ValueError(f"hidden_i is required for markov_head_type={self.head_type!r}")

        if self.head_type == "gated":
            assert self.gate_proj is not None
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_i, markov_embed], dim=-1)))
            return markov_embed, self.markov_bias(gate * markov_embed), None

        # rnn
        assert self.joint_proj is not None
        if state is None:
            state = markov_embed.new_zeros(markov_embed.shape[0], self.markov_rank)
        z = torch.cat([state, markov_embed, hidden_i], dim=-1)
        gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
        bias = self.markov_bias(torch.tanh(out_raw))
        return markov_embed, bias, new_state


class ConfidenceHead(torch.nn.Module):
    """Confidence projection; counted for compute only (does not drive acceptance)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class DsparkDraftModel(DflashDraftModel):
    """DFlash backbone plus optional Markov / Confidence heads."""

    def __init__(
        self,
        draft_hf_config: Qwen3Config,
        dflash_config: DflashConfig,
        dspark_config: DsparkConfig,
        *,
        layer_idx_offset: int = 0,
    ):
        super().__init__(draft_hf_config, dflash_config, layer_idx_offset=layer_idx_offset)
        self.dspark_config = dspark_config
        self.markov_head: Optional[MarkovHead] = None
        self.confidence_head: Optional[ConfidenceHead] = None

        markov_rank = int(dspark_config.markov_rank or 0)
        hidden_size = int(draft_hf_config.hidden_size)
        if markov_rank > 0:
            vocab_size = int(draft_hf_config.vocab_size)
            self.markov_head = MarkovHead(
                vocab_size,
                vocab_size,
                markov_rank,
                hidden_size=hidden_size,
                head_type=str(dspark_config.markov_head_type or "vanilla"),
            )
        if dspark_config.enable_confidence_head:
            conf_dim = hidden_size
            if dspark_config.confidence_head_with_markov and markov_rank > 0:
                conf_dim = conf_dim + markov_rank
            self.confidence_head = ConfidenceHead(conf_dim)


class DsparkWrapper(DflashWrapper):
    """DFlash prefill/decode path with DSpark sequential proposal sampling."""

    def __init__(
        self,
        dspark_config: DsparkConfig,
        dflash_config: DflashConfig,
        hf_config,
        model: torch.nn.Module,
        draft: DsparkDraftModel,
        draft_hf_config: Qwen3Config,
        target_layers: Optional[torch.nn.ModuleList] = None,
    ):
        super().__init__(
            dflash_config,
            hf_config,
            model,
            draft,
            draft_hf_config,
            target_layers=target_layers,
        )
        self.dspark_config = dspark_config

    def _exclude_anchor_from_lm_head(self) -> bool:
        """DSpark samples all ``block_size`` slots (``sample_from_anchor=True``)."""
        return False

    def _propose_draft_tokens(
        self,
        draft_hidden: torch.Tensor,
        draft_logits_b: torch.Tensor,
        batch_size: int,
        block: int,
        next_tokens: torch.Tensor,
    ):
        # Parent passes lm_seq-shaped logits; with full-block lm_head this is [B, block, V].
        draft_logits_b = draft_logits_b.reshape(batch_size, block, -1)
        draft_tokens, confidence = self._sample_sequential(draft_hidden, draft_logits_b, batch_size, block, next_tokens)
        # Formal side outputs keep residency under ``torch.compile``:
        # - next_tokens: verify ArgMax path
        # - confidence: stacked ConfidenceHead outputs (prevents Confidence DCE)
        if confidence is not None:
            return draft_tokens, next_tokens, confidence
        return draft_tokens, next_tokens

    def _sample_sequential(
        self,
        draft_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        batch_size: int,
        block: int,
        next_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Block-internal serial Markov / Confidence / ArgMax for compute modeling.

        Returns ``(draft_tokens, confidence)`` where ``confidence`` is ``[B, N]``
        when ConfidenceHead is enabled, else ``None``.
        """
        hidden_b = draft_hidden.reshape(batch_size, block, -1)

        prev = next_tokens[:, -1]
        draft_steps: list[torch.Tensor] = []
        conf_steps: list[torch.Tensor] = []
        markov_head = getattr(self.draft, "markov_head", None)
        confidence_head = getattr(self.draft, "confidence_head", None)
        with_markov = bool(self.dspark_config.confidence_head_with_markov)
        markov_state: Optional[torch.Tensor] = None

        for i in range(block):
            logits_i = base_logits[:, i]
            head_i = hidden_b[:, i]
            markov_embed = None
            if markov_head is not None:
                # gated/rnn read head_i; rnn also carries block-local state.
                markov_embed, bias, markov_state = markov_head(prev, head_i, markov_state)
                bias = self._align_markov_bias_vocab(logits_i, bias, self.draft.lm_head)
                logits_i = logits_i + bias
            if confidence_head is not None:
                if markov_embed is not None and with_markov:
                    conf_features = torch.cat([head_i, markov_embed], dim=-1)
                else:
                    conf_features = head_i
                # Keep conf_i live; stack below and return as a formal decode side output.
                conf_i = confidence_head(conf_features)
                if conf_i.dim() > 1 and conf_i.size(-1) == 1:
                    conf_i = conf_i.squeeze(-1)
                conf_steps.append(conf_i)
            draft_i = torch.argmax(logits_i, dim=-1)
            draft_steps.append(draft_i)
            prev = draft_i

        draft_tokens = torch.stack(draft_steps, dim=1)
        confidence = torch.stack(conf_steps, dim=1) if conf_steps else None
        return draft_tokens, confidence

    @staticmethod
    def _align_markov_bias_vocab(
        logits_i: torch.Tensor,
        bias: torch.Tensor,
        lm_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Match Markov bias last-dim to draft ``lm_head`` logits (unified TP layout)."""
        vocab_logits = logits_i.shape[-1]
        vocab_bias = bias.shape[-1]
        if vocab_logits == vocab_bias:
            return bias
        if vocab_bias > vocab_logits and vocab_bias % vocab_logits == 0:
            # Full-V bias vs local V/TP logits: take this rank's shard.
            rank = int(getattr(lm_head, "tp_rank", 0) or 0)
            return bias.narrow(-1, rank * vocab_logits, vocab_logits)
        if vocab_logits > vocab_bias and vocab_logits % vocab_bias == 0:
            tp_group = getattr(lm_head, "tp_group", None)
            if tp_group is not None:
                gathered = tp_group.all_gather(bias, dim=-1)
                return gathered[..., :vocab_logits]
        raise ValueError(
            f"DSpark Markov bias vocab {vocab_bias} incompatible with draft logits "
            f"vocab {vocab_logits}; markov_bias must share lm_head TP layout"
        )


def apply_cli_overrides_to_dspark_config(
    scfg: DsparkConfig,
    *,
    cli_block_size: Optional[int] = None,
    cli_num_draft_layers: Optional[int] = None,
    prefer_existing: bool = False,
) -> dict:
    """Resolve block/layers/aux onto ``scfg`` via the shared Dflash draft config file."""
    # Clamp before to_dflash_config(): DflashConfig rejects negatives in __post_init__.
    scfg.dspark_acceptance_length = min(
        max(float(scfg.dspark_acceptance_length), 0.0),
        float(scfg.dspark_block_size),
    )
    dcfg = scfg.to_dflash_config()
    source = apply_cli_overrides_to_source_and_dcfg(
        dcfg,
        cli_block_size=cli_block_size,
        cli_num_draft_layers=cli_num_draft_layers,
        prefer_existing=prefer_existing,
    )
    scfg.dspark_block_size = int(dcfg.dflash_block_size)
    scfg.num_draft_layers = int(dcfg.num_draft_layers)
    scfg.aux_hidden_state_layer_ids = list(dcfg.aux_hidden_state_layer_ids) if dcfg.aux_hidden_state_layer_ids else None
    scfg.layer_types = list(dcfg.layer_types) if dcfg.layer_types else None
    scfg.sliding_window = dcfg.sliding_window
    # Re-clamp DSpark acceptance to [0, block_size] after block may change.
    max_accept = float(scfg.dspark_block_size)
    scfg.dspark_acceptance_length = min(
        max(float(scfg.dspark_acceptance_length), 0.0),
        max_accept,
    )
    return source


def build_dspark_draft_and_wrapper(
    model,
    scfg: DsparkConfig,
    hf_config,
    *,
    num_target_hidden_layers: int,
    target_hidden_size: int,
    target_vocab_size: int,
    target_max_position_embeddings: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    target_layers: Optional[torch.nn.ModuleList] = None,
) -> DsparkWrapper:
    """Build DSpark draft (DFlash backbone + heads) and wrap the target model."""
    apply_cli_overrides_to_dspark_config(scfg, prefer_existing=True)
    dcfg = scfg.to_dflash_config()
    draft_hf_config = build_draft_hf_config(
        dcfg,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        target_max_position_embeddings=target_max_position_embeddings,
    )
    max_layer = max(dcfg.aux_hidden_state_layer_ids)
    if max_layer >= num_target_hidden_layers:
        raise ValueError(
            f"target_layer_ids max={max_layer} out of range for target num_hidden_layers={num_target_hidden_layers}"
        )

    layer_idx_offset = int(num_target_hidden_layers)
    draft = DsparkDraftModel(draft_hf_config, dcfg, scfg, layer_idx_offset=layer_idx_offset)
    embed, lm_head = resolve_target_embed_and_lm_head(model)
    draft.set_shared(embed, lm_head)
    if dtype is not None:
        for name, module in draft.named_children():
            if name in ("embed_tokens", "lm_head"):
                continue
            module.to(dtype=dtype)
    return DsparkWrapper(
        scfg,
        dcfg,
        hf_config,
        model._inner,
        draft,
        draft_hf_config,
        target_layers=target_layers,
    )
