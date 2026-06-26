import functools
from typing import Callable, Optional

import torch

from .. import ops  # noqa: F401
from ..model_config import MtpConfig
from .sampler import Sampler, SamplingMetadata, select_lm_head_hidden_states
from .utils import ModelWrapperBase


class MultiTokenPredictorLayer(torch.nn.Module):
    def __init__(self, hf_config, mtp_block: torch.nn.Module):
        super().__init__()
        self.emb_norm = torch.nn.RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.hidden_norm = torch.nn.RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.linear_proj = torch.nn.Linear(hf_config.hidden_size * 2, hf_config.hidden_size, bias=False)
        self.mtp_block = mtp_block

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.emb_norm(inputs_embeds)
        previous_hidden_states = self.hidden_norm(previous_hidden_states)

        hidden_states = self.linear_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states = self.mtp_block(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        return hidden_states


def _resolve_mtp_layer_cls(hf_config, mtp_block):
    """Select the appropriate MTP layer class based on HC (Hyper-Connection) config.

    V4 models have hc_mult > 1 and require HyperConnectedMultiTokenPredictorLayer
    to expand [B,S,D] -> [B,S,Hc,D] before the MTP block and reduce after.
    V3/V3.2 models use the base MultiTokenPredictorLayer.
    """
    hc_mult = int(getattr(mtp_block, "hc_mult", None) or getattr(hf_config, "hc_mult", 1) or 1)
    if hc_mult > 1:
        # Import here to avoid circular dependency at module level.
        from .deepseek_v4 import HyperConnectedMultiTokenPredictorLayer

        return HyperConnectedMultiTokenPredictorLayer
    return MultiTokenPredictorLayer


class MultiTokenPredictor(torch.nn.Module):
    def __init__(
        self,
        hf_config,
        num_mtp_layers,
        mtp_block_creator: Callable[[int], torch.nn.Module],
    ):
        super().__init__()
        self.mtp_start_layer_idx = hf_config.num_hidden_layers
        self.num_mtp_layers = num_mtp_layers
        # First block determines whether HC layers are needed; all subsequent
        # blocks share the same model family so the same choice applies to all.
        first_block = mtp_block_creator(self.mtp_start_layer_idx)
        layer_cls = _resolve_mtp_layer_cls(hf_config, first_block)
        self.layers = torch.nn.ModuleList(
            [
                layer_cls(
                    hf_config,
                    mtp_block_creator(idx),
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            ]
        )
        self.embed_tokens = torch.nn.Embedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
        )
        # TODO(jgong5): lm_head should share the weights with the main model and among MTP layers.
        #               Otherwise, the memory consumption would be higher.
        self.lm_head = torch.nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.layers[spec_step_idx](
            inputs_embeds,
            positions,
            previous_hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        intermediate_hidden_states = hidden_states
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        assert sampling_metadata is not None, "No sampling metadata given for MTP"
        hidden_states = select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="proposal")
        hidden_states = self.lm_head(hidden_states)
        return hidden_states, intermediate_hidden_states


class MtpWrapper(ModelWrapperBase):
    """For TensorCast only, simulate the MTP computation"""

    def __init__(self, mtp_config: MtpConfig, hf_config, model: torch.nn.Module):
        super().__init__(model)
        self.mtp_config = mtp_config
        self.hf_config = hf_config
        mtp_block_cls = self.get_mtp_block_cls()
        assert mtp_block_cls is not None, (
            f"unable to find mtp block class {self.mtp_config.mtp_block_module_name} in {self._inner}"
        )
        self.mtp = MultiTokenPredictor(
            hf_config,
            self.mtp_config.num_mtp_layers,
            functools.partial(mtp_block_cls, self.hf_config),
        )
        self.sampler = Sampler()
        self.rotary_emb = self.get_rotary_emb()
        if self.rotary_emb is None:
            raise ValueError(f"Unable to find rotary embedding module from {model}")

    def get_mtp_block_cls(self):
        for _, module in self._inner.named_modules():
            if type(module).__name__ == self.mtp_config.mtp_block_module_name:
                return type(module)
        return None

    def get_rotary_emb(self):
        for name, module in self._inner.named_modules():
            if name.endswith(".rotary_emb"):
                return module
        return None

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,  # NOTE: extra args should be torch.compile compatible
    ) -> torch.Tensor:
        sampling_metadata: Optional[SamplingMetadata] = kwargs.get("sampling_metadata")
        assert sampling_metadata is not None, "No sampling metadata given for MTP"
        logits, hidden_states = self._inner(
            input_ids,
            position_ids,
            inputs_embeds,
            output_intermediate_hidden_states=True,
            **kwargs,
        )
        next_tokens = self.sampler(logits, sampling_metadata)
        # The first target-model pass returns target tokens plus one bonus token per request.
        # This model assumes every speculative token is accepted and feeds the last returned token forward.
        output = torch.empty(
            [next_tokens.size(0), self.mtp_config.num_mtp_layers + 1],
            dtype=torch.long,
            device=next_tokens.device,
        )
        output[:, 0] = next_tokens[:, -1]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i in range(self.mtp_config.num_mtp_layers):
            input_ids = torch.ops.tensor_cast.shift_and_update_input_ids(
                input_ids, sampling_metadata.query_start_loc, next_tokens
            )
            logits, hidden_states = self.mtp.forward(
                input_ids,
                position_ids,
                hidden_states,
                inputs_embeds,
                position_embeddings=position_embeddings,
                spec_step_idx=i,
                **kwargs,
            )
            next_tokens = self.sampler(logits, sampling_metadata)
            output[:, i + 1] = next_tokens[:, -1]
        return output
