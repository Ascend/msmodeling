import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class SpecDecodeMetadata:
    """Shape metadata for speculative decode logits selection."""

    logits_indices: torch.Tensor
    num_active_requests: int
    num_speculative_tokens: int


@dataclasses.dataclass
class SamplingMetadata:
    """
    A simplified sampling data assuming the sampling parameters like top_k/top_k are the same across all
    requests.
    """

    query_start_loc: Optional[torch.Tensor] = None
    """(batch_size + 1,), the start location of each request in query Tensor. If not set,
    the request inputs have the same length indicated by the input_ids shape:
    (batch_size, query_length, hidden_size).
    """

    selected_token_indices: Optional[torch.Tensor] = dataclasses.field(
        default_factory=lambda: torch.tensor(-1, dtype=torch.long)
    )
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None
    top_k: Optional[int] = None  # None for greedy search
    # TODO: add more sampling params, e.g. top-k/top-p


def _index_select_token_dim(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices = indices.to(hidden_states.device)
    if indices.ndim == 0:
        indices = indices.unsqueeze(0)
    if hidden_states.ndim == 3:
        return hidden_states.index_select(1, indices)
    return hidden_states.index_select(0, indices)


def _has_explicit_selected_token_indices(indices: Optional[torch.Tensor]) -> bool:
    # SamplingMetadata's scalar -1 default means "no explicit selected rows".
    # Generated prefill metadata uses a vector of explicit selected rows.
    return indices is not None and indices.ndim > 0


def _select_logits_for_sampling(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices = indices.to(logits.device)
    selected_rows = indices.numel()
    if indices.ndim == 0:
        selected_rows = 1
    if logits.ndim == 3 and logits.size(1) == selected_rows:
        return logits.reshape(-1, logits.size(-1))
    if logits.ndim == 2 and logits.size(0) == selected_rows:
        return logits
    logits = _index_select_token_dim(logits, indices)
    if logits.ndim == 3:
        return logits.reshape(-1, logits.size(-1))
    return logits


def _spec_window_size(spec_metadata: SpecDecodeMetadata) -> int:
    if spec_metadata.num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive for spec decode metadata")
    if spec_metadata.num_active_requests <= 0:
        raise ValueError("num_active_requests must be positive for spec decode metadata")
    return spec_metadata.num_speculative_tokens + 1


def _proposal_logits_indices(spec_metadata: SpecDecodeMetadata) -> torch.Tensor:
    # _build_spec_decode_metadata appends each request's target+bonus rows as one
    # logits_indices window. The reshape is over that metadata ordering, not over
    # hidden_states storage layout; row selection uses explicit index_select later.
    return spec_metadata.logits_indices.reshape(spec_metadata.num_active_requests, _spec_window_size(spec_metadata))[
        :, -1
    ]


def _validate_spec_decode_indices(spec_metadata: SpecDecodeMetadata) -> None:
    spec_window = _spec_window_size(spec_metadata)
    expected_indices = spec_metadata.num_active_requests * spec_window
    actual_indices = spec_metadata.logits_indices.numel()
    if actual_indices != expected_indices:
        raise ValueError(
            "Spec decode logits_indices length must equal num_active_requests * "
            f"(num_speculative_tokens + 1), got {actual_indices} for "
            f"{spec_metadata.num_active_requests} requests and {spec_metadata.num_speculative_tokens} "
            f"speculative tokens; expected {expected_indices}."
        )


def select_lm_head_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: Optional[SamplingMetadata],
    mode: str = "target",
) -> torch.Tensor:
    if sampling_metadata is None:
        return hidden_states
    spec_metadata = sampling_metadata.spec_decode_metadata
    if spec_metadata is not None:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        _validate_spec_decode_indices(spec_metadata)
        if mode == "target":
            return _index_select_token_dim(hidden_states, spec_metadata.logits_indices)
        if mode == "proposal":
            return _index_select_token_dim(hidden_states, _proposal_logits_indices(spec_metadata))
        raise ValueError(f"Unsupported lm_head selection mode: {mode}")
    if _has_explicit_selected_token_indices(sampling_metadata.selected_token_indices):
        return _index_select_token_dim(hidden_states, sampling_metadata.selected_token_indices)
    return hidden_states


class Sampler(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata, **kwargs) -> torch.Tensor:
        """Return greedy next tokens.

        Spec decode models TensorCast dependency/shape flow only: verification rows
        are reduced with greedy argmax for target+bonus tokens, not probability
        acceptance/rejection sampling.
        """
        spec_metadata = sampling_metadata.spec_decode_metadata
        if spec_metadata is not None:
            logits = hidden_states.reshape(-1, hidden_states.size(-1))
            num_active_requests = spec_metadata.num_active_requests
            num_speculative_tokens = spec_metadata.num_speculative_tokens
            spec_window = _spec_window_size(spec_metadata)
            verification_rows = num_active_requests * spec_window
            if logits.size(0) == verification_rows:
                # Verification rows contain target speculative positions followed by one bonus position per request.
                verification_logits = logits.reshape(num_active_requests, spec_window, logits.size(-1))
                target_logits = verification_logits[:, :num_speculative_tokens, :]
                bonus_logits = verification_logits[:, num_speculative_tokens, :]
                # TensorCast models the spec-decode dependency/shape path here, not a full
                # probability-acceptance rejection sampler; target and bonus tokens use greedy argmax.
                target_tokens = torch.argmax(target_logits, dim=-1)
                bonus_tokens = torch.argmax(bonus_logits, dim=-1, keepdim=True)
                # Output shape: (num_active_requests, num_speculative_tokens + 1);
                # target tokens followed by one bonus token per request.
                return torch.cat([target_tokens, bonus_tokens], dim=-1)
            if logits.size(0) == num_active_requests:
                return torch.argmax(logits, dim=-1, keepdim=True)
            raise ValueError(
                "Spec decode logits rows must match active requests or "
                f"verification rows, got {logits.size(0)} rows for {num_active_requests} "
                f"requests and {num_speculative_tokens} speculative tokens. "
                f"Expected {verification_rows} verification rows or {num_active_requests} proposal rows."
            )
        selected_token_indices = sampling_metadata.selected_token_indices
        if _has_explicit_selected_token_indices(selected_token_indices):
            logits = _select_logits_for_sampling(hidden_states, selected_token_indices)
        elif sampling_metadata.query_start_loc is None:
            assert hidden_states.ndim == 3
            logits = hidden_states[:, -1, :]
        else:
            query_start_loc = sampling_metadata.query_start_loc.to(hidden_states.device)
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            logits = hidden_states.index_select(0, query_start_loc[1:] - 1)
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tokens
