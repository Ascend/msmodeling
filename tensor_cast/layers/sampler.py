import dataclasses
from typing import Optional

import torch


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

    selected_token_indices: Optional[torch.Tensor] = torch.tensor(-1, dtype=torch.long)
    top_k: Optional[int] = None  # None for greedy search
    # TODO: add more sampling params, e.g. top-k/top-p


class Sampler(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata, **kwargs) -> torch.Tensor:
        if sampling_metadata.query_start_loc is None:
            assert hidden_states.ndim == 3
            logits = hidden_states[:, -1, :]
        else:
            query_start_loc = sampling_metadata.query_start_loc.to(hidden_states.device)
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            logits = hidden_states.index_select(0, query_start_loc[1:] - 1)
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tokens
