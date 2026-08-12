from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("shift_and_update_input_ids")
def _(
    input_ids: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    next_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Creates a new input_ids tensor by shifting each query's tokens to the left
    and appending a new token at the end.

    Args:
        input_ids: A 1D tensor containing the concatenated tokens of all queries.
        query_start_loc: A 1D tensor of shape `(batch_size + 1)`, where
                         `query_start_loc[i]` is the starting index of the i-th
                         query in `input_ids`. If not set, the input_ids have the
                         same length indicated by the input_ids shape:
                         (batch_size, query_length, hidden_size).
        next_tokens: A 2D tensor of shape `(batch_size, sequence_length)`. The last
                     token from each sequence (`next_tokens[:, -1]`) will be used.

    Returns:
        A new 1D tensor `new_input_ids` with the transformed tokens.
    """
    if input_ids.is_meta:
        return torch.empty_like(input_ids)

    new_input_ids = input_ids.clone()
    update_tokens = next_tokens[:, -1] if next_tokens.ndim > 1 else next_tokens

    if query_start_loc is None:
        if input_ids.ndim == 1:
            new_input_ids[:-1] = input_ids[1:]
            new_input_ids[-1] = update_tokens.reshape(-1)[0]
            return new_input_ids
        new_input_ids[..., :-1] = input_ids[..., 1:]
        new_input_ids[..., -1] = update_tokens.to(input_ids.device)
        return new_input_ids

    flat_input_ids = input_ids.reshape(-1)
    flat_new_input_ids = new_input_ids.reshape(-1)
    starts = query_start_loc.to(device="cpu", dtype=torch.long).tolist()
    flat_update_tokens = update_tokens.reshape(-1).to(input_ids.device)
    for request_idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if end <= start:
            continue
        if end - start > 1:
            flat_new_input_ids[start : end - 1] = flat_input_ids[start + 1 : end]
        flat_new_input_ids[end - 1] = flat_update_tokens[request_idx]
    return new_input_ids
