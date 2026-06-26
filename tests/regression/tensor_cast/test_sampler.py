from types import SimpleNamespace

import pytest
import torch

from tensor_cast.layers.mtp import MtpWrapper
from tensor_cast.layers.sampler import (
    Sampler,
    SamplingMetadata,
    SpecDecodeMetadata,
    select_lm_head_hidden_states,
)
from tensor_cast.model_config import MtpConfig
from tensor_cast.transformers.model import CausalLmWrapper


def _spec_metadata(logits_indices=None, num_active_requests=2, num_speculative_tokens=2):
    if logits_indices is None:
        logits_indices = [2, 3, 4, 5, 6, 7]
    return SpecDecodeMetadata(
        logits_indices=torch.tensor(logits_indices, dtype=torch.long),
        num_active_requests=num_active_requests,
        num_speculative_tokens=num_speculative_tokens,
    )


def _lm_head_weight():
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


def _project(hidden_states):
    return hidden_states @ _lm_head_weight().T


def test_spec_decode_selects_target_and_proposal_rows():
    sampling_metadata = SamplingMetadata(spec_decode_metadata=_spec_metadata())
    hidden_states = torch.arange(8 * 2, dtype=torch.float32).view(1, 8, 2)

    target_hidden_states = select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="target")
    proposal_hidden_states = select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="proposal")

    assert target_hidden_states.tolist() == [
        [4.0, 5.0],
        [6.0, 7.0],
        [8.0, 9.0],
        [10.0, 11.0],
        [12.0, 13.0],
        [14.0, 15.0],
    ]
    assert proposal_hidden_states.tolist() == [[8.0, 9.0], [14.0, 15.0]]


def test_lm_head_selection_ignores_default_selected_token_sentinel():
    hidden_states = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    selected_hidden_states = select_lm_head_hidden_states(hidden_states, SamplingMetadata())

    assert selected_hidden_states is hidden_states


def test_spec_decode_selection_rejects_wrong_logits_indices_length():
    sampling_metadata = SamplingMetadata(
        spec_decode_metadata=_spec_metadata(logits_indices=[2, 4], num_active_requests=2, num_speculative_tokens=2)
    )
    hidden_states = torch.arange(8 * 2, dtype=torch.float32).view(1, 8, 2)

    with pytest.raises(ValueError, match="logits_indices length must equal"):
        select_lm_head_hidden_states(hidden_states, sampling_metadata, mode="target")


def test_spec_decode_sampler_returns_target_tokens_plus_bonus_token():
    logits = torch.zeros(1, 6, 8)
    for row, token_id in enumerate([1, 2, 3, 4, 5, 6]):
        logits[0, row, token_id] = 100.0 + row
    sampling_metadata = SamplingMetadata(spec_decode_metadata=_spec_metadata(logits_indices=list(range(6))))

    next_tokens = Sampler()(logits, sampling_metadata)

    assert next_tokens.tolist() == [[1, 2, 3], [4, 5, 6]]


class _FixedCausalInner(torch.nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **kwargs):
        return (self.hidden_states,)


def test_causal_lm_wrapper_projects_spec_decode_verification_rows():
    hidden_states = torch.arange(8 * 2, dtype=torch.float32).view(1, 8, 2)
    wrapper = CausalLmWrapper(
        SimpleNamespace(hidden_size=2, vocab_size=3),
        _FixedCausalInner(hidden_states),
    )
    with torch.no_grad():
        wrapper.lm_head.weight.copy_(_lm_head_weight())
    sampling_metadata = SamplingMetadata(spec_decode_metadata=_spec_metadata())

    logits = wrapper(
        input_ids=None,
        position_ids=torch.arange(8, dtype=torch.long).view(1, 8),
        sampling_metadata=sampling_metadata,
    )

    expected_hidden_states = hidden_states.view(-1, 2).index_select(0, _spec_metadata().logits_indices)
    assert logits.tolist() == _project(expected_hidden_states).tolist()


class _UnusedMtpBlock(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("test replaces the generated MTP layer")


class _FixedMtpLayer(torch.nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, *args, **kwargs):
        return self.hidden_states


class _FakeMtpBlock(torch.nn.Module):
    def __init__(self, hf_config, layer_idx=None):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class _FakeRotaryEmbedding(torch.nn.Module):
    def forward(self, hidden_states, position_ids):
        return torch.empty_like(hidden_states)


class _FixedMtpInner(torch.nn.Module):
    def __init__(self, logits, hidden_states, hf_config):
        super().__init__()
        self.logits = logits
        self.hidden_states = hidden_states
        self.block = _FakeMtpBlock(hf_config)
        self.layer = torch.nn.Module()
        self.layer.rotary_emb = _FakeRotaryEmbedding()

    def forward(self, input_ids, position_ids, inputs_embeds, **kwargs):
        assert kwargs["output_intermediate_hidden_states"]
        return self.logits, self.hidden_states


def _mtp_wrapper(logits, hidden_states, hf_config):
    wrapper = MtpWrapper(
        MtpConfig(num_mtp_layers=1, mtp_block_module_name="_FakeMtpBlock"),
        hf_config,
        _FixedMtpInner(logits, hidden_states, hf_config),
    )
    with torch.no_grad():
        wrapper.mtp.lm_head.weight.copy_(_lm_head_weight())
    return wrapper


def test_mtp_wrapper_forward_prefill_reuses_already_selected_logits():
    hf_config = SimpleNamespace(hidden_size=2, vocab_size=3, rms_norm_eps=1e-6, num_hidden_layers=1)
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 1] = 10.0
    target_hidden_states = torch.zeros(1, 6, 2)
    mtp_hidden_states = torch.zeros(1, 6, 2)
    mtp_hidden_states[0, 2, 0] = 10.0
    mtp_hidden_states[0, 5, 1] = 10.0
    wrapper = _mtp_wrapper(logits, target_hidden_states, hf_config)
    wrapper.mtp.layers = torch.nn.ModuleList([_FixedMtpLayer(mtp_hidden_states)])
    sampling_metadata = SamplingMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.long),
        selected_token_indices=torch.tensor([2, 5], dtype=torch.long),
    )

    output = wrapper(
        input_ids=torch.zeros(1, 6, dtype=torch.long),
        position_ids=torch.arange(6, dtype=torch.long).view(1, 6),
        inputs_embeds=torch.zeros(1, 6, 2),
        sampling_metadata=sampling_metadata,
    )

    assert output.tolist() == [[2, 0], [1, 1]]


def test_mtp_wrapper_forward_feeds_bonus_token_and_projects_mtp_proposal_rows():
    hf_config = SimpleNamespace(hidden_size=2, vocab_size=3, rms_norm_eps=1e-6, num_hidden_layers=1)
    logits = torch.zeros(1, 6, 3)
    logits[0, :, 0] = 1.0
    logits[0, 2, 2] = 10.0
    logits[0, 5, 2] = 10.0
    target_hidden_states = torch.zeros(1, 8, 2)
    mtp_hidden_states = torch.zeros(1, 8, 2)
    mtp_hidden_states[0, 4, 0] = 10.0
    mtp_hidden_states[0, 7, 1] = 10.0
    wrapper = _mtp_wrapper(logits, target_hidden_states, hf_config)
    wrapper.mtp.layers = torch.nn.ModuleList([_FixedMtpLayer(mtp_hidden_states)])
    sampling_metadata = SamplingMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.long),
        spec_decode_metadata=_spec_metadata(),
    )

    output = wrapper(
        input_ids=torch.zeros(1, 8, dtype=torch.long),
        position_ids=torch.arange(8, dtype=torch.long).view(1, 8),
        inputs_embeds=torch.zeros(1, 8, 2),
        sampling_metadata=sampling_metadata,
    )

    assert output.tolist() == [[2, 0], [2, 1]]
