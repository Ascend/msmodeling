# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Semantic operator activation policy tests."""

import pytest

from tools.model_diagnostics.domain import (
    ExecutionPhase,
    ModelRunContext,
    ParallelContext,
    TheoryOperatorSpec,
)
from tools.model_diagnostics.specification.builtin_activation import (
    DsaEnabledActivation,
    ExplicitMoeGateActivation,
    LmHeadTokenSelectionActivation,
    MlaPrefillKvProjectionActivation,
    MtpEnabledActivation,
    NonMtpLmHeadActivation,
    Qwen35DenseFfnActivation,
    Qwen35LinearGdnActivation,
    Qwen35MoeFfnActivation,
    VisionPrefillActivation,
)
from tools.model_diagnostics.specification.errors import SpecificationLoadError
from tools.model_diagnostics.specification.mtp_window import parse_num_mtp_tokens
from tools.model_diagnostics.specification.operator_activation import (
    OperatorActivationRegistry,
    OperatorActivationRequest,
)


def _request(
    *,
    phase: ExecutionPhase,
    query_length: int,
    num_mtp_tokens: object,
    model_config: dict[str, object] | None = None,
    parallel: ParallelContext | None = None,
) -> OperatorActivationRequest:
    context = ModelRunContext(
        model_name="test",
        entrypoint="text_generate",
        phase=phase,
        batch_size=1,
        query_length=query_length,
        context_length=None,
        parallel=parallel or ParallelContext(),
        model_config={"num_mtp_tokens": num_mtp_tokens, **(model_config or {})},
        quantization_config={},
    )
    return OperatorActivationRequest(
        spec_id="test_v1",
        model_category="test",
        region_id="output",
        stage_id="lm_head",
        operator=TheoryOperatorSpec(operator_name="selection"),
        context=context,
    )


@pytest.mark.parametrize(
    ("phase", "query_length", "num_mtp_tokens", "expected"),
    (
        (ExecutionPhase.PREFILL, 4, 0, True),
        (ExecutionPhase.DECODE, 1, 0, False),
        (ExecutionPhase.DECODE, 3, 2, False),
        (ExecutionPhase.DECODE, 2, 2, False),
    ),
)
def test_lm_head_token_selection_activation(
    phase: ExecutionPhase,
    query_length: int,
    num_mtp_tokens: int,
    expected: bool,
) -> None:
    assert (
        LmHeadTokenSelectionActivation().is_active(
            _request(
                phase=phase,
                query_length=query_length,
                num_mtp_tokens=num_mtp_tokens,
            )
        )
        is expected
    )


@pytest.mark.parametrize(
    ("phase", "query_length", "num_mtp_tokens", "expected"),
    (
        (ExecutionPhase.PREFILL, 4, 2, False),
        (ExecutionPhase.DECODE, 1, 0, False),
        (ExecutionPhase.DECODE, 3, 2, True),
        (ExecutionPhase.DECODE, 2, 2, False),
    ),
)
def test_mtp_enabled_activation(
    phase: ExecutionPhase,
    query_length: int,
    num_mtp_tokens: int,
    expected: bool,
) -> None:
    assert (
        MtpEnabledActivation().is_active(
            _request(
                phase=phase,
                query_length=query_length,
                num_mtp_tokens=num_mtp_tokens,
            )
        )
        is expected
    )


@pytest.mark.parametrize(
    ("phase", "query_length", "num_mtp_tokens", "expected"),
    (
        (ExecutionPhase.PREFILL, 4, 2, True),
        (ExecutionPhase.DECODE, 1, 0, True),
        (ExecutionPhase.DECODE, 3, 2, False),
    ),
)
def test_non_mtp_lm_head_activation(
    phase: ExecutionPhase,
    query_length: int,
    num_mtp_tokens: int,
    expected: bool,
) -> None:
    assert (
        NonMtpLmHeadActivation().is_active(
            _request(
                phase=phase,
                query_length=query_length,
                num_mtp_tokens=num_mtp_tokens,
            )
        )
        is expected
    )


@pytest.mark.parametrize(
    ("phase", "image_config", "expected"),
    (
        (
            ExecutionPhase.PREFILL,
            {"image_batch_size": 1, "image_height": 224, "image_width": 224},
            True,
        ),
        (ExecutionPhase.DECODE, {"image_batch_size": 1, "image_height": 224, "image_width": 224}, False),
        (ExecutionPhase.PREFILL, {"image_batch_size": 1}, False),
        (ExecutionPhase.PREFILL, {"image_batch_size": 1, "image_height": 224}, False),
    ),
)
def test_vision_prefill_requires_complete_image_dimensions(
    phase: ExecutionPhase,
    image_config: dict[str, object],
    expected: bool,
) -> None:
    request = _request(
        phase=phase,
        query_length=30,
        num_mtp_tokens=0,
        model_config=image_config,
    )

    assert VisionPrefillActivation().is_active(request) is expected


@pytest.mark.parametrize(("index_topk", "expected"), ((None, False), (2048, True)))
def test_dsa_enabled_activation(index_topk: int | None, expected: bool) -> None:
    request = _request(
        phase=ExecutionPhase.PREFILL,
        query_length=2,
        num_mtp_tokens=0,
        model_config={} if index_topk is None else {"index_topk": index_topk},
    )

    assert DsaEnabledActivation().is_active(request) is expected


@pytest.mark.parametrize(
    ("phase", "expected"),
    ((ExecutionPhase.PREFILL, True), (ExecutionPhase.DECODE, False)),
)
def test_mla_prefill_kv_projection_activation(phase: ExecutionPhase, expected: bool) -> None:
    request = _request(phase=phase, query_length=2, num_mtp_tokens=0)

    assert MlaPrefillKvProjectionActivation().is_active(request) is expected


@pytest.mark.parametrize(("model_type", "expected"), (("kimi_k2", False), ("deepseek_v3", True)))
def test_explicit_moe_gate_activation(model_type: str, expected: bool) -> None:
    request = _request(
        phase=ExecutionPhase.PREFILL,
        query_length=2,
        num_mtp_tokens=0,
        model_config={"model_type": model_type},
    )

    assert ExplicitMoeGateActivation().is_active(request) is expected


@pytest.mark.parametrize(
    ("model_type", "dense_ffn", "moe_ffn"),
    (
        ("qwen3_5_text", True, False),
        ("qwen3_5_moe_text", False, True),
        ("qwen3_next", False, True),
        ("qwen3", False, False),
    ),
)
def test_qwen35_ffn_stage_activation(
    model_type: str,
    dense_ffn: bool,
    moe_ffn: bool,
) -> None:
    request = _request(
        phase=ExecutionPhase.PREFILL,
        query_length=2,
        num_mtp_tokens=0,
        model_config={"model_type": model_type},
    )

    assert Qwen35DenseFfnActivation().is_active(request) is dense_ffn
    assert Qwen35MoeFfnActivation().is_active(request) is moe_ffn


@pytest.mark.parametrize(
    ("model_type", "gdn"),
    (
        ("qwen3_5_text", True),
        ("qwen3_5_moe_text", True),
        ("qwen3_next", True),
    ),
)
def test_qwen35_linear_stage_activation(
    model_type: str,
    gdn: bool,
) -> None:
    request = _request(
        phase=ExecutionPhase.PREFILL,
        query_length=2,
        num_mtp_tokens=0,
        model_config={"model_type": model_type},
    )

    assert Qwen35LinearGdnActivation().is_active(request) is gdn


@pytest.mark.parametrize("invalid", (-1, True, "2"))
def test_mtp_helpers_reject_invalid_mtp_count(invalid: object) -> None:
    context = _request(
        phase=ExecutionPhase.DECODE,
        query_length=3,
        num_mtp_tokens=invalid,
    ).context
    with pytest.raises(SpecificationLoadError, match="num_mtp_tokens"):
        parse_num_mtp_tokens(context)


def test_activation_registry_rejects_duplicate_and_unknown_ids() -> None:
    registry = OperatorActivationRegistry()
    policy = LmHeadTokenSelectionActivation()
    registry.register(policy)

    assert registry.resolve(policy.policy_id) is policy
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(policy)
    with pytest.raises(KeyError, match="unregistered"):
        registry.resolve("missing")
