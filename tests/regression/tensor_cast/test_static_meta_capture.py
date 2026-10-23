# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from collections import namedtuple
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensor_cast.compilation import get_backend
from tensor_cast.compilation.static_meta import (
    flatten_tensor_inputs,
    StaticMetaCaptureUnsupported,
    StaticMetaForward,
)
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.device import DeviceProfile
from tensor_cast.layers.sampler import SamplingMetadata, Sampler, SpecDecodeMetadata
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime


@dataclass
class _Metadata:
    tensor: torch.Tensor
    length: int
    derived: int = field(init=False)

    def __post_init__(self):
        self.derived = self.length * 2


def test_flatten_preserves_tensor_aliases_and_complete_metadata():
    tensor = torch.empty(4, device="meta")
    extra = torch.empty(2, device="meta")
    metadata = _Metadata(tensor, 4)
    metadata.extra = extra
    pair_type = namedtuple("Pair", "left right")
    inputs = {"meta": metadata, "pair": pair_type(tensor, extra), "static": {1, 3}}
    flat, rebuild = flatten_tensor_inputs(inputs)
    assert len(flat) == 2
    replacements = [torch.empty_like(t) for t in flat]
    with patch.object(_Metadata, "__post_init__", side_effect=AssertionError("must not recalculate metadata")):
        result = rebuild(replacements)
    assert isinstance(result["pair"], pair_type)
    assert result["pair"].left is result["meta"].tensor is replacements[0]
    assert result["pair"].right is result["meta"].extra is replacements[1]
    assert result["meta"].derived == 8
    assert result["static"] == {1, 3}
    assert metadata.tensor is tensor and metadata.extra is extra


@pytest.mark.parametrize("value", [object()])
def test_flatten_rejects_unknown_inputs(value):
    with pytest.raises(StaticMetaCaptureUnsupported):
        flatten_tensor_inputs({"input": value})


def test_host_metadata_is_lifted_without_changing_values_or_aliases():
    tensor = torch.arange(3)
    flat, rebuild = flatten_tensor_inputs((tensor, tensor))
    assert len(flat) == 1 and flat[0] is tensor
    assert rebuild(flat)[0] is rebuild(flat)[1] is tensor


def test_real_model_weights_are_not_supported():
    with pytest.raises(StaticMetaCaptureUnsupported, match="meta model parameters"):
        StaticMetaForward(torch.nn.Linear(2, 2), None)(x=torch.empty(1, 2, device="meta"))


def test_capture_restores_module_state_even_when_lowering_fails():
    class CapturingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.empty(8, device="meta"))
            self.scale = self.get_buffer("scale")

        def forward(self, x):
            # Model wrappers can retain aliases created while tracing. They
            # must not keep functional tensors after that trace has ended.
            self.capture_alias = x
            self.scale = self.scale + 1
            return x * self.scale

    model = CapturingModule()
    original_scale = model.scale
    backend = SimpleNamespace(compile_static_meta=lambda *args: (_ for _ in ()).throw(RuntimeError("lowering")))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="lowering"):
            StaticMetaForward(model, backend)(x=torch.empty(2, 8, device="meta"))
        assert model.scale is original_scale
        assert not hasattr(model, "capture_alias")


def _runner(**overrides):
    runner = ModelRunner.__new__(ModelRunner)
    runner.user_input = SimpleNamespace(
        do_compile=True,
        dynamic_shapes=False,
        allow_graph_break=False,
        device="test",
        dflash=False,
        dspark=False,
    )
    runner.user_input.__dict__.update(overrides)
    native = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4"),
        model_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)),
        is_vl_model=False,
    )
    runner.model = SimpleNamespace(_orig_mod=native)
    return runner


@pytest.mark.parametrize(
    "overrides",
    [
        {"do_compile": False},
        {"dynamic_shapes": True},
        {"allow_graph_break": True},
        {"dflash": True},
        {"dspark": True},
        {"enable_prefill_kv_offload": True},
        {"enable_decode_kv_offload": True},
    ],
)
def test_unsupported_runner_modes_keep_dynamo(overrides):
    runner = _runner(**overrides)
    assert not runner.enable_static_meta_capture()
    assert runner._static_meta_forward is None


def test_only_validated_static_v4_path_is_enabled():
    runner = _runner()
    with patch("tensor_cast.compilation.get_backend", return_value="backend"):
        assert runner.enable_static_meta_capture()
    assert runner._static_meta_forward.model is runner.model._orig_mod
    assert runner._static_meta_forward.backend == "backend"
    runner.model._orig_mod.hf_config.model_type = "other"
    assert not runner.enable_static_meta_capture()


@pytest.mark.parametrize(
    "config_path",
    ["enable_freezing", "multistream.enable", "passes.enable_sequence_parallel"],
)
def test_compiler_modes_outside_validation_keep_dynamo(config_path):
    value = config_path != "enable_freezing"
    with patch(f"tensor_cast.config.compilation.{config_path}", value):
        assert not _runner().enable_static_meta_capture()


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(8, 8, device="meta", dtype=torch.bfloat16))
        self.register_buffer("scale", torch.empty(8, device="meta", dtype=torch.bfloat16))
        self.register_buffer("quant_scale", torch.tensor(1.0))
        self.shared = torch.nn.Module()
        self.shared.register_buffer("scale", self.quant_scale)
        self.sampler = Sampler()

    def forward(self, x, metadata, sampling):
        value = x @ self.weight
        value = value * self.quant_scale
        value = torch.ops.tensor_cast.quantize(value, self.shared.scale, None)
        value = value.to(torch.bfloat16)
        value.add_(1)
        torch.ops.tensor_cast.apply_rope_inplace(value, metadata.tensor, metadata.tensor)
        value = torch.ops.tensor_cast.rms_norm(value, self.scale, 1e-6)
        tokens = self.sampler(value, sampling)
        return tokens, {"hidden": value}


def _capture(call, inputs, device):
    with (
        Runtime(AnalyticPerformanceModel(device), device, memory_tracker=MemoryTracker(device)) as runtime,
        torch.no_grad(),
    ):
        output = call(**inputs)
    signatures = [
        (
            str(event.op_invoke_info.func),
            event.op_invoke_info.cache_key,
            event.perf_results["analytic"].execution_time_s,
        )
        for event in runtime.event_list
        if event.perf_results["analytic"].execution_time_s > 0
    ]
    return output, signatures, runtime.total_execution_time_s(), runtime.memory_tracker.peak_mem_usage()


@pytest.mark.parametrize("lengths", [(1,), (4, 4), (1, 4)])
def test_static_fx_keeps_fused_ops_memory_and_speculative_rows(lengths):
    device = DeviceProfile.all_device_profiles["ATLAS_950_486T_96G"]
    model = _Toy().eval()
    backend = get_backend(device_name=device.name)
    compiled = torch.compile(model, backend=backend, dynamic=False, fullgraph=True)
    rows = sum(lengths)
    inputs = {
        "x": torch.empty(rows, 8, device="meta", dtype=torch.bfloat16),
        "metadata": _Metadata(torch.empty(rows, 8, dtype=torch.bfloat16), rows),
        "sampling": SamplingMetadata(
            selected_token_indices=None,
            spec_decode_metadata=SpecDecodeMetadata(
                torch.arange(rows, device="meta"), len(lengths), 3, verification_lens=lengths
            ),
        ),
    }
    old = _capture(compiled, inputs, device)
    new = _capture(StaticMetaForward(model, backend), inputs, device)
    assert new[0][0].shape == old[0][0].shape == (len(lengths), 4)
    assert new[0][1]["hidden"].shape == old[0][1]["hidden"].shape == (rows, 8)
    assert new[1:] == old[1:]
    repeated = _capture(StaticMetaForward(model, backend), inputs, device)
    assert repeated[1:] == old[1:]
