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

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

# pylint: disable=no-name-in-module
from tools.perf_data_collection.op_replay import replay_framework

if TYPE_CHECKING:
    from pathlib import Path


def test_build_inputs_honors_dtype_override(monkeypatch: pytest.MonkeyPatch):
    recorded_calls: list[tuple[tuple[int, ...], str, str]] = []

    monkeypatch.setattr(replay_framework, "init_runtime", lambda: None)
    monkeypatch.setattr(
        replay_framework,
        "get_runtime_modules",
        lambda: ("torch", "torch_npu"),
    )

    def fake_build_input_tensor(*, shape, input_format, dtype_name):
        recorded_calls.append((shape, input_format, dtype_name))
        return {
            "shape": shape,
            "input_format": input_format,
            "dtype_name": dtype_name,
        }

    monkeypatch.setattr(replay_framework, "build_input_tensor", fake_build_input_tensor)

    op = replay_framework.OpReplay(
        kernel_type="MaskedFill",
        description="test",
        usage_examples=["python test.py"],
        version_help="test",
        input_count=2,
        input_dtype_overrides={1: "DT_BOOL"},
    )

    tensors = op.build_inputs(
        {
            "Input Shapes": "2,3;2,3",
            "Input Formats": "ND;ND",
            "Input Data Types": "FLOAT16;INT64",
        }
    )

    assert [tensor["dtype_name"] for tensor in tensors] == ["DT_FLOAT16", "DT_BOOL"]
    assert recorded_calls == [
        ((2, 3), "ND", "DT_FLOAT16"),
        ((2, 3), "ND", "DT_BOOL"),
    ]


def test_resolve_api_supports_nested_torch_paths(monkeypatch: pytest.MonkeyPatch):
    class FakeFunctional:
        @staticmethod
        def softmax():
            return "softmax"

    class FakeNN:
        functional = FakeFunctional()

    class FakeTorch:
        nn = FakeNN()

    monkeypatch.setattr(
        replay_framework,
        "get_runtime_modules",
        lambda: (FakeTorch(), object()),
    )

    op = replay_framework.OpReplay(
        kernel_type="SoftmaxV2",
        api_path="torch.nn.functional.softmax",
        description="test",
        usage_examples=["python test.py"],
        version_help="test",
    )

    resolved = op.resolve_api()
    assert resolved() == "softmax"


def test_main_replays_each_row_repeat_count_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    csv_path = tmp_path / "Add.csv"
    csv_path.write_text(
        "Input Shapes,Input Formats,Input Data Types\n1,ND,FLOAT16\n",
        encoding="utf-8",
    )

    calls: list[int] = []

    monkeypatch.setattr(replay_framework, "ensure_npu_available", lambda: None)
    monkeypatch.setattr(replay_framework, "get_target_data_dir", lambda **_: tmp_path)
    monkeypatch.setattr(replay_framework, "get_replay_repeat_count", lambda _: 3)
    monkeypatch.setattr(replay_framework.OpReplay, "synchronize", lambda self: None)

    # run_row uses NPU event timing; patch get_runtime_modules so the test
    # does not require a real NPU.
    class FakeNpuEvent:
        def __init__(self, enable_timing=False):
            pass

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 0.001  # 1 us

    class FakeTorch:
        class npu:
            Event = FakeNpuEvent

    monkeypatch.setattr(replay_framework, "get_runtime_modules", lambda: (FakeTorch, None))

    def build_case(_row):
        return {"inputs": [], "kwargs": {}, "api": None}

    def run_case(_case):
        calls.append(1)
        return "ok"

    op = replay_framework.OpReplay(
        kernel_type="Add",
        description="test",
        usage_examples=["python test.py"],
        version_help="test",
        build_case=build_case,
        run_case=run_case,
    )

    monkeypatch.setattr(
        "sys.argv",
        ["Add_run.py", "--database-path", str(tmp_path)],
    )

    op.main()

    output = capsys.readouterr().out
    # run_row does warmup(2) + timed(3) = 5 total run_case calls
    assert len(calls) == 5
    # process_repeat_count is 1 (run_row handles repeats internally), so
    # total_rows = 1 row x 1 outer repeat = 1
    assert "Processed 1 Add rows" in output


def test_exact_runtime_match_records_original_query_signature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    recorded: list[dict[str, object]] = []

    # run_row uses NPU event timing; patch get_runtime_modules so the test
    # does not require a real NPU.
    class FakeNpuEvent:
        def __init__(self, enable_timing=False):
            pass

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 0.001

    class FakeTorch:
        class npu:
            Event = FakeNpuEvent

    monkeypatch.setattr(replay_framework, "get_runtime_modules", lambda: (FakeTorch, None))
    row = {
        "Input Shapes": "1,5,6144;5,6144;6144;",
        "Input Formats": "NCL;ND;ND;NULL",
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED",
        "Output Shapes": "1,5,6144;1,5,1;1,5,6144",
        "Output Data Types": "DT_BF16;FLOAT;DT_BF16",
        "Output Formats": "NCL;ND;NCL",
    }
    op = replay_framework.OpReplay(
        kernel_type="AddRmsNormBias",
        description="test",
        usage_examples=["python test.py"],
        version_help="test",
        build_case=lambda _row: {"inputs": [], "kwargs": {}, "api": None},
        run_case=lambda _case: "ok",
        exact_runtime_match=True,
    )
    monkeypatch.setattr(op, "synchronize", lambda: None)
    monkeypatch.setattr(
        replay_framework,
        "record_runtime_replay_case",
        lambda **kwargs: recorded.append(kwargs),
    )

    csv_path = tmp_path / "AddRmsNormBias.csv"
    original_row = dict(row)  # run_row mutates row (adds Average Duration)
    op.run_row(csv_path, 7, row)

    assert len(recorded) == 1
    assert recorded[0]["case_id"] == f"AddRmsNormBias:{csv_path}:7"
    # signature_context captures the original row data (before run_row adds
    # the measured Average Duration field).
    assert recorded[0]["signature_context"] == original_row


def test_runtime_row_warms_up_then_records_measurements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    recorded: list[dict[str, object]] = []
    calls: list[int] = []
    row = {"Runtime case_id": "runtime-1"}
    op = replay_framework.OpReplay(
        kernel_type="SparseFlashAttention",
        description="test",
        usage_examples=["python test.py"],
        version_help="test",
        build_case=lambda _row: {},
        run_case=lambda _case: calls.append(1) or "ok",
        format_success=lambda *_args: "ok",
        runtime_warmup_count=3,
    )
    monkeypatch.setattr(op, "synchronize", lambda: None)
    monkeypatch.setattr(
        replay_framework,
        "record_runtime_replay_case",
        lambda **kwargs: recorded.append(kwargs),
    )

    op.run_runtime_row(tmp_path / "SparseFlashAttention.csv", 2, row, repeat_count=2)

    assert len(calls) == 5
    assert recorded[0]["warmup_count"] == 3
    assert recorded[0]["repeat_count"] == 2


def test_runtime_replay_case_owns_fallback_identity_and_signature_context(
    tmp_path: Path,
):
    case = replay_framework.RuntimeReplayCase(
        kernel_type="AddRmsNormBias",
        csv_path=tmp_path / "AddRmsNormBias.csv",
        row_index=4,
        row={"Input Shapes": "2,16", "Average Duration(us)": "1.0"},
        exact_runtime_match=True,
    )

    assert case.case_id.endswith("AddRmsNormBias.csv:4")
    assert case.signature_context == {"Input Shapes": "2,16"}


def test_shard_spec_validates_range_and_delegates_stable_membership(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        replay_framework,
        "case_belongs_to_shard",
        lambda key, count, index: (key, count, index) == ("k", 4, 2),
    )

    assert replay_framework.ShardSpec(4, 2).accepts("k")
    assert not replay_framework.ShardSpec(4, 2).accepts("other")
    with pytest.raises(ValueError, match="case shard index"):
        replay_framework.ShardSpec(2, 2)
