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
    assert len(calls) == 3
    assert "Processed 3 Add rows" in output
