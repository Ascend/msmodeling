"""Tests for trace_to_csv.py — chrome trace JSON to human-readable CSV."""

import csv
import json
import sys
from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "tools"
        / "perf_data_collection"
        / "parsers"
    ),
)
import trace_to_csv as trace_to_csv_module  # noqa: E402


def _make_trace(tmp_path, events):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return str(path)


def _x(name, dur, pid=0, **kwargs):
    """Helper: create chrome trace X event."""
    return {
        "name": name,
        "ph": "X",
        "ts": 0,
        "dur": dur,
        "pid": pid,
        "tid": 0,
        "args": {k: str(v) for k, v in kwargs.items()},
    }


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class TestTraceToCSV:
    def test_main_writes_csv_via_argparse(self, tmp_path, monkeypatch):
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "aten.relu.default",
                    11,
                    kernel_type="Unary",
                    simulation_shapes=json.dumps([[4, 8]]),
                    source="MEASURED",
                ),
            ],
        )
        out = tmp_path / "cli.csv"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "trace_to_csv.py",
                "--trace",
                trace,
                "--output",
                str(out),
            ],
        )
        trace_to_csv_module.main()

        fieldnames, rows = _read_csv(out)
        assert fieldnames == trace_to_csv_module.COLUMNS
        assert rows == [
            {
                "op_name": "aten.relu.default",
                "kernel_type": "Unary",
                "simulation_shapes": "[[4, 8]]",
                "dur_us": "11",
                "source": "MEASURED",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            }
        ]

    def test_trace_to_csv_writes_full_row_contract(self, tmp_path):
        sub_kernel_shapes = json.dumps(
            [
                {
                    "kernel_type": "LayerNormKernel",
                    "kernel_shapes": [[2, 128, 4096], [4096], [4096]],
                    "shape_match_rule": "exact",
                },
                {
                    "kernel_type": "RmsNormKernel",
                    "kernel_shapes": [[2, 128, 4096], [4096]],
                    "shape_match_rule": "broadcast_rhs",
                },
            ]
        )
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "aten.native_layer_norm.default",
                    321,
                    kernel_type="LayerNormFusion",
                    simulation_shapes=json.dumps([[2, 128, 4096], [4096], [4096]]),
                    source="PARTIAL",
                    confidence="0.75",
                    composite="true",
                    sub_kernel_durations=json.dumps([100, 221]),
                    kernel_shapes=json.dumps([[2, 128, 4096], [4096], [4096]]),
                    shape_match_rule="broadcast_rhs",
                    sub_kernel_shapes=sub_kernel_shapes,
                ),
            ],
        )
        out = tmp_path / "row_contract.csv"
        trace_to_csv_module.trace_to_csv(trace, str(out))

        fieldnames, rows = _read_csv(out)
        assert fieldnames == trace_to_csv_module.COLUMNS
        assert rows == [
            {
                "op_name": "aten.native_layer_norm.default",
                "kernel_type": "LayerNormFusion",
                "simulation_shapes": "[[2, 128, 4096], [4096], [4096]]",
                "dur_us": "321",
                "source": "PARTIAL",
                "confidence": "0.75",
                "composite": "true",
                "sub_kernel_durations": "[100, 221]",
                "kernel_shapes": "[[2, 128, 4096], [4096], [4096]]",
                "shape_match_rule": "broadcast_rhs",
                "sub_kernel_shapes": sub_kernel_shapes,
            }
        ]

    def test_composite_ops_stay_one_row_per_op_and_keep_parseable_shapes(
        self, tmp_path
    ):
        composite_shapes = [
            {
                "kernel_type": "PadKernel",
                "kernel_shapes": [[120, 5120]],
                "shape_match_rule": "padding",
            },
            {
                "kernel_type": "SliceKernel",
                "kernel_shapes": [[64, 5120]],
                "shape_match_rule": "exact",
            },
        ]
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "tensor_cast.fake_composite",
                    50,
                    kernel_type="CompositeKernel",
                    simulation_shapes=json.dumps([[128, 5120]]),
                    source="PARTIAL",
                    confidence="0.50",
                    composite="true",
                    sub_kernel_durations=json.dumps([20, 30]),
                    kernel_shapes=json.dumps([[120, 5120], [64, 5120]]),
                    shape_match_rule="padding",
                    sub_kernel_shapes=json.dumps(composite_shapes),
                ),
                _x(
                    "aten.add.Tensor",
                    5,
                    simulation_shapes=json.dumps([[1, 4, 7168]]),
                ),
            ],
        )
        out = tmp_path / "composite.csv"
        trace_to_csv_module.trace_to_csv(trace, str(out))

        _, rows = _read_csv(out)
        assert rows == [
            {
                "op_name": "tensor_cast.fake_composite",
                "kernel_type": "CompositeKernel",
                "simulation_shapes": "[[128, 5120]]",
                "dur_us": "50",
                "source": "PARTIAL",
                "confidence": "0.50",
                "composite": "true",
                "sub_kernel_durations": "[20, 30]",
                "kernel_shapes": "[[120, 5120], [64, 5120]]",
                "shape_match_rule": "padding",
                "sub_kernel_shapes": json.dumps(composite_shapes),
            },
            {
                "op_name": "aten.add.Tensor",
                "kernel_type": "",
                "simulation_shapes": "[[1, 4, 7168]]",
                "dur_us": "5",
                "source": "",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            },
        ]
        assert json.loads(rows[0]["sub_kernel_shapes"]) == composite_shapes

    def test_missing_trace_events_writes_only_header(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text("{}")
        out = tmp_path / "out.csv"

        trace_to_csv_module.trace_to_csv(str(trace), str(out))

        fieldnames, rows = _read_csv(out)
        assert fieldnames == trace_to_csv_module.COLUMNS
        assert rows == []

    def test_missing_event_fields_use_csv_defaults(self, tmp_path):
        trace = _make_trace(tmp_path, [{"ph": "X"}])
        out = tmp_path / "defaults.csv"

        trace_to_csv_module.trace_to_csv(trace, str(out))

        _, rows = _read_csv(out)
        assert rows == [
            {
                "op_name": "",
                "kernel_type": "",
                "simulation_shapes": "",
                "dur_us": "0",
                "source": "",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            }
        ]

    def test_main_prints_csv_to_stdout_when_output_omitted(
        self, tmp_path, monkeypatch, capsys
    ):
        trace = _make_trace(
            tmp_path,
            [
                _x(
                    "aten.mm.default",
                    20,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes=json.dumps([[64, 128], [128, 256]]),
                ),
            ],
        )

        monkeypatch.setattr(
            sys,
            "argv",
            ["trace_to_csv.py", "--trace", trace],
        )
        trace_to_csv_module.main()

        output = capsys.readouterr().out.splitlines()
        reader = csv.DictReader(output)
        assert reader.fieldnames == trace_to_csv_module.COLUMNS
        assert list(reader) == [
            {
                "op_name": "aten.mm.default",
                "kernel_type": "MatMulV2",
                "simulation_shapes": "[[64, 128], [128, 256]]",
                "dur_us": "20",
                "source": "MEASURED",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            }
        ]

    def test_miss_event_no_kernel_type(self, tmp_path):
        trace = _make_trace(
            tmp_path,
            [
                _x("aten.add.Tensor", 2, simulation_shapes="[[1, 4, 7168]]"),
            ],
        )
        out = tmp_path / "out.csv"
        trace_to_csv_module.trace_to_csv(trace, str(out))

        _, rows = _read_csv(out)
        assert rows == [
            {
                "op_name": "aten.add.Tensor",
                "kernel_type": "",
                "simulation_shapes": "[[1, 4, 7168]]",
                "dur_us": "2",
                "source": "",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            }
        ]

    def test_skips_metadata_events(self, tmp_path):
        trace = _make_trace(
            tmp_path,
            [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 0,
                    "args": {"name": "empirical"},
                },
                {"name": "begin", "ph": "B", "pid": 0, "tid": 0, "ts": 1},
                {"name": "end", "ph": "E", "pid": 0, "tid": 0, "ts": 2},
                {"name": "missing_phase", "pid": 0, "tid": 0, "ts": 3},
                _x(
                    "aten.mm.default",
                    20,
                    kernel_type="MatMulV2",
                    source="MEASURED",
                    simulation_shapes="[]",
                ),
            ],
        )
        out = tmp_path / "out.csv"
        trace_to_csv_module.trace_to_csv(trace, str(out))

        _, rows = _read_csv(out)
        assert rows == [
            {
                "op_name": "aten.mm.default",
                "kernel_type": "MatMulV2",
                "simulation_shapes": "[]",
                "dur_us": "20",
                "source": "MEASURED",
                "confidence": "",
                "composite": "",
                "sub_kernel_durations": "",
                "kernel_shapes": "",
                "shape_match_rule": "",
                "sub_kernel_shapes": "",
            }
        ]
