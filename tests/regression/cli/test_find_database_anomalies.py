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

"""Regression tests for the read-only profiling database anomaly audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tensor_cast.performance_model.profiling_database.interpolation_index import (
    CandidateGroup,
    CandidatePoint,
)
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tools.perf_data_collection import find_database_anomalies
from tools.perf_data_collection.find_database_anomalies import (
    _LooContext,
    _csv_safe_cell,
    _loo_observations,
    _relative_latency_error,
    build_strict_row_signature,
    main,
    scan_database,
    write_reports,
)


MAPPING = """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
    alternate_kernel_types:
  "profiling.Cast":
    kernel_type: Cast
  "profiling.Duplicate":
    kernel_type: Duplicate
"""

HEADERS = [
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
    "Profiling Average Duration(us)",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    input_shapes: str,
    output_shapes: str,
    latency: str,
    *,
    output_dtype: str = "DT_BF16",
    output_format: str = "ND",
    profiling_latency: str = "0",
) -> dict[str, str]:
    return {
        "OP State": "static",
        "Accelerator Core": "AI_CORE",
        "Input Shapes": input_shapes,
        "Input Data Types": "DT_BF16;DT_BF16",
        "Input Formats": "ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": output_dtype,
        "Output Formats": output_format,
        "Average Duration(us)": latency,
        "Profiling Average Duration(us)": profiling_latency,
    }


@pytest.fixture
def audit_database(tmp_path: Path) -> Path:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(MAPPING, encoding="utf-8")

    _write_csv(
        database / "MatMulV2.csv",
        [
            _row("1,4;8,4", "1,8", "10"),
            _row("2,4;8,4", "2,8", "100"),
            _row("3,4;8,4", "3,8", "30"),
        ],
    )
    _write_csv(
        database / "Cast.csv",
        [
            _row("1,4", "1,4", "0", output_dtype="FLOAT"),
            _row("1,4", "1,4", "-1", output_dtype="INT8"),
        ],
    )
    _write_csv(
        database / "Duplicate.csv",
        [
            _row("4,4", "4,4", "5"),
            _row("4,4", "4,4", "7"),
        ],
    )
    _write_csv(
        database / "SameLatency.csv",
        [
            _row("5,5", "5,5", "5"),
            _row("5,5", "5,5", "0", profiling_latency="5"),
        ],
    )
    return database


def test_strict_signature_keeps_output_and_unknown_metadata() -> None:
    base = _row("1,4", "1,4", "1", output_dtype="FLOAT")
    same_semantics = dict(base, **{"Average Duration(us)": "2"})
    different_counter = dict(base, **{"Average aicore_time(us)": "999"})
    different_provenance = dict(base, **{"Runtime case_id": "case-2"})
    different_output = dict(base, **{"Output Data Types": "INT8"})
    different_metadata = dict(base, **{"Future Runtime Mode": "fast"})
    different_prefixed_metadata = dict(base, **{"Profiling Runtime Mode": "fast"})
    malformed_shape = dict(base, **{"Input Shapes": "1,,4"})

    signature = build_strict_row_signature("Cast", base, {"device": "TEST"})

    assert signature == build_strict_row_signature("Cast", same_semantics, {"device": "TEST"})
    assert signature == build_strict_row_signature("Cast", different_counter, {"device": "TEST"})
    assert signature == build_strict_row_signature("Cast", different_provenance, {"device": "TEST"})
    assert signature != build_strict_row_signature("Cast", different_output, {"device": "TEST"})
    assert signature != build_strict_row_signature("Cast", different_metadata, {"device": "TEST"})
    assert signature != build_strict_row_signature("Cast", different_prefixed_metadata, {"device": "TEST"})
    assert signature != build_strict_row_signature("Cast", malformed_shape, {"device": "TEST"})


@pytest.mark.parametrize("prefix", ["", " ", "\t", "\r", "\u2003"])
def test_csv_formula_cells_are_escaped_after_leading_whitespace(prefix: str) -> None:
    value = f"{prefix}=1+1"
    assert _csv_safe_cell(value) == "'" + value


def test_relative_latency_error_is_direction_neutral() -> None:
    assert _relative_latency_error(50.0, 10.0) == pytest.approx(4.0)
    assert _relative_latency_error(10.0, 50.0) == pytest.approx(4.0)


def test_scan_finds_deterministic_errors_duplicate_conflict_and_production_loo(
    audit_database: Path,
) -> None:
    result = scan_database(audit_database, residual_threshold=1.0)
    evidence_by_location = {
        (candidate.csv_path, candidate.row_number): set(candidate.evidence) for candidate in result.candidates
    }

    assert result.files_scanned == 4
    assert result.rows_scanned == 9
    assert result.residual_threshold == 1.0
    assert "UNMEASURED_PLACEHOLDER" in evidence_by_location[("Cast.csv", 2)]
    assert "INVALID_LATENCY_VALUE" in evidence_by_location[("Cast.csv", 3)]
    assert "DUPLICATE_LATENCY_CONFLICT" in evidence_by_location[("Duplicate.csv", 2)]
    assert "DUPLICATE_LATENCY_CONFLICT" in evidence_by_location[("Duplicate.csv", 3)]
    assert "DUPLICATE_LATENCY_CONFLICT" in evidence_by_location[("SameLatency.csv", 2)]
    assert "DUPLICATE_LATENCY_CONFLICT" in evidence_by_location[("SameLatency.csv", 3)]
    assert "MatMulV2" in result.loo_indexed_kernels
    assert "SameLatency" in result.deterministic_only_csvs

    loo_candidate = next(
        candidate
        for candidate in result.candidates
        if candidate.csv_path == "MatMulV2.csv" and candidate.row_number == 3
    )
    assert "LOCAL_LOO_RESIDUAL" in loo_candidate.evidence
    assert loo_candidate.predicted_latency_us == pytest.approx(20.0)
    assert loo_candidate.relative_error == pytest.approx(4.0)
    assert loo_candidate.interpolation_method == "linear_1d"
    assert loo_candidate.interpolation_axes == "M"

    at_threshold = scan_database(audit_database, residual_threshold=4.0)
    assert sum("LOCAL_LOO_RESIDUAL" in candidate.evidence for candidate in at_threshold.candidates) == 1

    above_threshold = scan_database(audit_database, residual_threshold=4.000001)
    assert not any("LOCAL_LOO_RESIDUAL" in candidate.evidence for candidate in above_threshold.candidates)

    repeated = scan_database(audit_database, residual_threshold=1.0)
    assert [(candidate.csv_path, candidate.row_number, candidate.evidence) for candidate in result.candidates] == [
        (candidate.csv_path, candidate.row_number, candidate.evidence) for candidate in repeated.candidates
    ]


def test_scan_reports_lossy_collection_signature_collisions(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "profiling.Cast":
    kernel_type: Cast
""",
        encoding="utf-8",
    )
    _write_csv(
        database / "Cast.csv",
        [
            _row("1,4", "1,4", "10", output_format="ND"),
            _row("1,4", "1,4", "12", output_format="NZ"),
        ],
    )

    result = scan_database(database, residual_threshold=None)
    collisions = [
        candidate for candidate in result.candidates if "COLLECTION_SIGNATURE_COLLISION" in candidate.evidence
    ]

    assert [(candidate.csv_path, candidate.row_number) for candidate in collisions] == [
        ("Cast.csv", 2),
        ("Cast.csv", 3),
    ]
    assert {candidate.status for candidate in collisions} == {"REVIEW_REGIME"}
    assert all("different strict row semantics" in candidate.reason for candidate in collisions)
    assert all("collection signature" in candidate.recommended_action for candidate in collisions)

    output = tmp_path / "audit"
    write_reports(result, output, remeasure_limit=10)
    summary = (output / "anomaly_summary.md").read_text(encoding="utf-8")
    assert "采集去重规则无法区分的数据行 / 算子 | 2 / 1" in summary
    with (output / "remeasure_manifest.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == []


def test_scan_rejects_git_lfs_pointer_as_one_file_error(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        """\
version: "test"
device: TEST_DEVICE
operator_mappings:
  "profiling.Cast":
    kernel_type: Cast
""",
        encoding="utf-8",
    )
    (database / "Cast.csv").write_text(
        """\
version https://git-lfs.github.com/spec/v1
oid sha256:4ca09ca0009de359ebc54021837466e736705aa0e148a92b034f366b91f23643
size 214708
""",
        encoding="utf-8",
    )

    result = scan_database(database, residual_threshold=None)

    assert result.rows_scanned == 0
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.csv_path == "Cast.csv"
    assert candidate.row_number == 1
    assert candidate.evidence == ["INVALID_CSV"]
    assert "Git LFS pointer" in candidate.reason


def test_cli_writes_stable_reports_without_modifying_database(
    audit_database: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    source_before = {path.name: path.read_bytes() for path in sorted(audit_database.glob("*.csv"))}

    assert (
        main(
            [
                "--database-path",
                str(audit_database),
                "--output-dir",
                str(output),
                "--residual-threshold",
                "1.0",
                "--remeasure-limit",
                "2",
            ]
        )
        == 0
    )

    candidates = output / "anomaly_candidates.csv"
    summary = output / "anomaly_summary.md"
    manifest = output / "remeasure_manifest.csv"
    first_contents = {path.name: path.read_bytes() for path in (candidates, summary, manifest)}

    assert (
        main(
            [
                "--database-path",
                str(audit_database),
                "--output-dir",
                str(output),
                "--residual-threshold",
                "1.0",
                "--remeasure-limit",
                "2",
            ]
        )
        == 0
    )
    assert first_contents == {path.name: path.read_bytes() for path in (candidates, summary, manifest)}
    assert source_before == {path.name: path.read_bytes() for path in sorted(audit_database.glob("*.csv"))}
    summary_text = summary.read_text(encoding="utf-8")
    assert "固定性能复测候选阈值 | 1（预测值与实测值至少相差 2 倍）" in summary_text
    assert "已确认的数据完整性问题" in summary_text
    assert "达到固定阈值的性能复测候选" in summary_text
    assert "Reported candidates" not in summary_text


@pytest.mark.parametrize("residual_threshold", [-1.0, float("inf"), float("nan")])
def test_scan_rejects_invalid_residual_threshold(
    audit_database: Path,
    residual_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        scan_database(audit_database, residual_threshold=residual_threshold)


def test_cli_skip_loo_runs_only_deterministic_checks(
    audit_database: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"

    assert (
        main(
            [
                "--database-path",
                str(audit_database),
                "--output-dir",
                str(output),
                "--skip-loo",
            ]
        )
        == 0
    )

    summary = (output / "anomaly_summary.md").read_text(encoding="utf-8")
    assert "固定性能复测候选阈值 | 未运行 LOO" in summary
    assert "达到固定残差阈值的数据行 | 0" in summary


def test_cli_rejects_output_inside_database(audit_database: Path) -> None:
    with pytest.raises(ValueError, match="outside the database directory"):
        main(
            [
                "--database-path",
                str(audit_database),
                "--output-dir",
                str(audit_database / "audit"),
            ]
        )


def test_attention_loo_reuses_q_token_matcher_and_sqrt_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    regime = (("dtype", "BF16"),)
    points = [
        CandidatePoint("Attention", {"seq": 1.0, "q_tokens": 16.0}, 10.0, regime, row_index=0),
        CandidatePoint("Attention", {"seq": 4.0, "q_tokens": 1.0}, 100.0, regime, row_index=1),
        CandidatePoint("Attention", {"seq": 9.0, "q_tokens": 16.0}, 30.0, regime, row_index=2),
    ]
    index = SimpleNamespace(_candidate_groups={regime: CandidateGroup(regime, points)}, points=points)
    source = SimpleNamespace(
        _kernel_overrides={},
        _attention_q_tokens_match=InterpolatingDataSource._attention_q_tokens_match,
        _latency_column_pure_candidate_group_attempts=(
            InterpolatingDataSource._latency_column_pure_candidate_group_attempts
        ),
        _candidate_latency_column_selection=(InterpolatingDataSource._candidate_latency_column_selection),
        _mark_sqrt_interpolation=InterpolatingDataSource._mark_sqrt_interpolation,
        _sqrt_seq_group=InterpolatingDataSource._sqrt_seq_group,
    )
    monkeypatch.setattr(
        find_database_anomalies,
        "_context_indexes",
        lambda _source, _context: [(index, (("seq",),), "sqrt_seq")],
    )

    analysis = _loo_observations(
        source,
        [_LooContext("attention", "Attention", "Attention")],
    )

    target_result = next(result for _error, _kernel, target, result in analysis.observations if target.row_index == 1)
    assert analysis.evaluated == 3
    assert analysis.predicted >= 1
    assert analysis.indexed_kernels == ("Attention",)
    assert target_result.latency_us == pytest.approx(20.0)
    assert target_result.method == "linear_1d_sqrt"


def test_loo_never_compares_different_latency_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    preferred = "preferred_latency_column"
    alternate = "alternate_latency_column"

    def point(axis: float, latency: float, selection: str, row_index: int, regime_name: str) -> CandidatePoint:
        regime = (("case", regime_name),)
        return CandidatePoint(
            "Compute",
            {"axis_0": axis},
            latency,
            regime,
            row_index=row_index,
            row_meta={"latency_column_selection": selection},
        )

    groups = {}
    for regime_name, target_source, neighbor_source, offset in (
        ("alternate_target", alternate, preferred, 0),
        ("preferred_target", preferred, alternate, 3),
    ):
        points = [
            point(1.0, 10.0, neighbor_source, offset, regime_name),
            point(2.0, 100.0, target_source, offset + 1, regime_name),
            point(3.0, 30.0, neighbor_source, offset + 2, regime_name),
        ]
        groups[points[0].regime_key] = CandidateGroup(points[0].regime_key, points)
    index = SimpleNamespace(
        _candidate_groups=groups, points=[point for group in groups.values() for point in group.points]
    )
    source = SimpleNamespace(
        _kernel_overrides={},
        _candidate_latency_column_selection=(InterpolatingDataSource._candidate_latency_column_selection),
        _latency_column_pure_candidate_group_attempts=(
            InterpolatingDataSource._latency_column_pure_candidate_group_attempts
        ),
    )
    monkeypatch.setattr(
        find_database_anomalies,
        "_context_indexes",
        lambda _source, _context: [(index, (("axis_0",),), None)],
    )

    analysis = _loo_observations(
        source,
        [_LooContext("generic_compute", "Compute", "Compute")],
    )

    assert analysis.observations == []
    assert analysis.evaluated == 6
    assert analysis.predicted == 0
    assert analysis.indexed_kernels == ("Compute",)
    assert len(analysis.row_outcomes) == 6
    assert all(outcome.predictions == 0 for outcome in analysis.row_outcomes.values())


def test_loo_evaluates_each_distinct_mapping_context(monkeypatch: pytest.MonkeyPatch) -> None:
    regime = (("dtype", "BF16"),)
    points = [
        CandidatePoint("Compute", {"axis_0": axis}, latency, regime, row_index=index)
        for index, (axis, latency) in enumerate(((1.0, 10.0), (2.0, 100.0), (3.0, 30.0)))
    ]
    index = SimpleNamespace(_candidate_groups={regime: CandidateGroup(regime, points)}, points=points)
    source = SimpleNamespace(
        _kernel_overrides={},
        _candidate_latency_column_selection=(InterpolatingDataSource._candidate_latency_column_selection),
        _latency_column_pure_candidate_group_attempts=(
            InterpolatingDataSource._latency_column_pure_candidate_group_attempts
        ),
    )
    monkeypatch.setattr(
        find_database_anomalies,
        "_context_indexes",
        lambda _source, _context: [(index, (("axis_0",),), None)],
    )

    analysis = _loo_observations(
        source,
        [
            _LooContext("generic_compute", "Compute", "PolicyA"),
            _LooContext("generic_compute", "Compute", "PolicyB"),
        ],
    )

    assert analysis.evaluated == 6
    assert analysis.predicted == 2
    assert len(analysis.observations) == 2


def test_matched_point_payload_lists_duplicate_contributors() -> None:
    point = CandidatePoint(
        "Compute",
        {"axis_0": 1.0},
        15.0,
        (("dtype", "BF16"),),
        row_index=0,
        row_meta={
            "latency_column": "Average Duration(us)",
            "duplicate_row_indices": [0, 2],
            "duplicate_row_meta": [
                {"raw_latency_us": 10.0, "latency_column": "Average Duration(us)"},
                {"raw_latency_us": 20.0, "latency_column": "Average Duration(us)"},
            ],
            "aggregation": "median",
        },
    )

    assert find_database_anomalies._matched_point_payload(point) == {
        "row_number": 2,
        "latency_us": 15.0,
        "latency_column": "Average Duration(us)",
        "aggregation": "median",
        "contributors": [
            {"row_number": 2, "latency_us": 10.0, "latency_column": "Average Duration(us)"},
            {"row_number": 4, "latency_us": 20.0, "latency_column": "Average Duration(us)"},
        ],
    }


def test_scan_rejects_missing_mapping_and_duplicate_headers(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    with pytest.raises(ValueError, match="op_mapping.yaml does not exist"):
        scan_database(database)

    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {broken: {kernel_type: Broken}}\n",
        encoding="utf-8",
    )
    (database / "Broken.csv").write_text(
        "Input Shapes,Input Shapes,Average Duration(us)\n1,1,1\n",
        encoding="utf-8",
    )
    result = scan_database(database, residual_threshold=None)

    assert result.files_scanned == 1
    assert result.rows_scanned == 0
    assert result.candidates[0].status == "CONFIRMED_FORMAT_ERROR"
    assert result.candidates[0].evidence == ["INVALID_CSV"]
    output = tmp_path / "audit"
    write_reports(result, output, remeasure_limit=1)
    with (output / "remeasure_manifest.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == []

    (database / "op_mapping.yaml").write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        scan_database(database)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ("version: test\n", "operator_mappings must be a non-empty mapping"),
        ("operator_mappings: {}\n", "operator_mappings must be a non-empty mapping"),
        ("operator_mappings: []\n", "operator_mappings must be a non-empty mapping"),
        ("operator_mappings: {bad: invalid}\n", "operator_mappings\\['bad'\\] must be a mapping"),
        (
            'operator_mappings: {bad: {composite: "yes"}}\n',
            "composite must be a boolean",
        ),
        ("operator_mappings: {bad: {}}\n", "kernel_type must be a non-empty string"),
        (
            "operator_mappings: {bad: {kernel_type: Compute, alternate_kernel_types: Alternate}}\n",
            "alternate_kernel_types must be a list",
        ),
        (
            "operator_mappings: {bad: {kernel_type: Compute, tc_input_count: 0}}\n",
            "tc_input_count must be a positive integer",
        ),
        (
            "operator_mappings: {bad: {kernel_type: Compute, query_mode: []}}\n",
            "query_mode must be a non-empty string",
        ),
    ],
)
def test_scan_rejects_invalid_loo_mapping_schema(tmp_path: Path, mapping: str, message: str) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(mapping, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        scan_database(database, residual_threshold=0.0)


def test_scan_reports_mapped_invalid_csv_without_aborting(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {broken: {kernel_type: Broken}}\n",
        encoding="utf-8",
    )
    (database / "Broken.csv").write_text(
        "Input Shapes,Input Shapes,Average Duration(us)\n1,1,1\n",
        encoding="utf-8",
    )

    result = scan_database(database, residual_threshold=0.0)

    assert result.loo_evaluated == 0
    assert result.deterministic_only_csvs == ("Broken",)
    assert result.candidates[0].evidence == ["INVALID_CSV"]


def test_scan_rejects_malformed_shape_metadata(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {malformed: {kernel_type: Malformed}}\n",
        encoding="utf-8",
    )
    _write_csv(database / "Malformed.csv", [_row("1,,4;8,4", "1,8", "10")])
    _write_csv(database / "ScalarInput.csv", [_row("1,4;", "1,4", "10")])

    result = scan_database(database, residual_threshold=0.0)

    candidate = result.candidates[0]
    assert len(result.candidates) == 1
    assert "Malformed" in result.deterministic_only_csvs
    assert candidate.evidence == ["INVALID_SHAPE_METADATA"]
    assert "empty dimension" in candidate.reason


def test_scan_detects_database_changes_during_scan(
    audit_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(("before", "after"))
    monkeypatch.setattr(find_database_anomalies, "_database_snapshot_hash", lambda *_args: next(snapshots))

    with pytest.raises(RuntimeError, match="changed during scan"):
        scan_database(audit_database, residual_threshold=None)


def test_scan_rejects_symlinked_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
    outside = tmp_path / "Outside.csv"
    _write_csv(outside, [_row("1,4", "1,4", "10")])
    link = database / "Linked.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        _write_csv(link, [_row("1,4", "1,4", "10")])
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path.name == link.name or original_is_symlink(path),
        )

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        scan_database(database, residual_threshold=None)


@pytest.mark.parametrize("residual_threshold", [None, 0.0])
def test_scan_reports_unclosed_csv_quote_without_aborting(
    tmp_path: Path,
    residual_threshold: float | None,
) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {broken: {kernel_type: Broken}}\n",
        encoding="utf-8",
    )
    (database / "Broken.csv").write_text(
        'Input Shapes,Average Duration(us)\n"1,4,10\n',
        encoding="utf-8",
    )
    _write_csv(database / "Unrelated.csv", [_row("1,4", "1,4", "10")])

    result = scan_database(database, residual_threshold=residual_threshold)

    assert result.files_scanned == 2
    assert any(candidate.evidence == ["INVALID_CSV"] for candidate in result.candidates)


def test_invalid_row_does_not_disable_valid_rows_in_same_kernel(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {mm: {kernel_type: MatMulV2}}\n",
        encoding="utf-8",
    )
    _write_csv(
        database / "MatMulV2.csv",
        [
            _row("1,4;8,4", "1,8", "10"),
            _row("2,4;8,4", "2,8", "-1"),
            _row("3,4;8,4", "3,8", "30"),
            _row("4,4;8,4", "4,8", "40"),
        ],
    )

    result = scan_database(database, residual_threshold=0.0)

    assert result.loo_predicted > 0
    assert "MatMulV2" in result.loo_predicted_kernels
    invalid = next(candidate for candidate in result.candidates if candidate.row_number == 3)
    assert "INVALID_LATENCY_VALUE" in invalid.evidence


def test_scan_reports_loo_abstentions_and_unique_row_denominators(audit_database: Path) -> None:
    result = scan_database(audit_database, residual_threshold=0.0)

    assert result.loo_rows_attempted >= result.loo_rows_predicted
    assert result.loo_rows_abstained == result.loo_rows_attempted - result.loo_rows_predicted
    assert result.loo_rows_abstained > 0
    assert any(candidate.status == "INSUFFICIENT_EVIDENCE" for candidate in result.candidates)


@pytest.mark.parametrize("residual_threshold", [None, 0.0])
def test_scan_reports_missing_primary_csv(
    tmp_path: Path,
    residual_threshold: float | None,
) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {missing: {kernel_type: MissingKernel, alternate_kernel_types: [AvailableAlternate]}}\n",
        encoding="utf-8",
    )
    _write_csv(database / "AvailableAlternate.csv", [_row("1,4", "1,4", "10")])

    result = scan_database(database, residual_threshold=residual_threshold)

    candidate = next(candidate for candidate in result.candidates if "MISSING_CSV" in candidate.evidence)
    assert candidate.csv_path == "MissingKernel.csv"
    assert candidate.status == "CONFIRMED_FORMAT_ERROR"
    assert candidate.evidence == ["MISSING_CSV"]
    assert (
        main(
            [
                "--database-path",
                str(database),
                "--output-dir",
                str(tmp_path / f"audit-{residual_threshold}"),
                *(["--skip-loo"] if residual_threshold is None else ["--residual-threshold", str(residual_threshold)]),
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "kernel_type",
    ["../Outside", "subdir/Outside", r"C:\\Outside", r"\\server\\share", "Outside.csv"],
)
def test_scan_rejects_mapping_kernel_paths(tmp_path: Path, kernel_type: str) -> None:
    database = tmp_path / "database"
    database.mkdir()
    mapping = {"operator_mappings": {"bad": {"kernel_type": kernel_type}}}
    (database / "op_mapping.yaml").write_text(
        find_database_anomalies.yaml.safe_dump(mapping),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single CSV stem"):
        scan_database(database, residual_threshold=0.0)


def test_scan_rejects_mapping_alternate_kernel_path(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {bad: {kernel_type: Primary, alternate_kernel_types: ['../Outside']}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single CSV stem"):
        scan_database(database, residual_threshold=0.0)


def test_production_index_parse_error_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        find_database_anomalies,
        "_context_indexes",
        lambda *_args: (_ for _ in ()).throw(pd.errors.ParserError("broken production CSV")),
    )

    analysis = _loo_observations(
        SimpleNamespace(),
        [_LooContext("generic_compute", "Broken", "Broken")],
    )

    assert analysis.observations == []
    assert "Broken" in analysis.index_errors


def test_scan_turns_production_index_parse_error_into_file_finding(
    audit_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = find_database_anomalies._context_indexes

    def fail_one_context(source, context):
        if context.kernel_type == "MatMulV2":
            raise pd.errors.ParserError("broken production CSV")
        return original(source, context)

    monkeypatch.setattr(find_database_anomalies, "_context_indexes", fail_one_context)

    result = scan_database(audit_database, residual_threshold=0.0)

    candidate = next(
        candidate
        for candidate in result.candidates
        if candidate.csv_path == "MatMulV2.csv" and candidate.row_number == 1
    )
    assert candidate.evidence == ["INVALID_CSV"]
    assert "production index rejected CSV" in candidate.reason


def test_overlapping_thresholded_residuals_are_reviewed_as_one_regime(tmp_path: Path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    (database / "op_mapping.yaml").write_text(
        "operator_mappings: {mm: {kernel_type: MatMulV2}}\n",
        encoding="utf-8",
    )
    _write_csv(
        database / "MatMulV2.csv",
        [
            _row(f"{axis},4;8,4", f"{axis},8", latency)
            for axis, latency in ((1, "10"), (2, "100"), (3, "30"), (4, "120"), (5, "50"))
        ],
    )

    result = scan_database(database, residual_threshold=1.0)
    residuals = [candidate for candidate in result.candidates if "LOCAL_LOO_RESIDUAL" in candidate.evidence]

    assert len(residuals) >= 2
    assert all(candidate.status == "REVIEW_REGIME" for candidate in residuals)


def test_latency_policy_details_are_preserved(
    audit_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = find_database_anomalies.ProfilingDataSource._effective_row_latency

    def effective_with_fallback(self, row, kernel_type, preferred_col):
        selected = original(self, row, kernel_type, preferred_col)
        if selected is None or kernel_type != "MatMulV2":
            return selected
        return (
            "Effective Duration(us)",
            selected[1] / 2,
            {
                "latency_selection": "profiling_core_envelope_fallback",
                "raw_latency_us": selected[1],
            },
        )

    monkeypatch.setattr(
        find_database_anomalies.ProfilingDataSource,
        "_effective_row_latency",
        effective_with_fallback,
    )

    result = scan_database(audit_database, residual_threshold=None)
    candidate = next(candidate for candidate in result.candidates if "LATENCY_POLICY_DIVERGENCE" in candidate.evidence)

    assert json.loads(candidate.exact_latency_details)["latency_selection"] == ("profiling_core_envelope_fallback")
    assert json.loads(candidate.candidate_latency_details)["latency_column_selection"]
    assert "registered exact policy" in candidate.reason
    assert candidate.status == "REVIEW_REGIME"
    assert candidate.recommended_action == ("Align exact and interpolation latency policies before using this row.")

    output = tmp_path / "audit"
    write_reports(result, output, remeasure_limit=len(result.candidates))
    with (output / "remeasure_manifest.csv").open(encoding="utf-8", newline="") as file:
        manifest_signatures = {row["strict_row_signature"] for row in csv.DictReader(file)}
    assert candidate.strict_row_signature not in manifest_signatures


def test_report_package_rolls_back_if_directory_publish_fails(
    audit_database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "audit"
    first_result = scan_database(audit_database, residual_threshold=None)
    write_reports(first_result, output, remeasure_limit=1)
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    original_replace = find_database_anomalies.os.replace
    failure_injected = False

    def fail_staging_publish(source, destination):
        nonlocal failure_injected
        source_path = Path(source)
        if (
            not failure_injected
            and source_path.is_dir()
            and source_path.name.startswith(".audit.staging-")
            and Path(destination) == output
        ):
            failure_injected = True
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(find_database_anomalies.os, "replace", fail_staging_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        write_reports(first_result, output, remeasure_limit=1)

    assert previous == {path.name: path.read_bytes() for path in output.iterdir()}
    assert not list(tmp_path.glob(".audit.staging-*"))
    assert not list(tmp_path.glob(".audit.previous-*"))
