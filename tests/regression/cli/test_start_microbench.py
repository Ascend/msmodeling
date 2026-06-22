"""Tests for tools/perf_data_collection/start_microbench.py.

Unit tests can run without NPU hardware.
End-to-end tests require NPU and are marked with @pytest.mark.npu.
"""

import csv
import importlib
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

# Add perf_data_collection to path for imports
PERF_DATA_COLLECTION_DIR = Path(__file__).resolve().parents[3] / "tools" / "perf_data_collection"
if str(PERF_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DATA_COLLECTION_DIR))

start_microbench = importlib.import_module("start_microbench")
aggregate_summary = start_microbench.aggregate_summary
build_msprof_cmd = start_microbench.build_msprof_cmd
GapRecord = start_microbench.GapRecord
get_cols = start_microbench.get_cols
get_sig = start_microbench.get_sig
md_table = start_microbench.md_table
print_report = start_microbench.print_report
run_msprof = start_microbench.run_msprof
should_skip_dispatch_ffn_msprof = start_microbench.should_skip_dispatch_ffn_msprof
update_csv = start_microbench.update_csv
update_db = start_microbench.update_db
UpdateResult = start_microbench.UpdateResult


# =============================================================================
# Unit Tests (No NPU Required)
# =============================================================================
class TestMdTable:
    """Tests for md_table function."""

    def test_empty_rows_returns_none(self):
        """Empty rows should return '_None_'."""
        result = md_table(["Col1", "Col2"], [])
        assert result == "_None_"

    def test_single_row(self):
        """Single row table should format correctly."""
        result = md_table(["Name", "Value"], [["foo", "bar"]])
        lines = result.split("\n")
        assert "Name" in lines[0]
        assert "foo" in lines[2]
        assert "bar" in lines[2]
        assert "---" in lines[1]  # separator line

    def test_multiple_rows(self):
        """Multiple rows should format correctly."""
        result = md_table(["Op", "Count"], [["Add", "5"], ["MatMul", "3"]])
        lines = result.split("\n")
        assert "Op" in lines[0]
        assert "Add" in lines[2]
        assert "MatMul" in lines[3]
        assert "5" in lines[2]

    def test_column_width_alignment(self):
        """Columns should align to widest value."""
        result = md_table(["Name", "Value"], [["a", "x"], ["longer_name", "y"]])
        lines = result.split("\n")
        # All lines should have same length for each column
        assert len(lines) == 4  # header, separator, 2 data rows


class TestDispatchFfnSkip:
    def test_skip_dfc_msprof_when_ep_size_exceeds_visible_devices(self):
        assert should_skip_dispatch_ffn_msprof(
            ["DispatchFFNCombine"],
            ep_size=16,
            nproc_per_node=None,
            visible_devices=2,
            update_mode="missing-only",
            has_prof_path=False,
        )

    def test_skip_dfc_msprof_for_full_run_when_ep_size_exceeds_visible_devices(self):
        assert should_skip_dispatch_ffn_msprof(
            None,
            ep_size=16,
            nproc_per_node=None,
            visible_devices=2,
            update_mode="missing-only",
            has_prof_path=False,
        )

    def test_do_not_skip_multinode_dfc_when_local_ranks_fit_visible_devices(self):
        assert not should_skip_dispatch_ffn_msprof(
            ["DispatchFFNCombine"],
            ep_size=32,
            nproc_per_node=16,
            visible_devices=16,
            update_mode="missing-only",
            has_prof_path=False,
        )

    def test_do_not_skip_dfc_msprof_for_mixed_operator_runs(self):
        assert not should_skip_dispatch_ffn_msprof(
            ["DispatchFFNCombine", "MatMulV2"],
            ep_size=16,
            nproc_per_node=None,
            visible_devices=2,
            update_mode="missing-only",
            has_prof_path=False,
        )


class TestGetCols:
    """Tests for get_cols function."""

    def test_none_returns_full_schema(self):
        """None input should return full default schema (62 columns)."""
        cols = get_cols(None)
        assert len(cols) == 62
        assert "Average Duration(us)" in cols
        assert "Profiling Average Duration(us)" in cols
        assert "Profiling Median Duration(us)" in cols
        assert "Profiling Std Duration(us)" in cols
        assert "MicroBench aicore_time(us)" in cols

    def test_excludes_legacy_columns(self):
        """Should exclude MicroBench Task/Kernel Duration columns."""
        cols = get_cols(
            [
                "OP State",
                "Input Shapes",
                "Average Duration(us)",
                "MicroBench Task Duration(us)",
                "MicroBench Kernel Duration(us)",
            ]
        )
        assert "MicroBench Task Duration(us)" not in cols
        assert "MicroBench Kernel Duration(us)" not in cols

    def test_converts_legacy_mb_dur_to_new(self):
        """Legacy 'MicroBench Duration(us)' should become 'Average Duration(us)'."""
        cols = get_cols(["OP State", "Input Shapes", "MicroBench Duration(us)"])
        assert "Average Duration(us)" in cols
        assert "MicroBench Duration(us)" not in cols

    def test_inserts_mb_cols_before_profiling_cols(self):
        """MicroBench columns should be inserted before their Profiling counterparts."""
        cols = get_cols(
            [
                "OP State",
                "Input Shapes",
                "Average Duration(us)",
                "Profiling Average aicore_time(us)",
            ]
        )
        mb_idx = cols.index("MicroBench aicore_time(us)")
        prof_idx = cols.index("Profiling Average aicore_time(us)")
        assert mb_idx < prof_idx


class TestGetSig:
    """Tests for profiler signature normalization."""

    def test_matmul_uses_file_op_name_when_op_state_is_dynamic(self):
        csv_row = {
            "OP State": "dynamic",
            "Input Shapes": "5,6144;2048,6144",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "5,2048",
            "Output Data Types": "DT_BF16",
        }
        profiler_row = {
            "OP Type": "MatMulV2",
            "Input Shapes": "5,6144;2048,6144",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "5,2048",
            "Output Data Types": "DT_BF16",
        }

        assert get_sig(csv_row, op_name="MatMulV2") == get_sig(profiler_row, op_name="MatMulV2")

    def test_matmul_family_accepts_transposed_profiler_rhs(self):
        csv_row = {
            "OP State": "static",
            "Input Shapes": "24,512;4096,512",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "24,4096",
            "Output Data Types": "DT_BF16",
        }
        profiler_row = {
            "OP Type": "MatMulCommon",
            "Input Shapes": "24,512;512,4096",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "24,4096",
            "Output Data Types": "DT_BF16",
        }

        assert get_sig(csv_row, op_name="MatMulV2") == get_sig(profiler_row, op_name="MatMulCommon")

    def test_index_ignores_csv_metadata_slots(self):
        csv_row = {
            "OP State": "dynamic",
            "Input Shapes": "64,1728;1;2;16",
            "Input Data Types": "DT_BF16;INT64;INT64;INT64",
            "Input Formats": "ND;ND;ND;ND",
            "Output Shapes": "16,1728",
            "Output Data Types": "DT_BF16",
        }
        profiler_row = {
            "OP State": "Index",
            "Input Shapes": "64,1728;16",
            "Input Data Types": "DT_BF16;INT64",
            "Input Formats": "ND;ND",
            "Output Shapes": "16,1728",
            "Output Data Types": "DT_BF16",
        }

        assert get_sig(csv_row, op_name="Index") == get_sig(profiler_row, op_name="Index")

    def test_slice_and_transpose_ignore_parameter_slots(self):
        slice_row = {
            "OP State": "Slice",
            "Input Shapes": "4,768;2;2",
            "Input Data Types": "DT_BF16;INT64;INT64",
            "Input Formats": "ND;ND;ND",
            "Output Shapes": "2,768",
            "Output Data Types": "DT_BF16",
        }
        slice_profiler_row = {
            "OP State": "Slice",
            "Input Shapes": "4,768",
            "Input Data Types": "DT_BF16",
            "Input Formats": "ND",
            "Output Shapes": "2,768",
            "Output Data Types": "DT_BF16",
        }
        transpose_row = {
            "OP State": "Transpose",
            "Input Shapes": "4,896;2",
            "Input Data Types": "DT_BF16;INT64",
            "Input Formats": "ND;ND",
            "Output Shapes": "896,4",
            "Output Data Types": "DT_BF16",
        }
        transpose_profiler_row = {
            "OP State": "Transpose",
            "Input Shapes": "4,896",
            "Input Data Types": "DT_BF16",
            "Input Formats": "ND",
            "Output Shapes": "896,4",
            "Output Data Types": "DT_BF16",
        }

        assert get_sig(slice_row, op_name="Slice") == get_sig(slice_profiler_row, op_name="Slice")
        assert get_sig(transpose_row, op_name="Transpose") == get_sig(transpose_profiler_row, op_name="Transpose")


class TestRunMsprof:
    """Tests for msprof wrapper behavior."""

    @staticmethod
    def _args(database_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            fail_fast=False,
            database_path=database_path,
            device="ATLAS_800_A3_752T_128G_DIE",
            vllm_version=None,
            torch_version=None,
            cann_version=None,
            repeat_count=1,
            update_mode="all",
            dispatch_ffn_combine_ep_size=16,
            dispatch_ffn_combine_nproc_per_node=None,
            dispatch_ffn_combine_nnodes=1,
            dispatch_ffn_combine_node_rank=0,
            dispatch_ffn_combine_master_addr="127.0.0.1",
            dispatch_ffn_combine_master_port=None,
        )

    def test_build_msprof_cmd_keeps_zero_node_rank(self, tmp_path: Path):
        cmd = build_msprof_cmd(
            tmp_path,
            self._args(tmp_path),
            ["DispatchFFNCombine"],
        )

        assert "--dispatch-ffn-combine-node-rank" in cmd
        assert cmd[cmd.index("--dispatch-ffn-combine-node-rank") + 1] == "0"

    def test_nonzero_msprof_exit_uses_generated_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """A failed msprof wrapper should not discard usable op_summary data."""
        import start_microbench

        def fake_run(cmd, check, cwd):
            profiler_root = next(tmp_path.glob("msprof_run_*"))
            output_dir = profiler_root / "PROF_001" / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_summary_001.csv").write_text(
                "OP Type,Task Duration(us)\nMatMulV2,1.0\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=255)

        monkeypatch.setattr(start_microbench.subprocess, "run", fake_run)

        profiler_root, prof_dirs = run_msprof(
            tmp_path,
            self._args(tmp_path),
            ["MatMulV2"],
        )

        assert profiler_root.exists()
        assert len(prof_dirs) == 1
        assert "Continuing with generated profiling data" in capsys.readouterr().out

    def test_nonzero_msprof_exit_without_summary_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Without op_summary output, a failed msprof run is still fatal."""
        import start_microbench

        monkeypatch.setattr(
            start_microbench.subprocess,
            "run",
            lambda cmd, check, cwd: SimpleNamespace(returncode=255),
        )

        with pytest.raises(RuntimeError, match="profiling data kept at"):
            run_msprof(tmp_path, self._args(tmp_path), ["MatMulV2"])

    def test_nonzero_combined_msprof_without_summary_falls_back_per_op(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """A combined profiler failure should retry selected ops separately."""
        import start_microbench

        calls: list[list[str]] = []

        def fake_run(cmd, check, cwd):
            calls.append(cmd)
            output_arg = next(item for item in cmd if item.startswith("--output="))
            profiler_root = Path(output_arg.split("=", 1)[1])
            if len(calls) > 1:
                output_dir = profiler_root / "PROF_001" / "mindstudio_profiler_output"
                output_dir.mkdir(parents=True)
                (output_dir / "op_summary_001.csv").write_text(
                    "OP Type,Task Duration(us)\nMatMulV2,1.0\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=255)

        monkeypatch.setattr(start_microbench.subprocess, "run", fake_run)

        profiler_root, prof_dirs = run_msprof(
            tmp_path,
            self._args(tmp_path),
            ["MatMulV2", "MaskedFill"],
        )

        assert profiler_root.exists()
        assert len(calls) == 3
        assert len(prof_dirs) == 2
        captured = capsys.readouterr().out
        assert "Retrying each selected operator" in captured

    def test_nonzero_full_msprof_without_summary_requires_explicit_ops(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """All-op profiler failure should not fan out into per-op msprof runs."""
        import start_microbench

        calls: list[list[str]] = []

        def fake_run(cmd, check, cwd):
            calls.append(cmd)
            return SimpleNamespace(returncode=255)

        monkeypatch.setattr(start_microbench.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="rerun with --ops"):
            run_msprof(tmp_path, self._args(tmp_path), None)

        assert len(calls) == 1


class TestUpdateCsv:
    """Tests for update_csv function."""

    def test_creates_new_csv_with_correct_columns(self, tmp_path: Path):
        """New CSV should have full default schema."""
        csv_path = tmp_path / "MatMulV2.csv"
        rows = [
            {
                "Input Shapes": "1024,1024;1024,1024",
                "Input Data Types": "FLOAT16;FLOAT16",
                "Input Formats": "ND;ND",
                "Average Duration(us)": "123.45",
                "Profiling Average Duration(us)": "130.0",
            }
        ]

        result = update_csv(csv_path, rows, mode="all", prune=False)

        assert csv_path.exists()
        assert result.added == 1
        assert result.updated == 0

        # Verify columns
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            assert "Average Duration(us)" in fieldnames
            assert "Profiling Average Duration(us)" in fieldnames

    def test_updates_existing_row(self, tmp_path: Path):
        """Existing row with matching signature should be updated."""
        csv_path = tmp_path / "Add.csv"
        # Create initial CSV
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Average Duration(us)": "",
                }
            )

        # Update with new data
        rows = [
            {
                "Input Shapes": "1024,1024",
                "Input Data Types": "FLOAT16;FLOAT16",
                "Input Formats": "ND;ND",
                "Average Duration(us)": "50.0",
                "Profiling Average Duration(us)": "55.0",
            }
        ]
        result = update_csv(csv_path, rows, mode="all", prune=False)

        assert result.updated == 1
        assert result.added == 0

        # Verify updated value
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["Average Duration(us)"] == "50.0"

    def test_matmul_common_merges_matmulv3_profile_alias(self, tmp_path: Path):
        csv_path = tmp_path / "MatMulCommon.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "Input Shapes": "1024,55296;6912,55296",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,6912",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "",
                }
            )

        result = update_db(
            tmp_path,
            {
                "MatMulV3": [
                    {
                        "Input Shapes": "1024,55296;6912,55296",
                        "Input Data Types": "DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND",
                        "Output Shapes": "1024,6912",
                        "Output Data Types": "DT_BF16",
                        "Average Duration(us)": "12.5",
                    }
                ]
            },
            ["MatMulCommon"],
            mode="missing-only",
            prune=False,
        )

        assert result[0].updated == 1
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Average Duration(us)"] == "12.5"

    def test_matmul_alias_rows_update_but_do_not_add_unmatched_rows(self, tmp_path: Path):
        csv_path = tmp_path / "MatMulV2.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": "24,512;4096,512",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "24,4096",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "",
                }
            )

        result = update_db(
            tmp_path,
            {
                "MatMulCommon": [
                    {
                        "Input Shapes": "24,512;512,4096",
                        "Input Data Types": "DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND",
                        "Output Shapes": "24,4096",
                        "Output Data Types": "DT_BF16",
                        "Average Duration(us)": "9.5",
                    },
                    {
                        "Input Shapes": "8,512;512,8192",
                        "Input Data Types": "DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND",
                        "Output Shapes": "8,8192",
                        "Output Data Types": "DT_BF16",
                        "Average Duration(us)": "99.0",
                    },
                ]
            },
            ["MatMulV2"],
            mode="all",
            prune=False,
        )

        assert result[0].updated == 1
        assert result[0].added == 0
        assert result[0].missing == []
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["Average Duration(us)"] == "9.5"

    def test_matmul_family_drift_does_not_report_false_missing_rows(self, tmp_path: Path):
        csv_path = tmp_path / "MatMulV2.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": "24,512;4096,512",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "24,4096",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "",
                }
            )

        result = update_db(
            tmp_path,
            {
                "MatMulV2": [
                    {
                        "Input Shapes": "5,512;512,2048",
                        "Input Data Types": "DT_BF16;DT_BF16",
                        "Input Formats": "ND;ND",
                        "Output Shapes": "5,2048",
                        "Output Data Types": "DT_BF16",
                        "Average Duration(us)": "9.5",
                    }
                ]
            },
            ["MatMulV2"],
            mode="missing-only",
            prune=False,
        )

        assert result[0].updated == 0
        assert result[0].added == 0
        assert result[0].missing == []

    def test_signature_normalizes_empty_shape_slots(self):
        generated_style = {
            "Input Shapes": '"1,128,1,512;();();2,16;()"',
            "Input Data Types": "DT_BF16;DT_UNDEFINED;DT_BF16;INT32;DT_UNDEFINED",
            "Input Formats": "ND;NULL;ND;ND;NULL",
            "Output Shapes": '"128,1,1,512;()"',
            "Output Data Types": "DT_BF16;FLOAT",
        }
        profiler_style = {
            "Input Shapes": '"1,128,1,512;;;2,16;"',
            "Input Data Types": "DT_BF16;;;INT32;",
            "Input Formats": "ND;;;ND;",
            "Output Shapes": '"128,1,1,512;"',
            "Output Data Types": "DT_BF16;",
        }

        assert get_sig(generated_style) == get_sig(profiler_style)

    def test_missing_only_mode_skips_valid_rows(self, tmp_path: Path):
        """missing-only mode should skip rows with valid duration."""
        csv_path = tmp_path / "Mul.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "512,512",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Average Duration(us)": "10.0",  # Already has valid duration
                }
            )

        rows = [
            {
                "Input Shapes": "512,512",
                "Input Data Types": "FLOAT16;FLOAT16",
                "Input Formats": "ND;ND",
                "Average Duration(us)": "15.0",
            }
        ]
        result = update_csv(csv_path, rows, mode="missing-only", prune=False)

        assert result.unchanged == 1
        assert result.updated == 0

    def test_prune_removes_invalid_rows(self, tmp_path: Path):
        """Prune should remove rows with only invalid durations."""
        csv_path = tmp_path / "Softmax.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024",
                    "Input Data Types": "FLOAT16",
                    "Input Formats": "ND",
                    "Average Duration(us)": "N/A",  # Invalid duration
                }
            )
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "2048",
                    "Input Data Types": "FLOAT16",
                    "Input Formats": "ND",
                    "Average Duration(us)": "5.0",  # Valid duration
                }
            )

        result = update_csv(csv_path, [], mode="all", prune=True)

        assert len(result.deleted) == 1
        # Verify only valid row remains
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["Input Shapes"] == "2048"

    def test_detects_duplicates(self, tmp_path: Path):
        """Should detect duplicate signatures in existing CSV."""
        csv_path = tmp_path / "Relu.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            # Two rows with same signature
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16",
                    "Input Formats": "ND",
                    "Average Duration(us)": "1.0",
                }
            )
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16",
                    "Input Formats": "ND",
                    "Average Duration(us)": "2.0",
                }
            )

        result = update_csv(csv_path, [], mode="all", prune=False)

        assert len(result.duplicates) == 1
        assert result.duplicates[0][1] == 2  # count = 2

    def test_legacy_mb_dur_migrated_on_rewrite(self, tmp_path: Path):
        """Legacy 'MicroBench Duration(us)' should be migrated when CSV is rewritten.

        Regression test: appending a new row should not clear the duration
        of existing rows that only have the legacy column name.
        """
        csv_path = tmp_path / "MatMulV2.csv"
        # Create CSV with legacy column name
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "MicroBench Duration(us)",  # legacy name
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "MicroBench Duration(us)": "12.34",  # legacy value
                }
            )

        # Append a new row with different signature
        new_rows = [
            {
                "Input Shapes": "2048,2048",
                "Input Data Types": "FLOAT16;FLOAT16",
                "Input Formats": "ND;ND",
                "Average Duration(us)": "56.78",
            }
        ]
        update_csv(csv_path, new_rows, mode="all", prune=False)

        # Verify the legacy row's duration is preserved under new column name
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["Input Shapes"]: r for r in reader}

        # Legacy row should have its duration migrated
        assert rows["1024,1024"]["Average Duration(us)"] == "12.34"
        # New row should have its duration
        assert rows["2048,2048"]["Average Duration(us)"] == "56.78"
        # Old column name should not exist
        assert "MicroBench Duration(us)" not in rows["1024,1024"]

    def test_legacy_mb_dur_preserved_on_prune(self, tmp_path: Path):
        """Legacy 'MicroBench Duration(us)' should prevent row from being pruned.

        Regression test: a row with only legacy duration should not be deleted
        when prune=True, because the value should be migrated before prune check.
        """
        csv_path = tmp_path / "Add.csv"
        # Create CSV with legacy column name and valid duration
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "MicroBench Duration(us)",  # legacy name
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "MicroBench Duration(us)": "10.0",  # valid duration
                }
            )
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "2048,2048",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "MicroBench Duration(us)": "N/A",  # invalid duration
                }
            )

        # Prune with no new rows
        result = update_csv(csv_path, [], mode="all", prune=True)

        # Row with valid legacy duration should be kept
        assert len(result.deleted) == 1  # Only the N/A row deleted
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Input Shapes"] == "1024,1024"
        assert rows[0]["Average Duration(us)"] == "10.0"

    def test_includes_extra_columns_for_new_csv(self, tmp_path: Path):
        """New CSV should include extra columns from rows_to_merge (e.g., EP Size)."""
        csv_path = tmp_path / "DispatchFFNCombine.csv"
        rows = [
            {
                "Input Shapes": "1024,4096;16,4096,2048",
                "Input Data Types": "FLOAT16;FLOAT16",
                "Input Formats": "ND;ND",
                "Average Duration(us)": "100.0",
                "EP Size": "8",  # Extra column for DispatchFFNCombine
            }
        ]

        update_csv(csv_path, rows, mode="all", prune=False)

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            assert "EP Size" in fieldnames

            rows = list(reader)
            assert rows[0]["EP Size"] == "8"

    def test_dispatch_ffn_legacy_csv_uses_incoming_ep_size_for_matching(self, tmp_path: Path):
        csv_path = tmp_path / "DispatchFFNCombine.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "static",
                    "Input Shapes": "4,16;2,16,8",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "4,16",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "",
                }
            )

        result = update_csv(
            csv_path,
            [
                {
                    "Input Shapes": "4,16;2,16,8",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "4,16",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "12.5",
                    "EP Size": "32",
                }
            ],
            mode="missing-only",
            prune=False,
        )

        assert result.updated == 1
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Average Duration(us)"] == "12.5"
        assert rows[0]["EP Size"] == "32"

    def test_match_only_unmatched_rows_warn(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "MatMulV2.csv"
        update_csv(
            csv_path,
            [],
            mode="all",
            prune=False,
            match_only_rows=[
                {
                    "Input Shapes": "4,16;16,8",
                    "Input Data Types": "DT_BF16;DT_BF16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "4,8",
                    "Output Data Types": "DT_BF16",
                    "Average Duration(us)": "12.5",
                }
            ],
        )

        assert "match-only profiling row did not match MatMulV2.csv" in capsys.readouterr().out

    def test_records_gap_between_mb_and_profiling(self, tmp_path: Path):
        """Should record gap when both MB and profiling durations are valid."""
        csv_path = tmp_path / "Gather.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Average Duration(us)",
                    "Profiling Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Input Shapes": "1024,512",
                    "Input Data Types": "FLOAT16",
                    "Input Formats": "ND",
                    "Average Duration(us)": "",
                    "Profiling Average Duration(us)": "100.0",
                }
            )

        rows = [
            {
                "Input Shapes": "1024,512",
                "Input Data Types": "FLOAT16",
                "Input Formats": "ND",
                "Average Duration(us)": "80.0",  # MB duration
                "Profiling Average Duration(us)": "100.0",  # Profiling duration
            }
        ]
        result = update_csv(csv_path, rows, mode="all", prune=False)

        assert len(result.gaps) == 1
        assert result.gaps[0].mb_us == 80.0
        assert result.gaps[0].prof_us == 100.0
        assert result.gaps[0].ratio == 0.8


class TestAggregateSummary:
    """Tests for aggregate_summary function."""

    def test_aggregates_op_type(self, tmp_path: Path):
        """Should aggregate rows by OP Type."""
        # Create a mock summary.csv with correct column names
        summary_dir = tmp_path / "msprof_run_001" / "summary"
        summary_dir.mkdir(parents=True)
        summary_csv = summary_dir / "summary.csv"

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP Type",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Task Duration(us)",
                    "aicore_time(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP Type": "MatMulV2",
                    "Input Shapes": "1024,1024;1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Task Duration(us)": "50.0",
                    "aicore_time(us)": "45.0",
                }
            )
            w.writerow(
                {
                    "OP Type": "Add",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Task Duration(us)": "10.0",
                    "aicore_time(us)": "8.0",
                }
            )

        result = aggregate_summary([summary_csv], ep_size=None)

        assert "MatMulV2" in result
        assert "Add" in result
        assert len(result["MatMulV2"]) == 1
        assert len(result["Add"]) == 1


class TestPrintReport:
    """Tests for print_report function."""

    def test_prints_overview_table(self, capsys):
        """Should print Overview table."""
        results = [UpdateResult(csv_path=Path("/tmp/test.csv"))]
        gaps = []

        print_report(results, gaps, status=None, to_file=None)

        captured = capsys.readouterr().out
        assert "# Profile Update Report" in captured
        assert "## Overview" in captured
        assert "CSV files touched" in captured

    def test_prints_operator_status(self, capsys):
        """Should print Operator Execution Status when provided."""
        results = [UpdateResult(csv_path=Path("/tmp/test.csv"))]
        gaps = []
        status = {
            "success": [{"op": "MatMulV2"}],
            "failed": [{"op": "Add", "reason": "NPU error"}],
            "skipped": [{"op": "Softmax"}],
        }

        print_report(results, gaps, status=status, to_file=None)

        captured = capsys.readouterr().out
        assert "## Operator Execution Status" in captured
        assert "Success: 1" in captured
        assert "Failed: 1" in captured
        assert "Skipped: 1" in captured

    def test_empty_tables_show_none(self, capsys):
        """Empty tables should show '_None_'."""
        results = [UpdateResult(csv_path=Path("/tmp/test.csv"))]
        gaps = []

        print_report(results, gaps, status=None, to_file=None)

        captured = capsys.readouterr().out
        # Deleted Empty Rows and Duplicate Signatures should show _None_
        assert "## Deleted Empty Rows\n_None_" in captured
        assert "## Duplicate Signatures\n_None_" in captured

    def test_writes_report_to_file(self, tmp_path: Path):
        """Should write report to file when to_file is provided."""
        results = [UpdateResult(csv_path=Path("/tmp/test.csv"), updated=5, added=2)]
        gaps = [GapRecord("MatMulV2", "MatMulV2.csv", "1024,1024", 80.0, 100.0, 20.0, 0.8)]

        report_result = print_report(results, gaps, status=None, to_file=tmp_path)

        assert report_result is not None
        report_path, csv_path = report_result
        assert report_path.exists()
        assert csv_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "# Profile Update Report" in content
        assert "## Overview" in content
        assert "## Duration Gap Hotspots" in content
        assert "MatMulV2" in content


# =============================================================================
# End-to-End Tests
# =============================================================================
class TestEndToEndWithMockedProfiling:
    """End-to-end tests with mocked profiling collection (no NPU required).

    These tests simulate the full pipeline with pre-generated profiling data
    returned from run_msprof.
    """

    @pytest.fixture
    def mock_prof_data(self, tmp_path: Path) -> Path:
        """Create mock profiling data directory structure.

        The structure matches msprof output:
        PROF_*/mindstudio_profiler_output/op_summary_*.csv
        """
        prof_dir = tmp_path / "PROF_001"
        output_dir = prof_dir / "mindstudio_profiler_output"
        output_dir.mkdir(parents=True)

        # Create op_summary_*.csv with mock profiling data
        summary_csv = output_dir / "op_summary_001.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP Type",
                    "Task Type",
                    "Task Duration(us)",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                    "aicore_time(us)",
                    "aic_total_cycles",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP Type": "MatMulV2",
                    "Task Type": "AICore",
                    "Task Duration(us)": "123.45",
                    "Input Shapes": "1024,1024;1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,1024",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "aicore_time(us)": "120.0",
                    "aic_total_cycles": "1000000",
                }
            )
            w.writerow(
                {
                    "OP Type": "Add",
                    "Task Type": "AICore",
                    "Task Duration(us)": "10.5",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,1024",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "aicore_time(us)": "8.0",
                    "aic_total_cycles": "50000",
                }
            )

        return prof_dir

    @pytest.fixture
    def temp_database(self, tmp_path: Path) -> Path:
        """Create a temporary database directory with CSV files."""
        db_path = tmp_path / "database"
        db_path.mkdir()

        # Create MatMulV2.csv with empty duration
        csv_path = db_path / "MatMulV2.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Accelerator Core",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Accelerator Core": "",
                    "Input Shapes": "1024,1024;1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,1024",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "Average Duration(us)": "",  # Empty - should be filled
                }
            )

        # Create Add.csv with existing duration (to test missing-only mode)
        add_csv = db_path / "Add.csv"
        with add_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Accelerator Core",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Accelerator Core": "",
                    "Input Shapes": "1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,1024",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "Average Duration(us)": "5.0",  # Already has valid duration
                }
            )

        return db_path

    @pytest.fixture(autouse=True)
    def mock_profile_collection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_prof_data: Path,
    ) -> None:
        profiler_root = tmp_path / "profiler_root"
        profiler_root.mkdir()
        monkeypatch.setattr(start_microbench, "ensure_custom_opp_env", lambda _ops: None)
        monkeypatch.setattr(start_microbench, "ensure_npu_available", lambda: None)
        monkeypatch.setattr(
            start_microbench,
            "run_msprof",
            lambda *_args, **_kwargs: (profiler_root, {mock_prof_data}),
        )

    def test_e2e_updates_database(self, temp_database: Path, capsys):
        """Test full pipeline updates database correctly."""
        # Simulate CLI args
        import sys

        # Import main function
        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(temp_database),
                "--update-mode",
                "all",
            ]
            main()

            # Verify MatMulV2.csv was updated
            matmul_csv = temp_database / "MatMulV2.csv"
            with matmul_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["Average Duration(us)"] == "123.450000"

            # Verify Add.csv was NOT updated in missing-only mode
            # (but we're using "all" mode, so it should be updated)
            add_csv = temp_database / "Add.csv"
            with add_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert rows[0]["Average Duration(us)"] == "10.500000"

            # Check console output
            captured = capsys.readouterr().out
            assert "# Profile Update Report" in captured
            assert "## Overview" in captured
        finally:
            sys.argv = old_argv

    def test_e2e_missing_only_mode(self, temp_database: Path, capsys):
        """Test missing-only mode only updates rows without valid duration."""
        import sys

        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(temp_database),
                "--update-mode",
                "missing-only",
            ]
            main()

            # MatMulV2 should be updated (empty duration)
            matmul_csv = temp_database / "MatMulV2.csv"
            with matmul_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert rows[0]["Average Duration(us)"] == "123.450000"

            # Add should NOT be updated (already has valid duration)
            add_csv = temp_database / "Add.csv"
            with add_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert rows[0]["Average Duration(us)"] == "5.0"  # Original value

            captured = capsys.readouterr().out
            assert "unchanged" in captured.lower() or "Unchanged" in captured
        finally:
            sys.argv = old_argv

    def test_e2e_creates_report_files(self, temp_database: Path):
        """Test that report markdown and CSV files are created."""
        import sys

        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(temp_database),
                "--update-mode",
                "all",
            ]
            main()

            # Check for report files
            reports_dir = temp_database / "reports"
            assert reports_dir.exists()

            md_files = list(reports_dir.glob("profile_update_report_*.md"))
            csv_files = list(reports_dir.glob("duration_gap_hotspots_full_*.csv"))
            assert len(md_files) == 1
            assert len(csv_files) == 1

            # Verify report content
            md_content = md_files[0].read_text(encoding="utf-8")
            assert "# Profile Update Report" in md_content
            assert "## Overview" in md_content
            assert "## Update Summary" in md_content
        finally:
            sys.argv = old_argv

    def test_e2e_prune_empty_duration_rows(self, tmp_path: Path):
        """Test --prune-empty-duration-rows removes invalid rows."""
        # Create database with rows that have only N/A durations
        db_path = tmp_path / "database"
        db_path.mkdir()

        csv_path = db_path / "MatMulV2.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "OP State",
                    "Accelerator Core",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                    "Average Duration(us)",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "OP State": "",
                    "Accelerator Core": "",
                    "Input Shapes": "1024,1024;1024,1024",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "1024,1024",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "Average Duration(us)": "N/A",  # Invalid - should be pruned
                }
            )
            w.writerow(
                {
                    "OP State": "",
                    "Accelerator Core": "",
                    "Input Shapes": "2048,2048;2048,2048",
                    "Input Data Types": "FLOAT16;FLOAT16",
                    "Input Formats": "ND;ND",
                    "Output Shapes": "2048,2048",
                    "Output Data Types": "FLOAT16",
                    "Output Formats": "ND",
                    "Average Duration(us)": "",  # Empty - should NOT be pruned (will be filled)
                }
            )

        import sys

        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(db_path),
                "--update-mode",
                "all",
                "--prune-empty-duration-rows",
            ]
            main()

            # Check that N/A row was removed
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Only the row with matching signature should remain
                assert len(rows) == 1
                assert rows[0]["Input Shapes"] == "1024,1024;1024,1024"
        finally:
            sys.argv = old_argv


@pytest.mark.npu
class TestEndToEndWithNPU:
    """End-to-end tests requiring NPU hardware.

    These tests run the actual profiling pipeline with msprof.
    Run with: pytest -m npu tests/regression/cli/test_start_microbench.py::TestEndToEndWithNPU

    Prerequisites:
    - NPU device available (torch_npu installed and device accessible)
    - msprof command available in PATH
    - Configure the class variables below before running tests
    """

    # Configure these paths before running NPU tests
    _VLLM_ASCEND_OPS = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
    )
    ASCEND_CUSTOM_OPP_PATH = f"{_VLLM_ASCEND_OPS}:${{ASCEND_CUSTOM_OPP_PATH}}"
    LD_LIBRARY_PATH = f"{_VLLM_ASCEND_OPS}/op_api/lib/:${{LD_LIBRARY_PATH}}"
    PROF_DATABASE_PATH = (
        "$(pwd)/tensor_cast/performance_model/profiling_database/data"
        "/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
    )
    # Device and version info
    DEVICE: str = "ATLAS_800_A3_752T_128G_DIE"
    VLLM_VERSION: str = "0.18.0"
    TORCH_VERSION: str = "2.9.0"
    CANN_VERSION: str = "8.5"

    def _setup_env(self) -> None:
        """Set environment variables from class constants.

        Expands ${VAR} references with existing environment variable values.
        Expands $(pwd) with current working directory.
        """
        import os

        if self.ASCEND_CUSTOM_OPP_PATH:
            old_ascend = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            path = self.ASCEND_CUSTOM_OPP_PATH.replace("${ASCEND_CUSTOM_OPP_PATH}", old_ascend)
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = path
        if self.LD_LIBRARY_PATH:
            old_ld = os.environ.get("LD_LIBRARY_PATH", "")
            path = self.LD_LIBRARY_PATH.replace("${LD_LIBRARY_PATH}", old_ld)
            os.environ["LD_LIBRARY_PATH"] = path
        if self.PROF_DATABASE_PATH:
            self.PROF_DATABASE_PATH = self.PROF_DATABASE_PATH.replace("$(pwd)", os.getcwd())

    def _check_npu_available(self) -> bool:
        """Check if NPU and msprof are available."""
        try:
            import torch
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                return False
        except ImportError:
            return False

        import shutil

        if not shutil.which("msprof"):
            return False

        return True

    @pytest.fixture
    def npu_database(self, tmp_path: Path) -> Path:
        """Get a database path for NPU testing.

        Uses PROF_DATABASE_PATH class variable, copies to tmp to avoid modifications.
        """
        self._setup_env()

        # Copy existing database to tmp
        import shutil

        db_path = tmp_path / "npu_database"
        shutil.copytree(self.PROF_DATABASE_PATH, db_path)
        return db_path

    def test_npu_with_prune_empty_duration(self, npu_database: Path, capsys):
        """Test NPU profiling with --prune-empty-duration-rows flag."""
        if not self._check_npu_available():
            pytest.skip("NPU or msprof not available")

        import sys

        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(npu_database),
                "--repeat-count",
                "1",
                "--device",
                self.DEVICE,
                "--vllm-version",
                self.VLLM_VERSION,
                "--torch-version",
                self.TORCH_VERSION,
                "--cann-version",
                self.CANN_VERSION,
                "--ops",
                "MatMulV2",
                "--prune-empty-duration-rows",
            ]
            main()

            # Verify profiling created output
            captured = capsys.readouterr().out
            assert "# Profile Update Report" in captured
            assert "## Overview" in captured

            # Check for report files
            reports_dir = npu_database / "reports"
            if reports_dir.exists():
                md_files = list(reports_dir.glob("profile_update_report_*.md"))
                assert len(md_files) >= 1

        finally:
            sys.argv = old_argv

    def test_npu_missing_only_mode(self, npu_database: Path, capsys):
        """Test NPU profiling with --update-mode missing-only flag."""
        if not self._check_npu_available():
            pytest.skip("NPU or msprof not available")

        import sys

        from start_microbench import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "start_microbench.py",
                "--database-path",
                str(npu_database),
                "--repeat-count",
                "1",
                "--device",
                self.DEVICE,
                "--vllm-version",
                self.VLLM_VERSION,
                "--torch-version",
                self.TORCH_VERSION,
                "--cann-version",
                self.CANN_VERSION,
                "--update-mode",
                "missing-only",
            ]
            main()

            # Verify output - either profiling ran or all data already valid
            captured = capsys.readouterr().out
            # If all CSV files already have valid durations, script outputs
            # "[SUMMARY] All target CSV files already have usable replay durations."
            # Otherwise, it outputs the profile update report.
            assert (
                "# Profile Update Report" in captured
                or "All target CSV files already have usable replay durations" in captured
            )

        finally:
            sys.argv = old_argv
