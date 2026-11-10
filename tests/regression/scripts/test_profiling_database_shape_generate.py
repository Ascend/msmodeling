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

"""Unit tests for the profiling_database_shape_generate skill scripts.

Covers the M1-M4 test plan of
``zhj_tmp/RFC-profiling-database-shape-generate.md`` (section 3.2):

- build_cases.py: command parsing / partial-param grid / preflight / sampling;
- generate_shape_db.py: append mode, parse skipping, retry loop, failures;
- mock_replay_durations.py: placeholder filling invariants and UTF-8-sig BOM;
- verify_db_coverage.py: command construction, log parsing, top-total scope,
  communication-op exclusion, stale-mock detection;
- generate_run_report.py: report sections, MISS aggregation and suggestions.

Only local/offline paths are exercised: model building, NPU replay and
``text_generate`` simulation runs are never executed here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "profiling_database_shape_generate"
SCRIPTS_DIR = SKILL_DIR / "scripts"


def _load_script(name: str):
    """Load a skill script by file path (the scripts are not an importable package)."""
    module_name = f"skill_profiling_database_shape_generate_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


build_cases = _load_script("build_cases")
generate_shape_db = _load_script("generate_shape_db")
mock_replay_durations = _load_script("mock_replay_durations")
verify_db_coverage = _load_script("verify_db_coverage")
generate_run_report = _load_script("generate_run_report")


@pytest.fixture(autouse=True)
def _test_device_profile(monkeypatch):
    """Register a minimal TEST_DEVICE profile so preflight/spec validation can run offline."""
    import torch

    from tensor_cast.device import DeviceProfile

    profile = SimpleNamespace(comm_grid=SimpleNamespace(grid=torch.empty(12)))
    monkeypatch.setitem(DeviceProfile.all_device_profiles, "TEST_DEVICE", profile)


def _write_cases_yaml(path: Path, workloads: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"workloads": workloads}, sort_keys=False), encoding="utf-8")
    return path


def _base_workload(**overrides) -> dict:
    workload = {
        "name": "glm52-case1",
        "model": "zai-org/GLM-5.2",
        "device": "TEST_DEVICE",
        "num_devices": 16,
        "input_length": 4096,
        "output_length": 1024,
        "batch_range": [1, 1],
        "tp_sizes": [16],
        "ep_sizes": [1],
        "quantize_linear_action": "W8A8_DYNAMIC",
    }
    workload.update(overrides)
    return workload


def _op_table_log() -> str:
    """A runtime.table_averages style log with one compute op and one communication op."""
    separator = "-" * 96
    return "\n".join(
        [
            separator,
            "  Name                      empirical total  empirical avg  analytic total  analytic avg  # of Calls",
            separator,
            "tensor_cast.matmul                  2.000ms        1.000ms         4.000ms        2.000ms         2",
            "tensor_cast.all_reduce            500.000us      250.000us       500.000us      250.000us         2",
            separator,
            "Total time for empirical: 2.500ms",
        ]
    )


def _miss_log() -> str:
    """A metrics_collector MISSes / MISS shapes style log."""
    return "\n".join(
        [
            "  MISSes (2 unique reasons):",
            "  [shape_mismatch] kernel found, no matching shape in CSV: tensor_cast.matmul (x3), tensor_cast.mla",
            "  [unmapped] not in op_mapping.yaml: tensor_cast.foo",
            "  PARTIAL: 2/10 (tensor_cast.gelu×2)",
            "  MISS shapes (2):",
            "  tensor_cast.matmul (1, 4096)",
            "  tensor_cast.foo (2, 3)",
            "  ... and 5 more",
        ]
    )


# ---------------------------------------------------------------------------
# build_cases.py
# ---------------------------------------------------------------------------


class TestBuildCasesHelpers:
    def test_split_command_keeps_quoted_paths(self):
        tokens = build_cases.split_command('python -m cli.inference.text_generate "C:/models/my model" --num-queries 1')
        assert tokens[0] == "python"
        assert "C:/models/my model" in tokens

    def test_strip_launcher_prefix(self):
        tokens = ["python", "-m", "cli.inference.text_generate", "Qwen/Qwen3-32B", "--tp-size", "8"]
        assert build_cases.strip_launcher_prefix(tokens) == ["Qwen/Qwen3-32B", "--tp-size", "8"]
        # 无 launcher marker 时原样返回
        plain = ["Qwen/Qwen3-32B", "--tp-size", "8"]
        assert build_cases.strip_launcher_prefix(plain) == plain


class TestMapArgsToSpec:
    def _args(self, **overrides) -> SimpleNamespace:
        base = dict(
            model_id="Qwen/Qwen3-32B",
            device="TEST_DEVICE",
            num_devices=8,
            num_queries=2,
            tp_size=8,
            ep_size=1,
            quantize_linear_action="W8A8_DYNAMIC",
            quantize_non_expert_linear_action="W8A8_DYNAMIC",
            quantize_attention_action="DISABLED",
            compile=False,
            decode=False,
            context_length=0,
            query_length=3584,
            moe_dp_size=None,
            dcp_size=1,
            pp_size=1,
            num_mtp_tokens=0,
            compilation_config=None,
            word_embedding_tp=None,
            enable_shared_expert_tp=False,
            prefix_cache_hit_rate=0.0,
            reserved_memory_gb=None,
            mxfp4_group_size=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_prefill_maps_query_length(self):
        warnings: list[str] = []
        workload = build_cases.map_args_to_spec(self._args(), warnings, 2048, "w1")
        assert workload["input_length"] == 3584
        assert workload["output_length"] == 2048
        assert workload["batch_range"] == [2, 2]
        assert workload["tp_sizes"] == [8]
        assert workload["ep_sizes"] == [1]
        assert workload["compile"] is False
        assert warnings == []

    def test_decode_derives_input_length_from_context(self):
        warnings: list[str] = []
        args = self._args(decode=True, context_length=4097, query_length=1)
        workload = build_cases.map_args_to_spec(args, warnings, 1024, "w1")
        # 实测算子 decode 命令 --context-length = input_length + output_length，
        # 反推 input_length = context_length - output_length
        assert workload["input_length"] == 4097 - 1024
        assert warnings == []
        # 反推的 input_length 再经 verify 正向构造，必须还原原始 decode 命令的 context
        command = verify_db_coverage.build_command(
            _verify_args(), workload, workload["input_length"], workload["output_length"], "decode"
        )
        assert command[command.index("--context-length") + 1] == "4097"

    def test_decode_requires_context_length_above_output(self):
        args = self._args(decode=True, context_length=1, query_length=1)
        with pytest.raises(SystemExit, match="context-length > output_length"):
            build_cases.map_args_to_spec(args, [], 1024, "w1")

    def test_prefill_context_length_warns(self):
        warnings: list[str] = []
        build_cases.map_args_to_spec(self._args(context_length=512), warnings, 1024, "w1")
        assert any("context-length 512" in message for message in warnings)

    def test_optional_fields_and_dropped_warnings(self):
        warnings: list[str] = []
        args = self._args(
            moe_dp_size=2,
            dcp_size=2,
            pp_size=2,
            num_mtp_tokens=1,
            speculative_method="mtp",
            num_speculative_tokens=2,
            compilation_config=["enable_sequence_parallel"],
            word_embedding_tp="row",
            enable_shared_expert_tp=True,
            prefix_cache_hit_rate=0.5,
            reserved_memory_gb=4,
            mxfp4_group_size=32,
            dump_input_shapes=True,
            performance_model="analytic",
            dp_size=4,
        )
        workload = build_cases.map_args_to_spec(args, warnings, 1024, "w1")
        assert workload["moe_dp_sizes"] == [2]
        assert workload["dcp_sizes"] == [2]
        assert workload["pp_sizes"] == [2]
        assert workload["num_mtp_tokens"] == [2]
        assert workload["compilation_config"] == ["enable_sequence_parallel"]
        assert workload["compile"] is True
        assert workload["word_embedding_tp"] == "row"
        assert workload["enable_shared_expert_tp"] is True
        assert workload["prefix_cache_hit_rate"] == 0.5
        assert workload["reserved_memory_gb"] == 4
        assert workload["mxfp4_group_size"] == 32
        assert any("dump-input-shapes" in message for message in warnings)
        assert any("performance-model" in message for message in warnings)
        assert any("dp-size" in message for message in warnings)

    def test_non_mtp_speculative_method_is_dropped(self):
        warnings: list[str] = []
        args = self._args(speculative_method="eagle", num_speculative_tokens=2)
        workload = build_cases.map_args_to_spec(args, warnings, 1024, "w1")
        assert "num_mtp_tokens" not in workload
        assert any("only mtp maps to num_mtp_tokens" in message for message in warnings)

    def test_enum_quantize_value_is_unwrapped(self):
        class _Enum:
            value = "W4A8_STATIC"

        workload = build_cases.map_args_to_spec(self._args(quantize_linear_action=_Enum()), [], 1024, "w1")
        assert workload["quantize_linear_action"] == "W4A8_STATIC"


class TestBuildCasesMode2:
    def _args(self, set_pairs: list[str], **overrides) -> SimpleNamespace:
        base = dict(
            set=set_pairs,
            input_length_grid="1024,2048",
            output_length_grid="512",
            batch_range_grid="1",
            quantize_grid="W8A8_DYNAMIC",
            max_workloads=64,
            name_prefix=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_requires_model_device_and_num_devices(self):
        with pytest.raises(SystemExit, match="mode 2 requires"):
            build_cases.build_mode2_workloads(self._args([]))

    def test_rejects_unknown_set_key(self):
        with pytest.raises(SystemExit, match="not a workload spec field"):
            build_cases.build_mode2_workloads(
                self._args(["model=Qwen/Qwen3-32B", "device=TEST_DEVICE", "num_devices=8", "unknown=1"])
            )

    def test_rejects_malformed_set_pair(self):
        with pytest.raises(SystemExit, match="expects key=value"):
            build_cases.build_mode2_workloads(self._args(["model"]))

    def test_grid_cartesian_product_and_defaults(self):
        workloads, warnings = build_cases.build_mode2_workloads(
            self._args(
                ["model=Qwen/Qwen3-32B", "device=TEST_DEVICE", "num_devices=8", "tp_sizes=[8]"],
            )
        )
        assert warnings == []
        assert len(workloads) == 2
        assert {w["input_length"] for w in workloads} == {1024, 2048}
        assert all(w["batch_range"] == [1, 1] for w in workloads)
        assert all(w["quantize_linear_action"] == "W8A8_DYNAMIC" for w in workloads)
        assert [w["name"] for w in workloads] == ["Qwen3-32B-combo1", "Qwen3-32B-combo2"]

    def test_overflow_is_deterministically_subsampled(self):
        workloads, warnings = build_cases.build_mode2_workloads(
            self._args(
                ["model=Qwen/Qwen3-32B", "device=TEST_DEVICE", "num_devices=8", "tp_sizes=[8]"],
                input_length_grid="1024,2048,4096,8192",
                max_workloads=2,
            )
        )
        assert len(workloads) == 2
        assert any("exceeds --max-workloads 2" in message for message in warnings)
        assert [w["input_length"] for w in workloads] == [1024, 4096]

    def test_set_field_overrides_grid(self):
        workloads, _ = build_cases.build_mode2_workloads(
            self._args(
                [
                    "model=Qwen/Qwen3-32B",
                    "device=TEST_DEVICE",
                    "num_devices=8",
                    "tp_sizes=[8]",
                    "input_length=777",
                ]
            )
        )
        assert [w["input_length"] for w in workloads] == [777]

    def test_explicit_set_name_gets_combo_index_suffix(self):
        # --set name 不能让所有 combo 同名：显式名称仅作为前缀并追加 combo 序号
        workloads, _ = build_cases.build_mode2_workloads(
            self._args(
                ["name=foo", "model=Qwen/Qwen3-32B", "device=TEST_DEVICE", "num_devices=8", "tp_sizes=[8]"],
            )
        )
        assert [w["name"] for w in workloads] == ["foo-combo1", "foo-combo2"]


class TestPreflight:
    def test_drops_workload_with_unregistered_device(self):
        workload = _base_workload(device="NOT_A_DEVICE")
        survivors, dropped, _notes = build_cases.preflight_workloads([workload])
        assert survivors == []
        assert any("not a registered DeviceProfile" in item for item in dropped)

    def test_drops_duplicate_workload_ids(self):
        first = _base_workload()
        second = _base_workload()  # identical name + fields -> same workload id
        survivors, dropped, notes = build_cases.preflight_workloads([first, second])
        assert survivors == [first]
        assert any("duplicate" in item for item in dropped)
        assert len(notes) == 1


class TestModelConfigMemoization:
    def test_memoizes_across_passes_and_restores(self, monkeypatch):
        from grid_generator import optimizer_spec

        calls: list[str] = []

        def counting(model_id: str) -> int:
            calls.append(model_id)
            return 61

        monkeypatch.setattr(optimizer_spec, "_resolve_num_hidden_layers", counting)
        monkeypatch.setattr(build_cases, "_NUM_HIDDEN_LAYERS_CACHE", {})
        with build_cases._memoized_model_config_lookup():
            assert optimizer_spec._resolve_num_hidden_layers("zai-org/GLM-5.2") == 61
        # 第二遍展开（final_validation）复用同一进程内缓存，不再解析模型 config
        with build_cases._memoized_model_config_lookup():
            assert optimizer_spec._resolve_num_hidden_layers("zai-org/GLM-5.2") == 61
        assert calls == ["zai-org/GLM-5.2"]
        # 上下文退出后恢复被替换的函数（缓存不改变后续调用语义）
        assert optimizer_spec._resolve_num_hidden_layers is counting


class TestBuildCasesMain:
    def test_command_and_set_are_mutually_exclusive(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_cases.py",
                "--output",
                str(tmp_path / "cases.yaml"),
                "--command",
                "python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 1 --query-length 128",
                "--set",
                "model=Qwen/Qwen3-32B",
            ],
        )
        with pytest.raises(SystemExit, match="mutually exclusive"):
            build_cases.main()

    def test_requires_command_or_set(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["build_cases.py", "--output", str(tmp_path / "cases.yaml")])
        with pytest.raises(SystemExit, match="provide --command"):
            build_cases.main()

    def test_mode1_dry_run_prints_payload(self, capsys, monkeypatch, tmp_path: Path):
        command = (
            "python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 1 --query-length 3584 "
            "--context-length 0 --device TEST_DEVICE --tp-size 8 --num-devices 8 "
            "--quantize-linear-action W8A8_DYNAMIC"
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_cases.py", "--output", str(tmp_path / "cases.yaml"), "--command", command, "--dry-run"],
        )
        build_cases.main()
        output = capsys.readouterr().out
        assert "dry-run payload" in output
        assert "input_length: 3584" in output
        assert not (tmp_path / "cases.yaml").exists()

    def test_mode1_writes_workload_yaml(self, monkeypatch, tmp_path: Path):
        output = tmp_path / "cases.yaml"
        command = (
            "python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 1 --query-length 3584 "
            "--context-length 0 --device TEST_DEVICE --tp-size 8 --num-devices 8 "
            "--quantize-linear-action W8A8_DYNAMIC"
        )
        monkeypatch.setattr(sys, "argv", ["build_cases.py", "--output", str(output), "--command", command])
        build_cases.main()
        text = output.read_text(encoding="utf-8")
        assert "generated by profiling_database_shape_generate/scripts/build_cases.py (command mode)" in text
        payload = yaml.safe_load(text)
        assert len(payload["workloads"]) == 1
        workload = payload["workloads"][0]
        assert workload["name"] == "Qwen3-32B-in3584-out2048"
        assert workload["input_length"] == 3584
        assert workload["output_length"] == 2048
        assert workload["tp_sizes"] == [8]
        assert workload["batch_range"] == [1, 1]

    def test_mode2_writes_workload_yaml(self, monkeypatch, tmp_path: Path):
        output = tmp_path / "cases.yaml"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_cases.py",
                "--output",
                str(output),
                "--set",
                "model=Qwen/Qwen3-32B",
                "--set",
                "device=TEST_DEVICE",
                "--set",
                "num_devices=8",
                "--set",
                "tp_sizes=[8]",
                "--input-length-grid",
                "1024,2048",
                "--output-length-grid",
                "512",
            ],
        )
        build_cases.main()
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert len(payload["workloads"]) == 2
        assert "(grid mode)" in output.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# generate_shape_db.py
# ---------------------------------------------------------------------------


class _FakeRun:
    """subprocess.run replacement that fakes parse/grid invocations and the report JSON."""

    def __init__(self, report_path: Path, failures_before_success: int = 0, always_fail: bool = False):
        self.report_path = report_path
        self.failures_before_success = failures_before_success
        self.always_fail = always_fail
        self.grid_calls: list[list[str]] = []
        self.parse_calls: list[list[str]] = []
        self.grid_kwargs: list[dict] = []

    def __call__(self, command, **kwargs):
        command = list(command)
        if len(command) > 1 and command[1] == "-c":
            self.grid_calls.append(command)
            self.grid_kwargs.append(kwargs)
            failed = self.always_fail or len(self.grid_calls) <= self.failures_before_success
            payload = {
                "workloads": {
                    "attempted": 1,
                    "succeeded": 0 if failed else 1,
                    "cached": 0,
                    "failed_workloads": ["w1"] if failed else [],
                }
            }
            self.report_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            self.parse_calls.append(command)
        return subprocess.CompletedProcess(command, 0)


class TestGenerateShapeDbHelpers:
    def test_resolve_target_model_from_first_workload(self, tmp_path: Path):
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        args = SimpleNamespace(target_model=None, cases_yaml=cases)
        assert generate_shape_db.resolve_target_model(args) == "zai-org/GLM-5.2"

    def test_resolve_target_model_prefers_explicit(self, tmp_path: Path):
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        args = SimpleNamespace(target_model="Qwen/Qwen3-32B", cases_yaml=cases)
        assert generate_shape_db.resolve_target_model(args) == "Qwen/Qwen3-32B"

    def test_resolve_target_model_requires_model_field(self, tmp_path: Path):
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [{"name": "no-model"}])
        args = SimpleNamespace(target_model=None, cases_yaml=cases)
        with pytest.raises(SystemExit, match="requires --target-model"):
            generate_shape_db.resolve_target_model(args)

    def test_build_grid_command_without_sweep(self, tmp_path: Path):
        args = SimpleNamespace(
            cases_yaml=tmp_path / "cases.yaml",
            rows=500,
            seed=3,
            with_model_sweep=False,
            target_model=None,
            report_path=None,
        )
        command = generate_shape_db.build_grid_command(args, tmp_path / "db")
        assert "--optimizer-args-file" in command
        assert "--target-models" not in command
        assert command[command.index("--rows") + 1] == "500"
        assert command[command.index("--seed") + 1] == "3"

    def test_build_grid_command_with_sweep(self, tmp_path: Path):
        args = SimpleNamespace(
            cases_yaml=tmp_path / "cases.yaml",
            rows=1000,
            seed=0,
            with_model_sweep=True,
            target_model="zai-org/GLM-5.2",
            report_path=tmp_path / "report.json",
        )
        command = generate_shape_db.build_grid_command(args, tmp_path / "db")
        assert command[command.index("--target-models") + 1] == "zai-org/GLM-5.2"
        assert command[command.index("--report-path") + 1] == str((tmp_path / "report.json").resolve())

    def test_run_grid_passes_tools_dir_via_env(self, monkeypatch, tmp_path: Path):
        report = tmp_path / "report.json"
        fake = _FakeRun(report)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        args = SimpleNamespace(
            cases_yaml=tmp_path / "cases.yaml",
            rows=10,
            seed=0,
            with_model_sweep=False,
            target_model=None,
            report_path=report,
            max_workers=2,
        )
        generate_shape_db.run_grid(args, tmp_path / "db", 1)
        env = fake.grid_kwargs[0]["env"]
        assert env["MSMODELING_PERF_TOOLS_DIR"] == str(generate_shape_db.TOOLS_DIR)
        # 路径不再内插进 python -c 源码，避免引号/反斜杠破坏子进程语法
        bootstrap = fake.grid_calls[0][2]
        assert str(generate_shape_db.TOOLS_DIR) not in bootstrap
        assert "os.environ['MSMODELING_PERF_TOOLS_DIR']" in bootstrap

    def test_run_grid_requires_generation_report(self, monkeypatch, tmp_path: Path):
        # generate_shape_grid 退出码为 0 但未写出报告时，run_grid 必须显式报错而非裸 traceback
        report = tmp_path / "missing-report.json"
        monkeypatch.setattr(
            generate_shape_db.subprocess,
            "run",
            lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
        )
        args = SimpleNamespace(
            cases_yaml=tmp_path / "cases.yaml",
            rows=10,
            seed=0,
            with_model_sweep=False,
            target_model=None,
            report_path=report,
            max_workers=2,
        )
        with pytest.raises(SystemExit, match="generation report not found"):
            generate_shape_db.run_grid(args, tmp_path / "db", 1)

    def test_database_has_data_rows(self, tmp_path: Path):
        db = tmp_path / "db"
        db.mkdir()
        assert generate_shape_db.database_has_data_rows(db) is False
        empty = db / "Empty.csv"
        empty.write_text("", encoding="utf-8")
        assert generate_shape_db.database_has_data_rows(db) is False
        header_only = db / "HeaderOnly.csv"
        header_only.write_text("Input Shapes\n", encoding="utf-8-sig")
        assert generate_shape_db.database_has_data_rows(db) is False
        with_data = db / "MatMul.csv"
        with_data.write_text("Input Shapes\n1,2,3\n", encoding="utf-8-sig")
        assert generate_shape_db.database_has_data_rows(db) is True


class TestGenerateShapeDbMain:
    def _db(self, tmp_path: Path, with_data: bool = False) -> Path:
        db = tmp_path / "db"
        db.mkdir()
        (db / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        if with_data:
            (db / "MatMul.csv").write_text("Input Shapes\n1,2,3\n", encoding="utf-8-sig")
        return db

    def _argv(self, db: Path, cases: Path, report: Path, extra: list[str] | None = None) -> list[str]:
        argv = [
            "generate_shape_db.py",
            "--database-path",
            str(db),
            "--cases-yaml",
            str(cases),
            "--report-path",
            str(report),
        ]
        return argv + (extra or [])

    def test_requires_existing_cases_yaml(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(db, tmp_path / "missing.yaml", tmp_path / "report.json"),
        )
        with pytest.raises(SystemExit, match="cases yaml not found"):
            generate_shape_db.main()

    def test_requires_op_mapping(self, monkeypatch, tmp_path: Path):
        db = tmp_path / "db"
        db.mkdir()
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        monkeypatch.setattr(sys, "argv", self._argv(db, cases, tmp_path / "report.json"))
        with pytest.raises(SystemExit, match="missing op_mapping.yaml"):
            generate_shape_db.main()

    def test_requires_kernel_details_csv(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        missing_csv = tmp_path / "no_such_kernel_details.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(
                db,
                cases,
                tmp_path / "report.json",
                ["--kernel-details-csv", str(missing_csv), "--max-attempts", "1"],
            ),
        )
        with pytest.raises(SystemExit, match="kernel_details csv not found"):
            generate_shape_db.main()

    def test_skip_parse_and_force_parse_are_mutually_exclusive(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(db, cases, tmp_path / "report.json", ["--skip-parse", "--force-parse"]),
        )
        with pytest.raises(SystemExit, match="mutually exclusive"):
            generate_shape_db.main()

    def test_skip_parse_runs_grid(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        report = tmp_path / "report.json"
        fake = _FakeRun(report)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(db, cases, report, ["--skip-parse", "--max-workers", "1"]),
        )
        generate_shape_db.main()
        assert fake.parse_calls == []
        assert len(fake.grid_calls) == 1

    def test_existing_rows_skip_parse_in_append_mode(self, monkeypatch, capsys, tmp_path: Path):
        db = self._db(tmp_path, with_data=True)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        report = tmp_path / "report.json"
        fake = _FakeRun(report)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        monkeypatch.setattr(sys, "argv", self._argv(db, cases, report))
        generate_shape_db.main()
        assert fake.parse_calls == []
        assert "already contains data rows" in capsys.readouterr().out

    def test_force_parse_reruns_parse_step(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path, with_data=True)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        kernel_details = tmp_path / "kernel_details.csv"
        kernel_details.write_text("Op Name,Input Shapes\n", encoding="utf-8")
        report = tmp_path / "report.json"
        fake = _FakeRun(report)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(
                db,
                cases,
                report,
                ["--force-parse", "--kernel-details-csv", str(kernel_details)],
            ),
        )
        generate_shape_db.main()
        assert len(fake.parse_calls) == 1
        assert "--profiling-path" in fake.parse_calls[0]
        assert len(fake.grid_calls) == 1

    def test_retries_failed_workloads_until_success(self, monkeypatch, capsys, tmp_path: Path):
        db = self._db(tmp_path)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        report = tmp_path / "report.json"
        fake = _FakeRun(report, failures_before_success=2)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(db, cases, report, ["--skip-parse", "--max-attempts", "3"]),
        )
        generate_shape_db.main()
        assert len(fake.grid_calls) == 3
        assert fake.parse_calls == []
        assert "retrying from cache" in capsys.readouterr().out

    def test_exits_after_max_attempts(self, monkeypatch, tmp_path: Path):
        db = self._db(tmp_path)
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])
        report = tmp_path / "report.json"
        fake = _FakeRun(report, always_fail=True)
        monkeypatch.setattr(generate_shape_db.subprocess, "run", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            self._argv(db, cases, report, ["--skip-parse", "--max-attempts", "2"]),
        )
        with pytest.raises(SystemExit, match="still incomplete after 2 attempts"):
            generate_shape_db.main()
        assert len(fake.grid_calls) == 2

    def test_run_raises_on_failed_command(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            generate_shape_db.subprocess,
            "run",
            lambda command, **kwargs: subprocess.CompletedProcess(command, 3),
        )
        with pytest.raises(SystemExit, match="command failed with exit code 3"):
            generate_shape_db.run(["python", "-m", "nope"])


# ---------------------------------------------------------------------------
# mock_replay_durations.py
# ---------------------------------------------------------------------------


class TestMockReplayDurations:
    def _csv(
        self, path: Path, rows: list[str], header: str = "Average Duration(us),Profiling Average Duration(us)"
    ) -> Path:
        path.write_text("\ufeff" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_fills_pending_rows_and_keeps_measured_rows(self, tmp_path: Path):
        path = self._csv(
            tmp_path / "MatMul.csv",
            [
                ",100",  # average missing -> filled; profiling already measured
                "200,300",  # fully measured -> untouched
                "0,50",  # zero average -> filled
            ],
        )
        total, mocked, has_cols = mock_replay_durations.mock_csv(path, 100.0)
        assert (total, mocked, has_cols) == (3, 2, True)
        frame = mock_replay_durations.pd.read_csv(path, encoding="utf-8-sig")
        assert frame["Average Duration(us)"].tolist() == [100.0, 200.0, 100.0]
        assert frame["Profiling Average Duration(us)"].tolist() == [100.0, 300.0, 50.0]

    def test_csv_without_latency_columns_reports_missing(self, tmp_path: Path):
        path = tmp_path / "NoLatency.csv"
        path.write_text("Input Shapes\n1,2,3\n", encoding="utf-8-sig")
        assert mock_replay_durations.mock_csv(path, 100.0) == (1, 0, False)

    def test_output_keeps_utf8_sig_bom(self, tmp_path: Path):
        path = self._csv(tmp_path / "MatMul.csv", [","])
        mock_replay_durations.mock_csv(path, 100.0)
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    def _source_db(self, tmp_path: Path, rows: list[str]) -> Path:
        source = tmp_path / "db"
        source.mkdir()
        (source / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        self._csv(source / "MatMul.csv", rows)
        (source / "notes.txt").write_text("keep me\n", encoding="utf-8")
        return source

    def test_main_requires_existing_source(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(
            sys,
            "argv",
            ["mock.py", "--database-path", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")],
        )
        with pytest.raises(SystemExit, match="database directory not found"):
            mock_replay_durations.main()

    def test_main_requires_op_mapping(self, monkeypatch, tmp_path: Path):
        source = tmp_path / "db"
        source.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["mock.py", "--database-path", str(source), "--output-dir", str(tmp_path / "out")],
        )
        with pytest.raises(SystemExit, match="missing op_mapping.yaml"):
            mock_replay_durations.main()

    def test_main_refuses_existing_output(self, monkeypatch, tmp_path: Path):
        source = self._source_db(tmp_path, [","])
        output = tmp_path / "out"
        output.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["mock.py", "--database-path", str(source), "--output-dir", str(output)],
        )
        with pytest.raises(SystemExit, match="already exists"):
            mock_replay_durations.main()

    def test_main_overwrite_builds_mock_copy(self, monkeypatch, capsys, tmp_path: Path):
        source = self._source_db(tmp_path, [",10", "20,30"])
        (source / "NoLatency.csv").write_text("Input Shapes\n1,2\n", encoding="utf-8-sig")
        output = tmp_path / "out"
        output.mkdir()
        (output / "stale.txt").write_text("old\n", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["mock.py", "--database-path", str(source), "--output-dir", str(output), "--overwrite"],
        )
        mock_replay_durations.main()
        assert not (output / "stale.txt").exists()
        assert (output / "notes.txt").read_text(encoding="utf-8") == "keep me\n"
        frame = mock_replay_durations.pd.read_csv(output / "MatMul.csv", encoding="utf-8-sig")
        assert frame["Average Duration(us)"].tolist() == [100.0, 20.0]
        assert frame["Profiling Average Duration(us)"].tolist() == [10.0, 30.0]
        output_text = capsys.readouterr().out
        assert "mocked 1/2 rows" in output_text
        assert "WARNING: no latency column" in output_text


# ---------------------------------------------------------------------------
# verify_db_coverage.py
# ---------------------------------------------------------------------------


def _verify_args(**overrides) -> SimpleNamespace:
    base = dict(
        model=None,
        device=None,
        num_devices=None,
        tp_size=None,
        ep_size=None,
        quantize_linear_action=None,
        word_embedding_tp=None,
        enable_shared_expert_tp=False,
        database_path=Path("/tmp/db_mockverify"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestVerifyHelpers:
    def test_is_communication_op(self):
        assert verify_db_coverage.is_communication_op("tensor_cast.all_reduce") is True
        assert verify_db_coverage.is_communication_op("tensor_cast.all_gather_into_tensor") is True
        assert verify_db_coverage.is_communication_op("tensor_cast.matmul") is False

    def test_first_candidate(self):
        assert verify_db_coverage._first_candidate([16, 8]) == 16
        assert verify_db_coverage._first_candidate([]) == 1
        assert verify_db_coverage._first_candidate(None) == 1
        assert verify_db_coverage._first_candidate(4) == 4

    def test_load_case_workloads_numbers_by_order(self, tmp_path: Path):
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload(), _base_workload(name="second")])
        workloads = verify_db_coverage.load_case_workloads(cases)
        assert sorted(workloads) == [1, 2]
        assert workloads[2]["name"] == "second"

    def test_load_case_workloads_requires_file(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="cases yaml not found"):
            verify_db_coverage.load_case_workloads(tmp_path / "missing.yaml")

    def test_load_case_workloads_requires_entries(self, tmp_path: Path):
        cases = tmp_path / "cases.yaml"
        cases.write_text("workloads: []\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="has no workloads"):
            verify_db_coverage.load_case_workloads(cases)

    def test_load_op_mapping_kernels_dedupes_candidates(self, tmp_path: Path):
        db = tmp_path / "db"
        db.mkdir()
        (db / "op_mapping.yaml").write_text(
            yaml.safe_dump(
                {
                    "operator_mappings": {
                        "tensor_cast.matmul": {
                            "kernel_type": "MatMulV3",
                            "primary_kernel_type": "MatMulV3",
                            "alternate_kernel_types": ["MatMulCommon", "MatMulV3"],
                        },
                        "tensor_cast.bad": "not-a-dict",
                    }
                }
            ),
            encoding="utf-8",
        )
        mapping = verify_db_coverage.load_op_mapping_kernels(db)
        assert mapping == {"tensor_cast.matmul": ["MatMulV3", "MatMulCommon"]}

    def test_load_op_mapping_kernels_requires_file(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="missing op_mapping.yaml"):
            verify_db_coverage.load_op_mapping_kernels(tmp_path)

    def test_load_replay_kernels(self, tmp_path: Path):
        replay = tmp_path / "op_replay"
        replay.mkdir()
        (replay / "MatMulV3_run.py").write_text("", encoding="utf-8")
        (replay / "notes.txt").write_text("", encoding="utf-8")
        assert verify_db_coverage.load_replay_kernels(replay) == {"MatMulV3"}
        with pytest.raises(SystemExit, match="op_replay directory not found"):
            verify_db_coverage.load_replay_kernels(tmp_path / "missing")


class TestBuildCommand:
    def test_prefill_command_inherits_workload(self):
        workload = _base_workload(enable_shared_expert_tp=True, word_embedding_tp="row", compile=True)
        command = verify_db_coverage.build_command(_verify_args(), workload, 4096, 1024, "prefill")
        assert command[command.index("--query-length") + 1] == "4096"
        assert command[command.index("--context-length") + 1] == "0"
        assert command[command.index("cli.inference.text_generate") + 1] == "zai-org/GLM-5.2"
        assert command[command.index("--tp-size") + 1] == "16"
        assert command[command.index("--num-devices") + 1] == "16"
        assert command[command.index("--quantize-linear-action") + 1] == "W8A8_DYNAMIC"
        assert "--enable-shared-expert-tp" in command
        assert command[command.index("--word-embedding-tp") + 1] == "row"
        assert "--compile" in command
        assert "--prefill" in command
        assert "--decode" not in command
        assert "--log-level" in command

    def test_decode_command_uses_full_window_context(self):
        workload = _base_workload()
        command = verify_db_coverage.build_command(_verify_args(), workload, 4096, 1024, "decode")
        assert command[command.index("--query-length") + 1] == "1"
        assert command[command.index("--context-length") + 1] == str(4096 + 1024)
        assert "--decode" in command
        assert "--prefill" not in command

    def test_cli_overrides_take_precedence(self):
        workload = _base_workload()
        args = _verify_args(model="Qwen/Qwen3-32B", device="TEST_DEVICE", num_devices=8, tp_size=8, ep_size=2)
        command = verify_db_coverage.build_command(args, workload, 1024, 512, "prefill")
        assert "Qwen/Qwen3-32B" in command
        assert command[command.index("--tp-size") + 1] == "8"
        assert command[command.index("--num-devices") + 1] == "8"
        assert command[command.index("--ep-size") + 1] == "2"

    def test_batch_range_uses_upper_bound_for_num_queries(self):
        workload = _base_workload(batch_range=[1, 64])
        command = verify_db_coverage.build_command(_verify_args(), workload, 4096, 1024, "prefill")
        assert command[command.index("--num-queries") + 1] == "64"

    def test_dense_workload_omits_model_specific_flags(self):
        workload = _base_workload(model="Qwen/Qwen3-32B")
        workload.pop("ep_sizes")
        command = verify_db_coverage.build_command(_verify_args(), workload, 1024, 512, "prefill")
        assert "--enable-shared-expert-tp" not in command
        assert "--word-embedding-tp" not in command
        # dense 模型不附加默认值 1 的 --ep-size 噪音 flag
        assert "--ep-size" not in command

    def test_compile_defaults_false_when_workload_omits_field(self):
        command = verify_db_coverage.build_command(_verify_args(), _base_workload(), 4096, 1024, "prefill")
        assert "--compile" not in command

    def test_quantize_and_mtp_flags_inherited_from_workload(self):
        workload = _base_workload(
            quantize_non_expert_linear_action="W4A8_DYNAMIC",
            quantize_attention_action="HIF8",
            num_mtp_tokens=[2],
        )
        command = verify_db_coverage.build_command(_verify_args(), workload, 4096, 1024, "decode")
        assert command[command.index("--quantize-non-expert-linear-action") + 1] == "W4A8_DYNAMIC"
        assert command[command.index("--quantize-attention-action") + 1] == "HIF8"
        assert command[command.index("--num-mtp-tokens") + 1] == "2"

    def test_disabled_quantize_and_missing_mtp_flags_omitted(self):
        workload = _base_workload(
            quantize_non_expert_linear_action="DISABLED",
            quantize_attention_action="DISABLED",
        )
        command = verify_db_coverage.build_command(_verify_args(), workload, 4096, 1024, "prefill")
        assert "--quantize-non-expert-linear-action" not in command
        assert "--quantize-attention-action" not in command
        assert "--num-mtp-tokens" not in command

    def test_missing_required_fields_fail(self):
        workload = _base_workload()
        workload.pop("tp_sizes")
        with pytest.raises(SystemExit, match="workload missing required fields"):
            verify_db_coverage.build_command(_verify_args(), workload, 1024, 512, "prefill")


class TestLogParsing:
    def test_parse_run_log_collects_misses_and_partial(self):
        parsed = verify_db_coverage.parse_run_log(_miss_log())
        funcs = [entry["func_name"] for entry in parsed["misses"]]
        assert funcs.count("tensor_cast.matmul") == 3
        assert "tensor_cast.mla" in funcs
        assert funcs.count("tensor_cast.foo") == 1
        reasons = {entry["func_name"]: entry["reason"] for entry in parsed["misses"]}
        assert reasons["tensor_cast.matmul"] == "shape_mismatch"
        assert reasons["tensor_cast.foo"] == "unmapped"
        assert parsed["partial_line"] == "PARTIAL: 2/10 (tensor_cast.gelu×2)"
        assert parsed["shapes_truncated"] == 5

    def test_parse_run_log_marks_shape_only_misses_as_per_shape(self):
        text = "\n".join(
            [
                "  MISSes (1 unique reasons):",
                "  [shape_mismatch] kernel found, no matching shape in CSV: tensor_cast.only_full",
                "  MISS shapes (1):",
                "  tensor_cast.partial_shape (2, 3)",
            ]
        )
        parsed = verify_db_coverage.parse_run_log(text)
        reasons = {entry["func_name"]: entry["reason"] for entry in parsed["misses"]}
        assert reasons == {"tensor_cast.only_full": "shape_mismatch", "tensor_cast.partial_shape": "per-shape"}

    def test_parse_op_table_reads_totals_and_calls(self):
        entries = verify_db_coverage.parse_op_table(_op_table_log())
        assert [entry["name"] for entry in entries] == ["tensor_cast.matmul", "tensor_cast.all_reduce"]
        matmul = entries[0]
        assert matmul["empirical"] == pytest.approx(2e-3)
        assert matmul["analytic"] == pytest.approx(4e-3)
        assert matmul["calls"] == 2
        # "s" 档：>=1s 的时长也按同一行格式解析
        seconds_log = "\n".join(
            [
                "-" * 96,
                "  Name            empirical total  empirical avg  analytic total  analytic avg  # of Calls",
                "-" * 96,
                "tensor_cast.matmul    1.500s    0.750s    1.500s    0.750s    2",
                "-" * 96,
            ]
        )
        seconds_entries = verify_db_coverage.parse_op_table(seconds_log)
        assert seconds_entries[0]["empirical"] == pytest.approx(1.5)
        assert seconds_entries[0]["analytic"] == pytest.approx(1.5)

    def test_parse_op_table_supports_nanosecond_unit(self):
        # runtime._format_time 对 <1us 的值输出 ns 后缀；缺该档会漏行并截断表体
        text = "\n".join(
            [
                "-" * 96,
                "  Name            empirical total  empirical avg  analytic total  analytic avg  # of Calls",
                "-" * 96,
                "tensor_cast.tiny    120.000ns    60.000ns    120.000ns    60.000ns    2",
                "tensor_cast.matmul    2.000ms    1.000ms    4.000ms    2.000ms    2",
                "-" * 96,
            ]
        )
        entries = verify_db_coverage.parse_op_table(text)
        assert [entry["name"] for entry in entries] == ["tensor_cast.tiny", "tensor_cast.matmul"]
        assert entries[0]["empirical"] == pytest.approx(120e-9)
        assert entries[0]["analytic"] == pytest.approx(120e-9)

    def test_parse_op_table_returns_empty_without_header(self):
        assert verify_db_coverage.parse_op_table("no table here") == []


class TestEvaluateTopTotal:
    def test_passes_when_compute_ops_hit_and_excludes_comm(self):
        entries = verify_db_coverage.parse_op_table(_op_table_log())
        passed, problems, info, ranked = verify_db_coverage.evaluate_top_total(entries, 10)
        assert passed is True
        assert problems == []
        assert any("1 communication op(s) excluded from judgment" in line for line in info)
        judged = {entry["name"]: entry["judged"] for entry in ranked}
        assert judged == {"tensor_cast.matmul": True, "tensor_cast.all_reduce": False}
        assert all("HIT" in line for line in info if "tensor_cast.matmul" in line)

    def test_fails_on_roofline_fallback(self):
        entries = [
            {"name": "tensor_cast.matmul", "empirical": 1e-3, "analytic": 1e-3, "calls": 1},
        ]
        passed, problems, _info, _ranked = verify_db_coverage.evaluate_top_total(entries, 10)
        assert passed is False
        assert len(problems) == 1
        assert "roofline fallback" in problems[0]

    def test_requires_op_summary_table(self):
        passed, problems, info, ranked = verify_db_coverage.evaluate_top_total([], 10)
        assert passed is False
        assert "op summary table not found" in problems[0]
        assert (info, ranked) == ([], [])

    def test_top_n_limits_ranking(self):
        entries = [{"name": f"op{i}", "empirical": 1e-3, "analytic": (i + 1) * 1e-3, "calls": 1} for i in range(5)]
        _passed, _problems, _info, ranked = verify_db_coverage.evaluate_top_total(entries, 2)
        assert [entry["name"] for entry in ranked] == ["op4", "op3"]

    def test_fails_when_top_n_contains_only_communication_ops(self):
        entries = [
            {"name": "tensor_cast.all_reduce", "empirical": 2e-3, "analytic": 2e-3, "calls": 2},
            {"name": "tensor_cast.all_gather", "empirical": 1e-3, "analytic": 1e-3, "calls": 1},
        ]
        passed, problems, info, ranked = verify_db_coverage.evaluate_top_total(entries, 10)
        assert passed is False
        assert problems == ["top-N contains no compute ops to judge"]
        assert ranked and all(entry["judged"] is False for entry in ranked)


class TestEvaluateMissScopes:
    def _parsed(self) -> dict:
        return {
            "misses": [
                {"func_name": "tensor_cast.matmul", "reason": "shape_mismatch"},
                {"func_name": "tensor_cast.mla", "reason": "shape_mismatch"},
                {"func_name": "tensor_cast.foo", "reason": "unmapped"},
            ],
            "partial_line": "PARTIAL: 1/4 (tensor_cast.mla)",
            "shapes_truncated": 0,
        }

    def test_replay_scope_only_fails_replay_supported_kernels(self):
        op_kernels = {"tensor_cast.matmul": ["MatMulV3"], "tensor_cast.foo": ["UnknownKernel"]}
        passed, problems, info = verify_db_coverage.evaluate(self._parsed(), op_kernels, {"MatMulV3"}, "replay")
        assert passed is False
        assert problems == ["tensor_cast.matmul kernel=MatMulV3 reason=shape_mismatch"]
        assert any("unmapped" in line for line in info)
        assert any("UnknownKernel" in line for line in info)
        assert any("partial ops" in line for line in info)

    def test_replay_scope_passes_when_all_misses_are_exempt(self):
        op_kernels = {"tensor_cast.matmul": ["MatMulV3"]}
        parsed = {"misses": [{"func_name": "tensor_cast.unknown", "reason": "unmapped"}], "partial_line": ""}
        passed, problems, info = verify_db_coverage.evaluate(parsed, op_kernels, {"MatMulV3"}, "replay")
        assert passed is True
        assert problems == []
        assert any("exempt" in line for line in info)

    def test_all_scope_fails_on_any_miss(self):
        passed, problems, info = verify_db_coverage.evaluate(self._parsed(), {}, set(), "all")
        assert passed is False
        assert len(problems) == 3
        assert info == []

    def test_all_scope_passes_without_misses(self):
        passed, problems, _info = verify_db_coverage.evaluate({"misses": []}, {}, set(), "all")
        assert (passed, problems) == (True, [])

    def test_classify_misses_groups_exempt_by_kernel(self):
        replayable, exempt = verify_db_coverage.classify_misses(
            self._parsed()["misses"],
            {"tensor_cast.matmul": ["MatMulV3"]},
            {"MatMulV3"},
        )
        assert replayable == [("tensor_cast.matmul", "shape_mismatch", "MatMulV3")]
        assert exempt == {("unmapped", "shape_mismatch"): 1, ("unmapped", "unmapped"): 1}


class TestMockFreshness:
    def _db_with_csv(self, path: Path, rows: int) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "MatMul.csv").write_text(
            "Input Shapes\n" + "\n".join(f"{i},2,3" for i in range(rows)) + "\n",
            encoding="utf-8-sig",
        )
        return path

    def test_row_counts_ignore_empty_csv(self, tmp_path: Path):
        db = self._db_with_csv(tmp_path / "db", 3)
        (db / "Empty.csv").write_text("", encoding="utf-8")
        assert verify_db_coverage.csv_row_counts(db) == {"MatMul.csv": 3}

    def test_reports_missing_and_fewer_rows(self, tmp_path: Path):
        source = self._db_with_csv(tmp_path / "source", 3)
        mock = self._db_with_csv(tmp_path / "mock", 1)
        problems = verify_db_coverage.check_mock_freshness(mock, source)
        assert problems == ["MatMul.csv: mock has 1 rows, source has 3"]

    def test_reports_more_rows(self, tmp_path: Path):
        source = self._db_with_csv(tmp_path / "source", 1)
        mock = self._db_with_csv(tmp_path / "mock", 3)
        problems = verify_db_coverage.check_mock_freshness(mock, source)
        assert problems == ["MatMul.csv: mock has 3 rows, source has 1"]

    def test_equal_row_counts_are_fresh(self, tmp_path: Path):
        source = self._db_with_csv(tmp_path / "source", 3)
        mock = self._db_with_csv(tmp_path / "mock", 3)
        assert verify_db_coverage.check_mock_freshness(mock, source) == []

    def test_reports_missing_csv(self, tmp_path: Path):
        source = self._db_with_csv(tmp_path / "source", 1)
        mock = tmp_path / "mock"
        mock.mkdir()
        assert verify_db_coverage.check_mock_freshness(mock, source) == [
            "MatMul.csv: missing in mock (source has 1 rows)"
        ]


class TestVerifyMain:
    def _cases(self, tmp_path: Path) -> Path:
        return _write_cases_yaml(tmp_path / "cases.yaml", [_base_workload()])

    def _mock_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "db_mockverify"
        db.mkdir()
        (db / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        return db

    def _log(self, db: Path, label: str, text: str) -> Path:
        log_dir = Path(f"{db}-verify-logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{label}_run.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_requires_op_mapping(self, monkeypatch, tmp_path: Path):
        db = tmp_path / "db_mockverify"
        db.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify.py", "--database-path", str(db), "--cases-yaml", str(self._cases(tmp_path))],
        )
        with pytest.raises(SystemExit, match="missing op_mapping.yaml"):
            verify_db_coverage.main()

    def test_unknown_case_id_is_rejected(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--cases",
                "5",
            ],
        )
        with pytest.raises(SystemExit, match="unknown cases"):
            verify_db_coverage.main()

    def test_invalid_case_id_value_is_rejected(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--cases",
                "1,a",
            ],
        )
        with pytest.raises(SystemExit, match="invalid case id"):
            verify_db_coverage.main()

    def test_workload_without_positive_lengths_is_rejected(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        workload = {k: v for k, v in _base_workload().items() if k not in ("input_length", "output_length")}
        cases = _write_cases_yaml(tmp_path / "cases.yaml", [workload])
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify.py", "--database-path", str(db), "--cases-yaml", str(cases)],
        )
        with pytest.raises(SystemExit, match="input_length"):
            verify_db_coverage.main()

    def test_invalid_phase_is_rejected(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "decode-ish",
            ],
        )
        with pytest.raises(SystemExit, match="phases"):
            verify_db_coverage.main()

    def test_dry_run_prints_command_without_summary(self, monkeypatch, capsys, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--dry-run",
            ],
        )
        verify_db_coverage.main()
        output = capsys.readouterr().out
        assert "case1-prefill" in output
        assert "dry-run" in output
        assert not (Path(f"{db}-verify-logs") / "verify-summary.json").exists()

    def test_skip_run_passes_and_writes_summary(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        self._log(db, "case1-prefill", _op_table_log())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--skip-run",
                "--top",
                "5",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 0
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert summary["miss_scope"] == "top-total"
        assert summary["top"] == 5
        assert summary["phases"][0]["passed"] is True
        assert summary["phases"][0]["top_ops"][0]["name"] == "tensor_cast.matmul"

    def test_skip_run_fails_on_roofline_fallback(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        fallback_log = _op_table_log().replace("4.000ms", "2.000ms")
        self._log(db, "case1-prefill", fallback_log)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--skip-run",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 1
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert summary["phases"][0]["passed"] is False
        assert "tensor_cast.matmul" in summary["phases"][0]["problems"][0]

    def test_subprocess_failure_is_recorded(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            verify_db_coverage.subprocess,
            "run",
            lambda command, **kwargs: subprocess.CompletedProcess(command, 2, stdout="", stderr="boom\n"),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 1
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert summary["phases"][0]["returncode"] == 2
        assert summary["phases"][0]["passed"] is False
        assert "exited with 2" in summary["phases"][0]["problems"][0]

    def test_run_success_captures_output_and_parses(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        monkeypatch.setattr(
            verify_db_coverage.subprocess,
            "run",
            lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=_op_table_log(), stderr=""),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 0
        log_file = Path(f"{db}-verify-logs") / "case1-prefill_run.log"
        assert log_file.is_file() and "empirical total" in log_file.read_text(encoding="utf-8")
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert summary["phases"][0]["returncode"] == 0
        assert summary["phases"][0]["top_ops"][0]["calls"] == 2

    def test_replay_scope_main_wiring(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        (db / "op_mapping.yaml").write_text(
            yaml.safe_dump({"operator_mappings": {"tensor_cast.matmul": {"kernel_type": "MatMulV3"}}}),
            encoding="utf-8",
        )
        replay = tmp_path / "op_replay"
        replay.mkdir()
        (replay / "MatMulV3_run.py").write_text("", encoding="utf-8")
        self._log(db, "case1-prefill", _miss_log())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--skip-run",
                "--miss-scope",
                "replay",
                "--op-replay-dir",
                str(replay),
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 1
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert summary["miss_scope"] == "replay"
        assert summary["top"] is None
        assert any("MatMulV3" in problem for problem in summary["phases"][0]["problems"])

    def test_cases_subset_runs_only_selected_case(self, monkeypatch, tmp_path: Path):
        db = self._mock_db(tmp_path)
        cases = _write_cases_yaml(
            tmp_path / "cases.yaml",
            [_base_workload(), _base_workload(name="glm52-case2")],
        )
        self._log(db, "case2-prefill", _op_table_log())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(cases),
                "--cases",
                "2",
                "--phases",
                "prefill",
                "--skip-run",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 0
        summary = json.loads((Path(f"{db}-verify-logs") / "verify-summary.json").read_text(encoding="utf-8"))
        assert [phase["label"] for phase in summary["phases"]] == ["case2-prefill"]
        assert summary["phases"][0]["case_id"] == 2

    def test_fresh_mock_prints_freshness_pass(self, monkeypatch, capsys, tmp_path: Path):
        db = self._mock_db(tmp_path)
        source = tmp_path / "db"
        source.mkdir()
        row = "Input Shapes\n1\n"
        (db / "MatMul.csv").write_text(row, encoding="utf-8-sig")
        (source / "MatMul.csv").write_text(row, encoding="utf-8-sig")
        self._log(db, "case1-prefill", _op_table_log())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--skip-run",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 0
        assert "mock freshness check passed" in capsys.readouterr().out

    def test_stale_mock_is_an_error(self, monkeypatch, tmp_path: Path):
        source = tmp_path / "db"
        source.mkdir()
        (source / "MatMul.csv").write_text("Input Shapes\n1\n2\n", encoding="utf-8-sig")
        db = self._mock_db(tmp_path)
        (db / "MatMul.csv").write_text("Input Shapes\n1\n", encoding="utf-8-sig")
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify.py", "--database-path", str(db), "--cases-yaml", str(self._cases(tmp_path))],
        )
        with pytest.raises(SystemExit, match="STALE"):
            verify_db_coverage.main()

    def test_allow_stale_downgrades_to_warning(self, monkeypatch, capsys, tmp_path: Path):
        source = tmp_path / "db"
        source.mkdir()
        (source / "MatMul.csv").write_text("Input Shapes\n1\n2\n", encoding="utf-8-sig")
        db = self._mock_db(tmp_path)
        (db / "MatMul.csv").write_text("Input Shapes\n1\n", encoding="utf-8-sig")
        self._log(db, "case1-prefill", _op_table_log())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify.py",
                "--database-path",
                str(db),
                "--cases-yaml",
                str(self._cases(tmp_path)),
                "--phases",
                "prefill",
                "--skip-run",
                "--allow-stale",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            verify_db_coverage.main()
        assert excinfo.value.code == 0
        assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generate_run_report.py
# ---------------------------------------------------------------------------


def _summary_payload(tmp_path: Path, passed: bool, miss_scope: str = "top-total") -> dict:
    top_ops = [
        {
            "name": "tensor_cast.matmul",
            "empirical": 2e-3,
            "analytic": 4e-3,
            "calls": 2,
            "hit": True,
            "judged": True,
        },
        {
            "name": "tensor_cast.all_reduce",
            "empirical": 5e-4,
            "analytic": 5e-4,
            "calls": 2,
            "hit": False,
            "judged": False,
        },
    ]
    misses: list[dict] = []
    problems: list[str] = []
    if not passed:
        top_ops.append(
            {
                "name": "tensor_cast.mlp",
                "empirical": 6e-3,
                "analytic": 6e-3,
                "calls": 1,
                "hit": False,
                "judged": True,
            }
        )
        problems = ["tensor_cast.mlp analytic=6.000ms (roofline fallback, empirical == analytic)"]
        misses = [{"func_name": "tensor_cast.mlp", "reason": "shape_mismatch"}]
    log_dir = tmp_path / "db_mockverify-verify-logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "case1-prefill_run.log"
    log_file.write_text("log\n", encoding="utf-8")
    return {
        "database": str(tmp_path / "db_mockverify"),
        "source_database": str(tmp_path / "db"),
        "cases_yaml": str(tmp_path / "cases.yaml"),
        "miss_scope": miss_scope,
        "top": 10,
        "phases": [
            {
                "label": "case1-prefill",
                "case_id": 1,
                "phase": "prefill",
                "command": ["python", "-m", "cli.inference.text_generate", "zai-org/GLM-5.2"],
                "log_file": str(log_file),
                "returncode": 0,
                "passed": passed,
                "problems": problems,
                "info": [
                    "top-3 ops by analytic total (1 communication op(s) excluded from judgment: "
                    "tensor_cast.all_reduce(500.000us)):"
                ],
                "top_ops": top_ops,
                "misses": misses,
                "partial_line": "",
            }
        ],
    }


class TestReportHelpers:
    def test_rel_path_inside_repo(self):
        assert generate_run_report.rel_path(REPO_ROOT / "tests") == str(Path("tests"))

    def test_rel_path_outside_repo_is_absolute(self, tmp_path: Path):
        assert generate_run_report.rel_path(tmp_path) == str(tmp_path)

    def test_format_duration_switches_unit_by_magnitude(self):
        assert generate_run_report._format_duration(2e-3) == "2.000ms"
        assert generate_run_report._format_duration(5e-4) == "0.500ms"
        assert generate_run_report._format_duration(1.2e-7) == "0.120us"

    def test_extract_comm_note(self, tmp_path: Path):
        entry = _summary_payload(tmp_path, True)["phases"][0]
        assert (
            generate_run_report.extract_comm_note(entry)
            == "1 communication op(s) excluded from judgment: tensor_cast.all_reduce(500.000us)"
        )
        assert generate_run_report.extract_comm_note({"info": []}) is None

    def test_render_generation_overview_missing_report(self, tmp_path: Path):
        assert generate_run_report.render_generation_overview(tmp_path / "missing.json") == [
            "- 生成报告不存在（生成阶段未运行或路径变更）。"
        ]

    def test_render_generation_overview_lists_totals(self, tmp_path: Path):
        report = tmp_path / "gen.json"
        report.write_text(
            json.dumps({"totals": {"demand_rows": 5}, "unsupported_kernel_demands": []}),
            encoding="utf-8",
        )
        lines = generate_run_report.render_generation_overview(report)
        assert lines[0] == f"- 报告文件：`{report}`"
        assert "- demand_rows: 5" in lines
        assert not any("unsupported_kernel_demands" in line for line in lines)

    def test_render_miss_in_top_lists_only_failed_compute_ops(self, tmp_path: Path):
        entry = _summary_payload(tmp_path, False)["phases"][0]
        lines = generate_run_report.render_miss_in_top(entry, 10)
        text = "\n".join(lines)
        assert "tensor_cast.mlp" in text
        assert "tensor_cast.all_reduce" not in text
        assert "MISS" in text

    def test_render_miss_in_top_without_misses(self, tmp_path: Path):
        entry = _summary_payload(tmp_path, True)["phases"][0]
        assert generate_run_report.render_miss_in_top(entry, 10) == ["- case TOP10中无 MISS 的计算算子。"]

    def test_render_miss_by_operator_aggregates_and_maps_root_cause(self, tmp_path: Path):
        summary = _summary_payload(tmp_path, False)
        lines = generate_run_report.render_miss_by_operator(summary)
        text = "\n".join(lines)
        assert "### tensor_cast.mlp" in text
        assert "MISS 原因：shape_mismatch" in text
        assert "根因方向：数据库缺该 shape 行" in text
        assert "MISS次数：1 次" in text
        assert "case1-prefill" in text

    def test_render_miss_by_operator_reports_all_hit(self, tmp_path: Path):
        summary = _summary_payload(tmp_path, True)
        assert generate_run_report.render_miss_by_operator(summary) == ["", "- 所有 TOP10 计算算子均命中，无 MISS。"]

    def test_replay_scope_miss_by_operator_uses_miss_list(self, tmp_path: Path):
        # replay/all 口径下 summary 无 top_ops：不得误报"所有 Top-10 计算算子均命中"
        summary = _summary_payload(tmp_path, False, miss_scope="replay")
        for phase in summary["phases"]:
            phase["top_ops"] = []
        text = "\n".join(generate_run_report.render_miss_by_operator(summary))
        assert "验收口径为 `replay`" in text
        assert "tensor_cast.mlp" in text
        assert "所有 Top-10 计算算子均命中" not in text

    def test_header_summary_ignores_none_passed(self, tmp_path: Path):
        # 防御：passed=None（dry-run 产物）不计为 FAIL
        summary = _summary_payload(tmp_path, True)
        summary["phases"][0]["passed"] = None
        lines = generate_run_report.render_header_summary(summary, 10)
        assert any("ALL PASS" in line for line in lines)

    def test_fold_phases_does_not_merge_replay_entries_with_different_misses(self, tmp_path: Path):
        # replay/all 口径下 top_ops 恒为空：不同 MISS 集合的同 phase 不得被折叠为"完全相同"
        summary = _summary_payload(tmp_path, False, miss_scope="replay")
        second = dict(summary["phases"][0])
        second["label"] = "case2-prefill"
        second["log_file"] = str(tmp_path / "db_mockverify-verify-logs" / "case2-prefill_run.log")
        second["misses"] = [{"func_name": "tensor_cast.attn", "reason": "shape_mismatch"}]
        summary["phases"].append(second)
        groups = generate_run_report.fold_phases(summary)
        assert [group[0]["label"] for group in groups] == ["case1-prefill", "case2-prefill"]

    def test_render_miss_by_operator_falls_back_for_unknown_reason(self, tmp_path: Path):
        summary = _summary_payload(tmp_path, False)
        summary["phases"][0]["misses"][0]["reason"] = "brand_new_reason"
        text = "\n".join(generate_run_report.render_miss_by_operator(summary))
        assert "MISS 原因：brand_new_reason" in text
        assert "原因未收录" in text

    def test_fold_phases_groups_identical_results(self, tmp_path: Path):
        summary = _summary_payload(tmp_path, True)
        second = json.loads(json.dumps(summary["phases"][0]))
        second["label"] = "case2-prefill"
        summary["phases"].append(second)
        groups = generate_run_report.fold_phases(summary)
        assert len(groups) == 1
        assert [entry["label"] for entry in groups[0]] == ["case1-prefill", "case2-prefill"]

    def test_fold_phases_keeps_different_phases_apart(self, tmp_path: Path):
        summary = _summary_payload(tmp_path, True)
        second = json.loads(json.dumps(summary["phases"][0]))
        second["label"] = "case1-decode"
        second["phase"] = "decode"
        summary["phases"].append(second)
        assert len(generate_run_report.fold_phases(summary)) == 2


class TestReportMain:
    def _prepare(self, tmp_path: Path, passed: bool) -> tuple[Path, Path]:
        db = tmp_path / "db"
        db.mkdir()
        (db / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        summary = _summary_payload(tmp_path, passed)
        summary["database"] = str(db)
        summary_path = tmp_path / "db_mockverify-verify-logs" / "verify-summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        generation = tmp_path / "db-shape-generation-report.json"
        generation.write_text(
            json.dumps(
                {
                    "totals": {"demand_rows": 10, "inserted_rows": 9, "skipped_rows": 1},
                    "unsupported_kernel_demands": [{"kernel_type": "MatMulCommon", "demand_count": 1}],
                }
            ),
            encoding="utf-8",
        )
        return db, summary_path

    def test_requires_op_mapping(self, monkeypatch, tmp_path: Path):
        db = tmp_path / "db"
        db.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["report.py", "--database-path", str(db)],
        )
        with pytest.raises(SystemExit, match="missing op_mapping.yaml"):
            generate_run_report.main()

    def test_requires_verify_summary(self, monkeypatch, tmp_path: Path):
        db = tmp_path / "db"
        db.mkdir()
        (db / "op_mapping.yaml").write_text("operator_mappings: {}\n", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["report.py", "--database-path", str(db), "--summary-json", str(tmp_path / "missing.json")],
        )
        with pytest.raises(SystemExit, match="verify summary not found"):
            generate_run_report.main()

    def test_pass_report_contains_all_sections(self, monkeypatch, tmp_path: Path):
        db, summary_path = self._prepare(tmp_path, passed=True)
        report_path = tmp_path / "out" / "run-report.md"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "report.py",
                "--database-path",
                str(db),
                "--summary-json",
                str(summary_path),
                "--generation-report",
                str(tmp_path / "db-shape-generation-report.json"),
                "--report-path",
                str(report_path),
            ],
        )
        generate_run_report.main()
        text = report_path.read_text(encoding="utf-8")
        for heading in (
            "## 1. 产物文件说明",
            "## 2. 仿真命令",
            "## 3. 仿真判定结果",
            "## 4. MISS 计算算子总览（按 analytic 总时长降序）",
            "## 5. 生成阶段概览",
            "## 结论",
        ):
            assert heading in text
        assert "ALL PASS" in text
        assert "tensor_cast.all_reduce" in text
        assert "demand_rows: 10" in text
        assert "unsupported_kernel_demands: 1 项" in text

    def test_fail_report_lists_misses_and_root_cause(self, monkeypatch, tmp_path: Path):
        db, summary_path = self._prepare(tmp_path, passed=False)
        report_path = tmp_path / "run-report.md"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "report.py",
                "--database-path",
                str(db),
                "--summary-json",
                str(summary_path),
                "--report-path",
                str(report_path),
            ],
        )
        generate_run_report.main()
        text = report_path.read_text(encoding="utf-8")
        assert "**FAIL**" in text
        assert "tensor_cast.mlp" in text
        assert "shape_mismatch" in text
        assert "数据库缺该 shape 行" in text
        assert "| 1 | `tensor_cast.mlp`" in text

    def test_replay_scope_report_lists_misses_without_top_table(self, monkeypatch, tmp_path: Path):
        # replay 口径 summary 不带 top_ops：报告第 3/4 节必须展示 MISS 清单而非"无 MISS"
        db, summary_path = self._prepare(tmp_path, passed=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["miss_scope"] = "replay"
        summary["top"] = None
        for phase in summary["phases"]:
            phase["top_ops"] = []
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        report_path = tmp_path / "run-report.md"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "report.py",
                "--database-path",
                str(db),
                "--summary-json",
                str(summary_path),
                "--report-path",
                str(report_path),
            ],
        )
        generate_run_report.main()
        text = report_path.read_text(encoding="utf-8")
        assert "## 4. MISS 总览（replay 口径，按算子聚合）" in text
        assert "本次验收口径为 `replay`" in text
        assert "tensor_cast.mlp" in text
        assert "所有 Top-10 计算算子均命中" not in text
        # 头部 FAIL 判定不得沿用 Top-N roofline 文案或误报"无缺口"
        assert "回放/查库缺口" in text
        assert "MISS 算子：tensor_cast.mlp" in text
        assert "无计算算子缺口" not in text
