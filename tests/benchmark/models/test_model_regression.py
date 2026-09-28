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

import contextlib
import io
import json
import logging
import re
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional
from unittest import mock

import torch
from parameterized import parameterized
from tensor_cast.core.compilation_config import apply_compilation_config
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig
from tests.helpers.model_assets import resolve_offline_model_id

logger = logging.getLogger(__name__)

CASE_DIR = Path(__file__).resolve().parent / "cases"
MAX_TEXT_BASELINE_INCREASE_PCT = 0.05
MAX_NON_TEXT_BASELINE_INCREASE_PCT = 0.20


class _TimeResultRow(NamedTuple):
    """One row of ``TestPerformanceRegression._time_results`` (Test 1).

    ``grouped=True`` marks a group sub-case: breaches are downgraded to WARN
    and the Test 1 summary excludes the row (Test 3 reports it in detail).
    """

    name: str
    result_overall: str
    actual: float
    init_time: float
    init_diff: float
    base_time: float
    base_diff: float
    status: str
    grouped: bool


@dataclass
class BasePerfRegressionCase:
    name: str
    description: str
    initial_time_s: float = 0.0
    baseline_time_s: float = 0.0
    initial_tolerance: float = 0.10
    baseline_tolerance: float = 0.20
    baseline_max_increase_pct: Optional[float] = None
    operator_top_n: int = 10
    operator_tolerance: float = 0.10
    operators: list[dict[str, float]] = None


@dataclass
class TextPerfRegressionCase(BasePerfRegressionCase):
    user_input: Optional[UserInputConfig] = None
    compilation_config: list[str] = field(default_factory=list)
    from_group: bool = False
    """Optional list of --compilation-config options to apply before running this case.

    Empty list (default) means "no --compilation-config" — all four flags default to
    False, matching the new unified default in :func:`apply_compilation_config`. Cases
    whose baseline was recorded under a different default (e.g. matmul-allreduce on)
    must opt in here to keep the baseline comparison valid.

    ``from_group`` marks a sub-case embedded in a group JSON: its operator baseline
    is read from the group JSON's ``sub_cases[key].operators``, not from a
    standalone ``{name}.json`` file.
    """


@dataclass
class VideoPerfRegressionCase(BasePerfRegressionCase):
    device: str = ""
    model_id: str = ""
    seq_len: int = 0
    batch_size: int = 0
    height: int = 0
    width: int = 0
    frame_num: int = 0
    sample_step: int = 0
    dtype: str = "float16"
    use_cfg: bool = False
    world_size: int = 1
    ulysses_size: int = 1
    cfg_parallel: bool = False
    quantize_linear_action: QuantizeLinearAction = QuantizeLinearAction.DISABLED


@dataclass
class GroupPerfRegressionCase:
    """Group-average precision judgment case.

    Declared by a standalone JSON with ``"type": "group"``. Sub-cases are
    embedded under ``sub_cases`` keyed by sub-case name; each value follows the
    same structure as a standalone text-case JSON (without ``name``/``type``).
    Sub-cases are executed sequentially inside the group's test method, then
    judged by the mean of per-sub-case |baseline diff| against the group
    tolerance.
    """

    name: str
    description: str = ""
    group_tolerance: Optional[float] = None
    """Group-level tolerance. None falls back to the minimum of the sub-cases'
    baseline_tolerance (conservative direction)."""
    sub_cases: dict[str, "TextPerfRegressionCase"] = field(default_factory=dict)


def _parse_total_time_s(table_result: str, performance_model_name: str = "analytic") -> float:
    pattern = rf"Total time for {performance_model_name}:\s*([\d.]+)\s*(ns|us|ms|s)"
    m = re.search(pattern, table_result)
    if not m:
        raise ValueError(f"Could not find 'Total time for {performance_model_name}' in output:\n{table_result}")
    value = float(m.group(1))
    unit = m.group(2)
    return value * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


def _is_baseline_time_acceptable(
    actual_time_s: float,
    baseline_time_s: float,
    tolerance: float,
    max_increase_pct: float,
    performance_model: str = "analytic",
) -> bool:
    """Allow Actual to exceed the baseline by the configured case-type limit.

    Profiling-mode cases (``performance_model == "profiling"``) use an empirical
    (measured) baseline, so the "analytic is optimal" increase cap does not
    apply and only the symmetric tolerance is enforced.
    """
    if baseline_time_s <= 0.0:
        return True
    diff_pct = (actual_time_s - baseline_time_s) / baseline_time_s
    if performance_model == "analytic":
        return diff_pct <= max_increase_pct and abs(diff_pct) <= tolerance
    return abs(diff_pct) <= tolerance


def _parse_top_operators(
    table_result: str,
    top_n: int = 10,
    performance_model_name: str = "analytic",
) -> list[tuple[str, float, int]]:
    lines = table_result.split("\n")
    data_started = False
    operators: list[tuple[str, float, int]] = []

    for line in lines:
        if f"{performance_model_name} total" in line and f"{performance_model_name} avg" in line:
            data_started = True
            continue
        if not data_started:
            continue
        if line.startswith("-") and operators:
            break
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        op_name = parts[0]
        time_str = parts[1] if len(parts) > 1 else ""
        calls_str = parts[3] if len(parts) > 3 else "0"

        m = re.match(r"([\d.]+)\s*(ns|us|ms|s)", time_str)
        if not m:
            continue

        value = float(m.group(1))
        unit = m.group(2)
        time_s = value * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
        num_calls = int(calls_str)
        operators.append((op_name, time_s, num_calls))

    return operators[:top_n]


def _resolve_table_performance_model_name(case: BasePerfRegressionCase) -> str:
    """Resolve the summary-table performance model name to parse for a case.

    Profiling-mode runs (``performance_model`` containing ``"profiling"``) report
    the empirical model as ``empirical`` in the table; analytic-only runs report
    ``analytic``. The guardrail target is the empirical total time when the
    profiling database is in use.
    """
    user_input = getattr(case, "user_input", None)
    performance_model = getattr(user_input, "performance_model", None) if user_input is not None else None
    if performance_model and "profiling" in performance_model:
        return "empirical"
    return "analytic"


def _run_case_simulation(case: BasePerfRegressionCase) -> tuple[str, float]:
    """Run one case simulation and return ``(table_result, actual_time_s)``.

    Shared by standalone cases (``test_performance_regression``) and group
    sub-cases (``test_group_precision_judgment``).
    """
    torch.compiler.reset()
    table_performance_model_name = _resolve_table_performance_model_name(case)

    if isinstance(case, VideoPerfRegressionCase):
        from cli.inference.video_generate import (
            run_inference as video_run_inference,
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            video_run_inference(
                device=case.device,
                model_id=case.model_id,
                batch_size=case.batch_size,
                seq_len=case.seq_len,
                height=case.height,
                width=case.width,
                frame_num=case.frame_num,
                sample_step=case.sample_step,
                dtype=case.dtype,
                use_cfg=case.use_cfg,
                world_size=case.world_size,
                ulysses_size=case.ulysses_size,
                cfg_parallel=case.cfg_parallel,
                quantize_linear_action=case.quantize_linear_action,
            )
        table_result = buf.getvalue()
    else:
        # Apply case-level compilation_config so the model is built under the
        # same flags the baseline was recorded under. An empty list explicitly
        # resets all four flags to False (the new unified default).
        apply_compilation_config(case.compilation_config)
        model_runner = ModelRunner(case.user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)

        if isinstance(result, ModelRunnerMetrics):
            table_result = result.table_result
        else:
            raise RuntimeError(f"Unexpected result type: {type(result)}")

    actual_time_s = _parse_total_time_s(table_result, performance_model_name=table_performance_model_name)
    logger.info(
        "[%s] Actual total time (%s): %.6fs (%.3fms)",
        case.name,
        table_performance_model_name,
        actual_time_s,
        actual_time_s * 1000,
    )
    return table_result, actual_time_s


def _format_time_anomaly(
    case: BasePerfRegressionCase,
    actual_time_s: float,
    initial_passed: bool,
    baseline_passed: bool,
    initial_diff_pct: float,
    baseline_diff_pct: float,
    breach_label: str = "FAIL",
) -> list[str]:
    """Build the human-readable detail lines for a time-tolerance breach."""
    lines = [
        f"  Description: {case.description}",
        f"  Actual:   {actual_time_s * 1000:.3f}ms",
    ]
    if case.initial_time_s > 0.0:
        lines.append(
            f"  vs Initial: {case.initial_time_s * 1000:.3f}ms "
            f"({initial_diff_pct * 100:+.2f}%, tolerance: ±{case.initial_tolerance * 100:.0f}%) "
            f"{'PASS' if initial_passed else breach_label}"
        )
    if case.baseline_time_s > 0.0:
        lines.append(
            f"  vs Baseline: {case.baseline_time_s * 1000:.3f}ms "
            f"({baseline_diff_pct * 100:+.2f}%, tolerance: ±{case.baseline_tolerance * 100:.0f}%) "
            f"{'PASS' if baseline_passed else breach_label}"
        )
    return lines


def _load_baseline_operators(case_name: str) -> Optional[dict[str, dict[str, float]]]:
    filepath = CASE_DIR / f"{case_name}.json"
    if not filepath.exists():
        return None
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    operators = data.get("operators", [])
    if not operators:
        return None
    return {
        op["name"]: {
            "total_time_s": op["total_time_s"],
            "num_calls": op.get("num_calls", 0),
        }
        for op in operators
    }


def _baseline_operators_from_case(case: BasePerfRegressionCase) -> Optional[dict[str, dict[str, float]]]:
    """Operator baseline embedded in the case object (group sub-cases read it
    from the group JSON's ``sub_cases[key].operators``).
    """
    if not case.operators:
        return None
    return {
        op["name"]: {
            "total_time_s": op["total_time_s"],
            "num_calls": op.get("num_calls", 0),
        }
        for op in case.operators
    }


def _save_baseline_operators(case_name: str, operators: list[tuple[str, float, int]]):
    filepath = CASE_DIR / f"{case_name}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Case file not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    data["operators"] = [
        {"name": name, "total_time_s": time_s, "num_calls": num_calls} for name, time_s, num_calls in operators
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _build_text_case(data: dict, name: str, from_group: bool = False) -> TextPerfRegressionCase:
    data = dict(data)
    data.pop("name", None)
    data.pop("type", None)
    ui_data = data.pop("user_input", {})
    if ui_data.get("model_id"):
        ui_data["model_id"] = resolve_offline_model_id(ui_data["model_id"])
    for key in ("quantize_linear_action", "quantize_attention_action"):
        if key in ui_data and isinstance(ui_data[key], str):
            if key == "quantize_linear_action":
                ui_data[key] = QuantizeLinearAction[ui_data[key]]
            else:
                ui_data[key] = QuantizeAttentionAction[ui_data[key]]
    user_input = UserInputConfig(**ui_data)
    compilation_config = data.pop("compilation_config", [])
    return TextPerfRegressionCase(
        name=name,
        user_input=user_input,
        compilation_config=compilation_config,
        from_group=from_group,
        **data,
    )


def _load_perf_regression_cases() -> tuple[list[BasePerfRegressionCase], list[GroupPerfRegressionCase]]:
    cases: list[BasePerfRegressionCase] = []
    group_cases: list[GroupPerfRegressionCase] = []
    for path in sorted(CASE_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        case_type = data.pop("type", "text")
        if case_type == "video":
            data["quantize_linear_action"] = QuantizeLinearAction[data["quantize_linear_action"]]
            if data.get("model_id") and not Path(data["model_id"]).is_absolute():
                candidate = (Path(__file__).resolve().parent / data["model_id"]).resolve()
                if not candidate.exists():
                    candidate = (Path(__file__).resolve().parents[2] / data["model_id"]).resolve()
                data["model_id"] = str(candidate)
            cases.append(VideoPerfRegressionCase(**data))
        elif case_type == "group":
            name = data.pop("name")
            sub_raw = data.pop("sub_cases", None) or {}
            if not sub_raw:
                logger.warning(
                    'Group case "%s" (%s) declares an empty "sub_cases"; it is an invalid '
                    "configuration and the group test will fail at runtime.",
                    name,
                    path.name,
                )
            sub_cases: dict[str, TextPerfRegressionCase] = {}
            for sub_name, sub_data in sub_raw.items():
                sub_cases[sub_name] = _build_text_case(sub_data, name=sub_name, from_group=True)
            group_cases.append(
                GroupPerfRegressionCase(
                    name=name,
                    description=data.pop("description", ""),
                    group_tolerance=data.pop("group_tolerance", None),
                    sub_cases=sub_cases,
                )
            )
        else:
            name = data.get("name", path.stem)
            cases.append(_build_text_case(data, name=name))
    return cases, group_cases


PERF_REGRESSION_CASES, GROUP_PERF_CASES = _load_perf_regression_cases()


class TestPerformanceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] [%(name)s] %(message)s",
        )
        cls._time_results: list[_TimeResultRow] = []
        cls._op_results: list[dict] = []
        cls._op_detail_rows: list[tuple] = []
        cls._group_rows: list[dict] = []
        # Per-group judgment rows, appended by test_group_precision_judgment:
        # {"group", "tolerance", "avg", "members": [subcase detail dict], "status"}
        # where each member carries the sub-case detail fields rendered by the
        # Test 3 summary: {"case", "actual", "init", "init_diff", "baseline",
        # "base_diff", "status"}.
        # Ensure a clean compilation state before any case runs.
        apply_compilation_config([])

    def setUp(self):
        # Reset compilation_config between cases so the previous case's flags
        # never leak into the next. Each case explicitly re-applies its own
        # compilation_config (if any) via apply_compilation_config below.
        apply_compilation_config([])

    def tearDown(self):
        # Do not leak case-level compilation flags to an unrelated test that
        # xdist schedules next in the same worker process.
        apply_compilation_config([])

    @classmethod
    def tearDownClass(cls):
        _print_time_summary(cls._time_results)
        _print_operator_summary(cls._op_results, cls._op_detail_rows)
        _print_group_summary(cls._group_rows)

    def _record_time_comparison(self, case, name: str, actual_time_s: float, grouped: bool) -> dict:
        """Test 1: total time comparison; append a row to ``_time_results``.

        ``grouped=True`` downgrades breach labels/results to WARN (group
        members are judged by the group average, not individually) and marks
        the row so the Test 1 summary excludes it; grouped sub-cases are
        reported in full detail by the Test 3 group summary.
        """
        initial_passed = True
        baseline_passed = True
        initial_diff_pct = 0.0
        baseline_diff_pct = 0.0

        if case.initial_time_s > 0.0:
            initial_diff_pct = (actual_time_s - case.initial_time_s) / case.initial_time_s
            initial_passed = abs(initial_diff_pct) <= case.initial_tolerance

        if case.baseline_time_s > 0.0:
            baseline_diff_pct = (actual_time_s - case.baseline_time_s) / case.baseline_time_s
            performance_model = _resolve_table_performance_model_name(case)
            if performance_model == "analytic":
                max_baseline_increase_pct = (
                    case.baseline_max_increase_pct
                    if case.baseline_max_increase_pct is not None
                    else (
                        MAX_NON_TEXT_BASELINE_INCREASE_PCT
                        if isinstance(case, VideoPerfRegressionCase)
                        else MAX_TEXT_BASELINE_INCREASE_PCT
                    )
                )
            else:
                # Profiling cases skip the increase cap; the placeholder is
                # ignored by _is_baseline_time_acceptable.
                max_baseline_increase_pct = 0.0
            judgment_mode = "profiling" if performance_model == "empirical" else "analytic"
            baseline_passed = _is_baseline_time_acceptable(
                actual_time_s,
                case.baseline_time_s,
                case.baseline_tolerance,
                max_baseline_increase_pct,
                performance_model=judgment_mode,
            )

        time_overall = initial_passed and baseline_passed

        if case.initial_time_s == 0.0 and case.baseline_time_s == 0.0:
            time_status = "NO_BASELINE"
            result_overall = "NO_BASELINE"
        elif not initial_passed and not baseline_passed:
            time_status = "WARN(BOTH)" if grouped else "FAIL(BOTH)"
            result_overall = "WARN" if grouped else "FAIL"
        elif not initial_passed:
            time_status = "WARN(INIT)" if grouped else "FAIL(INIT)"
            result_overall = "WARN" if grouped else "FAIL"
        elif not baseline_passed:
            time_status = "WARN(BASE)" if grouped else "FAIL(BASE)"
            result_overall = "WARN" if grouped else "FAIL"
        else:
            time_status = "PASS"
            result_overall = "PASS"

        self.__class__._time_results.append(
            _TimeResultRow(
                name=name,
                result_overall=result_overall,
                actual=actual_time_s,
                init_time=case.initial_time_s,
                init_diff=initial_diff_pct * 100,
                base_time=case.baseline_time_s,
                base_diff=baseline_diff_pct * 100,
                status=time_status,
                grouped=grouped,
            )
        )
        return {
            "initial_passed": initial_passed,
            "baseline_passed": baseline_passed,
            "initial_diff_pct": initial_diff_pct,
            "baseline_diff_pct": baseline_diff_pct,
            "time_overall": time_overall,
            "time_status": time_status,
        }

    def _compare_operators(self, case, name: str, table_result: str, baseline_operators: Optional[dict]) -> dict:
        """Test 2: Top-N operator-level comparison (warning only).

        Returns the per-case result dict. Standalone cases additionally feed
        the class-level Test 2 summary lists; group sub-cases only return the
        result (their operator details are rendered by the Test 3 group
        summary instead, see ``_print_group_summary``).
        """
        table_performance_model_name = _resolve_table_performance_model_name(case)
        actual_operators = _parse_top_operators(
            table_result, top_n=case.operator_top_n, performance_model_name=table_performance_model_name
        )

        detail_rows: list[tuple] = []
        violations: list[str] = []
        op_passed = None

        if baseline_operators is None:
            if getattr(case, "from_group", False):
                baseline_source = "the group JSON's sub_cases[key].operators"
            else:
                baseline_source = str(CASE_DIR / f"{name}.json")
            logger.warning(
                "[%s] No operator baseline found in %s. "
                "Please generate baseline explicitly before running regression tests.",
                name,
                baseline_source,
            )
            return {
                "case_name": name,
                "op_passed": None,
                "op_status": "NO_BASELINE",
                "violations": [],
                "detail_rows": detail_rows,
            }
        else:
            baseline_top_n = dict(
                sorted(
                    baseline_operators.items(),
                    key=lambda item: item[1]["total_time_s"],
                    reverse=True,
                )[: case.operator_top_n]
            )

            actual_op_names = {op_name for op_name, _, _ in actual_operators}
            baseline_op_names = set(baseline_top_n.keys())

            missing_from_actual = baseline_op_names - actual_op_names
            for op_name in sorted(missing_from_actual):
                bl = baseline_top_n[op_name]
                violations.append(
                    f"  MISSING OPERATOR: {op_name} (baseline={bl['total_time_s'] * 1000:.3f}ms, #calls={bl['num_calls']}) - Not found in current results"
                )
                detail_rows.append(
                    (
                        name,
                        op_name,
                        f"{bl['total_time_s'] * 1000:.3f}ms",
                        "MISSING",
                        "N/A",
                        "WARN",
                        str(bl["num_calls"]),
                        "-",
                    )
                )

            for op_name, actual_op_time, actual_num_calls in actual_operators:
                baseline = baseline_top_n.get(op_name)
                if baseline is None:
                    violations.append(
                        f"  NEW OPERATOR: {op_name} (actual={actual_op_time * 1000:.3f}ms, #calls={actual_num_calls}) - Not in baseline Top {case.operator_top_n}"
                    )
                    detail_rows.append(
                        (
                            name,
                            op_name,
                            "N/A",
                            f"{actual_op_time * 1000:.3f}ms",
                            "NEW",
                            "WARN",
                            "-",
                            str(actual_num_calls),
                        )
                    )
                    continue
                baseline_op_time = baseline["total_time_s"]
                baseline_num_calls = baseline["num_calls"]
                if baseline_op_time == 0.0:
                    detail_rows.append(
                        (
                            name,
                            op_name,
                            "0.000ms",
                            f"{actual_op_time * 1000:.3f}ms",
                            "N/A",
                            "PASS",
                            "0",
                            str(actual_num_calls),
                        )
                    )
                    continue
                op_diff_pct = (actual_op_time - baseline_op_time) / baseline_op_time
                time_fail = abs(op_diff_pct) > case.operator_tolerance
                calls_mismatch = baseline_num_calls > 0 and actual_num_calls != baseline_num_calls
                op_row_status = "PASS" if (not time_fail and not calls_mismatch) else "WARN"
                if op_row_status == "WARN":
                    violations.append(
                        f"  {op_name}: {op_diff_pct * 100:+.2f}% "
                        f"(baseline={baseline_op_time * 1000:.3f}ms, actual={actual_op_time * 1000:.3f}ms)"
                    )
                if baseline_num_calls > 0 and actual_num_calls != baseline_num_calls:
                    violations.append(
                        f"  {op_name}: #CALLS MISMATCH baseline={baseline_num_calls}, actual={actual_num_calls}"
                    )
                calls_str = f"{baseline_num_calls}/{actual_num_calls}{'!' if baseline_num_calls > 0 and actual_num_calls != baseline_num_calls else ''}"
                detail_rows.append(
                    (
                        name,
                        op_name,
                        f"{baseline_op_time * 1000:.3f}ms",
                        f"{actual_op_time * 1000:.3f}ms",
                        f"{op_diff_pct * 100:+.2f}%",
                        op_row_status,
                        str(baseline_num_calls),
                        calls_str,
                    )
                )

            op_passed = len(violations) == 0
            op_status = "PASS" if op_passed else "WARN"

            logger.info(
                "[%s] Top-%d Operator Comparison (tolerance: ±%.0f%%):",
                name,
                case.operator_top_n,
                case.operator_tolerance * 100,
            )
            logger.info(
                "  %-50s %10s %10s %10s %8s",
                "Operator",
                "Baseline",
                "Actual",
                "Diff%",
                "#Calls",
            )
            logger.info("  %s %s %s %s %s", "-" * 50, "-" * 10, "-" * 10, "-" * 10, "-" * 8)
            for op_name, actual_op_time, actual_num_calls in actual_operators:
                baseline = baseline_top_n.get(op_name)
                if baseline is None:
                    logger.info(
                        "  %-50s %10s %9.3fms %10s %8d",
                        op_name,
                        "N/A",
                        actual_op_time * 1000,
                        "NEW",
                        actual_num_calls,
                    )
                elif baseline["total_time_s"] == 0.0:
                    logger.info(
                        "  %-50s %9.3fms %9.3fms %10s %8d",
                        op_name,
                        0.0,
                        actual_op_time * 1000,
                        "N/A",
                        actual_num_calls,
                    )
                else:
                    diff = (actual_op_time - baseline["total_time_s"]) / baseline["total_time_s"] * 100
                    calls_flag = "!" if baseline["num_calls"] > 0 and actual_num_calls != baseline["num_calls"] else ""
                    logger.info(
                        "  %-50s %9.3fms %9.3fms %+9.2f%% %7d%s",
                        op_name,
                        baseline["total_time_s"] * 1000,
                        actual_op_time * 1000,
                        diff,
                        actual_num_calls,
                        calls_flag,
                    )

        result = {
            "case_name": name,
            "op_passed": op_passed,
            "op_status": op_status,
            "violations": violations,
            "detail_rows": detail_rows,
        }

        if op_passed is False:
            msg_parts = [
                f"\n[{case.name}] Operator-level differences detected (WARNING - not treated as test failure):"
            ]
            msg_parts.append(f"  Description: {case.description}")
            msg_parts.append(f"  Top-{case.operator_top_n} Operator Comparison (...)")
            for v in violations:
                msg_parts.append(v)
            logger.warning("\n".join(msg_parts))

        if getattr(case, "from_group", False):
            # Group sub-cases are rendered by the Test 3 group summary; keep
            # them out of the Test 2 summary to avoid duplicated reporting.
            return result
        self.__class__._op_results.append(result)
        self.__class__._op_detail_rows.extend(detail_rows)
        return result

    @parameterized.expand(
        [(case.name, case) for case in PERF_REGRESSION_CASES],
        skip_on_empty=True,
    )
    def test_performance_regression(self, name: str, case: BasePerfRegressionCase):
        table_result, actual_time_s = _run_case_simulation(case)

        # ============================================================
        # Test 1: Total time comparison (Initial vs Baseline)
        # ============================================================
        info = self._record_time_comparison(case, name, actual_time_s, grouped=False)

        # ============================================================
        # Test 2: Operator-level comparison (Top-N vs Initial Baseline)
        # ============================================================
        self._compare_operators(case, name, table_result, _load_baseline_operators(name))

        # ============================================================
        # Comprehensive judgment: standalone cases hard-fail on breach.
        # ============================================================
        if not info["time_overall"]:
            msg_parts = [f"\n[{case.name}] Total time regression anomaly detected!"]
            msg_parts.extend(
                _format_time_anomaly(
                    case,
                    actual_time_s,
                    info["initial_passed"],
                    info["baseline_passed"],
                    info["initial_diff_pct"],
                    info["baseline_diff_pct"],
                )
            )
            self.fail("\n".join(msg_parts))

    @parameterized.expand(
        [(group.name, group) for group in GROUP_PERF_CASES],
        skip_on_empty=True,
    )
    def test_group_precision_judgment(self, name: str, group: GroupPerfRegressionCase):
        self._run_group_case(name, group)

    def _run_group_case(self, name: str, group: GroupPerfRegressionCase):
        """Group-average precision judgment, one test per group.

        Sub-cases are executed sequentially (in JSON declaration order); each one
        uses the same execution/judgment helpers as standalone cases. After all
        sub-cases finish:

        - any sub-case ERROR (exception)            -> group FAIL;
        - empty ``sub_cases`` (invalid JSON)        -> group FAIL;
        - otherwise mean(|baseline diff|) vs group tolerance -> PASS/FAIL.
        """
        if not group.sub_cases:
            self.__class__._group_rows.append(
                {"group": name, "tolerance": None, "avg": None, "members": [], "status": "FAIL(EMPTY)"}
            )
            self.fail(
                f'Group case "{name}" declares no sub_cases. This is an invalid '
                "configuration: remove the JSON file or declare its members."
            )

        members: list[dict] = []
        errors: list[str] = []

        for sub_name, sub_case in group.sub_cases.items():
            op_result = None
            try:
                table_result, actual_time_s = _run_case_simulation(sub_case)
                info = self._record_time_comparison(sub_case, sub_name, actual_time_s, grouped=True)
                op_result = self._compare_operators(
                    sub_case, sub_name, table_result, _baseline_operators_from_case(sub_case)
                )
                members.append(
                    _make_group_member(sub_case, sub_name, actual_time_s, info, info["time_status"], op_result)
                )
            except Exception as e:  # noqa: BLE001 - keep executing remaining sub-cases
                logger.exception("[%s] Sub-case errored", sub_name)
                errors.append(f"{sub_name}: {e}")
                members.append(_make_group_member(sub_case, sub_name, None, None, "ERROR", None))

        # Effective tolerance: explicit group_tolerance, else the minimum of
        # the sub-cases' baseline_tolerance (conservative direction). Resolved
        # before the ERROR check so the summary row always carries the tolerance.
        if group.group_tolerance is not None:
            tolerance: Optional[float] = group.group_tolerance
        else:
            tolerances = [sub.baseline_tolerance for sub in group.sub_cases.values()]
            tolerance = min(tolerances)
            if len(set(tolerances)) > 1:
                logger.warning(
                    '[group=%s] "group_tolerance" not set; falling back to the minimum '
                    "of sub-case baseline_tolerance: %.2f%%",
                    name,
                    tolerance * 100,
                )

        if errors:
            completed = (
                ", ".join(
                    f"{m['case']}({m['base_diff'] * 100:+.2f}%)" if m["base_diff"] is not None else f"{m['case']}(N/A)"
                    for m in members
                )
                or "none"
            )
            msg_parts = [f"\n[group={name}] {len(errors)} sub-case(s) errored; group judgment fails:"]
            msg_parts.extend(f"  {e}" for e in errors)
            msg_parts.append(f"  Completed sub-cases baseline diff: {completed}")
            self.__class__._group_rows.append(
                {"group": name, "tolerance": tolerance, "avg": None, "members": members, "status": "FAIL(ERROR)"}
            )
            self.fail("\n".join(msg_parts))

        diffs = [abs(m["base_diff"]) for m in members if m["base_diff"] is not None]
        if len(diffs) != len(members):
            missing = [m["case"] for m in members if m["base_diff"] is None]
            self.__class__._group_rows.append(
                {"group": name, "tolerance": tolerance, "avg": None, "members": members, "status": "FAIL(NO_BASELINE)"}
            )
            self.fail(
                f"[group={name}] Cannot judge the group average: {len(missing)} sub-case(s) "
                f"have no baseline and contribute no diff: {', '.join(missing)}"
            )

        avg = sum(abs(d) * 100 for d in diffs) / len(diffs)
        # diffs are fractions; tolerance is a fraction too; avg is in percent.
        status = "FAIL" if avg > tolerance * 100 else "PASS"
        self.__class__._group_rows.append(
            {"group": name, "tolerance": tolerance, "avg": avg, "members": members, "status": status}
        )

        if status == "FAIL":
            msg_parts = [
                f"\n[group={name}] Group average precision judgment failed: "
                f"avg |baseline diff| = {avg:.2f}% > tolerance {tolerance * 100:.2f}%"
            ]
            for m in members:
                msg_parts.append(f"    {m['case']}: |diff| = {abs(m['base_diff']) * 100:.2f}%")
            self.fail("\n".join(msg_parts))


def _make_group_member(
    case: BasePerfRegressionCase,
    sub_name: str,
    actual_time_s: Optional[float],
    info: Optional[dict],
    status: str,
    op_result: Optional[dict] = None,
) -> dict:
    """Build one sub-case detail row of ``_group_rows``.

    ``info`` is the dict returned by ``_record_time_comparison``; it is ``None``
    when the sub-case errored before its time comparison completed, in which
    case the diff fields are reported as ``None``. ``op_result`` is the dict
    returned by ``_compare_operators``; ``None`` means the operator comparison
    did not complete (or was mocked out), rendered as ``N/A`` in the Op column.
    """
    op_status = "N/A"
    op_warn_count = 0
    op_rows: list[tuple] = []
    if op_result is not None:
        op_status = op_result["op_status"]
        op_warn_count = len(op_result["violations"])
        op_rows = op_result["detail_rows"]
    return {
        "case": sub_name,
        "actual": actual_time_s,
        "init": case.initial_time_s,
        "init_diff": info["initial_diff_pct"] if info and case.initial_time_s > 0.0 else None,
        "baseline": case.baseline_time_s,
        "base_diff": info["baseline_diff_pct"] if info and case.baseline_time_s > 0.0 else None,
        "status": status,
        "op_status": op_status,
        "op_warn_count": op_warn_count,
        "op_rows": op_rows,
    }


_GROUP_SUBCASE_HEADER = (
    f"{'Case':<20} {'Actual':>13}  {'Init':>13}  {'InitDiff':>9}  {'Baseline':>13}  {'BaseDiff':>9}  "
    f"{'Status':<12}  {'Op(TopN)':<12}"
)


def _format_group_op_column(member: dict) -> str:
    """Format the Op(TopN) column value of one Test 3 sub-case row."""
    op_status = member["op_status"]
    if op_status == "WARN":
        return f"WARN({member['op_warn_count']})"
    return str(op_status)


def _format_group_subcase_row(member: dict) -> str:
    """Format one sub-case detail row of the Test 3 group summary."""
    actual_str = f"{member['actual'] * 1000:.3f}ms" if member["actual"] is not None else "N/A"
    init_str = f"{member['init'] * 1000:.3f}ms" if member["init"] > 0 else "N/A"
    init_diff_str = f"{member['init_diff'] * 100:+.2f}%" if member["init_diff"] is not None else "N/A"
    base_str = f"{member['baseline'] * 1000:.3f}ms" if member["baseline"] > 0 else "N/A"
    base_diff_str = f"{member['base_diff'] * 100:+.2f}%" if member["base_diff"] is not None else "N/A"
    op_str = _format_group_op_column(member)
    return (
        f"{member['case']:<20} {actual_str:>13}  {init_str:>13}  {init_diff_str:>9}  "
        f"{base_str:>13}  {base_diff_str:>9}  {member['status']:<12}  {op_str:<12}"
    )


_GROUP_OP_WARN_HEADER = (
    f"  {'Case':<24} {'Operator':<44} {'Baseline':>10}  {'Actual':>10}  {'Diff':>10}  {'#Calls':>10}"
)


def _print_group_summary(group_rows: list[dict]):
    if not group_rows:
        return

    total = len(group_rows)
    passed = sum(1 for r in group_rows if r["status"] == "PASS")
    failed = sum(1 for r in group_rows if r["status"].startswith("FAIL"))

    _emit("")
    _emit("=" * 135)
    _emit("  [Test 3] Group Average Precision Judgment Summary")
    _emit("=" * 135)

    for row in group_rows:
        tol_str = f"{row['tolerance'] * 100:.2f}%" if row["tolerance"] is not None else "N/A"
        avg_str = f"{row['avg']:.2f}%" if row["avg"] is not None else "N/A"
        _emit(
            f"Group: {row['group']:<24} Tolerance: {tol_str:>8}   AvgDiff: {avg_str:>8}   GroupStatus: {row['status']}"
        )
        _emit("-" * 135)
        _emit(_GROUP_SUBCASE_HEADER)
        if not row["members"]:
            _emit(f"{'(no sub-cases)':<20}")
        for member in row["members"]:
            _emit(_format_group_subcase_row(member))
        # Operator WARN detail for the whole group (WARNING only, not part of
        # the group judgment). NO_BASELINE sub-cases are already visible via
        # the Op(TopN) column above.
        op_warn_rows = [
            (case_name, op_name, baseline_str, actual_str, diff_str, calls_str)
            for m in row["members"]
            for case_name, op_name, baseline_str, actual_str, diff_str, status, _, calls_str in m["op_rows"]
            if status == "WARN"
        ]
        if op_warn_rows:
            _emit("-" * 135)
            _emit("  Operator warnings in this group:")
            _emit(_GROUP_OP_WARN_HEADER)
            for case_name, op_name, baseline_str, actual_str, diff_str, calls_str in op_warn_rows:
                _emit(
                    f"  {case_name:<24} {op_name:<44} {baseline_str:>10}  {actual_str:>10}  {diff_str:>10}  {calls_str:>10}"
                )
        _emit("-" * 135)

    _emit(f"Total: {total} group(s) | Passed: {passed} | Failed: {failed}")
    _emit("=" * 135)
    _emit("")


class TestBaselineTimeThreshold(unittest.TestCase):
    def test_allows_text_actual_within_five_percent_of_baseline(self):
        baseline_time_s = 0.064380
        tolerance = 0.20

        self.assertTrue(_is_baseline_time_acceptable(0.060820, baseline_time_s, tolerance, 0.05))
        self.assertTrue(_is_baseline_time_acceptable(baseline_time_s, baseline_time_s, tolerance, 0.05))
        self.assertTrue(_is_baseline_time_acceptable(0.065000, baseline_time_s, tolerance, 0.05))
        self.assertFalse(_is_baseline_time_acceptable(0.067600, baseline_time_s, tolerance, 0.05))

    def test_allows_non_text_actual_within_twenty_percent_of_baseline(self):
        baseline_time_s = 0.064380
        tolerance = 0.20

        self.assertTrue(_is_baseline_time_acceptable(0.077000, baseline_time_s, tolerance, 0.20))
        self.assertFalse(_is_baseline_time_acceptable(0.078000, baseline_time_s, tolerance, 0.20))

    def test_keeps_baseline_tolerance_for_large_improvements(self):
        self.assertFalse(_is_baseline_time_acceptable(0.040000, 0.064380, 0.20, 0.05))

    def test_profiling_mode_skips_increase_cap(self):
        # Profiling (empirical) cases: no increase cap; only the symmetric
        # baseline_tolerance applies. +30% exceeds a 5% cap but is within 0.35.
        self.assertTrue(_is_baseline_time_acceptable(0.083694, 0.064380, 0.35, 0.05, performance_model="profiling"))
        # The symmetric tolerance is still enforced in both directions.
        self.assertFalse(_is_baseline_time_acceptable(0.090000, 0.064380, 0.35, 0.05, performance_model="profiling"))
        self.assertFalse(_is_baseline_time_acceptable(0.040000, 0.064380, 0.35, 0.05, performance_model="profiling"))


class TestGroupJudgmentLogic(unittest.TestCase):
    """Unit tests for the group-average judgment logic (no simulation)."""

    def _make_tc(self) -> TestPerformanceRegression:
        # Bypass __init__: the judgment helpers only rely on class-level state
        # and self.fail().
        tc = TestPerformanceRegression.__new__(TestPerformanceRegression)
        tc.__class__._time_results = []
        tc.__class__._group_rows = []
        return tc

    def _make_group(self, sub_specs, group_tolerance=None) -> GroupPerfRegressionCase:
        sub_cases = {}
        for name, baseline_time_s, baseline_tolerance in sub_specs:
            sub_cases[name] = TextPerfRegressionCase(
                name=name,
                description="",
                initial_time_s=0.0,
                baseline_time_s=baseline_time_s,
                baseline_tolerance=baseline_tolerance,
            )
        return GroupPerfRegressionCase(
            name="g",
            description="",
            group_tolerance=group_tolerance,
            sub_cases=sub_cases,
        )

    def _patch_simulation(self, actual_by_name):
        def fake_simulation(case):
            return "", actual_by_name[case.name]

        return mock.patch(f"{__name__}._run_case_simulation", side_effect=fake_simulation)

    def test_group_avg_is_absolute_mean(self):
        # +14% and -14% must not cancel out: avg |diff| = 14% > 10% -> FAIL.
        group = self._make_group([("a", 1.0, 0.50), ("b", 1.0, 0.50)], group_tolerance=0.10)
        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            self._patch_simulation({"a": 1.14, "b": 0.86}),
        ):
            with self.assertRaises(AssertionError) as ctx:
                tc._run_group_case("g", group)

        self.assertIn("avg |baseline diff| = 14.00% > tolerance 10.00%", str(ctx.exception))
        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "FAIL")
        self.assertAlmostEqual(row["avg"], 14.0)

    def test_avg_equal_to_tolerance_passes(self):
        # 8 -> 9 is exactly +12.5% in float; strict ">" judgment must PASS
        # when the average equals the tolerance.
        group = self._make_group([("a", 8.0, 0.50)], group_tolerance=0.125)
        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            self._patch_simulation({"a": 9.0}),
        ):
            tc._run_group_case("g", group)

        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "PASS")
        self.assertAlmostEqual(row["avg"], 12.5)

    def test_sub_case_error_fails_group_without_stopping(self):
        group = self._make_group([("bad", 1.0, 0.20), ("good", 1.0, 0.20)], group_tolerance=0.10)
        calls = []

        def fake_simulation(case):
            calls.append(case.name)
            if case.name == "bad":
                raise RuntimeError("boom")
            return "", 1.02

        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            mock.patch(f"{__name__}._run_case_simulation", side_effect=fake_simulation),
        ):
            with self.assertRaises(AssertionError) as ctx:
                tc._run_group_case("g", group)

        self.assertEqual(calls, ["bad", "good"])
        self.assertIn("boom", str(ctx.exception))
        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "FAIL(ERROR)")

    def test_empty_sub_cases_fails_group(self):
        group = self._make_group([])
        tc = self._make_tc()
        with self.assertRaises(AssertionError) as ctx:
            tc._run_group_case("g", group)

        self.assertIn("declares no sub_cases", str(ctx.exception))
        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "FAIL(EMPTY)")

    def test_tolerance_fallback_takes_minimum(self):
        group = self._make_group([("a", 1.0, 0.05), ("b", 1.0, 0.20)])
        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            self._patch_simulation({"a": 1.03, "b": 1.03}),
        ):
            with self.assertLogs(logger, level="WARNING") as logs:
                tc._run_group_case("g", group)

        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "PASS")
        self.assertAlmostEqual(row["tolerance"], 0.05)
        self.assertTrue(any("falling back to the minimum" in message for message in logs.output))

    def test_group_members_carry_subcase_details(self):
        group = self._make_group([("a", 1.0, 0.50), ("b", 1.0, 0.50)], group_tolerance=0.20)
        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            self._patch_simulation({"a": 1.14, "b": 0.86}),
        ):
            tc._run_group_case("g", group)

        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "PASS")
        members = row["members"]
        self.assertEqual([m["case"] for m in members], ["a", "b"])
        for m in members:
            self.assertEqual(
                set(m),
                {
                    "case",
                    "actual",
                    "init",
                    "init_diff",
                    "baseline",
                    "base_diff",
                    "status",
                    "op_status",
                    "op_warn_count",
                    "op_rows",
                },
            )
            # _compare_operators is mocked out -> op fields fall back to N/A.
            self.assertEqual(m["op_status"], "N/A")
            self.assertEqual(m["op_warn_count"], 0)
            self.assertEqual(m["op_rows"], [])
        self.assertAlmostEqual(members[0]["base_diff"], 0.14)
        self.assertAlmostEqual(members[1]["base_diff"], -0.14)
        # +14% breaches the text default 5% increase cap; grouped breaches are
        # downgraded to WARN(BASE) because the group is judged by its average.
        self.assertEqual(members[0]["status"], "WARN(BASE)")
        self.assertEqual(members[1]["status"], "PASS")
        self.assertEqual(members[0]["actual"], 1.14)
        self.assertEqual(members[0]["baseline"], 1.0)

    def test_errored_subcase_appears_as_error_member(self):
        group = self._make_group([("bad", 1.0, 0.20), ("good", 1.0, 0.20)], group_tolerance=0.10)

        def fake_simulation(case):
            if case.name == "bad":
                raise RuntimeError("boom")
            return "", 1.02

        tc = self._make_tc()
        with (
            mock.patch.object(TestPerformanceRegression, "_compare_operators", return_value=None),
            mock.patch(f"{__name__}._run_case_simulation", side_effect=fake_simulation),
        ):
            with self.assertRaises(AssertionError):
                tc._run_group_case("g", group)

        row = TestPerformanceRegression._group_rows[0]
        self.assertEqual(row["status"], "FAIL(ERROR)")
        by_case = {m["case"]: m for m in row["members"]}
        self.assertEqual(by_case["bad"]["status"], "ERROR")
        self.assertIsNone(by_case["bad"]["base_diff"])
        self.assertEqual(by_case["good"]["status"], "PASS")
        self.assertAlmostEqual(by_case["good"]["base_diff"], 0.02)


class TestSummaryRendering(unittest.TestCase):
    """Rendering tests for the Test 1 / Test 3 summary tables."""

    @staticmethod
    def _capture(func, *args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            func(*args)
        return buf.getvalue()

    def test_time_summary_excludes_grouped_subcases(self):
        standalone = _TimeResultRow("standalone", "PASS", 1.0, 0.9, 11.11, 1.0, 0.0, "PASS", False)
        grouped = _TimeResultRow("group-sub", "PASS", 1.0, 0.0, 0.0, 1.0, 2.0, "PASS", True)

        out = self._capture(_print_time_summary, [standalone, grouped])

        self.assertIn("standalone", out)
        self.assertNotIn("group-sub", out)
        self.assertIn("Total: 1 | Passed: 1", out)

    def test_time_summary_hidden_when_only_grouped_subcases(self):
        grouped = _TimeResultRow("group-sub", "PASS", 1.0, 0.0, 0.0, 1.0, 2.0, "PASS", True)

        out = self._capture(_print_time_summary, [grouped])

        self.assertEqual(out, "")

    def test_group_summary_renders_group_and_subcase_details(self):
        rows = [
            {
                "group": "glm5.2",
                "tolerance": 0.15,
                "avg": 4.16,
                "members": [
                    {
                        "case": "3.5k-decode",
                        "actual": 45.23,
                        "init": 48.90,
                        "init_diff": -0.075,
                        "baseline": 43.85,
                        "base_diff": 0.0315,
                        "status": "PASS",
                        "op_status": "WARN",
                        "op_warn_count": 1,
                        "op_rows": [
                            ("3.5k-decode", "mlp.linear.pro", "1.234ms", "1.500ms", "+21.56%", "WARN", "4", "4/4"),
                            ("3.5k-decode", "attn.rope", "0.512ms", "0.520ms", "+1.56%", "PASS", "2", "2/2"),
                        ],
                    },
                    {
                        "case": "3.5k-prefill",
                        "actual": 62.10,
                        "init": 60.00,
                        "init_diff": 0.035,
                        "baseline": 59.05,
                        "base_diff": 0.0517,
                        "status": "PASS",
                        "op_status": "PASS",
                        "op_warn_count": 0,
                        "op_rows": [],
                    },
                ],
                "status": "PASS",
            },
            {
                "group": "qwen3.5-moe",
                "tolerance": 0.10,
                "avg": 12.30,
                "members": [
                    {
                        "case": "64k-decode",
                        "actual": 128.40,
                        "init": 120.00,
                        "init_diff": 0.07,
                        "baseline": 114.30,
                        "base_diff": 0.123,
                        "status": "WARN(BASE)",
                        "op_status": "NO_BASELINE",
                        "op_warn_count": 0,
                        "op_rows": [],
                    },
                    {
                        # Errored sub-case: operator comparison never ran, so
                        # _make_group_member fills the op fields with N/A.
                        "case": "64k-prefill",
                        "actual": None,
                        "init": 98.00,
                        "init_diff": None,
                        "baseline": 84.20,
                        "base_diff": None,
                        "status": "ERROR",
                        "op_status": "N/A",
                        "op_warn_count": 0,
                        "op_rows": [],
                    },
                ],
                "status": "FAIL",
            },
        ]

        out = self._capture(_print_group_summary, rows)

        # Group-level result lines.
        self.assertIn("Group: glm5.2", out)
        self.assertIn("Tolerance:   15.00%", out)
        self.assertIn("AvgDiff:    4.16%", out)
        self.assertIn("GroupStatus: PASS", out)
        self.assertIn("Group: qwen3.5-moe", out)
        self.assertIn("GroupStatus: FAIL", out)
        # Sub-case detail header and rows.
        for token in ("Case", "Actual", "Init", "InitDiff", "Baseline", "BaseDiff", "Status"):
            self.assertIn(token, out)
        self.assertIn("3.5k-decode", out)
        self.assertIn("45230.000ms", out)
        self.assertIn("-7.50%", out)
        self.assertIn("+3.15%", out)
        self.assertIn("64k-decode", out)
        self.assertIn("WARN(BASE)", out)
        # Errored sub-case renders as N/A row.
        self.assertIn("64k-prefill", out)
        self.assertIn("ERROR", out)
        # Op(TopN) column: status per sub-case, WARN carries the violation count.
        self.assertIn("Op(TopN)", out)
        self.assertIn("WARN(1)", out)
        self.assertIn("NO_BASELINE", out)
        # Group operator WARN block: only WARN rows, PASS rows are omitted.
        self.assertIn("Operator warnings in this group:", out)
        self.assertIn("mlp.linear.pro", out)
        self.assertIn("+21.56%", out)
        self.assertIn("4/4", out)
        self.assertNotIn("attn.rope", out)
        # Footer counts groups, not sub-cases.
        self.assertIn("Total: 2 group(s) | Passed: 1 | Failed: 1", out)


def _emit(text: str):
    print(text)


def _print_time_summary(results: list[_TimeResultRow]):
    # Group sub-cases are judged (and reported in full detail) by the Test 3
    # group summary; Test 1 only covers standalone cases.
    results = [r for r in results if not r.grouped]
    if not results:
        return

    total = len(results)
    passed = sum(1 for r in results if r[1] == "PASS")
    failed = sum(1 for r in results if r[1] == "FAIL")
    warned = sum(1 for r in results if r[1] == "WARN")
    no_baseline = sum(1 for r in results if r[1] == "NO_BASELINE")

    _emit("")
    _emit("=" * 120)
    _emit("  [Test 1] Total Time Regression Summary")
    _emit("=" * 120)
    header = (
        f"{'Case':<36} {'Actual':>10}  "
        f"{'Init':>10}  {'InitDiff':>10}  "
        f"{'Baseline':>10}  {'BaseDiff':>10}  "
        f"{'Status':>10}"
    )
    _emit(header)
    _emit("-" * 120)

    for row in results:
        actual_str = f"{row.actual * 1000:.3f}ms"
        init_str = f"{row.init_time * 1000:.3f}ms" if row.init_time > 0 else "N/A"
        init_diff_str = f"{row.init_diff:+.2f}%" if row.init_time > 0 else "N/A"
        base_str = f"{row.base_time * 1000:.3f}ms" if row.base_time > 0 else "N/A"
        base_diff_str = f"{row.base_diff:+.2f}%" if row.base_time > 0 else "N/A"
        _emit(
            f"{row.name:<36} {actual_str:>10}  {init_str:>10}  {init_diff_str:>10}  {base_str:>10}  {base_diff_str:>10}  {row.status:>10}"
        )

    _emit("-" * 120)
    _emit(f"Total: {total} | Passed: {passed} | Failed: {failed} | Warned: {warned} | No Baseline: {no_baseline}")
    _emit("=" * 120)
    _emit("")


def _print_operator_summary(op_results: list[dict], op_detail_rows: list[tuple]):
    if not op_detail_rows:
        return

    total = len(op_results)
    passed = sum(1 for r in op_results if r["op_passed"] is True)
    warned = sum(1 for r in op_results if r["op_passed"] is False)
    no_baseline = sum(1 for r in op_results if r["op_passed"] is None)

    if warned == 0:
        _emit("*** All Operator Top-N Checks Passed (no warnings) ***")
        _emit("")
        return

    _emit("=" * 145)
    _emit("  [Test 2] Operator-Level Differences Summary (WARNING only, not treated as failure)")
    _emit("=" * 145)
    header = (
        f"{'Case':<32} {'Operator':<44} {'Baseline':>10} {'Actual':>10}  {'Diff':>10}  {'#Calls':>12} {'Status':>10}"
    )
    _emit(header)
    _emit("-" * 145)

    for (
        case_name,
        op_name,
        baseline_str,
        actual_str,
        diff_str,
        status,
        _,
        calls_str,
    ) in op_detail_rows:
        if status == "WARN":
            _emit(
                f"{case_name:<32} {op_name:<44} {baseline_str:>10}  {actual_str:>10}  {diff_str:>10}  {calls_str:>12}  {status:>10}"
            )

    _emit("-" * 145)
    _emit(f"Total Cases: {total} | Passed: {passed} | Warnings: {warned} | No Baseline: {no_baseline}")
    _emit("=" * 145)
    _emit("")
    _emit("*** Operator Differences Detected (WARNING only, test NOT failed) ***")
