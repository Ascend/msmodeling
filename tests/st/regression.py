import json
import logging
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from parameterized import parameterized

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig


logger = logging.getLogger(__name__)

REPORT_FILE = Path(__file__).resolve().parent / "regression_report.txt"

CASE_DIR = Path(__file__).resolve().parent / "cases"


@dataclass
class BasePerfRegressionCase:
    name: str
    description: str
    initial_time_s: float = 0.0
    baseline_time_s: float = 0.0
    initial_tolerance: float = 0.10
    baseline_tolerance: float = 0.20
    operator_top_n: int = 10
    operator_tolerance: float = 0.10
    operators: List[Dict[str, float]] = None


@dataclass
class TextPerfRegressionCase(BasePerfRegressionCase):
    user_input: Optional[UserInputConfig] = None


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


def _parse_total_time_s(table_result: str, model_name: str = "analytic") -> float:
    pattern = rf"Total time for {model_name}:\s*([\d.]+)\s*(ns|us|ms|s)"
    m = re.search(pattern, table_result)
    if not m:
        raise ValueError(f"Could not find 'Total time for {model_name}' in output:\n{table_result}")
    value = float(m.group(1))
    unit = m.group(2)
    return value * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


def _parse_top_operators(
    table_result: str,
    top_n: int = 10,
    model_name: str = "analytic",
) -> List[Tuple[str, float, int]]:
    lines = table_result.split("\n")
    data_started = False
    operators: List[Tuple[str, float, int]] = []

    for line in lines:
        if f"{model_name} total" in line and f"{model_name} avg" in line:
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


def _load_baseline_operators(case_name: str) -> Optional[Dict[str, Dict[str, float]]]:
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


def _save_baseline_operators(case_name: str, operators: List[Tuple[str, float, int]]):
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


def _load_perf_regression_cases() -> List[BasePerfRegressionCase]:
    cases: List[BasePerfRegressionCase] = []
    for path in sorted(CASE_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        case_type = data.pop("type", "text")
        if case_type == "video":
            data["quantize_linear_action"] = QuantizeLinearAction[data["quantize_linear_action"]]
            if data.get("model_id") and not Path(data["model_id"]).is_absolute():
                data["model_id"] = str(Path(__file__).resolve().parent / data["model_id"])
            cases.append(VideoPerfRegressionCase(**data))
        else:
            ui_data = data.pop("user_input", {})
            for key in ("quantize_linear_action", "quantize_attention_action"):
                if key in ui_data and isinstance(ui_data[key], str):
                    if key == "quantize_linear_action":
                        ui_data[key] = QuantizeLinearAction[ui_data[key]]
                    else:
                        ui_data[key] = QuantizeAttentionAction[ui_data[key]]
            user_input = UserInputConfig(**ui_data)
            cases.append(TextPerfRegressionCase(user_input=user_input, **data))
    return cases


PERF_REGRESSION_CASES = _load_perf_regression_cases()


class TestPerformanceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] [%(name)s] %(message)s",
        )
        cls._time_results: List[Tuple] = []
        cls._op_results: List[Dict] = []
        cls._op_detail_rows: List[Tuple] = []

    @classmethod
    def tearDownClass(cls):
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            _print_time_summary(cls._time_results, report_file=f)
            _print_operator_summary(cls._op_results, cls._op_detail_rows, report_file=f)

    @parameterized.expand(
        [(case.name, case) for case in PERF_REGRESSION_CASES],
        skip_on_empty=True,
    )
    def test_performance_regression(self, name: str, case: BasePerfRegressionCase):
        torch.compiler.reset()

        if isinstance(case, VideoPerfRegressionCase):
            import io
            from contextlib import redirect_stderr, redirect_stdout

            from cli.inference.video_generate import (
                run_inference as video_run_inference,
            )

            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
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
            actual_time_s = _parse_total_time_s(table_result)
        else:
            model_runner = ModelRunner(case.user_input)
            result = model_runner.run_inference(generate_inputs_func=generate_inputs)

            if isinstance(result, ModelRunnerMetrics):
                table_result = result.table_result
            else:
                self.fail(f"Unexpected result type: {type(result)}")

            actual_time_s = _parse_total_time_s(table_result)
        logger.info(
            "[%s] Actual total time: %.6fs (%.3fms)",
            name,
            actual_time_s,
            actual_time_s * 1000,
        )

        # ============================================================
        # Test 1: Total time comparison (Initial vs Baseline)
        # ============================================================
        initial_passed = True
        baseline_passed = True
        initial_diff_pct = 0.0
        baseline_diff_pct = 0.0

        if case.initial_time_s > 0.0:
            initial_diff_pct = (actual_time_s - case.initial_time_s) / case.initial_time_s
            initial_passed = abs(initial_diff_pct) <= case.initial_tolerance

        if case.baseline_time_s > 0.0:
            baseline_diff_pct = (actual_time_s - case.baseline_time_s) / case.baseline_time_s
            baseline_passed = abs(baseline_diff_pct) <= case.baseline_tolerance

        time_overall = initial_passed and baseline_passed

        if case.initial_time_s == 0.0 and case.baseline_time_s == 0.0:
            time_status = "NO_BASELINE"
        elif not initial_passed and not baseline_passed:
            time_status = "FAIL(BOTH)"
        elif not initial_passed:
            time_status = "FAIL(INIT)"
        elif not baseline_passed:
            time_status = "FAIL(BASE)"
        else:
            time_status = "PASS"

        self.__class__._time_results.append(
            (
                name,
                time_overall,
                actual_time_s,
                case.initial_time_s,
                initial_diff_pct * 100,
                case.baseline_time_s,
                baseline_diff_pct * 100,
                time_status,
            )
        )

        # ============================================================
        # Test 2: Operator-level comparison (Top-N vs Initial Baseline)
        # ============================================================
        actual_operators = _parse_top_operators(table_result, top_n=case.operator_top_n)
        baseline_operators = _load_baseline_operators(name)

        op_passed = None

        if baseline_operators is None:
            case_path = CASE_DIR / f"{name}.json"
            self.__class__._op_results.append(
                {
                    "case_name": name,
                    "op_passed": False,
                    "op_status": "NO_BASELINE",
                    "violations": [],
                }
            )
            self.fail(
                f"[{name}] No operator baseline found in case file: {case_path}. "
                "Please generate baseline explicitly before running regression tests. "
                "See README.md for the baseline lifecycle process."
            )
        else:
            baseline_top_n = dict(
                sorted(
                    baseline_operators.items(),
                    key=lambda item: item[1]["total_time_s"],
                    reverse=True,
                )[: case.operator_top_n]
            )

            violations: List[str] = []
            actual_op_names = {op_name for op_name, _, _ in actual_operators}
            baseline_op_names = set(baseline_top_n.keys())

            missing_from_actual = baseline_op_names - actual_op_names
            for op_name in sorted(missing_from_actual):
                bl = baseline_top_n[op_name]
                violations.append(
                    f"  MISSING OPERATOR: {op_name} (baseline={bl['total_time_s'] * 1000:.3f}ms, #calls={bl['num_calls']}) - Not found in current results"
                )
                self.__class__._op_detail_rows.append(
                    (
                        name,
                        op_name,
                        f"{bl['total_time_s'] * 1000:.3f}ms",
                        "MISSING",
                        "N/A",
                        "FAIL",
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
                    self.__class__._op_detail_rows.append(
                        (
                            name,
                            op_name,
                            "N/A",
                            f"{actual_op_time * 1000:.3f}ms",
                            "NEW",
                            "FAIL",
                            "-",
                            str(actual_num_calls),
                        )
                    )
                    continue
                baseline_op_time = baseline["total_time_s"]
                baseline_num_calls = baseline["num_calls"]
                if baseline_op_time == 0.0:
                    self.__class__._op_detail_rows.append(
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
                op_row_status = "PASS" if (not time_fail and not calls_mismatch) else "FAIL"
                if op_row_status == "FAIL":
                    violations.append(
                        f"  {op_name}: {op_diff_pct * 100:+.2f}% "
                        f"(baseline={baseline_op_time * 1000:.3f}ms, actual={actual_op_time * 1000:.3f}ms)"
                    )
                if baseline_num_calls > 0 and actual_num_calls != baseline_num_calls:
                    violations.append(
                        f"  {op_name}: #CALLS MISMATCH baseline={baseline_num_calls}, actual={actual_num_calls}"
                    )
                calls_str = f"{baseline_num_calls}/{actual_num_calls}{'!' if baseline_num_calls > 0 and actual_num_calls != baseline_num_calls else ''}"
                self.__class__._op_detail_rows.append(
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
            op_status = "PASS" if op_passed else "FAIL"
            self.__class__._op_results.append(
                {
                    "case_name": name,
                    "op_passed": op_passed,
                    "op_status": op_status,
                    "violations": violations,
                }
            )

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

            if not op_passed:
                pass

        # ============================================================
        # Comprehensive judgment
        # ============================================================
        if not time_overall:
            msg_parts = [f"\n[{case.name}] Total time regression anomaly detected!"]
            msg_parts.append(f"  Description: {case.description}")
            msg_parts.append(f"  Actual:   {actual_time_s * 1000:.3f}ms")
            if case.initial_time_s > 0.0:
                msg_parts.append(
                    f"  vs Initial: {case.initial_time_s * 1000:.3f}ms "
                    f"({initial_diff_pct * 100:+.2f}%, tolerance: ±{case.initial_tolerance * 100:.0f}%) "
                    f"{'PASS' if initial_passed else 'FAIL'}"
                )
            if case.baseline_time_s > 0.0:
                msg_parts.append(
                    f"  vs Baseline: {case.baseline_time_s * 1000:.3f}ms "
                    f"({baseline_diff_pct * 100:+.2f}%, tolerance: ±{case.baseline_tolerance * 100:.0f}%) "
                    f"{'PASS' if baseline_passed else 'FAIL'}"
                )
            self.fail("\n".join(msg_parts))

        if op_passed is False:
            violations = self.__class__._op_results[-1].get("violations", [])
            msg_parts = [f"\n[{case.name}] Operator-level regression anomaly detected!"]
            msg_parts.append(f"  Description: {case.description}")
            msg_parts.append(f"  Top-{case.operator_top_n} Operator Comparison (...)")
            for v in violations:
                msg_parts.append(v)
            self.fail("\n".join(msg_parts))


def _emit(text: str, report_file=None):
    print(text)
    if report_file is not None:
        report_file.write(text + "\n")


def _print_time_summary(results: List[Tuple], report_file=None):
    if not results:
        return

    total = len(results)
    passed = sum(1 for r in results if r[1] is True)
    failed = sum(1 for r in results if r[1] is False)
    no_baseline = sum(1 for r in results if r[1] is None)
    _ = no_baseline

    _emit("", report_file)
    _emit("=" * 120, report_file)
    _emit("  [Test 1] Total Time Regression Summary", report_file)
    _emit("=" * 120, report_file)
    header = (
        f"{'Case':<36} {'Actual':>10}  "
        f"{'Init':>10}  {'InitDiff':>10}  "
        f"{'Baseline':>10}  {'BaseDiff':>10}  "
        f"{'Status':>10}"
    )
    _emit(header, report_file)
    _emit("-" * 120, report_file)

    for name, ok, actual, init_time, init_diff, base_time, base_diff, status in results:
        actual_str = f"{actual * 1000:.3f}ms"
        init_str = f"{init_time * 1000:.3f}ms" if init_time > 0 else "N/A"
        init_diff_str = f"{init_diff:+.2f}%" if init_time > 0 else "N/A"
        base_str = f"{base_time * 1000:.3f}ms" if base_time > 0 else "N/A"
        base_diff_str = f"{base_diff:+.2f}%" if base_time > 0 else "N/A"
        _emit(
            f"{name:<36} {actual_str:>10}  {init_str:>10}  {init_diff_str:>10}  {base_str:>10}  {base_diff_str:>10}  {status:>10}",
            report_file,
        )

    _emit("-" * 120, report_file)
    _emit(
        f"Total: {total} | Passed: {passed} | Failed: {failed} | No Baseline: {no_baseline}",
        report_file,
    )
    _emit("=" * 120, report_file)
    _emit("", report_file)


def _print_operator_summary(op_results: List[Dict], op_detail_rows: List[Tuple], report_file=None):
    if not op_detail_rows:
        return

    total = len(op_results)
    passed = sum(1 for r in op_results if r["op_passed"] is True)
    failed = sum(1 for r in op_results if r["op_passed"] is False)
    no_baseline = sum(1 for r in op_results if r["op_passed"] is None)
    _ = no_baseline

    _emit("=" * 145, report_file)
    _emit("  [Test 2] Operator-Level Regression Summary", report_file)
    _emit("=" * 145, report_file)
    header = (
        f"{'Case':<32} {'Operator':<44} {'Baseline':>10} {'Actual':>10}  {'Diff':>10}  {'#Calls':>12} {'Status':>10}"
    )
    _emit(header, report_file)
    _emit("-" * 145, report_file)

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
        _emit(
            f"{case_name:<32} {op_name:<44} {baseline_str:>10}  {actual_str:>10}  {diff_str:>10}  {calls_str:>12}  {status:>10}",
            report_file,
        )

    _emit("-" * 145, report_file)
    _emit(
        f"Total Cases: {total} | Passed: {passed} | Failed: {failed} | No Baseline: {no_baseline}",
        report_file,
    )
    _emit("=" * 145, report_file)
    _emit("", report_file)

    if failed > 0:
        _emit("*** Operator Regression Anomaly Detected! ***", report_file)
    else:
        _emit("*** All Operator Checks Passed ***", report_file)
