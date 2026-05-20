import logging
import re
import unittest
from dataclasses import dataclass
from typing import List, Optional

import torch
from parameterized import parameterized

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import (
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig


logger = logging.getLogger(__name__)


@dataclass
class AutoBaselineCase:
    name: str
    description: str
    baseline_input: UserInputConfig
    compare_input: UserInputConfig
    tolerance: float = 0.05


@dataclass
class AutoBaselineResult:
    case_name: str
    baseline_time_s: float
    actual_time_s: float
    diff_pct: float
    tolerance: float
    passed: bool
    error: Optional[str] = None


def _parse_total_time_s(table_result: str, model_name: str = "analytic") -> float:
    pattern = rf"Total time for {model_name}:\s*([\d.]+)\s*(ns|us|ms|s)"
    m = re.search(pattern, table_result)
    if not m:
        raise ValueError(f"Could not find 'Total time for {model_name}' in output:\n{table_result}")
    value = float(m.group(1))
    unit = m.group(2)
    return value * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


def _run_single(user_input: UserInputConfig) -> float:
    torch.compiler.reset()
    model_runner = ModelRunner(user_input)
    result = model_runner.run_inference(generate_inputs_func=generate_inputs)
    if isinstance(result, ModelRunnerMetrics):
        return _parse_total_time_s(result.table_result)
    raise TypeError(f"Unexpected result type: {type(result)}")


def run_auto_baseline(
    baseline_input: UserInputConfig,
    compare_input: UserInputConfig,
    case_name: str = "auto_baseline",
    tolerance: float = 0.05,
) -> AutoBaselineResult:
    logger.info("\n%s", "=" * 60)
    logger.info("  AUTO BASELINE TEST: %s", case_name)
    logger.info("%s", "=" * 60)

    logger.info("[Run 1/2] Establishing baseline...")
    try:
        baseline_time_s = _run_single(baseline_input)
        logger.info("  Baseline time: %.3fms", baseline_time_s * 1000)
    except Exception as e:
        return AutoBaselineResult(
            case_name=case_name,
            baseline_time_s=0.0,
            actual_time_s=0.0,
            diff_pct=0.0,
            tolerance=tolerance,
            passed=False,
            error=f"Baseline run failed: {e}",
        )

    logger.info("[Run 2/2] Running comparison...")
    try:
        actual_time_s = _run_single(compare_input)
        logger.info("  Actual time:   %.3fms", actual_time_s * 1000)
    except Exception as e:
        return AutoBaselineResult(
            case_name=case_name,
            baseline_time_s=baseline_time_s,
            actual_time_s=0.0,
            diff_pct=0.0,
            tolerance=tolerance,
            passed=False,
            error=f"Comparison run failed: {e}",
        )

    if baseline_time_s <= 0:
        return AutoBaselineResult(
            case_name=case_name,
            baseline_time_s=baseline_time_s,
            actual_time_s=actual_time_s,
            diff_pct=0.0,
            tolerance=tolerance,
            passed=False,
            error=f"Invalid baseline time: {baseline_time_s}",
        )

    diff_pct = (actual_time_s - baseline_time_s) / baseline_time_s
    passed = abs(diff_pct) <= tolerance

    return AutoBaselineResult(
        case_name=case_name,
        baseline_time_s=baseline_time_s,
        actual_time_s=actual_time_s,
        diff_pct=diff_pct,
        tolerance=tolerance,
        passed=passed,
    )


def _print_result(result: AutoBaselineResult):
    logger.info("\n%s", "=" * 60)
    logger.info("  RESULT: %s", result.case_name)
    logger.info("%s", "=" * 60)

    if result.error:
        logger.error("  ERROR: %s", result.error)
        logger.info("  Status: FAIL")
        return

    logger.info("  Baseline:  %.3fms", result.baseline_time_s * 1000)
    logger.info("  Actual:    %.3fms", result.actual_time_s * 1000)
    logger.info("  Diff:      %+.2f%%", result.diff_pct * 100)
    logger.info("  Tolerance: ±%.0f%%", result.tolerance * 100)
    logger.info("  Status:    %s", "PASS" if result.passed else "FAIL")

    if not result.passed:
        direction = "slower" if result.diff_pct > 0 else "faster"
        logger.warning(
            "  Performance regression: %.2f%% %s than baseline!",
            abs(result.diff_pct) * 100,
            direction,
        )


AUTO_BASELINE_CASES: List[AutoBaselineCase] = [
    AutoBaselineCase(
        name="qwen3-8B_auto",
        description="Qwen3-8B decode, baseline ctx=1536 vs compare ctx=1500, TP=2, compile",
        baseline_input=UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-8B",
            num_queries=32,
            query_len=1,
            context_length=1536,
            do_compile=True,
            decode=True,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            tp_size=2,
            world_size=2,
        ),
        compare_input=UserInputConfig(
            device="ATLAS_800_A2_376T_64G",
            model_id="Qwen/Qwen3-8B",
            num_queries=32,
            query_len=1,
            context_length=1500,
            do_compile=True,
            decode=True,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
            tp_size=2,
            world_size=2,
        ),
        tolerance=0.05,
    ),
]


class TestAutoBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] [%(name)s] %(message)s",
        )

    @parameterized.expand(
        [(case.name, case) for case in AUTO_BASELINE_CASES],
        name_func=lambda func, num, p: f"{func.__name__}_{p.args[0]}",
    )
    def test_auto_baseline(self, _name: str, case: AutoBaselineCase):
        logger.info("Baseline config:  context_length=%d", case.baseline_input.context_length)
        logger.info("Compare config:   context_length=%d", case.compare_input.context_length)
        logger.info("Baseline config:  context_length=%d", case.baseline_input.context_length)
        logger.info("Compare config:   context_length=%d", case.compare_input.context_length)

        result = run_auto_baseline(
            baseline_input=case.baseline_input,
            compare_input=case.compare_input,
            case_name=case.name,
            tolerance=case.tolerance,
        )
        _print_result(result)

        self.assertTrue(
            result.passed,
            f"[{case.name}] Auto baseline FAILED: "
            f"diff={result.diff_pct * 100:+.2f}%, "
            f"tolerance=±{result.tolerance * 100:.0f}%" + (f", error={result.error}" if result.error else ""),
        )
