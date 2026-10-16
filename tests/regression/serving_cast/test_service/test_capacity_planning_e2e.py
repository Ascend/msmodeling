#!/usr/bin/env python3

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

"""Capacity planning E2E consistency harness (plus unit tests for its helpers).

Manual end-to-end check: each scenario runs the normal optimizer with a single
fixed parallel config, then runs the same command with ``--target-qps`` set to
the achieved QPS. The capacity planner should reference the same best config —
the only addition is the extrapolation step.

The parsing/verification helpers above and the ``Test*`` cases at the bottom are
covered by fast unit tests; the ``run_*`` scenario functions below drive real
models and profiling databases, so they are deliberately *not* named ``test_*``
and are only executed through this module's ``__main__`` entry point:

    pytest tests/regression/serving_cast/test_service/test_capacity_planning_e2e.py
    python -m tests.regression.serving_cast.test_service.test_capacity_planning_e2e
"""

import re
import subprocess
import sys
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

from prettytable import PrettyTable

from serving_cast.service.optimizer_summary import compute_request_qps

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: Relative tolerance for comparing TTFT/TPOT across the two runs.
#:
#: The parallel configuration has to match exactly, but the SLO metrics are
#: ranked differently: the normal run maximises throughput while the capacity run
#: minimises total devices, so a different batch/concurrency row can win. The
#: recorded E2E results show that drift (TPOT 19 -> 21 ms, TTFT 5734 -> 6371 ms,
#: about 11%), so this is loose enough to accept batch-selection drift and tight
#: enough to catch a wrong simulation path.
SLO_RELATIVE_TOLERANCE = 0.25

_SLO_PATTERNS = {
    "ttft": re.compile(r"TTFT:\s+([\d.]+)\s+ms"),
    "tpot": re.compile(r"TPOT:\s+([\d.]+)\s+ms"),
}


def _strip_ansi(text: str) -> str:
    """Drop ANSI styling so parsed values compare as plain text."""
    return _ANSI_RE.sub("", text)


def _extract_slo_fields(line: str) -> dict:
    """Extract TTFT/TPOT values from a log line."""
    result = {}
    for name, pattern in _SLO_PATTERNS.items():
        m = pattern.search(line)
        if m:
            result[name] = float(m.group(1))
    return result


@dataclass
class OptimizerBestRow:
    parallel: str
    batch_size: int
    concurrency: int
    token_s: float
    ttft: Optional[float]
    tpot: Optional[float]
    num_devices: int
    qps_req_s: Optional[float] = None
    """Per-instance request QPS as printed by the summary ("QPS (req/s)"), if present."""


def _column_spans(separator: str) -> list[tuple[int, int]]:
    """Character spans of each column, taken from a prettytable ``+---+`` row."""
    plus = [index for index, char in enumerate(separator) if char == "+"]
    return [(plus[i] + 1, plus[i + 1]) for i in range(len(plus) - 1)]


def _parse_first_table_row(output: str) -> dict:
    """Parse the first (top-ranked) data row of the summary table.

    Cells are sliced using the column spans of the header separator rather than
    split on ``|``, because the ``parallel`` column itself reads
    ``TP=8 | PP=1 | DP=1``. ANSI styling is stripped *before* slicing: the escape
    sequences occupy real positions in the raw text, so a styled cell shifts every
    later column relative to the separator row and the columns stop lining up.

    Returns:
        Column name -> cell text for the top row, or ``{}`` when no table exists.
    """
    lines = _strip_ansi(output).splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        names = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if "Top" not in names or "concurrency" not in names:
            continue
        separator_index = index + 1
        if separator_index >= len(lines) or not lines[separator_index].lstrip().startswith("+"):
            continue
        spans = _column_spans(lines[separator_index])
        data_index = separator_index + 1
        if len(spans) != len(names) or data_index >= len(lines):
            continue
        data_line = lines[data_index]
        if not data_line.lstrip().startswith("|"):
            continue
        return {name: data_line[start:end].strip() for name, (start, end) in zip(names, spans)}
    return {}


def _table_float(table: dict, key: str) -> Optional[float]:
    value = table.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _table_int(table: dict, key: str) -> Optional[int]:
    value = _table_float(table, key)
    return int(value) if value is not None else None


def _parse_best_row(output: str) -> Optional[OptimizerBestRow]:
    """Parse the normal run's best configuration.

    Throughput and the SLO metrics come from the "Overall Best Configuration"
    block. The parallel descriptor, concurrency and batch size are only printed
    in the Top-N table, so the table's first row (the top-ranked configuration)
    supplies them. Previously those fields were hard-coded to zero, which made
    decode scenarios derive ``--target-qps 0.0`` and made the parallel check
    unreachable.
    """
    best_section = False
    fields: dict = {}
    for line in output.splitlines():
        if "Overall Best Configuration" in line:
            best_section = True
        elif best_section and line.strip().startswith("---"):
            break
        elif best_section:
            line = line.strip()
            if "Best Throughput:" in line:
                m = re.search(r"([\d.]+)\s+tokens/s", line)
                if m:
                    fields["token_s"] = float(m.group(1))
            fields.update(_extract_slo_fields(line))
            if "Parallel:" in line:
                m = re.search(r"Parallel:\s+(.+?)\s*$", line)
                if m:
                    fields["parallel"] = m.group(1).strip()

    table = _parse_first_table_row(output)
    if "token_s" not in fields:
        throughput = _table_float(table, "Throughput (token/s)")
        if throughput is None:
            return None
        fields["token_s"] = throughput
    for metric in ("ttft", "tpot"):
        if metric not in fields:
            value = _table_float(table, "TTFT (ms)" if metric == "ttft" else "TPOT (ms)")
            if value is not None:
                fields[metric] = value

    return OptimizerBestRow(
        parallel=fields.get("parallel") or table.get("parallel", ""),
        batch_size=_table_int(table, "batch_size") or 0,
        concurrency=_table_int(table, "concurrency") or 0,
        token_s=fields["token_s"],
        ttft=fields.get("ttft"),
        tpot=fields.get("tpot"),
        num_devices=_table_int(table, "num_devices") or 0,
        qps_req_s=_table_float(table, "QPS (req/s)"),
    )


def _parse_capacity_best(output: str) -> Optional[dict]:
    """Parse the capacity planning summary block."""
    lines = output.splitlines()
    fields: dict = {}
    in_summary = False
    for line in lines:
        if "Capacity Planning Result" in line:
            in_summary = True
        elif in_summary and "推算对比表" in line:
            break
        elif in_summary:
            line = line.strip()
            if "Minimum Devices:" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    fields["minimum_devices"] = int(m.group(1))
            if "Achieved QPS:" in line:
                m = re.search(r"([\d.]+)", line)
                if m:
                    fields["achieved_qps"] = float(m.group(1))
            if "Parallel:" in line:
                # Capture to end of line: the value contains spaces, and a
                # non-greedy match truncated it to "TP=8", hiding real mismatches.
                m = re.search(r"Parallel:\s+(.+?)\s*$", line)
                if m:
                    fields["parallel"] = m.group(1).strip()
            fields.update(_extract_slo_fields(line))
    return fields if fields else None


def _run(cmd: list[str]) -> str:
    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print(result.stderr[-2000:])
        return ""
    return result.stdout + result.stderr


def _missing_metrics(row: OptimizerBestRow) -> list[str]:
    """Fields a parsed best row must carry for the check to mean anything.

    Decode-only rows are validated on TPOT, which needs the concurrency the QPS
    formula divides by; aggregation rows fall back to token throughput.
    """
    missing = []
    if not row.parallel:
        missing.append("parallel")
    if row.tpot is not None and row.ttft is None:
        if row.concurrency <= 0 and not (row.qps_req_s and row.qps_req_s > 0):
            missing.append("concurrency (or a QPS (req/s) column)")
        if row.tpot <= 0:
            missing.append("positive tpot")
    elif row.ttft is None and not row.token_s:
        missing.append("ttft or token/s")
    return missing


def _compute_req_qps(row: OptimizerBestRow, output_length: int) -> Optional[float]:
    """Per-instance request QPS, mirroring the capacity planner's producers.

    Decode rows are reconstructed from concurrency because the summary prints the
    QPS cell rounded to two decimals, which loses precision for small instances.
    Prefill rows consume the printed cell, since the producer may derive it from
    the prefill makespan rather than TTFT. Returns None when this row's mode
    lacks the inputs, so the caller fails loudly instead of passing
    ``--target-qps 0`` (which the CLI rejects).
    """
    if row.tpot is not None and row.ttft is None:
        # Decode: the planner uses concurrency / (tpot * output_length), so
        # reconstruct it exactly rather than consuming the two-decimal display cell.
        if row.concurrency > 0 and row.tpot > 0 and output_length > 0:
            return row.concurrency / (row.tpot * output_length) * 1000
        if row.qps_req_s is not None and row.qps_req_s > 0:
            return row.qps_req_s
        return None
    # Disaggregation prefill can derive QPS from the prefill makespan instead of
    # TTFT, which is not reconstructible here, so the printed cell is authoritative.
    if row.qps_req_s is not None and row.qps_req_s > 0:
        return row.qps_req_s
    if not row.token_s or output_length <= 0:
        return None
    return compute_request_qps(row.token_s, output_length)


def _normalize_parallel(text: str) -> str:
    """Canonical form of a parallel descriptor for equality comparison.

    Order and spacing are normalised, but the dimension set must match exactly:
    a substring test accepts "TP=4" for "TP=4 | PP=2", which is a different
    topology.
    """
    tokens = [token.strip().upper() for token in re.split(r"[|,]", text) if token.strip()]
    return "|".join(sorted(tokens))


def _relative_close(actual: float, expected: float, tolerance: float = SLO_RELATIVE_TOLERANCE) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tolerance


def _run_scenario(name: str, base_cmd: list[str], output_length: int) -> bool:
    print(f"\n[{name}]")

    output_normal = _run(base_cmd)
    best = _parse_best_row(output_normal)
    if best is None:
        print("FAIL: could not parse best row from normal output")
        return False

    missing = _missing_metrics(best)
    if missing:
        print(f"FAIL: normal output is missing required field(s): {', '.join(missing)}")
        return False

    print(f"  Normal best: token/s={best.token_s:.2f}, parallel={best.parallel}")
    if best.ttft:
        print(f"  TTFT={best.ttft:.2f} ms")
    if best.tpot:
        print(f"  TPOT={best.tpot:.2f} ms")

    target_qps = _compute_req_qps(best, output_length)
    if target_qps is None or target_qps <= 0:
        print("FAIL: could not derive a positive target QPS from the normal output")
        return False
    print(f"  Target QPS: {target_qps:.6f}")

    # Full precision on purpose: rounding to two decimals turned a small target
    # into 0.0, which the CLI rejects because --target-qps must be positive.
    cp_cmd = base_cmd + ["--target-qps", f"{target_qps:.6f}"]
    output_cp = _run(cp_cmd)
    cp_result = _parse_capacity_best(output_cp)
    if cp_result is None:
        print("FAIL: could not parse capacity planning output")
        return False

    print(f"  Capacity: min_devices={cp_result.get('minimum_devices')}, parallel={cp_result.get('parallel', '')}")

    for field in ("minimum_devices", "achieved_qps", "parallel"):
        if cp_result.get(field) in (None, ""):
            print(f"FAIL: capacity output is missing required field: {field}")
            return False

    if cp_result["minimum_devices"] <= 0:
        print(f"FAIL: capacity planning reported no feasible device count ({cp_result['minimum_devices']})")
        return False

    # 0.005 is half of the two-decimal display unit, so the rendered value is
    # allowed to round down without being judged as missing the target.
    if cp_result["achieved_qps"] < target_qps - 0.005:
        print(f"FAIL: achieved QPS {cp_result['achieved_qps']:.2f} is below the target {target_qps:.2f}")
        return False

    if _normalize_parallel(best.parallel) != _normalize_parallel(cp_result["parallel"]):
        print(f"FAIL: Parallel mismatch: normal={best.parallel}, capacity={cp_result['parallel']}")
        return False

    for metric, expected in (("ttft", best.ttft), ("tpot", best.tpot)):
        if expected is None:
            continue
        actual = cp_result.get(metric)
        if actual is None:
            print(f"FAIL: capacity output is missing {metric.upper()}")
            return False
        if not _relative_close(actual, expected):
            print(
                f"FAIL: {metric.upper()} moved beyond {SLO_RELATIVE_TOLERANCE:.0%}: "
                f"normal={expected:.2f} ms, capacity={actual:.2f} ms"
            )
            return False

    print("  PASS: config consistent")
    return True


# ============================================================
# Scenario 1: Kimi-K3 (aggregation, dspark, 64 die)
#   Config: TP=16, PP=1, DP=4, EP=64, DSpark=7/acc=2.58, compile
# ============================================================


def run_kimi_k3():
    base_cmd = [
        sys.executable,
        "-m",
        "cli.inference.throughput_optimizer",
        "moonshotai/Kimi-K3",
        "--device",
        "ATLAS_800_A3_752T_128G_DIE",
        "--num-devices",
        "64",
        "--input-length",
        "28000",
        "--output-length",
        "700",
        "--max-batched-tokens",
        "4096",
        "--tp-sizes",
        "16",
        "--ep-sizes",
        "64",
        "--moe-dp-sizes",
        "1",
        "--pp-sizes",
        "1",
        "--compilation-config",
        "enable_sequence_parallel",
        "--enable-shared-expert-tp",
        "--speculative-method",
        "dspark",
        "--num-speculative-tokens",
        "7",
        "--acceptance-length",
        "2.58",
        "--num-draft-layers",
        "5",
        "--dspark-markov-rank",
        "256",
        "--dspark-markov-head",
        "vanilla",
        "--quantize-linear-action",
        "W4A8_DYNAMIC",
        "--quantize-non-expert-linear-action",
        "W8A8_DYNAMIC",
        "--reserved-memory-gb",
        "9.6",
        "--prefix-cache-hit-rate",
        "0.1069",
        "--batch-range",
        "1",
        "1",
        "--jobs",
        "1",
        "--compile",
    ]
    assert _run_scenario("Kimi-K3 (aggregation, TP16/EP64, 64 die)", base_cmd, 700)


# ============================================================
# Scenario 2: GLM-5.1 prefill (disagg prefill, profiling, 32 die)
#   Config: TP=8, PP=4, DP=1, EP=4, chunk=24576, hit=15.16%
# ============================================================


def run_glm51_prefill():
    model = ".cache/modelscope/models/Eco-Tech/GLM-5___1-w8a8"
    db = (
        "tensor_cast/performance_model/profiling_database/data/"
        "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
    )

    base_cmd = [
        sys.executable,
        "-m",
        "cli.inference.throughput_optimizer",
        model,
        "--device",
        "ATLAS_800_A3_752T_128G_DIE",
        "--num-devices",
        "32",
        "--input-length",
        "51000",
        "--output-length",
        "700",
        "--disagg",
        "--ttft-limit",
        "10000",
        "--max-batched-tokens",
        "24576",
        "--prefix-cache-hit-rate",
        "0.1516",
        "--tp-sizes",
        "8",
        "--ep-sizes",
        "4",
        "--moe-dp-sizes",
        "1",
        "--pp-sizes",
        "4",
        "--pp-layer-partitions",
        "[[20,20,20,18]]",
        "--num-mtp-tokens",
        "0",
        "--batch-range",
        "1",
        "8",
        "--quantize-linear-action",
        "W8A8_DYNAMIC",
        "--quantize-attention-action",
        "DISABLED",
        "--enable-shared-expert-tp",
        "--compile",
        "--compilation-config",
        "enable_sequence_parallel",
        "enable_dispatch_ffn_combine",
        "--performance-model",
        "profiling",
        "--profiling-database-path",
        db,
        "--reserved-memory-gb",
        "10",
        "--jobs",
        "2",
    ]
    assert _run_scenario("GLM-5.1 prefill (TP8/PP4/EP4, profiling)", base_cmd, 700)


# ============================================================
# Scenario 3: GLM-5.1 decode (disagg decode, profiling, 32 die)
#   Config: TP=4, DP=8, EP=4, MTP=2
# ============================================================


def run_glm51_decode():
    model = ".cache/modelscope/models/Eco-Tech/GLM-5___1-w8a8"
    db = (
        "tensor_cast/performance_model/profiling_database/data/"
        "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"
    )

    base_cmd = [
        sys.executable,
        "-m",
        "cli.inference.throughput_optimizer",
        model,
        "--device",
        "ATLAS_800_A3_752T_128G_DIE",
        "--num-devices",
        "32",
        "--input-length",
        "51000",
        "--output-length",
        "700",
        "--disagg",
        "--tpot-limit",
        "50",
        "--max-batched-tokens",
        "51000",
        "--prefix-cache-hit-rate",
        "0.0665",
        "--tp-sizes",
        "4",
        "--ep-sizes",
        "4",
        "--moe-dp-sizes",
        "1",
        "--pp-sizes",
        "1",
        "--dcp-sizes",
        "1",
        "--num-mtp-tokens",
        "2",
        "--batch-range",
        "1",
        "32",
        "--quantize-linear-action",
        "W8A8_DYNAMIC",
        "--quantize-attention-action",
        "DISABLED",
        "--enable-shared-expert-tp",
        "--compile",
        "--compilation-config",
        "enable_dispatch_ffn_combine",
        "--performance-model",
        "profiling",
        "--profiling-database-path",
        db,
        "--reserved-memory-gb",
        "10",
        "--jobs",
        "4",
    ]
    assert _run_scenario("GLM-5.1 decode (TP4/EP4/MTP2, profiling)", base_cmd, 700)


# ============================================================
# Scenario 4: DeepSeek-V4-Pro (disagg prefill, dspark, 32 die)
#   Config: TP=2, PP=4, DP=4, EP=8
# ============================================================


def run_ds_v4_pro():
    base_cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "cli.inference.throughput_optimizer",
        "deepseek-ai/DeepSeek-V4-Pro",
        "--device",
        "ATLAS_800_A3_752T_128G_DIE",
        "--num-devices",
        "32",
        "--input-length",
        "32768",
        "--output-length",
        "300",
        "--prefix-cache-hit-rate",
        "0",
        "--quantize-linear-action",
        "W4A8_DYNAMIC",
        "--quantize-non-expert-linear-action",
        "W8A8_DYNAMIC",
        "--quantize-attention-action",
        "DISABLED",
        "--reserved-memory-gb",
        "10",
        "--tp-sizes",
        "2",
        "--pp-sizes",
        "4",
        "--ep-sizes",
        "8",
        "--moe-dp-sizes",
        "1",
        "--speculative-method",
        "dspark",
        "--num-speculative-tokens",
        "5",
        "--num-draft-layers",
        "3",
        "--acceptance-length",
        "5",
        "--dspark-markov-rank",
        "256",
        "--max-batched-tokens",
        "8192",
        "--batch-range",
        "1",
        "30",
        "--ttft-limit",
        "7000",
        "--disagg",
        "--performance-model",
        "analytic",
        "--enable-shared-expert-tp",
        "--compile",
        "--compilation-config",
        "enable_multistream",
        "enable_matmul_allreduce",
        "enable_dispatch_ffn_combine",
        "--jobs",
        "2",
        "--max-search-combinations",
        "0",
        "--log-level",
        "info",
    ]
    assert _run_scenario("DeepSeek-V4-Pro prefill (TP2/PP4/EP8, dspark)", base_cmd, 300)


if __name__ == "__main__":
    scenarios = [
        run_kimi_k3,
        run_glm51_prefill,
        run_glm51_decode,
        run_ds_v4_pro,
    ]

    failed = []
    for func in scenarios:
        try:
            func()
            print(f"  {func.__name__}: PASS")
        except Exception as e:
            print(f"  {func.__name__}: ERROR: {e}")
            failed.append(func.__name__)

    if failed:
        print(f"\n{len(failed)} scenario(s) failed: {failed}")
        sys.exit(1)
    print("\nAll scenarios passed.")


def _render_best_config_table(
    *,
    token_s=500.0,
    ttft=300.0,
    tpot=40.0,
    parallel="TP=8 | PP=1 | DP=1",
    concurrency=64,
    num_devices=8,
    batch_size=64,
    qps=None,
):
    """Render a summary table in the same shape the optimizer prints.

    Built with PrettyTable rather than a literal string so the fixture cannot
    drift from the real output format (``parallel`` contains ``|`` and the
    throughput cell is ANSI-wrapped in production).
    """
    columns = [
        "Top",
        "Throughput (token/s)",
        "TTFT (ms)",
        "TPOT (ms)",
        "concurrency",
        "num_devices",
        "parallel",
        "batch_size",
    ]
    row = [
        1,
        f"\033[1m{token_s:.2f}\033[0m",
        "" if ttft is None else f"{ttft:.2f}",
        "" if tpot is None else f"{tpot:.2f}",
        concurrency,
        num_devices,
        parallel,
        batch_size,
    ]
    if qps is not None:
        columns.insert(2, "QPS (req/s)")
        row.insert(2, f"{qps:.2f}")
    table = PrettyTable()
    table.field_names = columns
    table.add_row(row)
    return table.get_string()


def _normal_output(
    *,
    token_s=500.0,
    ttft=300.0,
    tpot=40.0,
    parallel="TP=8 | PP=1 | DP=1",
    concurrency=64,
    num_devices=8,
    batch_size=64,
    qps=None,
):
    """Full normal-run output: best-configuration block plus the Top-N table."""
    lines = ["  Overall Best Configuration:", f"    Best Throughput: {token_s:.2f} tokens/s"]
    if ttft is not None:
        lines.append(f"    TTFT: {ttft:.2f} ms")
    if tpot is not None:
        lines.append(f"    TPOT: {tpot:.2f} ms")
    lines.append("  " + "-" * 76)
    lines.append(
        _render_best_config_table(
            token_s=token_s,
            ttft=ttft,
            tpot=tpot,
            parallel=parallel,
            concurrency=concurrency,
            num_devices=num_devices,
            batch_size=batch_size,
            qps=qps,
        )
    )
    lines.append("*" * 80)
    return "\n".join(lines)


def _capacity_output(*, minimum_devices=8, achieved_qps=1.0, parallel="TP=8 | PP=1 | DP=1", ttft=None, tpot=None):
    """Capacity planning summary block."""
    lines = ["Capacity Planning Result", f"  Minimum Devices: {minimum_devices}"]
    lines.append(f"  Achieved QPS:     {achieved_qps:.2f} req/s")
    if parallel is not None:
        lines.append(f"  Parallel:         {parallel}")
    if ttft is not None:
        lines.append(f"  TTFT:             {ttft:.2f} ms")
    if tpot is not None:
        lines.append(f"  TPOT:             {tpot:.2f} ms")
    lines.append("  推算对比表 (按总卡数升序, 同卡数取单卡QPS最高)")
    return "\n".join(lines)


class TestExtractSloFields(unittest.TestCase):
    def test_extract_ttft(self):
        result = _extract_slo_fields("  TTFT: 123.45 ms")
        self.assertEqual(result, {"ttft": 123.45})

    def test_extract_tpot(self):
        result = _extract_slo_fields("  TPOT: 67.89 ms")
        self.assertEqual(result, {"tpot": 67.89})

    def test_extract_both(self):
        line = "  TTFT: 100.0 ms  TPOT: 50.0 ms"
        result = _extract_slo_fields(line)
        self.assertEqual(result, {"ttft": 100.0, "tpot": 50.0})

    def test_no_match(self):
        result = _extract_slo_fields("  some other text")
        self.assertEqual(result, {})


class TestParseFirstTableRow(unittest.TestCase):
    def test_reads_every_module_level_column(self):
        row = _parse_first_table_row(_normal_output(concurrency=32, num_devices=4, batch_size=16))
        self.assertEqual(row["concurrency"], "32")
        self.assertEqual(row["num_devices"], "4")
        self.assertEqual(row["batch_size"], "16")

    def test_parallel_cell_keeps_the_embedded_pipes(self):
        """The parallel cell contains '|', so it cannot be split on '|'."""
        row = _parse_first_table_row(_normal_output(parallel="TP=4 | PP=2 | DP=1"))
        self.assertEqual(row["parallel"], "TP=4 | PP=2 | DP=1")

    def test_throughput_cell_is_ansi_free(self):
        row = _parse_first_table_row(_normal_output(token_s=1234.5))
        self.assertEqual(row["Throughput (token/s)"], "1234.50")

    def test_qps_column_is_read_when_present(self):
        row = _parse_first_table_row(_normal_output(ttft=None, tpot=20.0, qps=3.5))
        self.assertEqual(row["QPS (req/s)"], "3.50")

    def test_no_table_returns_empty(self):
        self.assertEqual(_parse_first_table_row("no table here"), {})

    def test_table_without_summary_columns_is_ignored(self):
        other = "| Step | Status |\n+------+--------+\n| 1 | ok |\n"
        self.assertEqual(_parse_first_table_row(other), {})

    def test_header_without_separator_is_ignored(self):
        self.assertEqual(_parse_first_table_row("| Top | concurrency |\nplain text\n"), {})

    def test_column_count_mismatch_is_ignored(self):
        self.assertEqual(_parse_first_table_row("| Top | concurrency |\n+---+\n| 1 | 2 |\n"), {})

    def test_missing_data_row_is_ignored(self):
        self.assertEqual(_parse_first_table_row("| Top | concurrency |\n+-----+------------+\n"), {})

    def test_non_row_after_header_is_ignored(self):
        self.assertEqual(_parse_first_table_row("| Top | concurrency |\n+-----+------------+\ngarbage\n"), {})


class TestParseBestRow(unittest.TestCase):
    def test_parses_block_and_table_together(self):
        row = _parse_best_row(_normal_output(token_s=500.0, ttft=300.0, tpot=40.0))
        self.assertIsNotNone(row)
        self.assertEqual(row.token_s, 500.0)
        self.assertEqual(row.ttft, 300.0)
        self.assertEqual(row.tpot, 40.0)
        self.assertEqual(row.parallel, "TP=8 | PP=1 | DP=1")
        # Previously hard-coded to 0, which broke the decode target-QPS formula.
        self.assertEqual(row.concurrency, 64)
        self.assertEqual(row.num_devices, 8)
        self.assertEqual(row.batch_size, 64)

    def test_decode_row_has_no_ttft_but_has_qps(self):
        row = _parse_best_row(_normal_output(ttft=None, tpot=20.0, concurrency=128, qps=4.0))
        self.assertIsNone(row.ttft)
        self.assertEqual(row.tpot, 20.0)
        self.assertEqual(row.concurrency, 128)
        self.assertEqual(row.qps_req_s, 4.0)

    def test_parallel_line_inside_block_is_kept_in_full(self):
        output = "Overall Best Configuration\nBest Throughput: 500.0 tokens/s\nParallel: TP=8 | PP=2\n---\n"
        row = _parse_best_row(output)
        self.assertEqual(row.parallel, "TP=8 | PP=2")

    def test_table_only_output_falls_back_to_the_table(self):
        """Some summary layouts print the table without a best-config block."""
        output = _render_best_config_table(token_s=777.0, ttft=111.0, tpot=None)
        row = _parse_best_row(output)
        self.assertIsNotNone(row)
        self.assertEqual(row.token_s, 777.0)
        self.assertEqual(row.ttft, 111.0)
        self.assertIsNone(row.tpot)

    def test_no_token_s(self):
        output = "Overall Best Configuration\n---\nTTFT: 300.0 ms\n---"
        self.assertIsNone(_parse_best_row(output))

    def test_parse_empty(self):
        self.assertIsNone(_parse_best_row(""))


class TestParseCapacityBest(unittest.TestCase):
    SAMPLE = """
Capacity Planning Result
  Minimum Devices: 32
  Achieved QPS:     105.0 req/s
  Parallel:         TP=8 | PP=1 | DP=1
  TTFT:             300.0 ms
推算对比表
"""

    def test_parse_full(self):
        result = _parse_capacity_best(self.SAMPLE)
        self.assertIsNotNone(result)
        self.assertEqual(result["minimum_devices"], 32)
        self.assertAlmostEqual(result["achieved_qps"], 105.0)
        # The full descriptor is captured, not just the first token.
        self.assertEqual(result["parallel"], "TP=8 | PP=1 | DP=1")
        self.assertAlmostEqual(result["ttft"], 300.0)

    def test_parse_empty(self):
        self.assertIsNone(_parse_capacity_best(""))


class TestComputeReqQps(unittest.TestCase):
    def test_prefill_uses_the_printed_qps_cell(self):
        """A prefill producer may use the phase makespan, so the cell is authoritative."""
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=64,
            token_s=500.0,
            ttft=500.0,
            tpot=None,
            num_devices=8,
            qps_req_s=2.5,
        )
        self.assertAlmostEqual(_compute_req_qps(row, 512), 2.5)

    def test_decode_prefers_the_exact_formula_over_the_rounded_cell(self):
        """The printed cell is 2-decimal; the formula keeps full precision."""
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=7,
            token_s=500.0,
            ttft=None,
            tpot=700.0,
            num_devices=8,
            qps_req_s=0.01,
        )
        self.assertAlmostEqual(_compute_req_qps(row, 512), 7 / (700.0 * 512) * 1000)

    def test_decode_falls_back_to_the_printed_cell(self):
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=0,
            token_s=500.0,
            ttft=None,
            tpot=700.0,
            num_devices=8,
            qps_req_s=0.014,
        )
        self.assertAlmostEqual(_compute_req_qps(row, 512), 0.014)

    def test_token_s_fallback(self):
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=0,
            token_s=500.0,
            ttft=300.0,
            tpot=None,
            num_devices=8,
        )
        qps = _compute_req_qps(row, 512)
        self.assertAlmostEqual(qps, 500.0 / 512)

    def test_tpot_based(self):
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=64,
            token_s=500.0,
            ttft=None,
            tpot=40.0,
            num_devices=8,
        )
        expected = 64 / (40.0 * 512) * 1000
        self.assertAlmostEqual(_compute_req_qps(row, 512), expected)

    def test_decode_without_concurrency_is_unverifiable(self):
        """The old code returned 0 here, producing an unusable --target-qps 0."""
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=0,
            token_s=500.0,
            ttft=None,
            tpot=40.0,
            num_devices=8,
        )
        self.assertIsNone(_compute_req_qps(row, 512))

    def test_aggregation_without_throughput_is_unverifiable(self):
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=64,
            token_s=0.0,
            ttft=300.0,
            tpot=None,
            num_devices=8,
        )
        self.assertIsNone(_compute_req_qps(row, 512))

    def test_tpot_zero_is_unverifiable(self):
        row = OptimizerBestRow(
            parallel="TP=8",
            batch_size=0,
            concurrency=64,
            token_s=500.0,
            ttft=None,
            tpot=0.0,
            num_devices=8,
        )
        self.assertIsNone(_compute_req_qps(row, 512))


class TestMissingMetrics(unittest.TestCase):
    def _row(self, **overrides):
        defaults = dict(
            parallel="TP=8",
            batch_size=1,
            concurrency=64,
            token_s=500.0,
            ttft=300.0,
            tpot=40.0,
            num_devices=8,
        )
        defaults.update(overrides)
        return OptimizerBestRow(**defaults)

    def test_complete_row_is_accepted(self):
        self.assertEqual(_missing_metrics(self._row()), [])

    def test_decode_row_needs_concurrency_or_qps_cell(self):
        missing = _missing_metrics(self._row(ttft=None, concurrency=0))
        self.assertTrue(any("concurrency" in item for item in missing))

    def test_decode_row_with_qps_cell_needs_no_concurrency(self):
        self.assertEqual(_missing_metrics(self._row(ttft=None, concurrency=0, qps_req_s=1.5)), [])

    def test_decode_row_needs_positive_tpot(self):
        missing = _missing_metrics(self._row(ttft=None, tpot=0.0))
        self.assertIn("positive tpot", missing)

    def test_missing_parallel_is_reported(self):
        self.assertIn("parallel", _missing_metrics(self._row(parallel="")))

    def test_aggregation_without_throughput_is_reported(self):
        missing = _missing_metrics(self._row(ttft=None, tpot=None, token_s=0.0))
        self.assertIn("ttft or token/s", missing)


class TestParallelNormalization(unittest.TestCase):
    def test_spacing_and_order_are_irrelevant(self):
        self.assertEqual(_normalize_parallel("TP=8|PP=1|DP=1"), _normalize_parallel("dp=1 | tp=8 | pp=1"))

    def test_subset_is_not_equal(self):
        """A substring test accepted "TP=4" for "TP=4 | PP=2"."""
        self.assertNotEqual(_normalize_parallel("TP=4"), _normalize_parallel("TP=4 | PP=2"))

    def test_different_dimension_value_is_not_equal(self):
        self.assertNotEqual(_normalize_parallel("TP=4"), _normalize_parallel("TP=8"))


class TestRelativeClose(unittest.TestCase):
    def test_within_tolerance(self):
        self.assertTrue(_relative_close(110.0, 100.0, 0.25))

    def test_beyond_tolerance(self):
        self.assertFalse(_relative_close(200.0, 100.0, 0.25))

    def test_small_drift_is_accepted_at_default_tolerance(self):
        # Observed in the recorded E2E run: TPOT 19 -> 21 ms.
        self.assertTrue(_relative_close(21.0, 19.0, SLO_RELATIVE_TOLERANCE))

    def test_zero_reference_requires_exact(self):
        self.assertTrue(_relative_close(0.0, 0.0))
        self.assertFalse(_relative_close(1.0, 0.0))


class TestRunWithMock(unittest.TestCase):
    @patch(f"{__name__}.subprocess.run")
    def test_run_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = _run(["echo", "hello"])
        self.assertIn("ok", result)

    @patch(f"{__name__}.subprocess.run")
    def test_run_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = _run(["false"])
        self.assertEqual(result, "")


class TestRunScenarioWithMock(unittest.TestCase):
    """A scenario only passes when every field it asserts is actually present."""

    def _run_pair(self, normal_output, cp_output):
        with patch(f"{__name__}._run") as mock_run:
            mock_run.side_effect = [normal_output, cp_output]
            return _run_scenario("test", ["cmd"], 512), mock_run

    def test_aggregation_pass_with_all_metrics(self):
        # target QPS = 500 / 512 = 0.9766, achieved 1.2 clears it.
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0, ttft=300.0, tpot=40.0),
            _capacity_output(achieved_qps=1.2, ttft=320.0, tpot=42.0),
        )
        self.assertTrue(ok)

    def test_decode_pass_uses_unrounded_positive_target(self):
        """Regression: concurrency was hard-coded to 0, so the target was 0.0.

        A decode row with 0.014 req/s per instance must produce a positive,
        non-zero target instead of the CLI rejecting ``--target-qps 0``.
        """
        ok, mock_run = self._run_pair(
            _normal_output(ttft=None, tpot=700.0, concurrency=7),
            _capacity_output(achieved_qps=1.008, tpot=720.0),
        )
        self.assertTrue(ok)
        cp_cmd = mock_run.call_args_list[1].args[0]
        target = float(cp_cmd[cp_cmd.index("--target-qps") + 1])
        self.assertGreater(target, 0)
        self.assertAlmostEqual(target, 7 / (700.0 * 512) * 1000, places=6)

    def test_zero_throughput_target_is_rejected(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=0.0, ttft=300.0),
            _capacity_output(achieved_qps=1.2),
        )
        self.assertFalse(ok)

    def test_decode_without_concurrency_fails(self):
        ok, _ = self._run_pair(
            _normal_output(ttft=None, tpot=700.0, concurrency=0, qps=None),
            _capacity_output(achieved_qps=1.0, tpot=700.0),
        )
        self.assertFalse(ok)

    def test_parallel_mismatch_fails(self):
        ok, _ = self._run_pair(
            _normal_output(parallel="TP=8 | PP=1 | DP=1"),
            _capacity_output(parallel="TP=4 | PP=1 | DP=1"),
        )
        self.assertFalse(ok)

    def test_subset_parallel_is_not_consistent(self):
        """Same-Parallel check must not degrade to a substring comparison."""
        ok, _ = self._run_pair(
            _normal_output(parallel="TP=4"),
            _capacity_output(parallel="TP=4 | PP=2"),
        )
        self.assertFalse(ok)

    def test_slo_drift_beyond_tolerance_fails(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0, ttft=300.0),
            _capacity_output(achieved_qps=1.2, ttft=900.0),
        )
        self.assertFalse(ok)

    def test_slo_drift_within_tolerance_passes(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0, ttft=5734.0, tpot=None),
            _capacity_output(achieved_qps=1.2, ttft=6371.0),
        )
        self.assertTrue(ok)

    def test_missing_capacity_slo_metric_fails(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0, ttft=300.0),
            _capacity_output(achieved_qps=1.2, ttft=None),
        )
        self.assertFalse(ok)

    def test_achieved_qps_below_target_fails(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0),  # target 0.9766
            _capacity_output(achieved_qps=0.5),
        )
        self.assertFalse(ok)

    def test_zero_minimum_devices_fails(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0),
            _capacity_output(minimum_devices=0, achieved_qps=1.2),
        )
        self.assertFalse(ok)

    def test_missing_capacity_parallel_fails(self):
        ok, _ = self._run_pair(
            _normal_output(token_s=500.0),
            _capacity_output(achieved_qps=1.2, parallel=None),
        )
        self.assertFalse(ok)

    def test_missing_capacity_achieved_qps_fails(self):
        cp_output = "Capacity Planning Result\n  Minimum Devices: 8\n  Parallel:         TP=8\n"
        ok, _ = self._run_pair(_normal_output(token_s=500.0), cp_output)
        self.assertFalse(ok)

    def test_parse_failure(self):
        ok, _ = self._run_pair("garbage", "garbage")
        self.assertFalse(ok)


class TestScenarioFunctionsWithMock(unittest.TestCase):
    @patch(f"{__name__}._run_scenario")
    def test_run_kimi_k3(self, mock_scenario):
        mock_scenario.return_value = True
        run_kimi_k3()
        mock_scenario.assert_called_once()

    @patch(f"{__name__}._run_scenario")
    def test_run_glm51_prefill(self, mock_scenario):
        mock_scenario.return_value = True
        run_glm51_prefill()
        mock_scenario.assert_called_once()

    @patch(f"{__name__}._run_scenario")
    def test_run_glm51_decode(self, mock_scenario):
        mock_scenario.return_value = True
        run_glm51_decode()
        mock_scenario.assert_called_once()

    @patch(f"{__name__}._run_scenario")
    def test_run_ds_v4_pro(self, mock_scenario):
        mock_scenario.return_value = True
        run_ds_v4_pro()
        mock_scenario.assert_called_once()
