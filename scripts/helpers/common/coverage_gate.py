"""Coverage totals from .coverage and threshold check.

Merges former coverage_gate.py (load_totals) and check_ut_gate.py (GateConfig,
check_thresholds, check_ut_gate). Single subprocess call per gate check.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.helpers._config import Config
from scripts.helpers._paths import REPO_ROOT

DEFAULT_COVERAGE_DATA = REPO_ROOT / ".coverage"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateConfig:
    line_threshold: float
    branch_threshold: float

    @classmethod
    def from_config(cls, cfg: Config) -> GateConfig:
        return cls(
            line_threshold=cfg.line_threshold,
            branch_threshold=cfg.branch_threshold,
        )


@dataclass(frozen=True, slots=True)
class CoverageTotals:
    line_percent: float
    branch_percent: float


# ---------------------------------------------------------------------------
# Threshold check (pure — no I/O)
# ---------------------------------------------------------------------------


def check_thresholds(
    line_pct: float,
    branch_pct: float,
    config: GateConfig,
) -> list[str]:
    """Return failure messages. Empty list means all thresholds met."""
    failures: list[str] = []
    if line_pct < config.line_threshold:
        failures.append(f"line coverage {line_pct:.1f}% < {config.line_threshold}%")
    if branch_pct < config.branch_threshold:
        failures.append(f"branch coverage {branch_pct:.1f}% < {config.branch_threshold}%")
    return failures


# ---------------------------------------------------------------------------
# Coverage data loading
# ---------------------------------------------------------------------------


def load_totals(coverage_data: Path) -> CoverageTotals:
    """Run ``coverage json`` subprocess, parse totals.

    Raises FileNotFoundError if data file missing, RuntimeError on subprocess failure.
    """
    if not coverage_data.is_file():
        raise FileNotFoundError(f"coverage data not found: {coverage_data}")

    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "-o",
        "-",
        f"--data-file={coverage_data}",
    ]
    logger.debug("Running coverage: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "(no output)"
        raise RuntimeError(f"coverage json failed (exit {result.returncode}): {detail}")

    data = json.loads(result.stdout)
    totals = data["totals"]
    line = float(totals["percent_covered_display"].rstrip("%"))
    num_branches = totals["num_branches"]
    if num_branches == 0:
        branch = 100.0
    else:
        branch = 100.0 * totals["covered_branches"] / num_branches
    return CoverageTotals(line_percent=line, branch_percent=branch)


# ---------------------------------------------------------------------------
# Gate convenience
# ---------------------------------------------------------------------------


def check_ut_gate(
    coverage_data: Path = DEFAULT_COVERAGE_DATA,
    config: GateConfig | None = None,
) -> tuple[bool, str]:
    """Return (passed, message). Loads totals via subprocess.

    Prefer load_totals + check_thresholds when caller already has CoverageTotals.
    """
    if not coverage_data.is_file():
        return False, f"coverage data not found: {coverage_data}"

    try:
        totals = load_totals(coverage_data)
    except (FileNotFoundError, RuntimeError) as exc:
        return False, str(exc)

    cfg = config if config is not None else GateConfig.from_config(Config.from_env())
    failures = check_thresholds(totals.line_percent, totals.branch_percent, cfg)

    if failures:
        return False, "Coverage gate failed: " + "; ".join(failures)
    return (
        True,
        f"Coverage gate passed: line={totals.line_percent:.1f}% branch={totals.branch_percent:.1f}%",
    )
