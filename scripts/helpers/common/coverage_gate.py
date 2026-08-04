"""Coverage totals from .coverage and threshold check."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    branch = 100.0 if num_branches == 0 else 100.0 * totals["covered_branches"] / num_branches
    return CoverageTotals(line_percent=line, branch_percent=branch)
