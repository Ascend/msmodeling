# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""Domain exceptions for optix optimizer CLI and orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from optix.optimizer.scheduler import Scheduler

from optix.logging import format_evaluation_failure


class OptimizerError(Exception):
    """Base class for optimizer failures that should terminate a CLI run."""


class ConfigFileNotFoundError(OptimizerError):
    """Raised when ``--config`` points to a path that does not exist."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Custom config file not found: {path}")


class InvalidConfigError(OptimizerError):
    """Raised when a custom TOML config file cannot be parsed."""

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        super().__init__(f"Invalid TOML config file '{path}': {cause}")


class BenchmarkUnavailableError(OptimizerError):
    """Raised when the selected benchmark executable is missing from PATH."""

    def __init__(
        self,
        policy: str,
        executable: str,
        guidance: str,
        *,
        default_policy: str,
    ) -> None:
        self.policy = policy
        self.executable = executable
        self.default_policy = default_policy
        super().__init__(
            f"Selected benchmark '-b {policy}' requires executable '{executable}' on PATH "
            f"(default -b is {default_policy}). Executable not found. {guidance}"
        )


class SimulatorUnavailableError(OptimizerError):
    """Raised when the selected simulator executable is missing from PATH."""

    def __init__(
        self,
        policy: str,
        executable: str,
        guidance: str,
        *,
        default_policy: str,
    ) -> None:
        self.policy = policy
        self.executable = executable
        self.default_policy = default_policy
        super().__init__(
            f"Selected simulator '-e {policy}' requires executable '{executable}' on PATH "
            f"(default -e is {default_policy}). Executable not found. {guidance}"
        )


class BenchmarkResultError(OptimizerError):
    """Raised when benchmark output CSV files are missing or ambiguous."""


class BaselineRunError(OptimizerError):
    """Raised when the default baseline service or benchmark run fails."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] | None = None,
        return_code: int | None = None,
        log_path: str | Path | None = None,
        cause: Exception | Any | None = None,
    ) -> None:
        self.command = list(command or [])
        self.return_code = return_code
        self.log_path = log_path
        self.cause = cause
        super().__init__(message)

    @classmethod
    def from_scheduler(cls, scheduler: Scheduler) -> BaselineRunError:
        """Build a baseline failure from scheduler simulator state."""
        simulator = scheduler.simulator
        command = list(getattr(simulator, "command", None) or [])
        log_path = getattr(simulator, "run_log", None)
        return_code = None
        process = getattr(simulator, "process", None)
        if process is not None:
            return_code = process.returncode
        cause = scheduler.error_info
        message = format_evaluation_failure(scheduler, cause)
        return cls(
            message,
            command=command,
            return_code=return_code,
            log_path=log_path,
            cause=cause,
        )


class NoFeasibleSolutionError(OptimizerError):
    """Raised when PSO finishes without any feasible candidate."""

    def __init__(self) -> None:
        super().__init__("No feasible solution found")
