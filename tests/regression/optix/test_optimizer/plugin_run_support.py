"""Shared helpers for plugin ``.run()`` and simulator command tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


def prepare_plugin_for_run(
    plugin: Any,
    *,
    command: list[str],
    work_path: str,
    run_log: str,
    process_name: str = "",
    run_log_fp: Any | None = None,
    env: dict[str, Any] | None = None,
) -> None:
    """Set attributes required before ``CustomProcess.run()``; fail fast on bad command."""
    if not command:
        raise ValueError("command must be a non-empty list")

    from optix.config.constant import ProcessState, Stage

    plugin.command = command
    plugin.work_path = work_path
    plugin.run_log = run_log
    plugin.run_log_fp = run_log_fp if run_log_fp is not None else MagicMock()
    plugin.process_name = process_name
    plugin.env = env if env is not None else {}
    if not hasattr(plugin, "_process_stage"):
        plugin._process_stage = ProcessState(stage=Stage.stop)
    if not hasattr(plugin, "process"):
        plugin.process = None


def make_mindie_simulator_config(
    tmp_dir: str,
    config_data: dict[str, Any] | None = None,
) -> MagicMock:
    """Build Mindie ``Simulator`` mock config with on-disk JSON paths under *tmp_dir*."""
    if config_data is None:
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
    config_path = Path(tmp_dir) / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    bak_path = Path(tmp_dir) / "config.json.bak"

    mock_config = MagicMock()
    mock_config.config_path = config_path
    mock_config.config_bak_path = bak_path
    mock_config.process_name = "mindie"
    mock_config.command = MagicMock()
    return mock_config
