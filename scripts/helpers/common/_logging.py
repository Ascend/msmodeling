"""Shared logging setup and env audit for ci_gate and nightly entry points."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.helpers._config import Config

_LOG_FORMAT = "%(levelname)-5s %(asctime)s.%(msecs)03d [%(filename)s:%(lineno)d] %(message)s"
_LOG_DATE_FORMAT = "%m-%d %H:%M:%S"


def setup_logger(name: str = "ci_gate") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        stream=sys.stderr,
    )
    return logging.getLogger(name)


def log_env_audit(cfg: Config, logger: logging.Logger) -> None:
    logger.info("=== Environment audit ===")
    _log_env_var(logger, "MSMODELING_TEST_MAP_PATH", cfg.test_map_path)
    _log_env_var(logger, "MSMODELING_TEST_BASE_BRANCH", cfg.base_branch)
    _log_env_var(logger, "MSMODELING_TEST_LINE_THRESHOLD", cfg.line_threshold)
    _log_env_var(logger, "MSMODELING_TEST_BRANCH_THRESHOLD", cfg.branch_threshold)
    _log_env_var(logger, "MSMODELING_BENCHMARK_PARALLEL", cfg.benchmark_parallel)
    _log_env_var(
        logger,
        "FEISHU_WEBHOOK_URL",
        "(configured)" if cfg.feishu_webhook_url else "(not set)",
    )
    _log_env_var(logger, "MSMODELING_CACHE", cfg.msmodeling_cache)
    _log_env_var(logger, "MSMODELING_TEST_WEIGHTS_PRUNE", cfg.weights_prune)
    _log_env_var(logger, "MSMODELING_OFFLINE", os.environ.get("MSMODELING_OFFLINE", "(not set)"))
    logger.info("==========================")


def _log_env_var(logger: logging.Logger, key: str, value: object) -> None:
    source = "env" if key in os.environ else "default"
    logger.info("  %s = %s  [%s]", key, value, source)
