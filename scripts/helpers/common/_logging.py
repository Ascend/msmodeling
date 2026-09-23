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


def _env_var_source(key: str) -> str:
    if key not in os.environ:
        return "default"
    if os.environ[key] == "":
        return "empty"
    return "env"


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
    _log_env_var(logger, "GITCODE_OWNER", cfg.gitcode_owner or "(not set)")
    _log_env_var(logger, "GITCODE_REPO", cfg.gitcode_repo or "(not set)")
    _log_env_var(
        logger,
        "GITCODE_PR_NUMBER",
        cfg.gitcode_pr_number if cfg.gitcode_pr_number is not None else "(not set)",
    )
    _log_env_var(
        logger,
        "GITCODE_PAT",
        "(configured)" if cfg.gitcode_pat else "(not set)",
    )
    _log_env_var(logger, "MSMODELING_OFFLINE", os.environ.get("MSMODELING_OFFLINE", ""))
    logger.info("==========================")


def _log_env_var(logger: logging.Logger, key: str, value: object) -> None:
    logger.info("  %s = %s  [%s]", key, value, _env_var_source(key))
