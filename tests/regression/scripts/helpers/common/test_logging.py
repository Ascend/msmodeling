"""Tests for common._logging — setup_logger, log_env_audit."""

from __future__ import annotations

import logging

import pytest

from scripts.helpers._config import Config
from scripts.helpers.common._logging import log_env_audit


def _audit_config(**overrides: object) -> Config:
    """Build Config for log_env_audit; tolerates pre/post merge field differences."""
    base: dict[str, object] = {
        "test_map_path": "/tmp/test_map.json",
        "base_branch": "develop",
        "line_threshold": 60.0,
        "branch_threshold": 40.0,
        "benchmark_parallel": False,
        "feishu_webhook_url": "",
        "weights_prune": False,
        "msmodeling_cache": ".msmodeling_cache",
        "gitcode_owner": "",
        "gitcode_repo": "",
        "gitcode_pr_number": None,
        "gitcode_pat": "",
    }
    names = set(Config.model_fields)
    base.update(overrides)
    return Config(**{key: value for key, value in base.items() if key in names})


def test_log_env_audit_masks_feishu_webhook_when_configured(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSMODELING_OFFLINE", "1")
    cfg = _audit_config(
        feishu_webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/secret-token",
    )
    logger = logging.getLogger("test_logging")

    with caplog.at_level(logging.INFO, logger="test_logging"):
        log_env_audit(cfg, logger)

    combined = caplog.text
    assert "secret-token" not in combined
    assert "open.feishu.cn" not in combined
    assert "(configured)" in combined


def test_log_env_audit_shows_not_set_when_webhook_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test_logging")

    with caplog.at_level(logging.INFO, logger="test_logging"):
        log_env_audit(_audit_config(), logger)

    assert "(not set)" in caplog.text


def test_log_env_audit_uses_gitcode_env_keys(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITCODE_OWNER", "Ascend")
    logger = logging.getLogger("test_logging")

    with caplog.at_level(logging.INFO, logger="test_logging"):
        log_env_audit(_audit_config(gitcode_owner="Ascend"), logger)

    assert "GITCODE_OWNER = Ascend  [env]" in caplog.text


def test_log_env_audit_marks_default_empty_and_env_sources(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MSMODELING_BENCHMARK_PARALLEL", raising=False)
    monkeypatch.setenv("MSMODELING_TEST_WEIGHTS_PRUNE", "0")
    monkeypatch.setenv("MSMODELING_OFFLINE", "")
    logger = logging.getLogger("test_logging")

    with caplog.at_level(logging.INFO, logger="test_logging"):
        log_env_audit(_audit_config(weights_prune=False), logger)

    combined = caplog.text
    assert "MSMODELING_BENCHMARK_PARALLEL = False  [default]" in combined
    assert "MSMODELING_TEST_WEIGHTS_PRUNE = False  [env]" in combined
    assert "MSMODELING_OFFLINE =   [empty]" in combined
