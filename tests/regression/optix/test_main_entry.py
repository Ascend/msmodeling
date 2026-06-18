"""Regression tests for ``python -m optix`` (optix.__main__)."""

from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest


def test_optix_main_module_delegates_to_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_optimizer_main() -> None:
        calls.append("optimizer")

    monkeypatch.setattr("optix.configure_logger", lambda: None)
    with patch("optix.optimizer.optimizer.main", side_effect=_fake_optimizer_main):
        runpy.run_module("optix", run_name="__main__", alter_sys=True)

    assert calls == ["optimizer"]
