"""Unit tests for sim_warmup module."""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

from services.sim_warmup import _lock, ensure_sim_stack_warmed


def _ensure_mock_modules() -> dict[str, MagicMock]:
    """Pre-populate ``sys.modules`` so local imports inside the function body
    (``import torch._dynamo``, etc.) resolve to ``MagicMock`` instances.
    """
    inserted: dict[str, MagicMock] = {}
    for mod_name in [
        "torch",
        "torch._dynamo",
        "tensor_cast",
        "tensor_cast.core",
        "tensor_cast.core.model_runner",
        "tensor_cast.core.user_config",
        "tensor_cast.core.quantization",
        "tensor_cast.core.quantization.datatypes",
        "tensor_cast.device_profiles",
        "serving_cast",
        "serving_cast.parallel_runner",
        "transformers",
    ]:
        if mod_name not in sys.modules:
            m = MagicMock()
            sys.modules[mod_name] = m
            inserted[mod_name] = m
    return inserted


def _cleanup_mock_modules(inserted: dict[str, MagicMock]) -> None:
    for mod_name in inserted:
        sys.modules.pop(mod_name, None)


class TestEnsureSimStackWarmed:
    """Tests for ensure_sim_stack_warmed function."""

    def test_ensure_sim_stack_warmed_is_callable(self):
        """Function can be called without error."""
        inserted = _ensure_mock_modules()
        try:
            import services.sim_warmup as sw

            sw._warmed = False
            ensure_sim_stack_warmed()
            assert sw._warmed is True
        finally:
            _cleanup_mock_modules(inserted)

    def test_ensure_sim_stack_warmed_thread_safe(self):
        """Multiple threads can safely call concurrently."""
        inserted = _ensure_mock_modules()
        try:
            import services.sim_warmup as sw

            sw._warmed = False

            def warm_up():
                ensure_sim_stack_warmed()

            threads = [threading.Thread(target=warm_up) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert sw._warmed is True  # at least one thread warmed it
        finally:
            _cleanup_mock_modules(inserted)
            sw._warmed = False

    def test_lock_exists(self):
        """Module has a lock for thread safety."""
        assert _lock is not None
        assert isinstance(_lock, type(threading.Lock()))

    def test_double_call_skips_warmup(self, monkeypatch):
        """Second call hits the fast-path early return (line 28)."""
        inserted = _ensure_mock_modules()
        try:
            import services.sim_warmup as sw

            sw._warmed = False
            import_counter = {"n": 0}
            original_import = __import__

            def counting_import(name, *a, **kw):
                if any(m in name for m in ("tensor_cast", "serving_cast", "torch._dynamo")):
                    import_counter["n"] += 1
                return original_import(name, *a, **kw)

            monkeypatch.setattr("builtins.__import__", counting_import)
            ensure_sim_stack_warmed()
            count_after_first = import_counter["n"]
            ensure_sim_stack_warmed()
            assert import_counter["n"] == count_after_first  # no new sim imports
            assert sw._warmed is True
        finally:
            _cleanup_mock_modules(inserted)
            sw._warmed = False  # restore

    def test_double_checked_lock_branch(self, monkeypatch):
        """The inner ``if _warmed: return`` (line 31) fires when a caller passes
        the outer check but finds _warmed True inside the lock — i.e. a peer
        warmed it while this caller waited on the lock.

        We simulate the race deterministically: a flag whose truthiness flips
        False (outer check, line 28) -> True (inner check, line 31) between the
        two reads, so the caller enters the lock but returns early without
        re-warming.
        """
        inserted = _ensure_mock_modules()
        try:
            import services.sim_warmup as sw

            class FlipBool:
                """Falsy on first read, truthy thereafter (peer warmed mid-flight)."""

                def __init__(self):
                    self._reads = 0

                def __bool__(self):
                    self._reads += 1
                    return self._reads > 1

            flip = FlipBool()
            monkeypatch.setattr(sw, "_warmed", flip)
            # Outer check (read 1) -> False -> enters lock; inner check (read 2) ->
            # True -> returns early at line 31 (no re-warm).
            ensure_sim_stack_warmed()
            assert flip._reads >= 2
        finally:
            _cleanup_mock_modules(inserted)
