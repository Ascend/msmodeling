"""One-time, single-threaded import of the simulation stack.

Why this exists: the runner adapters do their heavy imports (``tensor_cast``,
``serving_cast``, ``transformers``) lazily inside ``run()`` so the FastAPI app
boots without the sim stack. Under concurrent job execution that laziness races:
CPython does not fully guard ``from X import Y`` when module ``X`` is in
``sys.modules`` but its ``__init__`` is still executing in another thread — the
second thread gets a half-initialized module and raises
``cannot import name 'PretrainedConfig'``.

Fix: the first job to run imports the whole sim stack under a lock, in ONE
thread. Once warm, every submodule the adapters lazily import is fully
initialized in ``sys.modules``, so concurrent ``run()`` calls never race.
"""

from __future__ import annotations

import threading

_warmed = False
_lock = threading.Lock()


def ensure_sim_stack_warmed() -> None:
    """Import the sim stack once (idempotent, lock-guarded). Safe to call from
    every job's worker thread; only the first caller pays the import cost.
    """
    global _warmed
    if _warmed:
        return
    with _lock:
        if _warmed:  # another thread warmed it while we waited
            return
        # Exact submodules the adapters lazily import (see runners/*.py).
        import torch._dynamo  # noqa: F401  (text compile path)
        import tensor_cast.core.model_runner  # noqa: F401
        import tensor_cast.core.user_config  # noqa: F401
        import tensor_cast.core.quantization.datatypes  # noqa: F401
        import tensor_cast.device_profiles  # noqa: F401  (registers builtins)
        import serving_cast.parallel_runner  # noqa: F401
        import transformers  # noqa: F401  (PretrainedConfig etc.)

        _warmed = True
