"""Runner adapters + factory.

Every adapter module imports its heavy deps (torch / tensor_cast /
serving_cast) INSIDE ``run``, so this package imports cleanly without the
simulation stack (the FastAPI app stays bootable).
"""

from .registry import (
    UnknownRunnerError,
    create_runner,
    get_runner_class,
    runner_class_for_module,
)

__all__ = [
    "UnknownRunnerError",
    "create_runner",
    "get_runner_class",
    "runner_class_for_module",
]
