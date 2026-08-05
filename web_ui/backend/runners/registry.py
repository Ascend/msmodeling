"""Runner factory/registry.

Maps the seeded ``modules.runner_class`` strings to adapter instances:

    ModelRunner           -> TextGenerateRunnerAdapter
    VideoGenerateRunner   -> VideoGenerateRunnerAdapter
    ParallelRunner        -> ThroughputOptimizerAdapter

Adapters are instantiated lazily (their modules import nothing heavy at top
level — every torch/tensor_cast/serving_cast import is inside ``run``), so this
registry imports cleanly without the simulation stack. ``application/ports``
defines ``RunnerPort``; the adapters satisfy it structurally.
"""

from __future__ import annotations

from typing import Callable

from services.repositories import RunnerPort
from runners.text_generate import TextGenerateRunnerAdapter
from runners.throughput_optimizer import ThroughputOptimizerAdapter
from runners.video_generate import VideoGenerateRunnerAdapter


# runner_class (from modules table) -> adapter factory
_REGISTRY: dict[str, type] = {
    "ModelRunner": TextGenerateRunnerAdapter,
    "VideoGenerateRunner": VideoGenerateRunnerAdapter,
    "ParallelRunner": ThroughputOptimizerAdapter,
}

# module_id -> runner_class (so a module id resolves directly too)
_MODULE_TO_RUNNER_CLASS: dict[str, str] = {
    "text_generate": "ModelRunner",
    "video_generate": "VideoGenerateRunner",
    "throughput_optimizer": "ParallelRunner",
}


class UnknownRunnerError(KeyError):
    """Raised when a runner_class string has no registered adapter."""


def get_runner_class(name: str) -> type:
    """Return the adapter class registered for ``name`` (a ``runner_class``).

    Raises :class:`UnknownRunnerError` if no adapter is registered for ``name``.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise UnknownRunnerError(name)
    return cls


def runner_class_for_module(module_id: str) -> str:
    """Return the ``runner_class`` string for ``module_id`` (empty if unknown)."""
    return _MODULE_TO_RUNNER_CLASS.get(module_id, "")


def create_runner(runner_class_or_module: str) -> RunnerPort:
    """Instantiate the adapter for a ``runner_class`` or ``module_id``."""
    if runner_class_or_module in _REGISTRY:
        return _REGISTRY[runner_class_or_module]()  # type: ignore[return-value]
    cls_name = _MODULE_TO_RUNNER_CLASS.get(runner_class_or_module)
    if cls_name and cls_name in _REGISTRY:
        return _REGISTRY[cls_name]()  # type: ignore[return-value]
    raise UnknownRunnerError(runner_class_or_module)


def register_adapter(key: str, factory: Callable[[], RunnerPort]) -> None:
    """Register/override the adapter factory for ``key`` (a runner_class OR a
    module_id).

    ``create_runner`` short-circuits when ``key`` is already in ``_REGISTRY``, so
    registering under a module_id makes ``create_runner(module_id)`` invoke
    ``factory()``. Primarily a TEST SEAM for injecting a fake runner so the
    ``run_job`` flow can be exercised without the torch/tensor_cast simulation
    stack; production registration passes the adapter class itself.
    """
    _REGISTRY[key] = factory  # type: ignore[assignment]
