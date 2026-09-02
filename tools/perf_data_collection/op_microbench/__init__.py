"""Runtime support for the standalone operator microbenchmark CLI."""

from .adapters import (
    AdapterSpec,
    AdapterValidationError,
    get_adapter,
)

__all__ = [
    "AdapterSpec",
    "AdapterValidationError",
    "get_adapter",
]
