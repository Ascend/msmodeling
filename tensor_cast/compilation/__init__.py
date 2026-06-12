from . import patterns  # noqa: F401
from .compile_backend import CompilerBackend

_backend_by_device = {}


def get_backend(*, device_name=None):
    """
    Get the compilation backend for 'torch.compile'.

    Returns:
        Callable: The compilation backend function.
    """
    backend = _backend_by_device.get(device_name)
    if backend is None:
        backend = CompilerBackend(device_name=device_name)
        _backend_by_device[device_name] = backend
    return backend
