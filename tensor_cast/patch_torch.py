import contextlib
import logging
import threading

import torch

from .performance_model.utils import run_once

logger = logging.getLogger(__name__)

_meta_autocast_enabled = threading.local()


def _is_meta_autocast_enabled():
    return getattr(_meta_autocast_enabled, "value", False)


def _set_meta_autocast_enabled(enabled):
    _meta_autocast_enabled.value = enabled


_in_support_autocast_for_meta = False
_DTYPE_TO_TYPE_ORIGINAL_ATTR = "_tensor_cast_original_dtype_to_type"


@contextlib.contextmanager
def support_autocast_for_meta():
    """
    PyTorch doesn't support "meta" device for autocast. This prevents us from
    running a PyTorch model that calls autocast, such as those models in the
    Transformers. We have to patch PyTorch to work it around.
    """
    import torch.fx.experimental._config as config

    meta_nonzero_assume_all_nonzero_orig = config.meta_nonzero_assume_all_nonzero
    get_autocast_dtype_orig = torch.get_autocast_dtype
    _C_is_autocast_available_orig = torch._C._is_autocast_available
    is_autocast_enabled_orig = torch.is_autocast_enabled
    set_autocast_enabled_orig = torch.set_autocast_enabled

    def get_autocast_dtype(device):
        if device == "meta":
            return torch.half
        else:
            return get_autocast_dtype_orig(device)

    def is_autocast_available(device):
        if device == "meta":
            return True
        else:
            return _C_is_autocast_available_orig(device)

    def is_autocast_enabled(device):
        if device == "meta":
            return _is_meta_autocast_enabled()
        else:
            return is_autocast_enabled_orig(device)

    def set_autocast_enabled(device, enabled):
        if device == "meta":
            _set_meta_autocast_enabled(enabled)
        else:
            set_autocast_enabled_orig(device, enabled)

    global _in_support_autocast_for_meta
    if _in_support_autocast_for_meta:
        yield
        return
    _in_support_autocast_for_meta = True
    torch.get_autocast_dtype = get_autocast_dtype
    torch.is_autocast_enabled = is_autocast_enabled
    torch.set_autocast_enabled = set_autocast_enabled
    torch._C._is_autocast_available = is_autocast_available
    config.meta_nonzero_assume_all_nonzero = True
    try:
        yield
    finally:
        torch._C._is_autocast_available = _C_is_autocast_available_orig
        torch.set_autocast_enabled = set_autocast_enabled_orig
        torch.is_autocast_enabled = is_autocast_enabled_orig
        torch.get_autocast_dtype = get_autocast_dtype_orig
        _in_support_autocast_for_meta = False
        config.meta_nonzero_assume_all_nonzero = meta_nonzero_assume_all_nonzero_orig


@contextlib.contextmanager
def specialize_float():
    """
    Patch torch._dynamo.config.specialize_float to True, so that the float dtype
    information can be preserved in the graph. We assume floats are specialized
    in our pattern matching passes like RMSNorm for params like eps.
    """
    old_flag = torch._dynamo.config.specialize_float
    torch._dynamo.config.specialize_float = True
    try:
        yield
    finally:
        torch._dynamo.config.specialize_float = old_flag


@contextlib.contextmanager
def patch_fallback_node_due_to_unsupported_type():
    """
    Patch torch._inductor.pattern_matcher.fallback_node_due_to_unsupported_type to always return False,
    so that the pattern matching passes can be applied without being blocked due to meta tensors.
    """
    import torch._inductor.pattern_matcher as pattern_matcher

    if not hasattr(pattern_matcher, "fallback_node_due_to_unsupported_type"):
        yield
        return

    original_func = pattern_matcher.fallback_node_due_to_unsupported_type

    def always_false(*args, **kwargs):
        return False

    pattern_matcher.fallback_node_due_to_unsupported_type = always_false
    try:
        yield
    finally:
        pattern_matcher.fallback_node_due_to_unsupported_type = original_func


@contextlib.contextmanager
def prepare_freezing():
    """
    Prepare PyTorch Dynamo for graph freezing by enabling the relevant config.
    We need this for the `freeze()` call from inductor to work properly.
    """
    old_flag = torch._dynamo.config.prepare_freezing
    torch._dynamo.config.prepare_freezing = True
    try:
        yield
    finally:
        torch._dynamo.config.prepare_freezing = old_flag


@contextlib.contextmanager
def patch_dtype_abbrs():
    """
    Patch torch.utils._dtype_abbrs in order to support FX graph dump with int4 dtype used
    by MXFP4.
    """
    try:
        from torch.utils._dtype_abbrs import dtype_abbrs
    except ModuleNotFoundError:
        yield
        return

    original_dtype_abbrs = dict(dtype_abbrs)
    dtype_abbrs.update(
        {
            torch.int4: "i4",
        }
    )
    try:
        yield
    finally:
        dtype_abbrs.clear()
        dtype_abbrs.update(original_dtype_abbrs)


@contextlib.contextmanager
def patch_dtype_to_type():
    """This patch tries to fix the FX graph tracing issue when int4 dtype is used.
    For example, the `torch.cat` fails with int4 tensors because the dtype_to_type
    function in torch._prims_common does not support int4 dtype.
    """
    try:
        from torch import _prims_common
    except ModuleNotFoundError:
        yield
        return

    current_dtype_to_type = _prims_common.dtype_to_type
    if hasattr(current_dtype_to_type, _DTYPE_TO_TYPE_ORIGINAL_ATTR):
        yield
        return

    original_dtype_to_type = current_dtype_to_type

    def dtype_to_type_patched(dtype: torch.dtype) -> type:
        if dtype == torch.int4:
            return int
        return original_dtype_to_type(dtype)

    setattr(dtype_to_type_patched, _DTYPE_TO_TYPE_ORIGINAL_ATTR, original_dtype_to_type)
    _prims_common.dtype_to_type = dtype_to_type_patched
    try:
        yield
    finally:
        if _prims_common.dtype_to_type is dtype_to_type_patched:
            _prims_common.dtype_to_type = original_dtype_to_type


@contextlib.contextmanager
def patch_masked_scatter():
    """Patch Tensor.masked_scatter to work with meta device tensors."""
    try:
        original_masked_scatter = torch.Tensor.masked_scatter

        def masked_scatter_meta_safe(self, mask, source):
            if isinstance(self, torch.Tensor) and self.device.type == "meta":
                run_once(
                    "tensor_cast.patch_torch.masked_scatter.meta",
                    logger.warning,
                    "TensorCast: masked_scatter on meta is bypassed (returns empty_like); "
                    "shape/dtype preserved and op time is ~0.",
                )
                return torch.empty_like(self)
            return original_masked_scatter(self, mask, source)

        torch.Tensor.masked_scatter = masked_scatter_meta_safe
    except Exception:
        # Fallback to plain context if patching fails
        yield
        return

    try:
        yield
    finally:
        torch.Tensor.masked_scatter = original_masked_scatter


@contextlib.contextmanager
def patch_torch():
    """
    Apply all patches to PyTorch.
    """
    with (
        support_autocast_for_meta(),
        specialize_float(),
        patch_fallback_node_due_to_unsupported_type(),
        patch_dtype_abbrs(),
        patch_dtype_to_type(),
        prepare_freezing(),
        patch_masked_scatter(),
    ):
        yield
