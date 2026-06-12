import logging
from typing import Callable, List, Optional, Union

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo


_op_estimator_table = {}

logger = logging.getLogger(__name__)


def register_op_estimator(op, device_names: Optional[Union[str, List[str]]], override: Optional[bool] = False):
    if not isinstance(device_names, (list, tuple)):
        device_names = [device_names]

    def decorator(estimator):
        for device_name in device_names:
            if device_name not in _op_estimator_table:
                _op_estimator_table[device_name] = {}
            if op in _op_estimator_table[device_name]:
                if override:
                    logger.warning(
                        "Overwriting existing estimator for op %s (device: %s)",
                        op,
                        device_name,
                    )
                else:
                    raise ValueError(f"Op {op} already registered for device {device_name}")
            _op_estimator_table[device_name][op] = estimator
        return estimator

    return decorator


def get_op_estimator(
    op, device_name: Optional[str]
) -> Callable[[OpInvokeInfo, DeviceProfile], PerformanceModel.Result]:
    if device_name not in _op_estimator_table:
        device_name = None
    if op not in _op_estimator_table[device_name]:
        op = None
    return _op_estimator_table[device_name][op]
