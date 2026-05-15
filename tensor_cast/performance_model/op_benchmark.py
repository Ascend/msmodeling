from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.utils._cxx_pytree import tree_map

from ..config import performance_model as perf_config
from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .utils import is_noop_self_copy_op, is_view_op


_op_impl_registry = {}


def get_op_impl(op, device_str):
    return _op_impl_registry.get((op, device_str))


def register_op_impl(op, device_str):
    def decorator(benchmark_functor):
        key = (op, device_str)
        if key in _op_impl_registry:
            raise ValueError(f"Implementation for {op} on {device_str} already registered")
        _op_impl_registry[key] = benchmark_functor
        return benchmark_functor

    return decorator


@register_op_impl(torch.ops.tensor_cast.quantize.default, torch.device("cpu"))
def _(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    output = torch.round(x / scale)
    if offset is not None:
        output = output + offset
    return output.to(dtype=out_dtype)


class OpBenchmarkBase(ABC):
    def __init__(self, device_profile: DeviceProfile):
        self.device_profile = device_profile

    @abstractmethod
    def benchmark(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # Benchmark the given op and return the estimated latency in seconds
        pass


class OpBenchmark(OpBenchmarkBase):
    def __init__(self, device_profile: DeviceProfile):
        super().__init__(device_profile)
        self.runtime_device = self.infer_runtime_device()

    def benchmark(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        if is_view_op(op_invoke_info.func) or is_noop_self_copy_op(op_invoke_info.func, op_invoke_info.args):
            return PerformanceModel.Result(0.0)
        if op_invoke_info.func.namespace == "tensor_cast":
            op_impl = get_op_impl(op_invoke_info.func, self.runtime_device)
            if op_impl is None:
                raise ValueError(f"No implementation registered for {op_invoke_info.func} on {self.runtime_device}")
        else:
            op_impl = op_invoke_info.func
        return self.do_bench(op_impl, op_invoke_info.args, op_invoke_info.kwargs)

    def do_bench(self, op_impl, args, kwargs) -> PerformanceModel.Result:
        # construct real inputs for all the meta tensors on the given device
        real_args, real_kwargs = tree_map(
            lambda t: (torch.empty_like(t, device=self.runtime_device) if isinstance(t, torch.Tensor) else t),
            (args, kwargs),
        )
        # warm up
        for _ in range(perf_config.empirical.warmup_runs):
            op_impl(*real_args, **real_kwargs)

        # benchmark
        import time

        start_time = time.perf_counter()
        for _ in range(perf_config.empirical.benchmark_runs):
            op_impl(*real_args, **real_kwargs)
        end_time = time.perf_counter()
        avg_latency_s = (end_time - start_time) / perf_config.empirical.benchmark_runs

        return PerformanceModel.Result(execution_time_s=avg_latency_s)

    def infer_runtime_device(self) -> torch.device:
        if (device_override := perf_config.empirical.runtime_device_override) is not None:
            return device_override
        if self.device_profile.name == "TEST_DEVICE":
            return torch.device("cpu")
        if self.device_profile.vendor == "HUAWEI":
            return torch.device("npu")
        elif self.device_profile.vendor == "NVIDIA":
            return torch.device("cuda")
        else:
            raise ValueError(f"Unsupported benchmarking ops on vendor: {self.device_profile.vendor}")
