# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from serving_cast.stime import current_task_name, get_logger, now

logger = get_logger(__name__)

try:
    from ms_service_profiler import Level, Profiler
except ImportError:
    try:
        from ms_service_profiler import Profiler
    except ImportError as e:
        raise ImportError("Please install ms_service_profiler") from e
    Level = getattr(Profiler, "Level", None)
    if Level is None:
        raise ImportError("ms_service_profiler.Profiler has no Level; upgrade ms_service_profiler") from None


def parse_main_func() -> None:
    """Run ``ms_service_profiler.parse`` main (``sys.argv`` is configured by caller)."""
    from ms_service_profiler.parse import main as _parse_main

    _parse_main()


class SimProfiler(Profiler):
    def event(self, event_name):
        self.metric("logical_start_time", now())
        self.metric("logical_end_time", now())
        self.metric("logical_pid", current_task_name())

        return super().event(event_name)

    def span_start(self, span_name):
        self.metric("logical_start_time", now())
        self.metric("logical_pid", current_task_name())

        return super().span_start(span_name)

    def span_end(self):
        self.metric("logical_end_time", now())
        self.metric("logical_pid", current_task_name())

        return super().span_end()
