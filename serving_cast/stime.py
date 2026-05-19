# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
A module for managing and synchronizing logical time in a multi-threaded environment
"""

import logging
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable, Optional

import salabim as sim

# Singleton salabim Environment (call sites use _get_sim_env() so pylint sees sim.Environment).
_salabim_env: Optional[sim.Environment] = None


def _get_sim_env() -> sim.Environment:
    global _salabim_env
    if _salabim_env is None:
        _salabim_env = sim.Environment()
    return _salabim_env


# 1. Singleton simulation environment (public API for tests / external use)
class SimulationEnv:
    @classmethod
    def clear(cls) -> None:
        global _salabim_env
        _salabim_env = None

    def __new__(cls) -> sim.Environment:
        return _get_sim_env()


def init_simulation():
    """
    initialize simulation environment
    """
    SimulationEnv.clear()
    _get_sim_env()


def start_simulation():
    """
    Start simulation
    """
    _get_sim_env().run()


def stop_simulation():
    """
    Stop simulation
    """
    _get_sim_env().main().activate()


def current_task_name():
    return _get_sim_env().current_component().name()


def current_task():
    return _get_sim_env().current_component()


# 2. time related functions/classes
def now() -> float:
    """
    Returns the current logical timestamp.
    """
    return _get_sim_env().now()


def elapse(ts: float):
    """
    Explicitly advance the logical time of the current Task.

    Args:
        ts (float): The logical duration (in seconds) to set.
    """
    _get_sim_env().current_component().hold(ts)


class DurationDecorator:
    """
    A decorator to specify a duration to elapse after a function call.

    When the decorated function returns, the calling Task's logical time
    is advanced by the specified duration.

    Usage:
        @stime.DurationDecorator(5.0)
        def some_long_running_function():
            ...

    Args:
        ts (float): The logical duration (in seconds) to set.
    """

    def __init__(self, ts: float):
        if ts < 0.0:
            raise ValueError("Cannot set negative time")
        self._duration = ts

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            elapse(self._duration)
            return result

        return wrapper


class Duration:
    """
    A context manager to add a fixed duration to the execution time of a block of code.

    Upon exiting the 'with' block, the current Task's timestamp is
    advanced by the specified duration.

    Usage:
        with stime.Duration(2.5):
            # This block of code logically takes 2.5 seconds.
            ...

    Args:
        ts (float): The logical duration (in seconds) to set.
    """

    def __init__(self, ts: float):
        if ts < 0.0:
            raise ValueError("Cannot set negative time")
        self._duration = ts

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapse(self._duration)


# 3. Task
class Task(sim.Component):
    @abstractmethod
    def process(self):
        raise NotImplementedError

    def wait(self):
        """
        sleep current task
        """
        self.passivate()

    def notify(self):
        """
        awake current task
        """
        if self.ispassive():
            self.activate()


class CallableTask(Task):
    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def process(self):
        self.func(*self.args, **self.kwargs)


# 4. logging
def get_logger(logger_name: str):
    class SimulationTimeFilter(logging.Filter):
        def __init__(self):
            super().__init__("sim_time")

        def filter(self, record):
            try:
                record.sim_time = _get_sim_env().now()
            except Exception:
                record.sim_time = 0.0
            try:
                record.task_name = _get_sim_env().current_component().name()
            except Exception:
                record.task_name = ""
            return True  # always return True to ensure the record is processed

    customed_logger = logging.getLogger(logger_name)
    if getattr(customed_logger, "_serving_cast_sim_handler_installed", False):
        return customed_logger

    handler = logging.StreamHandler()
    sim_filter = SimulationTimeFilter()
    handler.addFilter(sim_filter)
    formatter = logging.Formatter(
        "[%(sim_time)8.2f][T%(task_name)s] %(levelname)-8s %(filename)s:%(lineno)d: %(message)s"
    )
    handler.setFormatter(formatter)
    customed_logger.addHandler(handler)
    customed_logger.propagate = False
    customed_logger._serving_cast_sim_handler_installed = True
    return customed_logger
