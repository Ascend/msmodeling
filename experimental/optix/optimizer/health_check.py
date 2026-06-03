from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Tuple, List
import subprocess

from ..config.config import get_settings, ErrorSeverity, ErrorType

LOG_ERROR_MESSAGE = "Detected {error_type} in logs"
BENCHMARK_LOG_ERROR_MESSAGE = "Detected {error_type} in benchmark logs"


class FatalError(subprocess.SubprocessError):
    """Fatal error, no retry (OOM, device failure, etc.)"""

    pass


class RetryableError(subprocess.SubprocessError):
    """Retryable error (network jitter, IO error, etc.)"""

    pass


class ServiceHookPoint(Enum):
    """Service framework hook point"""

    STARTUP_POLLING = "startup_polling"
    RUNTIME_MONITOR = "runtime_monitor"


class BenchmarkHookPoint(Enum):
    """Benchmark framework hook point"""

    RUNTIME_MONITOR = "runtime_monitor"


@dataclass
class ErrorContext:
    """Error context information"""

    error_type: Any
    severity: Any
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckContext:
    """Health check context"""

    simulator: Any
    benchmark: Any
    scheduler: Any
    current_time: float
    elapsed_time: float
    startup: bool = False


@dataclass
class HealthCheckResult:
    """Health check result"""

    is_healthy: bool
    error_context: Optional[ErrorContext] = None


def _check_log_patterns(
    log_content: str,
    patterns_dict: Dict[ErrorType, List[str]],
    severity: ErrorSeverity,
    error_message_format: str,
    log_snippet_length: int,
) -> Optional[HealthCheckResult]:
    """Check error patterns in logs (common function)

    Args:
        log_content: Log content
        patterns_dict: Error pattern dictionary {ErrorType: [pattern1, pattern2, ...]}
        severity: Error severity level
        error_message_format: Error message format string
        log_snippet_length: Log snippet length

    Returns:
        Returns HealthCheckResult(is_healthy=False) if an error is detected, otherwise returns None
    """
    log_lower = log_content.lower()
    for error_type, patterns in patterns_dict.items():
        for pattern in patterns:
            if pattern.lower() in log_lower:
                return HealthCheckResult(
                    is_healthy=False,
                    error_context=ErrorContext(
                        error_type=error_type,
                        severity=severity,
                        message=error_message_format.format(error_type=error_type.value),
                        details={"log_snippet": log_content[-log_snippet_length:]},
                    ),
                )
    return None


class HealthCheckHook:
    """Health check hook base class"""

    def __init__(self):
        self._hooks: Dict[Enum, List[Tuple[int, Callable, str]]] = {}

    def register(
        self, hook_point: Enum, func: Optional[Callable] = None, *, priority: int = 0, name: Optional[str] = None
    ):
        """Register hook function (implemented in base class, subclasses directly inherit)"""

        def decorator(f):
            hook_name = name or f.__name__
            if hook_point not in self._hooks:
                self._hooks[hook_point] = []
            self._hooks[hook_point].append((priority, f, hook_name))
            return f

        return decorator(func) if func else decorator

    def run(self, hook_point: Enum, context: HealthCheckContext) -> HealthCheckResult:
        """Execute all checks for the specified hook point"""
        if hook_point not in self._hooks:
            return HealthCheckResult(is_healthy=True)
        hooks = sorted(self._hooks[hook_point], key=lambda x: x[0])
        for priority, hook_func, hook_name in hooks:
            try:
                result = hook_func(context)
                if isinstance(result, HealthCheckResult):
                    if not result.is_healthy:
                        return result
            except Exception as e:
                return HealthCheckResult(
                    is_healthy=False,
                    error_context=ErrorContext(
                        error_type=ErrorType.UNKNOWN,
                        severity=ErrorSeverity.FATAL,
                        message=f"Hook {hook_name} raised unexpected exception: {type(e).__name__}: {str(e)}",
                    ),
                )

        return HealthCheckResult(is_healthy=True)


class ServiceHealthCheckHook(HealthCheckHook):
    """Service framework health check hook (only inherits, no need to re-implement register)"""

    pass


class BenchmarkHealthCheckHook(HealthCheckHook):
    """Benchmark framework health check hook (only inherits, no need to re-implement register)"""

    pass


class ServiceHealthChecks:
    """Predefined health checks for service framework"""

    @staticmethod
    def check_log_errors(context: HealthCheckContext) -> HealthCheckResult:
        """Check error messages in logs"""
        if not hasattr(context.simulator, 'get_last_log'):
            return HealthCheckResult(is_healthy=True)
        settings = get_settings()
        config = settings.health_check.service_errors
        log_content = context.simulator.get_last_log(number=settings.health_check.log_snippet_length)
        # Check fatal errors
        result = _check_log_patterns(
            log_content=log_content,
            patterns_dict=config.fatal_patterns,
            severity=ErrorSeverity.FATAL,
            error_message_format=LOG_ERROR_MESSAGE,
            log_snippet_length=settings.health_check.log_snippet_length,
        )
        if result:
            return result
        # Check retryable errors
        result = _check_log_patterns(
            log_content=log_content,
            patterns_dict=config.retryable_patterns,
            severity=ErrorSeverity.RETRYABLE,
            error_message_format=LOG_ERROR_MESSAGE,
            log_snippet_length=settings.health_check.log_snippet_length,
        )
        if result:
            return result
        return HealthCheckResult(is_healthy=True)


class BenchmarkHealthChecks:
    """Predefined health checks for benchmark framework"""

    @staticmethod
    def check_log_errors(context: HealthCheckContext) -> HealthCheckResult:
        """Check error messages in benchmark logs"""
        if not hasattr(context.benchmark, 'get_last_log'):
            return HealthCheckResult(is_healthy=True)
        settings = get_settings()
        config = settings.health_check.benchmark_errors
        log_content = context.benchmark.get_last_log(number=settings.health_check.log_snippet_length)
        # Check fatal errors
        result = _check_log_patterns(
            log_content=log_content,
            patterns_dict=config.fatal_patterns,
            severity=ErrorSeverity.FATAL,
            error_message_format=BENCHMARK_LOG_ERROR_MESSAGE,
            log_snippet_length=settings.health_check.log_snippet_length,
        )
        if result:
            return result
        # Check retryable errors
        result = _check_log_patterns(
            log_content=log_content,
            patterns_dict=config.retryable_patterns,
            severity=ErrorSeverity.RETRYABLE,
            error_message_format=BENCHMARK_LOG_ERROR_MESSAGE,
            log_snippet_length=settings.health_check.log_snippet_length,
        )
        if result:
            return result
        return HealthCheckResult(is_healthy=True)


service_health_checks_hooks = [
    (ServiceHookPoint.STARTUP_POLLING, ServiceHealthChecks.check_log_errors, 10),
    (ServiceHookPoint.RUNTIME_MONITOR, ServiceHealthChecks.check_log_errors, 10),
]

benchmark_health_checks_hooks = [(BenchmarkHookPoint.RUNTIME_MONITOR, BenchmarkHealthChecks.check_log_errors, 10)]
