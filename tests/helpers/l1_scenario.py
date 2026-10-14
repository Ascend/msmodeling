"""One build and one forward per model-level (L1) scenario.

WHY THIS EXISTS:
    Model-level tests used to construct ``ModelRunner`` once per assertion group,
    so the same model path was built and run several times only to assert numbers,
    then structure, then a capability. ``ModelRunner.__init__`` calls
    ``build_model`` directly and does not consult
    ``tests.helpers.model_cache``, so those repeats were real builds.

    This module runs a scenario once and hands the single result to every
    assertion group.

SIGNATURE POLICY:
    ``tests.helpers.model_cache.user_config_build_cache_key`` lists build fields
    explicitly and is incomplete: ``context_length``, ``acceptance_length``,
    ``dynamic_shapes`` and ``dspark_markov_rank`` all reach the build path but are
    absent from it, so two different graphs can collide on one key.

    ``L1BuildSignature`` inverts the default: every ``UserInputConfig`` field is a
    build field unless it is named in ``RUN_FIELDS`` or ``OBSERVABILITY_FIELDS``.
    A new config field therefore joins the build signature automatically. That
    direction is safe: an unnecessary field splits the cache and costs a build,
    while a missing field silently shares a wrong graph.

    Demoting a field to ``RUN_FIELDS`` is a claim that it cannot change the built
    graph. ``tests/helpers/tests/test_l1_scenario.py`` pins that claim
    structurally: it compares the module-name/type inventory and ``weight_size``
    of a one-layer model. That check is not semantic. It does not compare
    operator sequences, bindings, constants, or runtime events, so a field such
    as ``decode``, ``prefix_cache_hit_rate``, or ``mtp_acceptance_rate`` can
    still change execution without failing it. Image and profiling fields are
    covered by the dataclass inventory only, since they need a vision model or
    a profiling database.

COMPILATION:
    ``apply_compilation_config`` mutates process-global config and resets absent
    options, so it is part of the build signature rather than of the config.
    Each build and forward re-applies its signature's config. This lets one
    executor cache scenarios with different compilation configs and repairs
    drift caused by tests that mutate the same process-global config directly.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Iterable, Optional

from torch import nn

from tensor_cast.core.compilation_config import apply_compilation_config
from tensor_cast.core.input_generator import generate_inputs_varlen
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.runtime import Runtime

# Workload and runtime fields. Each name here is a claim that the field cannot
# change the built graph, verified by test_l1_scenario.py.
RUN_FIELDS = frozenset(
    {
        "num_queries",
        "query_len",
        "decode",
        "prefix_cache_hit_rate",
        "block_size",
        "reserved_memory_gb",
        "mtp_acceptance_rate",
        "image_batch_size",
        "image_height",
        "image_width",
        "performance_model",
        "profiling_database",
        "disable_profiling_interpolation",
    }
)

# Reporting side channels: they do not change the graph or the numeric metrics,
# but they do change emitted artifacts (chrome traces, shape dumps, log files).
# They stay out of the build signature and belong in the run signature, so two
# observability settings cannot share one L1RunArtifact.
OBSERVABILITY_FIELDS = frozenset(
    {
        "chrome_trace",
        "graph_log_url",
        "log_level",
        "dump_input_shapes",
        "dump_op_bound_results",
    }
)


# Run fields that ModelRunner turns into derived state at construction time, so a
# reused runner has to rebuild its performance models when they change.
_PERF_MODEL_FIELDS = frozenset({"performance_model", "profiling_database", "disable_profiling_interpolation"})


def _config_field_names() -> tuple[str, ...]:
    return tuple(config_field.name for config_field in fields(UserInputConfig))


def build_field_names() -> tuple[str, ...]:
    """Config fields that participate in the build signature."""
    return tuple(name for name in _config_field_names() if name not in RUN_FIELDS | OBSERVABILITY_FIELDS)


def _normalize(value: object) -> Any:
    """Reduce a config value to something hashable and comparable."""
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple((f.name, _normalize(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
        return tuple(sorted((key, _normalize(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = tuple(_normalize(item) for item in value)
        return tuple(sorted(items, key=repr)) if isinstance(value, (set, frozenset)) else items
    return value


def _extract(user_config: UserInputConfig, names: Iterable[str]) -> tuple[tuple[str, Any], ...]:
    return tuple((name, _normalize(getattr(user_config, name))) for name in names)


def materialize_requests(requests: object) -> Optional[list[Any]]:
    """Freeze ``requests`` once so a generator is not consumed twice.

    ``L1RunSignature`` and ``run_inference`` both iterate the value. A one-shot
    iterator would be exhausted by the signature and then handed empty to the
    forward pass.
    """
    if requests is None:
        return None
    return list(requests)


def _generate_inputs_identity(generate_inputs_func: Callable) -> str:
    """Identity of a named module-level input generator.

    Every lambda in a module has ``__qualname__ == "<lambda>"``, so two
    different lambdas would share one run signature and silently reuse the
    wrong artifact. ``functools.partial`` has no ``__qualname__``. Both are
    rejected. Pass a module-level function.
    """
    if isinstance(generate_inputs_func, types.FunctionType) and generate_inputs_func.__name__ == "<lambda>":
        raise TypeError("generate_inputs_func must be a named module-level function, not a lambda")
    qualname = getattr(generate_inputs_func, "__qualname__", "")
    module = getattr(generate_inputs_func, "__module__", None)
    if (
        not isinstance(generate_inputs_func, types.FunctionType)
        or module in (None, "__main__")
        or "<locals>" in qualname
    ):
        raise TypeError(
            f"generate_inputs_func must be a named module-level function, not {type(generate_inputs_func).__name__}"
        )
    return f"{module}.{qualname}"


@dataclass(frozen=True)
class L1BuildSignature:
    """Everything that determines the built graph and its compiled form."""

    config_fields: tuple[tuple[str, Any], ...]
    compilation_config: tuple[str, ...]

    @classmethod
    def from_user_config(
        cls,
        user_config: UserInputConfig,
        *,
        compilation_config: Iterable[str] = (),
    ) -> L1BuildSignature:
        return cls(
            config_fields=_extract(user_config, build_field_names()),
            compilation_config=tuple(sorted(compilation_config)),
        )

    def describe(self) -> str:
        values = dict(self.config_fields)
        return f"{values.get('model_id')}@{values.get('device')}"


@dataclass(frozen=True)
class L1RunSignature:
    """Everything that determines the forward pass, given a built graph."""

    config_fields: tuple[tuple[str, Any], ...]
    observability_fields: tuple[tuple[str, Any], ...]
    generate_inputs_func: str
    with_sampler: bool
    requests: Optional[tuple[Any, ...]]

    @classmethod
    def from_user_config(
        cls,
        user_config: UserInputConfig,
        *,
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
        requests: object = None,
    ) -> L1RunSignature:
        materialized = materialize_requests(requests)
        return cls(
            config_fields=_extract(user_config, sorted(RUN_FIELDS)),
            observability_fields=_extract(user_config, sorted(OBSERVABILITY_FIELDS)),
            generate_inputs_func=_generate_inputs_identity(generate_inputs_func),
            with_sampler=with_sampler,
            requests=None if materialized is None else _normalize(materialized),
        )


@dataclass
class L1RunArtifact:
    """The single result of one scenario, shared by every assertion group.

    ``runtime`` is the completed ``Runtime`` handed to ``run_inference``'s
    observer. It is exposed so structural assertions and diagnostics capture can
    read operator evidence from the same forward pass that produced ``metrics``,
    instead of running their own.
    """

    build_signature: L1BuildSignature
    run_signature: L1RunSignature
    metrics: ModelRunnerMetrics
    runtime: Runtime
    model: nn.Module

    @property
    def runtime_event_list(self) -> list:
        return list(self.metrics.runtime_event_list or [])

    @property
    def table_result(self) -> object:
        return self.metrics.table_result

    def operator_names(self) -> tuple[str, ...]:
        """Operator names in invocation order, from this run's own events."""
        if self.runtime is None:
            raise RuntimeError("observer did not run; runtime events are unavailable")
        names = []
        for event in self.runtime.event_list:
            invocation = getattr(event, "op_invoke_info", None)
            func = getattr(invocation, "func", None)
            if func is not None:
                names.append(str(func))
        return tuple(names)


@dataclass
class L1ExecutionCounters:
    """Evidence that a scenario built once and ran once.

    ``build_miss`` counts real ``ModelRunner`` constructions, which is where
    ``build_model`` is called, so it cannot drift from the actual build count.
    """

    build_miss: int = 0
    build_hit: int = 0
    forward_count: int = 0
    run_hit: int = 0
    # Full signature, not describe(). describe() is only model_id@device, so
    # compile / quant / parallel variants of one model would share a bucket.
    forwards_by_build: dict[L1BuildSignature, int] = field(default_factory=dict)

    def record_build(self, *, hit: bool, signature: L1BuildSignature) -> None:
        if hit:
            self.build_hit += 1
        else:
            self.build_miss += 1
            self.forwards_by_build.setdefault(signature, 0)

    def record_forward(self, signature: L1BuildSignature) -> None:
        self.forward_count += 1
        self.forwards_by_build[signature] = self.forwards_by_build.get(signature, 0) + 1


class L1ScenarioExecutor:
    """Build each distinct graph once and run each distinct workload once.

    Scope is one executor instance. Tests get a session-scoped instance from the
    ``l1_executor`` fixture so reuse spans a whole worker session. Its caches are
    unbounded, so model-dense suites should construct a module- or function-local
    instance and call ``reset()`` when that shorter reuse window ends.
    """

    def __init__(self) -> None:
        self._runners: dict[L1BuildSignature, ModelRunner] = {}
        self._runner_baselines: dict[L1BuildSignature, dict[str, Any]] = {}
        self._artifacts: dict[tuple[L1BuildSignature, L1RunSignature], L1RunArtifact] = {}
        self.counters = L1ExecutionCounters()

    def runner(
        self,
        user_config: UserInputConfig,
        *,
        compilation_config: Iterable[str] = (),
    ) -> ModelRunner:
        """Return the shared ``ModelRunner`` for this build signature.

        This is not a forward entry point. Counted forwards go through ``run()``
        only; ``L1ExecutionCounters`` does not see ``run_inference`` calls made
        on the returned runner. A check that needs a second forward must use
        its own executor, so it does not perturb the counts of the executor
        that produced the artifact under test.
        """
        signature = L1BuildSignature.from_user_config(user_config, compilation_config=compilation_config)
        cached = self._runners.get(signature)
        if cached is not None:
            self.counters.record_build(hit=True, signature=signature)
            self._rebind_runner(signature, cached, user_config)
            return cached

        self._apply_compilation(signature.compilation_config)
        runner = ModelRunner(user_config)
        self._runners[signature] = runner
        self._runner_baselines[signature] = self._snapshot_runner(runner)
        self.counters.record_build(hit=False, signature=signature)
        return runner

    def run(
        self,
        user_config: UserInputConfig,
        *,
        compilation_config: Iterable[str] = (),
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
        requests: object = None,
    ) -> L1RunArtifact:
        """Run the scenario at most once and return its shared artifact."""
        materialized_requests = materialize_requests(requests)
        build_signature = L1BuildSignature.from_user_config(user_config, compilation_config=compilation_config)
        run_signature = L1RunSignature.from_user_config(
            user_config,
            generate_inputs_func=generate_inputs_func,
            with_sampler=with_sampler,
            requests=materialized_requests,
        )
        cached = self._artifacts.get((build_signature, run_signature))
        if cached is not None:
            self.counters.run_hit += 1
            return cached

        runner = self.runner(user_config, compilation_config=compilation_config)
        # Compilation options are process-global and other tests can mutate them
        # without going through this executor. Re-apply immediately before every
        # forward, including forwards on cached runners.
        self._apply_compilation(build_signature.compilation_config)
        # run_inference invokes the observer after Runtime.__exit__, so event_list
        # is already populated and the artifact needs no second forward.
        captured: dict[str, object] = {}
        metrics = runner.run_inference(
            requests=materialized_requests,
            generate_inputs_func=generate_inputs_func,
            with_sampler=with_sampler,
            runtime_observer=lambda runtime: captured.__setitem__("runtime", runtime),
        )
        runtime = captured.get("runtime")
        if not isinstance(runtime, Runtime):
            raise RuntimeError("observer did not run; runtime events are unavailable")
        self.counters.record_forward(build_signature)
        artifact = L1RunArtifact(
            build_signature=build_signature,
            run_signature=run_signature,
            metrics=metrics,
            runtime=runtime,
            model=runner.model,
        )
        self._artifacts[(build_signature, run_signature)] = artifact
        return artifact

    def reset(self) -> None:
        """Drop every cached runner and artifact, and reset global compile state."""
        self._runners.clear()
        self._runner_baselines.clear()
        self._artifacts.clear()
        apply_compilation_config(())
        self.counters = L1ExecutionCounters()

    @staticmethod
    def _apply_compilation(compilation_config: tuple[str, ...]) -> None:
        """Align process-global compilation options for the next operation."""
        apply_compilation_config(compilation_config)

    @staticmethod
    def _snapshot_runner(runner: ModelRunner) -> dict[str, Any]:
        # run_inference mutates model_weight_size_gb in place for a vision model
        # that receives no image, so a reused runner must be restored or the next
        # run reports a smaller model.
        return {"model_weight_size_gb": runner.model_weight_size_gb}

    def _rebind_runner(
        self,
        signature: L1BuildSignature,
        runner: ModelRunner,
        user_config: UserInputConfig,
    ) -> None:
        """Point a reused runner at the current workload and clear run residue.

        ``run_inference`` reads workload and reporting fields off
        ``runner.user_input``, so a runner cached by build signature would
        otherwise keep serving the first scenario's ``num_queries``,
        ``query_len``, ``block_size`` and trace settings. The incoming config is
        equal on every build field by construction, so it can replace the stored
        one wholesale; derived state computed in ``__init__`` from run fields is
        then recomputed.
        """
        for name, value in self._runner_baselines[signature].items():
            setattr(runner, name, value)

        previous = runner.user_input
        runner.user_input = user_config
        if user_config.num_queries != 0:
            runner.request_info_default = [user_config.get_request_info()]
        else:
            runner.request_info_default = None
        if _extract(previous, sorted(_PERF_MODEL_FIELDS)) != _extract(user_config, sorted(_PERF_MODEL_FIELDS)):
            runner.perf_models = ModelRunner.create_performance_models(user_config, runner.device_profile)
