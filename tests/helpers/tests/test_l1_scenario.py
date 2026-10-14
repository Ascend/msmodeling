"""Contract tests for the L1 scenario signatures and executor bookkeeping.

These run without building a model. The model-level check that a scenario
builds once and runs once is deferred to the first migration PR.
"""

from __future__ import annotations

import dataclasses
import functools

import pytest

import tensor_cast.ops  # noqa: F401 — register custom ops before building
from tensor_cast import config as tensor_cast_config
from tensor_cast.core.input_generator import generate_inputs, generate_inputs_varlen
from tensor_cast.core.model_builder import build_model
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tests.helpers.l1_scenario import (
    OBSERVABILITY_FIELDS,
    RUN_FIELDS,
    L1BuildSignature,
    L1ExecutionCounters,
    L1RunArtifact,
    L1RunSignature,
    L1ScenarioExecutor,
    build_field_names,
    materialize_requests,
)
from tests.helpers import l1_scenario as l1_scenario_module
from tests.helpers.model_cache import user_config_build_cache_key

# Fields the legacy cache key omits even though the build path reads them.
LEGACY_KEY_BLIND_SPOTS = ("context_length", "dynamic_shapes", "acceptance_length", "dspark_markov_rank")


def make_config(**overrides) -> UserInputConfig:
    base = {
        "model_id": "Qwen/Qwen3-32B",
        "device": "TEST_DEVICE",
        "num_queries": 1,
        "query_len": 32,
        "context_length": 32,
    }
    base.update(overrides)
    return UserInputConfig(**base)


def config_field_names() -> set[str]:
    return {config_field.name for config_field in dataclasses.fields(UserInputConfig)}


def test_declared_run_and_observability_fields_exist():
    assert RUN_FIELDS <= config_field_names()
    assert OBSERVABILITY_FIELDS <= config_field_names()
    assert not RUN_FIELDS & OBSERVABILITY_FIELDS


def test_build_signature_covers_every_field_not_explicitly_demoted():
    """A new UserInputConfig field must join the build signature by default.

    This is the guard that keeps the signature from drifting incomplete the way
    user_config_build_cache_key did.
    """
    assert set(build_field_names()) == config_field_names() - RUN_FIELDS - OBSERVABILITY_FIELDS


def test_build_signature_is_strictly_more_complete_than_legacy_cache_key():
    legacy_field_count = len(user_config_build_cache_key(make_config()))
    build_fields = set(build_field_names())

    for name in LEGACY_KEY_BLIND_SPOTS:
        assert name in build_fields, f"{name} reaches the build path and must be in the signature"
    assert len(build_fields) > legacy_field_count


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_id", "Qwen/Qwen3-8B"),
        ("device", "ATLAS_800_A2_376T_64G"),
        ("tp_size", 8),
        ("do_compile", True),
        ("context_length", 4096),
        ("dynamic_shapes", True),
        ("acceptance_length", 2.0),
        ("dspark_markov_rank", 512),
        ("quantize_linear_action", QuantizeLinearAction.DISABLED),
    ],
)
def test_build_field_change_changes_build_signature(field_name, value):
    baseline = L1BuildSignature.from_user_config(make_config())
    changed = L1BuildSignature.from_user_config(make_config(**{field_name: value}))
    assert baseline != changed


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("num_queries", 16),
        ("query_len", 1),
        ("decode", True),
        ("block_size", 64),
        ("reserved_memory_gb", 4.0),
        ("chrome_trace", "trace.json"),
        ("dump_input_shapes", True),
    ],
)
def test_run_or_observability_field_change_keeps_build_signature(field_name, value):
    baseline = L1BuildSignature.from_user_config(make_config())
    changed = L1BuildSignature.from_user_config(make_config(**{field_name: value}))
    assert baseline == changed


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chrome_trace", "trace.json"),
        ("graph_log_url", "http://example.invalid/graph"),
        ("log_level", "DEBUG"),
        ("dump_input_shapes", True),
        ("dump_op_bound_results", True),
    ],
)
def test_observability_field_change_changes_run_signature(field_name, value):
    """Emitted artifacts are part of the run identity, not a free cache hit."""
    baseline = L1RunSignature.from_user_config(make_config())
    changed = L1RunSignature.from_user_config(make_config(**{field_name: value}))
    assert baseline != changed
    assert set(dict(changed.observability_fields)) == OBSERVABILITY_FIELDS


# Demoting a field to RUN_FIELDS claims it cannot change the built graph. The
# signature tests above only check that the claim is applied consistently. The
# comparison below is structural, not semantic: module name/type inventory and
# weight_size. It does not compare operator sequences, bindings, constants, or
# runtime events. decode, prefix_cache_hit_rate, and mtp_acceptance_rate can
# change execution without changing this fingerprint. One decoder layer is
# enough to expose a structural difference.
RUN_FIELD_ALTERNATIVES = [
    ("num_queries", 8),
    ("query_len", 1),
    ("decode", True),
    ("block_size", 64),
    ("prefix_cache_hit_rate", 0.5),
    ("reserved_memory_gb", 4.0),
    ("mtp_acceptance_rate", [0.5, 0.5]),
    ("performance_model", ["analytic"]),
]


def graph_fingerprint(user_config: UserInputConfig):
    model = build_model(user_config)
    return tuple((name, type(module).__name__) for name, module in model.named_modules()), model.weight_size


@pytest.fixture(scope="module")
def single_layer_baseline():
    return graph_fingerprint(make_config(num_hidden_layers_override=1))


def test_every_demoted_field_is_exercised_by_a_behavioural_case():
    """Keep the behavioural coverage list honest as RUN_FIELDS changes."""
    exercised = {name for name, _ in RUN_FIELD_ALTERNATIVES}
    # Image fields need a vision model, and profiling fields need a database
    # fixture; both are covered structurally only.
    deferred = {"image_batch_size", "image_height", "image_width", "profiling_database"}
    deferred |= {"disable_profiling_interpolation"}
    assert exercised | deferred == RUN_FIELDS


@pytest.mark.parametrize(("field_name", "value"), RUN_FIELD_ALTERNATIVES)
def test_run_field_change_does_not_change_the_built_graph(field_name, value, single_layer_baseline):
    """Structural demotion check only; see the note above RUN_FIELD_ALTERNATIVES."""
    changed = graph_fingerprint(make_config(num_hidden_layers_override=1, **{field_name: value}))
    assert changed == single_layer_baseline, f"{field_name} changes the built graph and is not a run field"


def test_compilation_config_is_part_of_the_build_signature():
    config = make_config(do_compile=True)
    plain = L1BuildSignature.from_user_config(config)
    compiled = L1BuildSignature.from_user_config(config, compilation_config=["enable_fusion"])
    assert plain != compiled


def test_compilation_config_order_does_not_matter():
    config = make_config()
    first = L1BuildSignature.from_user_config(config, compilation_config=["b", "a"])
    second = L1BuildSignature.from_user_config(config, compilation_config=["a", "b"])
    assert first == second


def test_signatures_are_hashable_dictionary_keys():
    config = make_config()
    build = L1BuildSignature.from_user_config(config)
    run = L1RunSignature.from_user_config(config)
    assert {(build, run): "artifact"}[(build, run)] == "artifact"


def test_enum_and_list_values_normalize_to_comparable_signatures():
    """Equal configs must produce equal signatures even with mutable field values."""
    first = L1BuildSignature.from_user_config(make_config(mtp_acceptance_rate=[0.9, 0.6]))
    second = L1BuildSignature.from_user_config(make_config(mtp_acceptance_rate=[0.9, 0.6]))
    assert first == second


def test_run_signature_separates_input_generator_and_sampler():
    config = make_config()
    varlen = L1RunSignature.from_user_config(config, generate_inputs_func=generate_inputs_varlen)
    plain = L1RunSignature.from_user_config(config, generate_inputs_func=generate_inputs)
    sampled = L1RunSignature.from_user_config(config, generate_inputs_func=generate_inputs_varlen, with_sampler=True)

    assert varlen != plain
    assert varlen != sampled


def test_run_signature_separates_explicit_request_lists():
    config = make_config()
    without = L1RunSignature.from_user_config(config)
    with_requests = L1RunSignature.from_user_config(config, requests=[config.get_request_info()])
    assert without != with_requests


def test_materialize_requests_keeps_a_generator_usable_twice():
    config = make_config()
    info = config.get_request_info()
    generator = (item for item in (info,))

    materialized = materialize_requests(generator)

    assert materialized == [info]
    assert list(generator) == []
    signature = L1RunSignature.from_user_config(config, requests=materialized)
    assert signature.requests is not None
    assert materialized == [info]


def test_counters_attribute_forwards_to_their_build():
    counters = L1ExecutionCounters()
    signature = L1BuildSignature.from_user_config(make_config())

    counters.record_build(hit=False, signature=signature)
    counters.record_forward(signature)
    counters.record_build(hit=True, signature=signature)
    counters.record_forward(signature)

    assert counters.build_miss == 1
    assert counters.build_hit == 1
    assert counters.forward_count == 2
    assert counters.forwards_by_build[signature] == 2
    assert signature.describe().endswith("@TEST_DEVICE")
    assert signature.describe() != repr(signature)


def test_forwards_of_one_model_stay_split_by_full_build_signature():
    """describe() collapses compile variants; the counter must not."""
    counters = L1ExecutionCounters()
    plain = L1BuildSignature.from_user_config(make_config())
    compiled = L1BuildSignature.from_user_config(make_config(), compilation_config=["enable_multistream"])

    assert plain.describe() == compiled.describe()
    counters.record_forward(plain)
    counters.record_forward(compiled)

    assert counters.forwards_by_build[plain] == 1
    assert counters.forwards_by_build[compiled] == 1


def test_run_signature_rejects_lambda_generate_inputs_func():
    with pytest.raises(TypeError, match="not a lambda"):
        L1RunSignature.from_user_config(make_config(), generate_inputs_func=lambda *_args, **_kwargs: None)


def test_run_signature_rejects_partial_generate_inputs_func():
    with pytest.raises(TypeError, match="named module-level function"):
        L1RunSignature.from_user_config(
            make_config(),
            generate_inputs_func=functools.partial(generate_inputs_varlen),
        )


def test_operator_names_fails_when_observer_did_not_run():
    artifact = L1RunArtifact(
        build_signature=L1BuildSignature.from_user_config(make_config()),
        run_signature=L1RunSignature.from_user_config(make_config()),
        metrics=object(),
        runtime=None,
        model=object(),
    )
    with pytest.raises(RuntimeError, match="observer did not run"):
        artifact.operator_names()


class _FakeRuntime:
    pass


class _FakeModelRunner:
    def __init__(self, user_input):
        self.user_input = user_input
        self.model_weight_size_gb = 1.0
        self.model = object()
        self.request_info_default = [user_input.get_request_info()]

    def run_inference(self, *, runtime_observer, **_kwargs):
        runtime_observer(_FakeRuntime())
        return object()


def test_one_executor_supports_multiple_compilation_configs(monkeypatch):
    applied = []
    monkeypatch.setattr(l1_scenario_module, "ModelRunner", _FakeModelRunner)
    monkeypatch.setattr(l1_scenario_module, "apply_compilation_config", lambda config: applied.append(config))

    executor = L1ScenarioExecutor()
    plain = executor.runner(make_config(), compilation_config=())
    multistream = executor.runner(make_config(), compilation_config=("enable_multistream",))

    assert plain is not multistream
    assert len(executor._runners) == 2
    assert applied == [(), ("enable_multistream",)]


def test_forward_reapplies_compilation_config_after_external_drift(monkeypatch):
    applied = []
    monkeypatch.setattr(l1_scenario_module, "ModelRunner", _FakeModelRunner)
    monkeypatch.setattr(l1_scenario_module, "Runtime", _FakeRuntime)
    monkeypatch.setattr(l1_scenario_module, "apply_compilation_config", lambda config: applied.append(config))

    executor = L1ScenarioExecutor()
    compilation = ("enable_multistream",)
    executor.run(make_config(query_len=32), compilation_config=compilation)
    l1_scenario_module.apply_compilation_config(("enable_sequence_parallel",))
    executor.run(make_config(query_len=64), compilation_config=compilation)

    assert applied == [
        compilation,
        compilation,
        ("enable_sequence_parallel",),
        compilation,
    ]


def test_reset_clears_caches_and_counters():
    executor = L1ScenarioExecutor()
    signature = L1BuildSignature.from_user_config(make_config())
    executor._runners[signature] = object()
    executor._runner_baselines[signature] = {"model_weight_size_gb": 1.0}
    executor.counters.record_build(hit=False, signature=signature)

    executor.reset()

    assert not executor._runners
    assert not executor._artifacts
    assert executor.counters.build_miss == 0


def test_reset_restores_default_compilation_config():
    executor = L1ScenarioExecutor()
    try:
        executor._apply_compilation(("enable_multistream",))
        assert tensor_cast_config.compilation.multistream.enable is True
    finally:
        executor.reset()

    assert tensor_cast_config.compilation.multistream.enable is False
