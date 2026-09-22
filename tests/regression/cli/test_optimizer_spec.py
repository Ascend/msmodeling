"""Tests for the optimizer-args workload spec loader."""

from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
import pytest
import torch

from serving_cast.service.utils import build_pp_search_candidates
from tensor_cast.device import DeviceProfile
from tools.perf_data_collection.grid_generator.optimizer_spec import (
    OptimizerSpecError,
    load_optimizer_spec,
)


@pytest.fixture(autouse=True)
def _test_device_profile(monkeypatch):
    profile = SimpleNamespace(comm_grid=SimpleNamespace(grid=torch.empty(12)))
    monkeypatch.setitem(DeviceProfile.all_device_profiles, "TEST_DEVICE", profile)


def _write_spec(tmp_path: Path, content: str, suffix: str = ".yaml") -> Path:
    spec_path = tmp_path / f"workloads{suffix}"
    spec_path.write_text(content, encoding="utf-8")
    return spec_path


def test_loads_multiple_scenarios_and_expands_parallel_combinations(tmp_path: Path) -> None:
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            """
workloads:
  - name: prefill-20k
    model: zai-org/GLM-5.1
    device: TEST_DEVICE
    num_devices: 4
    input_length: 20000
    output_length: 1024
    disagg: true
    ttft_limit: 10000
    max_batched_tokens: 20000
    batch_range: [1, 32]
    tp_sizes: [1, 2, 4]
    ep_sizes: [4]
    moe_dp_sizes: [1]
    quantize_linear_action: W8A8_DYNAMIC
    compile: true
    compilation_config: [enable_sequence_parallel, enable_dispatch_ffn_combine]
    enable_shared_expert_tp: true
    reserved_memory_gb: 10
  - model: org/model
    device: TEST_DEVICE
    num_devices: 1
    input_length: 8
    output_length: 1
""",
        )
    )

    assert len(spec.workload_summaries) == 2
    prefill = spec.workload_summaries[0]
    assert (prefill.name, prefill.model, prefill.num_devices) == ("prefill-20k", "zai-org/GLM-5.1", 4)
    assert prefill.requested_combinations == 3
    assert prefill.expanded_workloads == 3
    assert prefill.skipped_combinations == 0
    # One subprocess per parallel combination, like the internal policy.
    assert all(
        len(scenario.tp_sizes) == len(scenario.ep_sizes) == len(scenario.moe_dp_sizes) == 1
        for scenario in spec.scenarios
    )
    assert {scenario.sweep_name for scenario in spec.scenarios} == {"prefill-20k", "actual_2"}
    baseline = next(scenario for scenario in spec.scenarios if scenario.sweep_name == "actual_2")
    assert baseline.batch_range == (1, 512)
    assert not baseline.disagg


def test_spec_scenario_command_mirrors_optimizer_cli_flags(tmp_path: Path) -> None:
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            """
workloads:
  - name: decode-80k
    model: zai-org/GLM-5.1
    device: TEST_DEVICE
    num_devices: 32
    input_length: 80000
    output_length: 1024
    disagg: true
    tpot_limit: 70
    max_batched_tokens: 320000
    batch_range: [1, 32]
    tp_sizes: [2]
    # EP must equal TP x DP = 32 (issue #456 domain conservation); the former
    # EP=4 was a domain-broken combo now rejected at candidate generation.
    ep_sizes: [32]
    moe_dp_sizes: [1]
    dcp_sizes: [1]
    num_mtp_tokens: [0, 2]
    mtp_acceptance_rates: [0.9, 0.6]
    quantize_linear_action: W8A8_DYNAMIC
    quantize_attention_action: DISABLED
    reserved_memory_gb: 10
    compile: true
    compilation_config: [enable-sequence-parallel, enable_dispatch_ffn_combine]
    enable_shared_expert_tp: true
""",
        )
    )

    command = spec.scenarios[0].command(tmp_path)
    text = " ".join(command)
    assert "--disagg --tpot-limit 70" in text
    assert "--max-batched-tokens 320000" in text
    assert "--mtp-acceptance-rates 0.9 0.6" in text
    assert "--reserved-memory-gb 10" in text
    assert "--enable-shared-expert-tp" in text
    assert "--compile" in command
    # Display-name tokens are canonicalized exactly like the optimizer CLI.
    cc_index = command.index("--compilation-config")
    assert command[cc_index + 1 : cc_index + 3] == [
        "enable_sequence_parallel",
        "enable_dispatch_ffn_combine",
    ]
    assert "--ttft-limit" not in text
    # The profiled database is always managed by generate_shape_grid.
    assert "--performance-model profiling" in text


def test_json_spec_is_supported(tmp_path: Path) -> None:
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            """
{"workloads": [{
    "model": "org/model",
    "device": "TEST_DEVICE",
    "num_devices": 2,
    "input_length": 128,
    "output_length": 4,
    "tp_sizes": [1, 2],
    "quantize_linear_action": "DISABLED"
}]}
""",
            suffix=".json",
        )
    )

    assert len(spec.scenarios) == 2
    assert {scenario.quantize_linear_action for scenario in spec.scenarios} == {"DISABLED"}


_SCENARIO_YAML = """
model: org/model
device: TEST_DEVICE
num_devices: 1
input_length: 8
output_length: 1
"""


def test_digest_is_content_stable(tmp_path: Path) -> None:
    first_path = tmp_path / "a.yaml"
    second_path = tmp_path / "b.yaml"
    first_path.write_text(_SCENARIO_YAML, encoding="utf-8")
    second_path.write_text(_SCENARIO_YAML, encoding="utf-8")
    assert load_optimizer_spec(first_path).digest == load_optimizer_spec(second_path).digest
    changed_path = tmp_path / "c.yaml"
    changed_path.write_text(_SCENARIO_YAML.replace("input_length: 8", "input_length: 16"), encoding="utf-8")
    assert load_optimizer_spec(first_path).digest != load_optimizer_spec(changed_path).digest


def test_unknown_field_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(OptimizerSpecError, match="field 'tp_sz': unknown field"):
        load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML + "\ntp_sz: [1]\n"))


def test_targeted_errors_for_historical_optimizer_flags(tmp_path: Path) -> None:
    for field, fragment in (
        ("dp_sizes: [8]", "no independent --dp-sizes"),
        ("speculative_method: mtp", "speculative"),
        ("num_speculative_tokens: [2]", "speculative"),
        ("performance_model: profiling", "managed by generate_shape_grid"),
        ("profiling_database_path: some/dir", "managed by generate_shape_grid"),
        ("image_height: 224", "Multimodal"),
        ("enable_optimize_prefill_decode_ratio: true", "PD-ratio"),
    ):
        with pytest.raises(OptimizerSpecError, match=fragment):
            load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML + f"\n{field}\n"))


def test_pp_sizes_pass_through_and_expand_per_combination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 8,
    )
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4")
            + "pp_sizes: [1, 2]\ntp_sizes: [1, 2]\nep_sizes: [1]\n",
        )
    )

    # One subprocess per (tp, pp) combination; each expanded workload carries a
    # single pp value and forwards it as --pp-sizes.
    expanded_pp = sorted(scenario.pp_sizes for scenario in spec.scenarios)
    assert expanded_pp == [(1,), (1,), (2,), (2,)]
    for scenario in spec.scenarios:
        command = scenario.command(tmp_path)
        if scenario.pp_sizes:
            pp_index = command.index("--pp-sizes")
            assert command[pp_index + 1] == str(scenario.pp_sizes[0])
        else:
            assert "--pp-sizes" not in command
    workload_ids = {scenario.workload_id for scenario in spec.scenarios}
    assert len(workload_ids) == len(spec.scenarios)
    assert any("pp=1" in workload_id for workload_id in workload_ids)
    assert any("pp=2" in workload_id for workload_id in workload_ids)


def test_pp_sizes_validation_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 8,
    )
    base = _SCENARIO_YAML
    four = _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4")
    cases = [
        (base, "pp_sizes: [10]", "larger than 'num_devices'"),
        (four, "pp_sizes: [2]\nnum_mtp_tokens: [2]", "num_mtp_tokens to include 0"),
        (_SCENARIO_YAML.replace("num_devices: 1", "num_devices: 16"), "pp_sizes: [16]", "num_hidden_layers"),
    ]
    for scenario_text, addition, fragment in cases:
        with pytest.raises(OptimizerSpecError, match=fragment):
            load_optimizer_spec(_write_spec(tmp_path, scenario_text + addition + "\n"))


def test_pp_layer_partitions_validated_and_forwarded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 8,
    )
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4")
            + "pp_sizes: [2]\ntp_sizes: [2]\nep_sizes: [1]\npp_layer_partitions: [[5, 3]]\n",
        )
    )
    command = spec.scenarios[0].command(tmp_path)
    assert command[command.index("--pp-layer-partitions") + 1] == "[[5,3]]"
    assert spec.scenarios[0].pp_layer_partitions == ((5, 3),)

    error_cases = [
        ("pp_sizes: [2]\ntp_sizes: [2]\nep_sizes: [1]\npp_layer_partitions: [[5, 4]]", "expected num_hidden_layers 8"),
        (
            "pp_sizes: [2]\ntp_sizes: [2]\nep_sizes: [1]\npp_layer_partitions: [[4, 4], [6]]",
            "does not match any requested pp_sizes",
        ),
        (
            "pp_sizes: [2, 4]\ntp_sizes: [1]\nep_sizes: [1]\npp_layer_partitions: [[5, 3]]",
            r"no partition of length \[4\]",
        ),
        ("pp_layer_partitions: [[4, 4]]", "requires pp_sizes to be provided"),
    ]
    for addition, fragment in error_cases:
        with pytest.raises(OptimizerSpecError, match=fragment):
            load_optimizer_spec(
                _write_spec(tmp_path, _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4") + addition + "\n")
            )


def test_pp_sizes_empty_list_uses_powers_of_two_range(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 64,
    )
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4") + "pp_sizes: []\ntp_sizes: [1]\nep_sizes: [1]\n",
        )
    )
    assert {scenario.pp_sizes for scenario in spec.scenarios} == {(1,), (2,), (4,)}


def test_pp_only_spec_uses_stage_local_tp_ep_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 8,
    )
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 8") + "pp_sizes: [1, 2, 4]\n",
        )
    )

    assert {(scenario.pp_sizes[0], scenario.tp_sizes[0], scenario.ep_sizes[0]) for scenario in spec.scenarios} == {
        (1, 8, 8),
        (2, 4, 4),
        (4, 2, 2),
    }
    summary = spec.workload_summaries[0]
    assert summary.requested_combinations == 3
    assert summary.expanded_workloads == 3
    assert summary.skipped_combinations == 0


def test_invalid_values_fail_closed(tmp_path: Path) -> None:
    cases = [
        ("device: UNKNOWN_DEVICE", "not a registered DeviceProfile"),
        ("num_devices: 0", "must be >= 1"),
        ("input_length: distribution.yaml", "variable-length distribution"),
        ("input_length: 12.5", "must be an integer"),
        ("output_length: -1", "must be >= 1"),
        ("tp_sizes: [1, 8]", "larger than 'num_devices'"),
        ("batch_range: [8, 1]", "min must be <= max"),
        ("batch_range: [0, 4]", "must be >= 1"),
        ("num_mtp_tokens: [10]", "must be in 0-5"),
        ("quantize_linear_action: INT4", "must be one of"),
        ("quantize_attention_action: INT4", "must be one of"),
        ("word_embedding_tp: diagonal", "must be one of"),
        ("compilation_config: [enable_warp_specialize]", "invalid choice"),
        ("ttft_limit: 0", "must be > 0.0"),
        ("prefix_cache_hit_rate: 1.0", "must be in [0, 1)"),
    ]
    for addition, fragment in cases:
        with pytest.raises(OptimizerSpecError, match=re.escape(fragment)):
            load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML + f"\n{addition}\n"))


def test_mtp_tokens_beyond_acceptance_rates_fail_closed(tmp_path: Path) -> None:
    addition = "num_mtp_tokens: [6]\n"
    with pytest.raises(OptimizerSpecError, match="must be in 0-5 for mtp_acceptance_rates length 4"):
        load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML + addition))
    # Explicit rates make the same scenario legal (6 tokens need 5+ rates).
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML + addition + "mtp_acceptance_rates: [0.9, 0.9, 0.9, 0.9, 0.9]\n",
        )
    )
    assert spec.scenarios[0].mtp_tokens == (6,)


def test_no_legal_parallel_combination_fails_closed(tmp_path: Path) -> None:
    spec_text = _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 3") + "tp_sizes: [2]\nep_sizes: [3]\n"
    with pytest.raises(OptimizerSpecError, match="no valid parallel combination exists under"):
        load_optimizer_spec(_write_spec(tmp_path, spec_text))


def test_illegal_combinations_are_filtered_and_counted(tmp_path: Path) -> None:
    spec_text = _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4") + "tp_sizes: [1, 3]\n"
    spec = load_optimizer_spec(_write_spec(tmp_path, spec_text))
    summary = spec.workload_summaries[0]
    assert summary.requested_combinations == 2
    assert summary.expanded_workloads == 1
    assert summary.skipped_combinations == 1


def test_default_search_ranges_match_cli_omitted_flags(tmp_path: Path) -> None:
    spec = load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML))
    scenario = spec.scenarios[0]
    # CLI backward-compat default: TP searches powers of two, EP/MOE-DP fixed.
    assert scenario.tp_sizes == (1,)
    assert scenario.ep_sizes == (1,)
    assert scenario.moe_dp_sizes == (1,)
    assert scenario.dcp_sizes == (1,)
    assert scenario.mtp_tokens == (0,)

    multi_device = _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 4")
    spec = load_optimizer_spec(_write_spec(tmp_path, multi_device))
    # Every parallel combination becomes its own single-value workload. EP
    # mirrors the CLI default of a fixed EP=num_devices when the flag is absent.
    assert {scenario.tp_sizes for scenario in spec.scenarios} == {(1,), (2,), (4,)}
    assert all(scenario.dcp_sizes == (1,) for scenario in spec.scenarios)
    assert all(scenario.ep_sizes == (4,) and scenario.moe_dp_sizes == (1,) for scenario in spec.scenarios)


def test_report_dict_contains_workload_summaries(tmp_path: Path) -> None:
    spec = load_optimizer_spec(_write_spec(tmp_path, _SCENARIO_YAML))
    payload = spec.to_report_dict()
    assert payload["workloads"][0]["name"] == "actual_1"
    assert len(payload["digest"]) == 64


def test_duplicate_workloads_are_explicit_in_summary(tmp_path: Path) -> None:
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            """
workloads:
  - name: repeated
    model: org/model
    device: TEST_DEVICE
    num_devices: 1
    input_length: 8
    output_length: 1
  - name: repeated
    model: org/model
    device: TEST_DEVICE
    num_devices: 1
    input_length: 8
    output_length: 1
""",
        )
    )

    assert len(spec.scenarios) == 1
    assert sum(summary.expanded_workloads for summary in spec.workload_summaries) == len(spec.scenarios)
    assert [summary.skipped_duplicate_workloads for summary in spec.workload_summaries] == [0, 1]


def test_spec_parallel_expansion_matches_optimizer_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.perf_data_collection.grid_generator.optimizer_spec._resolve_num_hidden_layers",
        lambda model_id: 8,
    )
    spec = load_optimizer_spec(
        _write_spec(
            tmp_path,
            _SCENARIO_YAML.replace("num_devices: 1", "num_devices: 8")
            + "pp_sizes: [1, 2, 4]\n"
            + "tp_sizes: [1, 2]\n"
            + "ep_sizes: [1, 2]\n"
            + "moe_dp_sizes: [1]\n"
            + "dcp_sizes: [1, 2]\n"
            + "num_mtp_tokens: [0]\n",
        )
    )
    expected = build_pp_search_candidates(
        num_devices=8,
        tp_sizes=[1, 2],
        pp_sizes=[1, 2, 4],
        num_hidden_layers=8,
        ep_sizes=[1, 2],
        moe_dp_sizes=[1],
        num_mtp_token_sizes=[0],
        dcp_sizes=[1, 2],
    )

    observed = {
        (
            scenario.tp_sizes[0],
            scenario.pp_sizes[0],
            scenario.ep_sizes[0],
            scenario.moe_dp_sizes[0],
            scenario.dcp_sizes[0],
            scenario.mtp_tokens[0],
            scenario.pp_layer_partitions[0] if scenario.pp_layer_partitions else None,
        )
        for scenario in spec.scenarios
    }
    expected_values = {
        (
            candidate.tp_size,
            candidate.pp_size,
            candidate.ep_size,
            candidate.moe_dp_size,
            candidate.dcp_size,
            candidate.num_mtp_tokens,
            candidate.layer_partition,
        )
        for candidate in expected
    }
    assert observed == expected_values
