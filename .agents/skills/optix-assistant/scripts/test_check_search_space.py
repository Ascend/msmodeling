"""Ownership: shared-precheck (test). Unit tests for check_search_space.py and candidate validation.

验证参数定义、资源可行性与逐轮候选门禁；检查不修改输入，不注入推荐值。
"""

import copy
import json
import shutil
from pathlib import Path

import pytest

from check_search_space import check_parameters, load_configured_parameters
from config_writer import main as write_config
from validate_candidates import check_ref, validate_round, validate_search_space


def make_context(tmp_path, fields, others=""):
    path = tmp_path / "config.toml"
    path.write_text("[vllm.command]\nothers = " + repr(others) + "\n" + fields)
    context = {
        "source": {"config_toml": str(path)},
        "engine": "vllm",
        "hardware": {"world_size": 8},
        "model_info": {"is_moe": True},
    }
    return context


def test_interior_parallel_values_are_feasible(tmp_path):
    fields = ""
    for name in ("tp", "dp"):
        fields += f'[[vllm.target_field]]\nname="{name}"\ndtype="int"\nvalue=2\nmin=1\nmax=8\nconfig_position="run"\n'
    assert not check_parameters(make_context(tmp_path, fields))


def test_fixed_parallel_layout_must_match_resources(tmp_path):
    fields = '[[vllm.target_field]]\nname="tp"\ndtype="int"\nmin=4\nmax=4\nvalue=4\nconfig_position="run"\n'
    assert any(x["level"] == "error" for x in check_parameters(make_context(tmp_path, fields)))


def test_enabled_speculation_requires_base(tmp_path):
    fields = '[[vllm.target_field]]\nname="num_speculative_tokens"\ndtype="int"\nmin=0\nmax=7\nvalue=0\nconfig_position="run"\n'
    issues = check_parameters(make_context(tmp_path, fields, "--tensor-parallel-size 8"))
    assert any("method/model" in x["msg"] for x in issues)


@pytest.mark.parametrize("base", [{"method": "mtp"}, {"model": "/weights/draft model"}])
def test_speculation_written_by_config_writer_is_recognized(tmp_path, base):
    fields = '[[vllm.target_field]]\nname="num_speculative_tokens"\ndtype="int"\nmin=1\nmax=4\nvalue=2\nconfig_position="run"\n'
    context = make_context(tmp_path, fields, "--tensor-parallel-size 8")
    config = tmp_path / "config.toml"
    assert (
        write_config(["--config", str(config), "--set", f"vllm.command.others.speculative-config={json.dumps(base)}"])
        == 0
    )
    before = config.read_bytes()
    original_context = copy.deepcopy(context)
    section, definitions = load_configured_parameters(context)
    assert isinstance(section["command"]["others"], str)
    assert not check_parameters(context)
    assert not check_parameters(context, fields=definitions, section=section, params={"num_speculative_tokens": 3})
    assert definitions[0]["name"] == "num_speculative_tokens"
    assert definitions[0]["default"] == 2
    assert config.read_bytes() == before and context == original_context


@pytest.mark.parametrize("spec", [{"method": "mtp"}, {"model": "/weights/draft"}])
def test_speculation_equals_form_and_pcp_are_recognized(tmp_path, spec):
    fields = '[[vllm.target_field]]\nname="num_speculative_tokens"\ndtype="int"\nmin=1\nmax=4\nvalue=2\nconfig_position="run"\n'
    context = make_context(tmp_path, fields)
    section, definitions = load_configured_parameters(context)
    section["command"]["others"] = (
        '--tensor-parallel-size=2 --prefill-context-parallel-size=4 --enforce-eager '
        f"--speculative-config='{json.dumps(spec)}'"
    )
    assert not check_parameters(context, fields=definitions, section=section)


@pytest.mark.parametrize("base", [{"method": " "}, {"model": ""}, {"method": 123}, {}])
def test_enabled_speculation_rejects_empty_or_invalid_base(tmp_path, base):
    fields = '[[vllm.target_field]]\nname="num_speculative_tokens"\ndtype="int"\nmin=1\nmax=4\nvalue=2\nconfig_position="run"\n'
    context = make_context(tmp_path, fields)
    section, definitions = load_configured_parameters(context)
    section["command"]["others"] = f"--tensor-parallel-size 8 --speculative-config='{json.dumps(base)}'"
    issues = check_parameters(context, fields=definitions, section=section)
    assert any("method/model" in x["msg"] for x in issues)


def test_others_object_is_rejected_instead_of_treated_as_command(tmp_path):
    fields = '[[vllm.target_field]]\nname="tp"\ndtype="int"\nmin=8\nmax=8\nvalue=8\nconfig_position="run"\n'
    context = make_context(tmp_path, fields)
    section, definitions = load_configured_parameters(context)
    section["command"]["others"] = {"speculative-config": {"method": "mtp"}}
    assert any(
        "command-line string" in x["msg"] for x in check_parameters(context, fields=definitions, section=section)
    )


def setup_run(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[vllm.command]\nothers="--tensor-parallel-size 1"\n'
        '[[vllm.target_field]]\nname="MAX_NUM_SEQS"\ndtype="int"\n'
        'min=1\nmax=64\nvalue=16\nconfig_position="run"\n'
    )
    context = {
        "engine": "vllm",
        "source": {"config_toml": str(config)},
        "hardware": {"world_size": 1},
        "constraints": {"world_size": 1, "expressions": ["1*1*1*1==$WORLD_SIZE"]},
    }
    context["search_space"] = {
        "parameters": [
            {"name": "MAX_NUM_SEQS", "dtype": "int", "min": 1, "max": 64, "default": 16, "config_position": "run"}
        ],
        "constants": [],
    }
    data = {
        "round": 1,
        "directions": [
            {
                "direction_id": "d1",
                "type": "local_search",
                "trial_type": "optimization",
                "hypothesis": "Increase batch capacity within the configured range.",
                "evidence_analysis": "Initial configuration.",
                "refs_analysis": "Read methods/scheduling/request-scheduling.md; batch capacity is applicable to this baseline.",
                "candidates_analysis": "Space unchanged; choose batch 32.",
            }
        ],
        "routing_review": {
            "hypothesis": "首轮无实测数据，瓶颈未知，先按无基线分支建立基线。",
            "l1_layers_matched": ["通用层"],
            "layers_deferred": {"通用层": "首轮仅建立基线，机制层动作暂缓。"},
        },
        "candidates": [{"candidate_id": "r1-a", "direction_id": "d1", "params": {"MAX_NUM_SEQS": 32}}],
    }
    return context, data


def save(tmp_path, context, data):
    (tmp_path / "context.json").write_text(json.dumps(context))
    (tmp_path / f'candidates.round-{data["round"]}.json').write_text(json.dumps(data))


def extra(**updates):
    return {
        "name": "enforce_eager",
        "dtype": "bool",
        "config_position": "run",
        "default": True,
        "min": 0,
        "max": 1,
        **updates,
    }


def test_validation_uses_current_context_without_modifying_inputs(tmp_path):
    context, data = setup_run(tmp_path)
    context["search_space"]["parameters"].append(extra())
    save(tmp_path, context, data)
    before = copy.deepcopy(data)
    files_before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    validate_round(data, tmp_path, 1)
    assert data == before
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files_before
    context["search_space"]["parameters"][-1]["default"] = False
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    assert data == before


@pytest.mark.parametrize("key", ["name", "dtype", "config_position", "default", "min", "max"])
def test_missing_definition_fields_are_rejected(tmp_path, key):
    context, _ = setup_run(tmp_path)
    new = extra()
    new.pop(key)
    context["search_space"]["parameters"].append(new)
    with pytest.raises(ValueError):
        validate_search_space(context["search_space"], load_configured_parameters(context)[1])


@pytest.mark.parametrize("change", ["delete", "position", "dtype"])
def test_toml_definition_drift_is_rejected(tmp_path, change):
    context, _ = setup_run(tmp_path)
    item = context["search_space"]["parameters"][0]
    if change == "delete":
        context["search_space"]["parameters"].clear()
    elif change == "position":
        item["config_position"] = "env"
    elif change == "dtype":
        item["dtype"] = "float"
    with pytest.raises(ValueError):
        validate_search_space(context["search_space"], load_configured_parameters(context)[1])


def test_extra_parameters_can_be_added_changed_and_removed(tmp_path):
    context, data = setup_run(tmp_path)
    context["search_space"]["parameters"].append(extra())
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    context["search_space"]["parameters"][-1]["default"] = False
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    context["search_space"]["parameters"].pop()
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    assert len(context["search_space"]["parameters"]) == 1
    data["candidates"][0]["params"]["enforce_eager"] = True
    with pytest.raises(ValueError, match="Unknown parameter"):
        validate_round(data, tmp_path, 1)


def test_plain_string_metadata_cannot_bypass_coordinate_compatibility(tmp_path):
    context, _ = setup_run(tmp_path)
    context["search_space"]["parameters"].append(extra(dtype="str", default="text"))
    with pytest.raises(ValueError, match="use enum"):
        validate_search_space(context["search_space"], load_configured_parameters(context)[1])


def test_failed_previous_round_and_future_candidate_are_blocked(tmp_path):
    context, first = setup_run(tmp_path)
    save(tmp_path, context, first)
    second = copy.deepcopy(first)
    second["round"] = 2
    second["candidates"][0]["candidate_id"] = "r2-a"
    save(tmp_path, context, second)
    with pytest.raises(ValueError, match="Future candidate"):
        validate_round(first, tmp_path, 1)
    with pytest.raises(ValueError, match="unfinished"):
        validate_round(second, tmp_path, 2)
    (tmp_path / "results.round-1.json").write_text(json.dumps({"trials": [{"fitness": 1, "error": None}]}))
    with pytest.raises(ValueError, match="missing context.round-1.json"):
        validate_round(second, tmp_path, 2)
    shutil.copyfile(tmp_path / "context.json", tmp_path / "context.round-1.json")
    validate_round(second, tmp_path, 2)
    with pytest.raises(ValueError, match="already succeeded"):
        validate_round(first, tmp_path, 1)


def test_round_zero_check_needs_no_space_and_does_not_create_one(tmp_path):
    context, _ = setup_run(tmp_path)
    context.pop("search_space")
    before = copy.deepcopy(context)
    config = Path(context["source"]["config_toml"])
    config_before = config.read_bytes()
    assert not [issue for issue in check_parameters(context) if issue["level"] == "error"]
    assert context == before
    assert config.read_bytes() == config_before


def test_optimization_requires_agent_to_create_space(tmp_path):
    context, data = setup_run(tmp_path)
    agent_space = context.pop("search_space")
    save(tmp_path, context, data)
    before = (tmp_path / "context.json").read_bytes()
    with pytest.raises(ValueError, match="Missing context.search_space"):
        validate_round(data, tmp_path, 1)
    assert (tmp_path / "context.json").read_bytes() == before
    context["search_space"] = agent_space
    save(tmp_path, context, data)
    before = (tmp_path / "context.json").read_bytes()
    validate_round(data, tmp_path, 1)
    assert (tmp_path / "context.json").read_bytes() == before


def test_config_check_and_active_space_check_have_separate_inputs(tmp_path):
    context, data = setup_run(tmp_path)
    context["search_space"]["parameters"][0]["config_position"] = "env"
    assert not [issue for issue in check_parameters(context) if issue["level"] == "error"]
    save(tmp_path, context, data)
    with pytest.raises(ValueError, match="config_position conflicts"):
        validate_round(data, tmp_path, 1)


@pytest.mark.parametrize("gamma", [0, 1, 7])
def test_candidate_speculation_requires_base_only_when_enabled(tmp_path, gamma):
    context, _ = setup_run(tmp_path)
    section, fields = load_configured_parameters(context)
    fields.append(extra(name="num_speculative_tokens", dtype="int", default=0, max=7))
    params = {f["name"]: f["default"] for f in fields}
    params["num_speculative_tokens"] = gamma
    issues = check_parameters(context, fields=fields, section=section, params=params)
    assert any("method/model" in item["msg"] for item in issues) is (gamma > 0)


def test_fixed_command_speculation_also_requires_method(tmp_path):
    context, data = setup_run(tmp_path)
    path = Path(context["source"]["config_toml"])
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'others="--tensor-parallel-size 1"',
            "others=\"--tensor-parallel-size 1 --speculative-config '{\\\"num_speculative_tokens\\\": 2}'\"",
        ),
        encoding="utf-8",
    )
    save(tmp_path, context, data)
    with pytest.raises(ValueError, match="method/model"):
        validate_round(data, tmp_path, 1)


def test_context_constraints_must_use_runtime_parameter_names(tmp_path):
    from optix.optimizer.agentic.validation import validate_candidate

    context, data = setup_run(tmp_path)
    context["constraints"]["expressions"] = ["$TP*$DP*$PP*$PCP==$WORLD_SIZE"]
    save(tmp_path, context, data)
    with pytest.raises(ValueError, match="Constraint failed"):
        validate_round(data, tmp_path, 1)
    context["constraints"]["expressions"] = ["1*1*1*1==$WORLD_SIZE"]
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    persisted = json.loads((tmp_path / "context.json").read_text())
    assert validate_candidate(persisted, data["candidates"][0])["valid"]
    assert not any(f["name"].lower() in {"tp", "dp", "pp", "pcp"} for f in persisted["search_space"]["parameters"])


@pytest.mark.parametrize("param_name", ["tp", "tensor_parallel_size"])
def test_actual_parallel_values_are_checked_against_fixed_command(tmp_path, param_name):
    from optix.optimizer.agentic.validation import validate_candidate

    context, data = setup_run(tmp_path)
    context["hardware"]["world_size"] = context["constraints"]["world_size"] = 8
    config = Path(context["source"]["config_toml"])
    config.write_text(
        config.read_text(encoding="utf-8").replace("--tensor-parallel-size 1", "--data-parallel-size 2"),
        encoding="utf-8",
    )
    context["search_space"]["parameters"].append(extra(name=param_name, dtype="int", min=1, max=8, default=4))
    context["constraints"]["expressions"] = [f"${param_name.upper()}*2*1*1==$WORLD_SIZE"]
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    assert validate_candidate(json.loads((tmp_path / "context.json").read_text()), data["candidates"][0])["valid"]
    data["candidates"][0]["params"][param_name] = 2
    # Even an incomplete expression must not bypass the Skill's resource check.
    context["constraints"]["expressions"] = []
    save(tmp_path, context, data)
    with pytest.raises(ValueError, match="Candidate parallel values"):
        validate_round(data, tmp_path, 1)


@pytest.mark.parametrize("content", ["{", "[]", '{}', '{"search_space": {}}'])
def test_previous_round_archive_must_be_readable_context(tmp_path, content):
    context, data = setup_run(tmp_path)
    data["round"] = 2
    save(tmp_path, context, data)
    (tmp_path / "results.round-1.json").write_text(json.dumps({"trials": [{"fitness": 1, "error": None}]}))
    (tmp_path / "context.round-1.json").write_text(content)
    with pytest.raises(ValueError):
        validate_round(data, tmp_path, 2)


def test_archived_context_references_are_checked(tmp_path):
    ref = "context.round-1.json"
    assert check_ref(ref, tmp_path)
    (tmp_path / ref).write_text('{}')
    assert not check_ref(ref, tmp_path)


def test_skill_ref_rejects_path_traversal(tmp_path):
    assert check_ref("knowledge/../../../etc/hostname", tmp_path)
    assert check_ref("references/../../AGENTS.md", tmp_path)
    assert not check_ref("references/pso-mode.md", tmp_path)


def test_skill_ref_rejects_symlink_escaping_skill_root(tmp_path, monkeypatch):
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    root = tmp_path / "skill"
    (root / "knowledge").mkdir(parents=True)
    (root / "references").mkdir()
    (root / "references" / "pso-mode.md").write_text("card")
    (root / "knowledge" / "link.md").symlink_to(outside)
    (root / "knowledge" / "inner-link.md").symlink_to(root / "references" / "pso-mode.md")
    monkeypatch.setattr("validate_candidates.SKILL_ROOT", root)
    assert check_ref("knowledge/link.md", tmp_path)
    assert not check_ref("references/pso-mode.md", tmp_path)
    assert not check_ref("knowledge/inner-link.md", tmp_path)


def test_failed_round_can_retry_before_its_archive_is_updated(tmp_path):
    context, data = setup_run(tmp_path)
    context["search_space"]["parameters"].append(extra())
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    result_path = tmp_path / "results.round-1.json"
    result_path.write_text(json.dumps({"trials": [{"error": "failed", "fitness": float("inf")}]}))
    archive_path = tmp_path / "context.round-1.json"
    shutil.copyfile(tmp_path / "context.json", archive_path)
    context["search_space"]["parameters"][-1]["default"] = False
    save(tmp_path, context, data)
    validate_round(data, tmp_path, 1)
    assert json.loads(archive_path.read_text())["search_space"]["parameters"][-1]["default"] is True
    result_path.write_text(json.dumps({"trials": [{"error": None, "fitness": 1}]}))
    shutil.copyfile(tmp_path / "context.json", archive_path)
    assert json.loads(archive_path.read_text())["search_space"]["parameters"][-1]["default"] is False
    with pytest.raises(ValueError, match="already succeeded"):
        validate_round(data, tmp_path, 1)
