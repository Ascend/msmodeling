# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Regression for service intent -> effective argv -> archived export."""

import json
import shlex
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from optix.serve_plan import render_serve_flags, validate_measured_speculation
from optix.optimizer.agentic.parameters import validate_service_parameters
from optix.optimizer.agentic.validation import validate_candidate
from optix.optimizer.pd_adapter import prepare_pd_fields
from optix.optimizer.multihost_adapter import prepare_multihost_fields
from check_search_space import check_parameters
from tests.regression.optix.test_pd_agent.test_plugin_render import simulator, field
import export_best_config as export


ROOT = Path(__file__).resolve().parents[4]


def schema(name, default=2):
    return {"name": name, "dtype": "int", "min": 0, "max": 4, "default": default, "config_position": "run"}


@pytest.mark.parametrize(
    "name", ["num_speculative_tokens", "num_speculative_tokens_prefill", "num_speculative_tokens_decode"]
)
def test_preflight_rejects_positive_domain_without_base(name):
    pd = name.endswith(("_prefill", "_decode"))
    context = {"engine": "vllm", "deployment": {"type": "pd" if pd else "single"}, "model_info": {}}
    issues = check_parameters(
        context, fields=[schema(name, 0)], section={"command": {"others": "--quantization ascend"}}
    )
    assert any("requires method/model" in i["msg"] for i in issues)
    issues = check_parameters(context, fields=[schema(name, 0)], section={"command": {"others": ""}}, params={name: 0})
    assert not [i for i in issues if i["level"] == "error"]


@pytest.mark.parametrize("others", ["", "--quantization ascend"])
def test_candidate_validation_rejects_missing_base(tmp_path, others):
    path = tmp_path / "config.toml"
    path.write_text("[vllm.command]\nothers = " + json.dumps(others), encoding="utf-8")
    context = {
        "engine": "vllm",
        "source": {"config_toml": str(path)},
        "search_space": {"parameters": [schema("num_speculative_tokens")]},
    }
    result = validate_candidate(context, {"candidate_id": "mtp", "params": {"num_speculative_tokens": 2}})
    assert not result["valid"] and any("requires method/model" in e for e in result["errors"])


@pytest.mark.parametrize("raw", ["[]", "null", "3", "bad-json"])
@pytest.mark.parametrize("flag", ["--speculative-config", "--compilation_config", "--additional-config"])
def test_malformed_containers_fail(raw, flag):
    with pytest.raises(ValueError, match="JSON object"):
        render_serve_flags((), shlex.join([flag, raw]))


@pytest.mark.parametrize("value", [-1, 1.5, True, "nan", "inf"])
def test_invalid_gamma(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        render_serve_flags([field("num_speculative_tokens", value)], "--speculative-config '{\"method\":\"mtp\"}'")


def test_alias_equals_merge_and_off():
    others = "--speculative_config=" + shlex.quote('{"method":"mtp","num_speculative_tokens":1}')
    argv = render_serve_flags([field("num_speculative_tokens", 3)], others)
    assert json.loads(argv[1]) == {"method": "mtp", "num_speculative_tokens": 3}
    assert render_serve_flags([field("num_speculative_tokens", 0)], others) == []


def test_direct_suffix_rejected():
    with pytest.raises(ValueError, match="PD adapter"):
        render_serve_flags([field("num_speculative_tokens_decode", 2)])


def test_pd_real_scripts_and_role_overrides(tmp_path):
    sim = simulator(tmp_path)
    sim.vllm_others = "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
    fields = [
        field("num_speculative_tokens", 2),
        field("num_speculative_tokens_decode", 0),
        field("MAX_NUM_SEQS_prefill", 16),
        field("enforce_eager_decode", True),
    ]
    out, _ = prepare_pd_fields(sim, fields)
    sim.vllm_others = ""
    sim.update_config(out)
    sim._generate_all_scripts()
    scripts = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.glob("*.sh")}
    for name, script in scripts.items():
        if name.startswith("P-"):
            validate_measured_speculation(script, 2)
        elif name.startswith("D-"):
            validate_measured_speculation(script, 0)
    assert any(n.startswith("P-") for n in scripts)
    assert any(n.startswith("D-") for n in scripts)


def test_pd_runtime_rejects_orphan(tmp_path):
    sim = simulator(tmp_path)
    with pytest.raises(ValueError, match="requires method/model"):
        prepare_pd_fields(sim, [field("num_speculative_tokens_decode", 2)])


def test_multihost_runtime_rejects_orphan():
    sim = S(remote_executor=S(config=S(env={}, workers=[], chips_per_node=1)), config=S(command=S(others="")))
    with pytest.raises(ValueError, match="requires method/model"):
        prepare_multihost_fields(sim, [field("num_speculative_tokens", 2)])


def test_single_export_reuses_planner():
    with pytest.raises(ValueError, match="requires method/model"):
        export.render_serve_command("m", "m", "0", 1, {"num_speculative_tokens": 2}, [])
    result = export.render_serve_command(
        "m", "m", "0", 1, {"num_speculative_tokens": 2}, [{"flag": "--speculative-config", "value": '{"method":"mtp"}'}]
    )
    validate_measured_speculation(shlex.join(result), 2)


def test_archived_export_does_not_repair_old_failure():
    with pytest.raises(ValueError, match="Measured script"):
        export._check_measured_speculation(
            "vllm serve model --max-num-seqs 16", {"num_speculative_tokens_decode": 2}, "decode"
        )
    script = "vllm serve model --speculative_config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
    export._check_measured_speculation(script, {"num_speculative_tokens": 3})
    with pytest.raises(ValueError, match="OFF"):
        export._check_measured_speculation(script, {"num_speculative_tokens": 0})


def test_env_service_subkey_rejected():
    ctx = {
        "engine": "vllm",
        "search_space": {"parameters": [{**schema("num_speculative_tokens"), "config_position": "env"}]},
    }
    with pytest.raises(ValueError, match="config_position=run"):
        validate_service_parameters(ctx, {"num_speculative_tokens": 2})


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_candidate_validation_rejects_orphan(tmp_path, role):
    from optix.optimizer.agentic.pd import configuration_fingerprint

    path = tmp_path / "config.toml"
    path.write_text('[vllm.command]\nothers="--quantization ascend"', encoding="utf-8")
    name = "num_speculative_tokens_" + role
    context = {
        "engine": "vllm",
        "source": {"config_toml": str(path)},
        "deployment": {
            "type": "pd",
            "config_sha256": configuration_fingerprint(path),
            "plugin_config": {
                "nodes": [
                    {"role": "prefill", "ssh_ip": "p", "gpu_ids": [0]},
                    {"role": "decode", "ssh_ip": "d", "gpu_ids": [0]},
                ]
            },
        },
        "search_space": {"parameters": [schema(name)]},
    }
    result = validate_candidate(context, {"candidate_id": "mtp", "params": {name: 2}})
    assert not result["valid"] and any(role + ":" in e and "requires method/model" in e for e in result["errors"])


@pytest.mark.parametrize("position", ["prefill", "decode"])
def test_pd_export_checks_before_writing_and_preserves_script(tmp_path, position):
    from tests.regression.optix.test_pd_agent.test_export_shutdown import snapshot

    plan, context, best = snapshot()
    best["params"] = {"num_speculative_tokens_" + position: 2}
    with pytest.raises(ValueError, match="Measured script"):
        export._export_pd_snapshot(tmp_path, context, best, {})
    assert not (tmp_path / "best_deployment").exists()
    script = "#!/bin/bash\nvllm serve m --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'\n"
    plan["files"][position + ".sh"] = script
    export._export_pd_snapshot(tmp_path, context, best, {})
    assert (tmp_path / "best_deployment" / (position + ".sh")).read_text(encoding="utf-8") == script


def test_multihost_export_rejects_old_ineffective_gamma(tmp_path):
    best = {"metadata": {"launch_plan": {"start_node.sh": "vllm serve m"}}, "params": {"num_speculative_tokens": 2}}
    with pytest.raises(ValueError, match="Measured script"):
        export._export_multihost_snapshot(tmp_path, {"hardware": {"num_nodes": 1}}, best, {})
    assert not (tmp_path / "best_deployment").exists()


@pytest.mark.parametrize("mode", ["single", "multihost", "pd"])
def test_legal_candidate_complete_base(mode):
    name = "num_speculative_tokens_decode" if mode == "pd" else "num_speculative_tokens"
    ctx = {"engine": "vllm", "deployment": {"type": mode}, "search_space": {"parameters": [schema(name)]}}
    validate_service_parameters(ctx, {name: 2}, others="--speculative-config '{\"method\":\"mtp\"}'")


@pytest.mark.parametrize("mode", ["multihost", "pd"])
def test_runtime_rejects_env_speculative_subkey(tmp_path, mode):
    if mode == "pd":
        sim = simulator(tmp_path)
        call = prepare_pd_fields
        name = "num_speculative_tokens_decode"
    else:
        sim = S(remote_executor=S(config=S(env={}, workers=[], chips_per_node=1)), config=S(command=S(others="")))
        call = prepare_multihost_fields
        name = "num_speculative_tokens"
    with pytest.raises(ValueError, match="config_position=run"):
        call(sim, [field(name, 2, "env")])


def test_single_runtime_rejects_env_before_launch():
    from optix.optimizer.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = "vllm"
    scheduler.simulate_run_info = [field("num_speculative_tokens", 2, "env")]
    with pytest.raises(ValueError, match="config_position=run"):
        scheduler.run_simulate(None, ())


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_owned_option_rejected_by_preflight_and_runtime(tmp_path, role):
    name = "HOST_" + role
    context = {"engine": "vllm", "deployment": {"type": "pd"}}
    with pytest.raises(ValueError, match="plugin-owned"):
        validate_service_parameters(context, {name: "127.0.0.1"}, fields=[schema(name, "127.0.0.1")])
    with pytest.raises(ValueError, match="plugin-owned"):
        prepare_pd_fields(simulator(tmp_path), (field(name, "127.0.0.1"),))


def test_pd_owned_static_option_rejected_by_both_paths(tmp_path):
    context = {"engine": "vllm", "deployment": {"type": "pd"}}
    with pytest.raises(ValueError, match="plugin-owned"):
        validate_service_parameters(context, {}, fields=[], others="--port 1234")
    sim = simulator(tmp_path)
    sim.vllm_others = "--port 1234"
    with pytest.raises(ValueError, match="plugin-owned"):
        prepare_pd_fields(sim, ())
