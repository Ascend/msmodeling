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


import json
from pathlib import Path
from types import SimpleNamespace as S
import pytest
import tomlkit
import config_writer as writer
from optix.serve_plan import render_serve_flags, validate_service_positions
from optix.optimizer.agentic.parameters import validate_service_parameters


ROOT = Path(__file__).resolve().parents[4]


def config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[vllm.command]\nothers = "--quantization ascend"\n', encoding="utf-8")
    return path


def prepare(path, tmp_path, definition=None, *extra):
    return writer.main(
        [
            "--config",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--prepare-speculation",
            json.dumps(definition or {"method": "mtp"}),
            *extra,
        ]
    )


def test_prepare_and_candidate_override(tmp_path):
    path = config(tmp_path)
    assert prepare(path, tmp_path) == 0
    others = tomlkit.parse(path.read_text(encoding="utf-8"))["vllm"]["command"]["others"]
    assert render_serve_flags([], others) == ["--quantization", "ascend"]
    args = render_serve_flags([S(name="num_speculative_tokens", value=3, config_position="run")], others)
    assert json.loads(args[args.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 3}
    before = path.read_text(encoding="utf-8")
    assert prepare(path, tmp_path) == 0
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "definition",
    [
        {"model": "draft"},
        {"method": ""},
        {"method": "mtp", "model": ""},
        {"method": "mtp", "num_speculative_tokens": 2},
    ],
)
def test_invalid_definition_preserves_file(tmp_path, definition):
    path = config(tmp_path)
    original = path.read_text(encoding="utf-8")
    assert prepare(path, tmp_path, definition) == 2
    assert path.read_text(encoding="utf-8") == original
    assert not Path(str(path) + ".bak").exists()


@pytest.mark.parametrize(
    "marker", ["pd_experiment.json", "baseline.json", "pd_controller.json", ".pd_dispatch/pd_controller.json"]
)
def test_frozen_run_rejected(tmp_path, marker):
    path = config(tmp_path)
    frozen = tmp_path / "run" / marker
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}", encoding="utf-8")
    assert prepare(path, tmp_path) == 2
    assert "speculative" not in path.read_text(encoding="utf-8")


def test_dry_run_and_failed_check_do_not_publish(tmp_path, monkeypatch):
    path = config(tmp_path)
    original = path.read_text(encoding="utf-8")
    assert prepare(path, tmp_path, None, "--dry-run") == 0
    assert path.read_text(encoding="utf-8") == original
    monkeypatch.setattr(writer, "run_checks", lambda *a: False)
    assert prepare(path, tmp_path, None, "--check") == 1
    assert path.read_text(encoding="utf-8") == original
    assert not Path(str(path) + ".bak").exists()


@pytest.mark.parametrize("suffix", ["", "_prefill", "_decode"])
def test_gpu_channel_uses_shared_rules(tmp_path, suffix):
    name = "GPU_MEMORY_UTILIZATION" + suffix
    field = {"name": name, "dtype": "float", "min": 0.8, "max": 0.95, "default": 0.9, "config_position": "env"}
    context = {"engine": "vllm", "deployment": {"type": "pd" if suffix else "single"}}
    with pytest.raises(ValueError, match="config_position=run"):
        validate_service_parameters(context, {name: 0.9}, fields=[field], others="")
    with pytest.raises(ValueError, match="config_position=run"):
        validate_service_positions([S(**field)])
    path = config(tmp_path)
    assert (
        writer.main(
            [
                "--config",
                str(path),
                "--target-field",
                "engine=" + ("vllm_pd" if suffix else "vllm"),
                "name=" + name,
                "dtype=float",
                "value=0.9",
                "config_position=env",
            ]
        )
        == 2
    )
    assert not Path(str(path) + ".bak").exists()


def test_gpu_run_renders_cli():
    assert render_serve_flags([S(name="GPU_MEMORY_UTILIZATION", value=0.9, config_position="run")]) == [
        "--gpu-memory-utilization",
        "0.9",
    ]


def test_atomic_failure_preserves_original(tmp_path, monkeypatch):
    path = config(tmp_path)
    original = path.read_text(encoding="utf-8")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(writer.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        writer.dump_toml(str(path), tomlkit.parse('[vllm]\nx=1\n'))
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "base",
    [
        {"method": "other", "num_speculative_tokens": 0},
        {"method": "mtp", "num_speculative_tokens": 2},
        {"method": "mtp", "model": "old", "num_speculative_tokens": 0},
    ],
)
def test_conflicting_base_not_overwritten(tmp_path, base):
    import shlex

    path = config(tmp_path)
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    doc["vllm"]["command"]["others"] = "--speculative-config " + shlex.quote(json.dumps(base))
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    original = path.read_text(encoding="utf-8")
    assert prepare(path, tmp_path) == 2
    assert path.read_text(encoding="utf-8") == original
    assert not Path(str(path) + ".bak").exists()


def test_pd_prepared_base_reaches_both_role_scripts(tmp_path):
    from tests.regression.optix.test_pd_agent.test_plugin_render import simulator, field
    from optix.optimizer.pd_adapter import prepare_pd_fields
    from optix.serve_plan import validate_measured_speculation

    path = config(tmp_path)
    assert prepare(path, tmp_path) == 0
    sim = simulator(tmp_path)
    sim.vllm_others = tomlkit.parse(path.read_text(encoding="utf-8"))["vllm"]["command"]["others"]
    projected, _ = prepare_pd_fields(
        sim,
        [
            field("num_speculative_tokens_prefill", 1),
            field("num_speculative_tokens_decode", 3),
            field("GPU_MEMORY_UTILIZATION_prefill", 0.85),
            field("GPU_MEMORY_UTILIZATION_decode", 0.9),
        ],
    )
    sim.vllm_others = ""
    sim.update_config(projected)
    sim._generate_all_scripts()
    for script in tmp_path.glob("*.sh"):
        if script.name.startswith(("P-", "D-")):
            prefill = script.name.startswith("P-")
            text = script.read_text(encoding="utf-8")
            validate_measured_speculation(text, 1 if prefill else 3)
            assert "--gpu-memory-utilization " + ("0.85" if prefill else "0.9") in text
