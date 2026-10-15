"""Ownership: agent (test). Unit tests for export_best_config.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("export_best_config.py")
REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def legacy_candidate_archive(tmp_path):
    # Export joins results to the candidate archive when the engine has no type metadata.
    (tmp_path / "candidates.round-1.json").write_text(
        json.dumps({"round": 1, "candidates": [{"candidate_id": "r1-a", "params": {}}]})
    )


def load_module():
    spec = importlib.util.spec_from_file_location("export_best_config", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_context(run_dir, config, params, metadata=None, round_no=1):
    definitions = {name: {"name": name, "config_position": "run", **(metadata or {}).get(name, {})} for name in params}
    context = {
        "engine": "vllm",
        "source": {"config_toml": str(config)},
        "search_space": {"parameters": list(definitions.values()), "constants": []},
    }
    (run_dir / f"context.round-{round_no}.json").write_text(json.dumps(context))
    return definitions


def test_flag_maps_match_custom_command():
    # 防漂移：本脚本须在脱离 msmodeling 仓库的环境可运行，不能直接 import
    # optix.config.custom_command，因此两张映射表是副本——这里断言副本与
    # 源一致，任一侧改动另一侧没跟上都会让本测试失败。
    if not (REPO_ROOT / "optix" / "config" / "custom_command.py").exists():
        pytest.skip("not running inside the msmodeling repo")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from optix.config.custom_command import FLAG_NAME_MAP, JSON_SUBKEY_MAP

    module = load_module()
    assert module._FLAG_NAME_MAP == FLAG_NAME_MAP
    assert module._JSON_SUBKEY_MAP == JSON_SUBKEY_MAP


def _container_value(argv, flag):
    values = [argv[i + 1] for i, tok in enumerate(argv) if tok == flag]
    assert len(values) == 1, f"{flag} must appear exactly once, got {values}"
    return json.loads(values[0])


def test_moe_switches_render_into_additional_config():
    # 回归现场：enable_shared_expert_dp / multistream_overlap_shared_expert
    # 不是 vLLM 顶层 CLI 参数，必须并入 --additional-config 且与 config.toml
    # others 的固定底座（cpu_binding/flashcomm1/fused_mc2/weight_nz_mode）合并，
    # 不得渲染成顶层 flag、也不得与底座拆成两条重复容器。
    module = load_module()
    argv = module.render_serve_command(
        "model-path",
        "served",
        "127.0.0.1",
        "8000",
        params={
            "tp": 8,
            "enable_shared_expert_dp": True,
            "multistream_overlap_shared_expert": True,
        },
        fixed_flags=[
            {"flag": "--additional-config", "value": '{"enable_cpu_binding": true, "enable_fused_mc2": true}'},
            {"flag": "--seed", "value": "1024"},
        ],
    )
    assert "--enable-shared-expert-dp" not in argv
    assert "--multistream-overlap-shared-expert" not in argv
    additional = _container_value(argv, "--additional-config")
    assert additional["enable_shared_expert_dp"] is True
    assert additional["multistream_overlap_shared_expert"] is True
    assert additional["enable_cpu_binding"] is True
    assert additional["enable_fused_mc2"] is True
    assert "--seed" in argv


def test_speculative_container_merge_and_atomicity():
    # 既有语义回归：固定底座 method/model + 搜索侧 num_speculative_tokens 合并；
    # num_speculative_tokens==0 整体丢弃投机容器。
    module = load_module()
    argv = module.render_serve_command(
        "model-path",
        "served",
        "127.0.0.1",
        "8000",
        params={"num_speculative_tokens": 5, "speculative_enforce_eager": True},
        fixed_flags=[
            {"flag": "--speculative-config", "value": '{"method": "eagle3", "model": "draft-path"}'},
        ],
    )
    spec = _container_value(argv, "--speculative-config")
    assert spec == {"method": "eagle3", "model": "draft-path", "num_speculative_tokens": 5, "enforce_eager": True}

    argv_off = module.render_serve_command(
        "model-path",
        "served",
        "127.0.0.1",
        "8000",
        params={"num_speculative_tokens": 0},
        fixed_flags=[
            {"flag": "--speculative-config", "value": '{"method": "eagle3", "model": "draft-path"}'},
        ],
    )
    assert "--speculative-config" not in argv_off


def test_export_uses_archived_context_config(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        '[vllm.command]\nmodel="model path"\nserved_model_name="served"\nhost="10.0.0.1"\nport="9000"\n'
        + "others=\"--speculative-config '{\\\"method\\\":\\\"mtp\\\"}'\"\n"
    )
    archive_context(tmp_path, config, {"tp": 8, "num_speculative_tokens": 1, "CONCURRENCY": 8})
    (tmp_path / "results.round-1.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "candidate_id": "r1-a",
                        "fitness": 1,
                        "slo_violated": False,
                        "params": {"tp": 8, "num_speculative_tokens": 1, "CONCURRENCY": 8},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    load_module().main()
    command = json.loads((tmp_path / "serve_command.json").read_text())
    assert set(command) == {"argv", "command"}
    assert "--concurrency" not in command["argv"]
    assert _container_value(command["argv"], "--speculative-config")["method"] == "mtp"
    handoff = json.loads((tmp_path / "optimizer_config_handoff.json").read_text())
    assert "engine_fixed_params_verification" not in handoff
    assert all("verified" not in entry for entry in handoff["config_fixed_params"])
    assert handoff["recommended_params"]["num_speculative_tokens"] == 1
    import shlex

    shell_command = (tmp_path / "serve_command.sh").read_text().splitlines()[-1]
    assert shlex.split(shell_command) == command["argv"]


def test_render_serve_command_respects_parameter_positions():
    module = load_module()
    argv = module.render_serve_command(
        "model",
        "served",
        "host",
        "9000",
        params={"HCCL_BUFFSIZE": 200, "kv_cache_dtype": "fp8", "CONCURRENCY": 8},
        fixed_flags=[],
        parameter_metadata={
            "HCCL_BUFFSIZE": {"config_position": "env"},
            "kv_cache_dtype": {"config_position": "run"},
            "CONCURRENCY": {"config_position": "env"},
        },
    )
    assert "--hccl-buffsize" not in argv
    assert "--concurrency" not in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"


@pytest.mark.parametrize("position", ["BackendConfig.ScheduleConfig.maxBatchSize", "unsupported"])
def test_export_rejects_unsupported_config_position(position):
    with pytest.raises(ValueError, match="unsupported config_position"):
        load_module().render_serve_command(
            "model",
            "served",
            "host",
            "9000",
            {"custom": 1},
            [],
            parameter_metadata={"custom": {"config_position": position}},
        )


def test_export_environment_matches_winning_trial(tmp_path, monkeypatch):
    import os
    import shlex
    import subprocess

    config = tmp_path / "config.toml"
    config.write_text(
        '[vllm.command]\nmodel="model path"\nserved_model_name="served"\nhost="10.0.0.1"\nport="9000"\n'
        '[vllm.env]\nSTATIC_FLAG="keep"\nDROP_FLAG="configured"\n'
        '[[vllm.target_field]]\nname="HCCL_BUFFSIZE"\ndtype="int"\nconfig_position="env"\nvalue=50\n'
        '[[vllm.target_field]]\nname="custom_env"\ndtype="str"\nconfig_position="run"\nvalue="later"\n'
    )
    quoted = "space ' quote; $NOT_EXPANDED `not_executed`"
    metadata = {
        "custom_env": {"value": quoted, "dtype": "str", "config_position": "env"},
        "DROP_FLAG": {"value": "", "dtype": "str", "config_position": "env"},
        "kv_cache_dtype": {"value": "fp8", "dtype": "str", "config_position": "run"},
    }
    params = {"HCCL_BUFFSIZE": 200, "custom_env": quoted, "DROP_FLAG": "", "kv_cache_dtype": "fp8", "CONCURRENCY": 8}
    definitions = archive_context(
        tmp_path,
        config,
        params,
        {
            "HCCL_BUFFSIZE": {"config_position": "env"},
            **metadata,
        },
    )
    (tmp_path / "results.round-1.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "candidate_id": "r1-a",
                        "fitness": 1,
                        "slo_violated": False,
                        "params": params,
                        "metadata": {"parameter_metadata": metadata},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    load_module().main()

    command = json.loads((tmp_path / "serve_command.json").read_text())
    handoff = json.loads((tmp_path / "optimizer_config_handoff.json").read_text())
    expected_env = {"STATIC_FLAG": "keep", "DROP_FLAG": None, "HCCL_BUFFSIZE": "200", "CUSTOM_ENV": quoted}
    assert command["env"] == handoff["serve_env"] == expected_env
    assert handoff["recommended_params"] == params
    assert handoff["parameter_metadata"] == definitions
    assert shlex.split(command["command"]) == command["argv"]
    assert not {"--hccl-buffsize", "--custom-env", "--drop-flag", "--concurrency"} & set(command["argv"])

    # Replay the exported shell against a local argv/env recorder, not a service.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = bin_dir / "vllm"
    recorder.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': {"
        "k: os.environ.get(k) for k in ['STATIC_FLAG', 'DROP_FLAG', 'HCCL_BUFFSIZE', 'CUSTOM_ENV']}}))\n"
    )
    recorder.chmod(0o755)
    result = subprocess.run(
        ["bash", str(tmp_path / "serve_command.sh")],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""), "DROP_FLAG": "inherited"},
    )
    replay = json.loads(result.stdout)
    assert replay["argv"] == command["argv"][1:]
    assert replay["env"] == expected_env


@pytest.mark.parametrize("live_context", ["changed", "missing", "malformed"])
def test_export_uses_winning_context_archive(tmp_path, monkeypatch, live_context):
    config = tmp_path / "config.toml"
    config.write_text('[vllm.command]\nmodel="model"\nserved_model_name="served"\nhost="10.0.0.1"\nport="9000"\n')
    original = {
        "name": "CUSTOM_ENV",
        "dtype": "enum",
        "choices": ["first"],
        "default": "first",
        "min": 0,
        "max": 1,
        "config_position": "env",
    }
    later = {**original, "config_position": "run", "choices": ["later"], "default": "later"}
    if live_context == "changed":
        (tmp_path / "context.json").write_text(
            json.dumps(
                {
                    "engine": "mindie",
                    "source": {"config_toml": str(tmp_path / "unrelated.toml")},
                    "search_space": {"parameters": [later], "constants": []},
                }
            )
        )
    elif live_context == "malformed":
        (tmp_path / "context.json").write_text('{')
    for number, definition, fitness in [(1, original, 1), (2, later, 2)]:
        (tmp_path / f"candidates.round-{number}.json").write_text(
            json.dumps(
                {
                    "round": number,
                    "candidates": [{"candidate_id": f"r{number}-a", "params": {}}],
                }
            )
        )
        (tmp_path / f"context.round-{number}.json").write_text(
            json.dumps(
                {
                    "engine": "vllm",
                    "source": {"config_toml": str(config)},
                    "search_space": {"parameters": [definition], "constants": []},
                }
            )
        )
        (tmp_path / f"results.round-{number}.json").write_text(
            json.dumps(
                {
                    "trials": [
                        {
                            "candidate_id": f"r{number}-a",
                            "fitness": fitness,
                            "slo_violated": False,
                            "params": {"CUSTOM_ENV": definition["default"]},
                            "metadata": {"round_id": number},
                        }
                    ]
                }
            )
        )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    load_module().main()
    command = json.loads((tmp_path / "serve_command.json").read_text())
    assert command["env"] == {"CUSTOM_ENV": "first"}
    assert "--custom-env" not in command["argv"]
    handoff = json.loads((tmp_path / "optimizer_config_handoff.json").read_text())
    assert handoff["parameter_metadata"] == {"CUSTOM_ENV": original}
    assert command["argv"][2] == "model"


def test_export_archive_checks_positions_without_revalidating_domains():
    entry = {"name": "CUSTOM_ENV", "config_position": "env"}
    data = {"search_space": {"parameters": [entry], "constants": []}}
    best = {"candidate_id": "r1-a", "params": {"CUSTOM_ENV": "historical"}}
    assert load_module()._round_parameter_metadata(data, best) == {"CUSTOM_ENV": entry}


@pytest.mark.parametrize(
    "problem", ["missing_space", "missing_group", "duplicate", "missing_position", "missing_parameter"]
)
def test_export_rejects_incomplete_or_ambiguous_archive(problem):
    entry = {"name": "CUSTOM_ENV", "config_position": "env"}
    data = {"search_space": {"parameters": [entry], "constants": []}}
    best = {"candidate_id": "r1-a", "params": {"CUSTOM_ENV": "historical"}}
    if problem == "missing_space":
        data.pop("search_space")
    elif problem == "missing_group":
        data["search_space"].pop("constants")
    elif problem == "duplicate":
        data["search_space"]["constants"].append(dict(entry))
    elif problem == "missing_position":
        entry.pop("config_position")
    else:
        data["search_space"]["parameters"] = []
    with pytest.raises(ValueError):
        load_module()._round_parameter_metadata(data, best)


def test_export_falls_back_to_live_context_when_winning_archive_is_missing(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[vllm.command]\nmodel="model"\nserved_model_name="served"\n')
    (tmp_path / "context.json").write_text(
        json.dumps(
            {
                "engine": "vllm",
                "source": {"config_toml": str(config)},
                "search_space": {"parameters": [{"name": "tp", "config_position": "run"}], "constants": []},
            }
        )
    )
    (tmp_path / "results.round-1.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "candidate_id": "r1-a",
                        "fitness": 1,
                        "slo_violated": False,
                        "params": {"tp": 8},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    load_module().main()
    assert "context.round-1.json 缺失，回退读取 context.json" in capsys.readouterr().err
    handoff = json.loads((tmp_path / "optimizer_config_handoff.json").read_text())
    assert handoff["parameter_metadata"] == {"tp": {"name": "tp", "config_position": "run"}}
    assert json.loads((tmp_path / "serve_command.json").read_text())["argv"]


@pytest.mark.parametrize("archive", ["{", "[]", "{}"])
def test_export_does_not_fall_back_when_winning_archive_is_invalid(tmp_path, monkeypatch, archive):
    config = tmp_path / "config.toml"
    config.write_text('[vllm.command]\nmodel="model"\nserved_model_name="served"\n')
    (tmp_path / "context.json").write_text(
        json.dumps(
            {
                "engine": "vllm",
                "source": {"config_toml": str(config)},
                "search_space": {"parameters": [{"name": "tp", "config_position": "run"}], "constants": []},
            }
        )
    )
    (tmp_path / "context.round-1.json").write_text(archive)
    (tmp_path / "results.round-1.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "candidate_id": "r1-a",
                        "fitness": 1,
                        "slo_violated": False,
                        "params": {"tp": 8},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        load_module().main()
    assert exc.value.code == 1
    assert not (tmp_path / "best_result.json").exists()
    assert not (tmp_path / "serve_command.json").exists()


def _write_minimal_run(tmp_path, trials):
    config = tmp_path / "config.toml"
    config.write_text('[vllm.command]\nmodel="model"\nserved_model_name="served"\n')
    (tmp_path / "context.round-1.json").write_text(
        json.dumps(
            {
                "engine": "vllm",
                "source": {"config_toml": str(config)},
                "search_space": {"parameters": [{"name": "tp", "config_position": "run"}], "constants": []},
            }
        )
    )
    (tmp_path / "results.round-1.json").write_text(json.dumps({"trials": trials}))


def test_export_fails_when_all_trials_failed(tmp_path, monkeypatch):
    # 回归：全部 trial 失败（error 非空、fitness 非有限）时不得导出"最佳配置"，
    # 也不生成 best_result / handoff / serve_command 等部署交接产物。
    _write_minimal_run(
        tmp_path,
        [
            {"candidate_id": "r1-a", "fitness": float("inf"), "error": "startup failed", "params": {"tp": 8}},
            {"candidate_id": "r1-a", "fitness": float("inf"), "error": "startup failed", "params": {"tp": 4}},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        load_module().main()
    assert exc.value.code
    assert not (tmp_path / "best_result.json").exists()
    assert not (tmp_path / "serve_command.json").exists()
    assert not (tmp_path / "optimizer_config_handoff.json").exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_export_skips_failed_and_non_finite_fitness(tmp_path, monkeypatch, bad):
    _write_minimal_run(
        tmp_path,
        [
            {"candidate_id": "r1-a", "fitness": bad, "params": {"tp": 8}},
            {"candidate_id": "r1-a", "fitness": 2, "params": {"tp": 4}},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["export_best_config.py", "--run-dir", str(tmp_path)])
    load_module().main()
    best = json.loads((tmp_path / "best_result.json").read_text())
    assert best["fitness"] == 2
    assert best["params"] == {"tp": 4}
