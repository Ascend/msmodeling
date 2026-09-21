# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

from optix.config.custom_command import (
    CONTAINER_FLAGS,
    JSON_SUBKEY_MAP,
    AisBenchCommand,
    AisBenchCommandConfig,
    VllmBenchmarkCommand,
    VllmBenchmarkCommandConfig,
    VllmCommand,
    VllmCommandConfig,
    param_to_cli_flag,
)


def _container_dict(cmd: list[str], flag: str) -> dict[str, Any]:
    """Return the merged JSON dict carried by ``flag`` in a rendered command."""
    return json.loads(cmd[cmd.index(flag) + 1])


def _require_resolve_mindie_argv() -> Callable[[Mapping[str, str]], list[str]]:
    import optix.deploy_env as deploy_env_module

    resolve = getattr(deploy_env_module, "resolve_mindie_argv", None)
    assert resolve is not None, "resolve_mindie_argv is not implemented in optix.deploy_env"
    return cast("Callable[[Mapping[str, str]], list[str]]", resolve)


class TestVllmCommand:
    def test_init_success_without_which(self) -> None:
        config = VllmCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            others="",
        )
        command = VllmCommand(config)
        assert command.command_config == config

    def test_command_property(self) -> None:
        config = VllmCommandConfig(
            host="127.0.0.1",
            port="8080",
            model="/path/to/model",
            served_model_name="my-model",
            others="--gpu-memory-utilization 0.9",
        )
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "vllm"
        assert "serve" in cmd
        assert "/path/to/model" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    def test_command_no_others(self) -> None:
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        cmd_obj = VllmCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "vllm"
        assert "--gpu-memory-utilization" not in cmd

    def test_resolved_field_renders_capacity_flags(self) -> None:
        # Default PSO vLLM config declares MAX_NUM_BATCHED_TOKENS / MAX_NUM_SEQS
        # as config_position="run" fields; they must render as serve flags.
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        fields = [
            SimpleNamespace(name="MAX_NUM_BATCHED_TOKENS", value=8192),
            SimpleNamespace(name="MAX_NUM_SEQS", value=64),
        ]
        cmd = VllmCommand(config, fields).command
        assert "--max-num-batched-tokens" in cmd
        assert "8192" in cmd
        assert "--max-num-seqs" in cmd
        assert "64" in cmd

    def test_resolved_field_renders_compilation_config_json(self) -> None:
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        fields = [
            SimpleNamespace(name="COMPILATION_CONFIG", value='{"cudagraph_mode": "FULL_DECODE_ONLY"}'),
        ]

        cmd = VllmCommand(config, fields).command

        assert cmd.count("--compilation-config") == 1
        assert cmd[cmd.index("--compilation-config") + 1] == '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

    def test_resolved_field_empty_string_skips_compilation_config(self) -> None:
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        fields = [SimpleNamespace(name="COMPILATION_CONFIG", value="")]

        cmd = VllmCommand(config, fields).command

        assert "--compilation-config" not in cmd

    def test_resolved_field_empty_no_capacity_flags(self) -> None:
        # Command is fully field-driven: no resolved_field means no dynamic flags.
        # Guards against reintroducing hardcoded capacity flags (dual source).
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        cmd = VllmCommand(config, []).command
        assert "--max-num-batched-tokens" not in cmd
        assert "--max-num-seqs" not in cmd

    def test_benchmark_only_fields_never_render_as_serve_flags(self) -> None:
        # CONCURRENCY / REQUESTRATE are consumed by the benchmark command through
        # the $CONCURRENCY / $REQUESTRATE placeholders (rendered to
        # --max-concurrency / --request-rate in `vllm bench serve`). They are not
        # vLLM serve flags: rendering them here crashes `vllm serve`. The guard is
        # name-based (BENCHMARK_ONLY_FIELDS), so it must hold even when the field
        # is declared config_position="run" — position alone cannot save it.
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        fields = [
            SimpleNamespace(name="CONCURRENCY", value=100),
            SimpleNamespace(name="REQUESTRATE", value=20),
            SimpleNamespace(name="MAX_NUM_SEQS", value=64),
        ]

        cmd = VllmCommand(config, fields).command
        joined = " ".join(cmd).lower()

        assert "concurrency" not in joined
        assert "request-rate" not in joined
        # Not a blanket skip: an ordinary run field in the same batch must render.
        assert "--max-num-seqs" in cmd

    def test_container_flags_cover_every_json_subkey_container(self) -> None:
        # Structural guard for the whole class of bug: a JSON sub-key param renders
        # `--<container> '{...}'`, so its container MUST be in CONTAINER_FLAGS for the
        # merge to happen. A container missing from the tuple makes the sub-key emit a
        # *duplicate* flag, and _dedupe_plain_flags then keeps only the last dict —
        # silently dropping the search-side value (or the `others` base keys).
        # Fails whenever JSON_SUBKEY_MAP gains a container without CONTAINER_FLAGS.
        containers = {json_container for json_container, _ in JSON_SUBKEY_MAP.values()}
        missing = sorted("--" + container for container in containers if "--" + container not in CONTAINER_FLAGS)
        assert not missing, f"containers missing from CONTAINER_FLAGS: {missing}"

    def test_moe_additional_config_subkeys_merge_into_one_container(self) -> None:
        # vLLM Ascend MoE switches (enable_shared_expert_dp /
        # multistream_overlap_shared_expert) have NO top-level CLI flag; they are
        # sub-keys of --additional-config. Rendering them as top-level flags makes
        # `vllm serve` reject the launch (unrecognized arguments).
        config = VllmCommandConfig(
            host="localhost",
            port="8000",
            model="m",
            served_model_name="m",
            others="""--additional-config '{"enable_cpu_binding": true, "weight_nz_mode": 2}'""",
        )
        fields = [
            SimpleNamespace(name="enable_shared_expert_dp", value=True),
            SimpleNamespace(name="multistream_overlap_shared_expert", value=True),
        ]

        cmd = VllmCommand(config, fields).command

        assert "--enable-shared-expert-dp" not in cmd
        assert "--multistream-overlap-shared-expert" not in cmd
        # Single merged occurrence: argparse last-wins would otherwise drop one dict.
        assert cmd.count("--additional-config") == 1
        assert _container_dict(cmd, "--additional-config") == {
            "enable_cpu_binding": True,
            "weight_nz_mode": 2,
            "enable_shared_expert_dp": True,
            "multistream_overlap_shared_expert": True,
        }

    def test_additional_config_subkey_renders_without_others_baseline(self) -> None:
        # Same params with no preset baseline must still produce the container (an
        # empty `others` must not make the sub-key vanish) — and must not degrade to a
        # bogus top-level flag.
        config = VllmCommandConfig(host="localhost", port="8000", model="m", served_model_name="m", others="")
        fields = [
            SimpleNamespace(name="enable_shared_expert_dp", value=True),
            SimpleNamespace(name="multistream_overlap_shared_expert", value=True),
        ]

        cmd = VllmCommand(config, fields).command

        assert cmd.count("--additional-config") == 1
        assert _container_dict(cmd, "--additional-config") == {
            "enable_shared_expert_dp": True,
            "multistream_overlap_shared_expert": True,
        }

    def test_additional_config_subkey_false_renders_explicit_off(self) -> None:
        # additional_config.get(key, False) semantics: an explicit false must reach
        # vLLM as an off switch instead of being dropped (which would let the preset
        # baseline turn the feature back on).
        config = VllmCommandConfig(
            host="localhost",
            port="8000",
            model="m",
            served_model_name="m",
            others="""--additional-config '{"enable_shared_expert_dp": true}'""",
        )
        fields = [SimpleNamespace(name="enable_shared_expert_dp", value=False)]

        cmd = VllmCommand(config, fields).command

        assert _container_dict(cmd, "--additional-config") == {"enable_shared_expert_dp": False}

    def test_additional_config_subkey_maps_to_container_flag(self) -> None:
        # Runtime validation resolves a param back to its flag via param_to_cli_flag;
        # it must agree with the command side or a legal param is rejected as
        # "not supported by the current vllm runtime".
        assert param_to_cli_flag("enable_shared_expert_dp") == "additional-config"
        assert param_to_cli_flag("multistream_overlap_shared_expert") == "additional-config"


class TestVllmBenchmarkCommand:
    def test_init_success_without_which(self) -> None:
        config = VllmBenchmarkCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            dataset_name="sharegpt",
            num_prompts=100,
            result_dir="/tmp/results",
            others="",
        )
        cmd_obj = VllmBenchmarkCommand(config)
        assert cmd_obj.benchmark_command_config == config

    def test_command_property(self) -> None:
        config = VllmBenchmarkCommandConfig(
            host="localhost",
            port="8000",
            model="test-model",
            served_model_name="test",
            dataset_name="sharegpt",
            num_prompts=100,
            result_dir="/tmp/results",
            others="--extra-arg value",
        )
        cmd_obj = VllmBenchmarkCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "vllm"
        assert "bench" in cmd
        assert "serve" in cmd
        assert "--save-result" in cmd
        assert "--extra-arg" in cmd
        assert "value" in cmd
        assert "$CONCURRENCY" in cmd
        assert "$REQUESTRATE" in cmd


class TestAisBenchCommand:
    def test_init_success_without_which(self) -> None:
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="",
        )
        cmd_obj = AisBenchCommand(config)
        assert cmd_obj.aisbench_command_config == config

    def test_command_property(self) -> None:
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="--extra-arg value",
        )
        cmd_obj = AisBenchCommand(config)
        cmd = cmd_obj.command
        assert cmd[0] == "ais_bench"
        assert "--models" in cmd
        assert "model1" in cmd
        assert "--mode" in cmd
        assert "perf" in cmd
        assert "--work-dir" in cmd
        assert "/work" in cmd
        assert "--debug" in cmd
        assert "--extra-arg" in cmd
        assert "value" in cmd

    def test_command_no_others(self):
        """Test command when others is empty"""
        config = AisBenchCommandConfig(
            models="model1",
            mode="perf",
            work_dir="/work",
            others="",
        )
        cmd_obj = AisBenchCommand(config)
        cmd = cmd_obj.command
        assert "--extra-arg" not in cmd


class TestResolveMindieArgv:
    @patch("optix.deploy_env.os.path.isfile")
    def test_default_path_exists(self, mock_isfile: Any) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        mock_isfile.return_value = True
        argv = resolve_mindie_argv({})
        assert "mindieservice_daemon" in argv[0]

    @patch("optix.deploy_env.shutil.which")
    @patch("optix.deploy_env.os.path.isfile")
    def test_fallback_to_mindie_llm_server_in_deploy_path(
        self, mock_isfile: Any, mock_which: Any, tmp_path: Path
    ) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        deploy_bin = tmp_path / "deploy" / "bin"
        deploy_bin.mkdir(parents=True)
        mindie_server = deploy_bin / "mindie_llm_server"
        mindie_server.write_text("#!/bin/sh\n", encoding="utf-8")
        mindie_server.chmod(0o755)

        mock_isfile.return_value = False
        mock_which.return_value = str(mindie_server.resolve())
        env = {"PATH": f"{deploy_bin}:/usr/bin"}
        argv = resolve_mindie_argv(env)
        assert argv == [str(mindie_server.resolve())]

    @patch("optix.deploy_env.shutil.which")
    @patch("optix.deploy_env.os.path.isfile")
    def test_raises_when_no_command_found(self, mock_isfile: Any, mock_which: Any) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        mock_isfile.return_value = False
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError):
            resolve_mindie_argv({"PATH": "/usr/bin"})

    @patch("optix.deploy_env.os.path.isfile")
    def test_custom_install_path(self, mock_isfile: Any, tmp_path: Path) -> None:
        resolve_mindie_argv = _require_resolve_mindie_argv()
        custom_root = tmp_path / "custom" / "mindie-service"
        custom_daemon = custom_root / "bin" / "mindieservice_daemon"
        custom_daemon.parent.mkdir(parents=True)
        custom_daemon.write_text("#!/bin/sh\n", encoding="utf-8")
        custom_daemon.chmod(0o755)

        mock_isfile.side_effect = lambda path: Path(path) == custom_daemon
        env = {"MIES_INSTALL_PATH": str(custom_root), "PATH": "/usr/bin"}
        argv = resolve_mindie_argv(env)
        assert str(custom_daemon) in argv[0]
