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
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optix.config import config as config_module
from optix.config.config import OptimizerConfigField
from optix.optimizer.optimizer import main


def _build_plugin_args(config):
    return ["optix", "-b", "vllm_benchmark", "-e", "vllm", "-c", config]


def _enter_common_patches(stack, tmp_path_custom_config):
    """Enter common stubs shared by the tests; returns (mock_logger, mock_pso) for assertions."""
    stack.enter_context(patch("optix.optimizer.register.register_ori_functions"))
    stack.enter_context(patch("optix.plugins.load_general_plugins"))
    stack.enter_context(patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: MagicMock()}))
    stack.enter_context(patch.dict("optix.optimizer.register.benchmarks", {"vllm_benchmark": lambda **kw: MagicMock()}))
    stack.enter_context(patch.object(config_module, "get_settings", return_value=MagicMock()))
    mock_logger = stack.enter_context(patch("optix.optimizer.optimizer.logger"))
    mock_pso = stack.enter_context(patch("optix.optimizer.optimizer.PSOOptimizer"))
    stack.enter_context(patch.object(sys, "argv", _build_plugin_args(tmp_path_custom_config)))
    return mock_logger, mock_pso


def test_plugin_main_with_missing_custom_config_returns(tmp_path):
    missing_config = tmp_path / "missing.toml"

    with ExitStack() as stack:
        mock_logger, mock_pso = _enter_common_patches(stack, str(missing_config))
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    _fmt, exc = mock_logger.error.call_args.args
    assert str(exc) == f"Custom config file not found: {missing_config.resolve()}"
    mock_pso.assert_not_called()


def test_plugin_main_with_invalid_custom_config_raises(tmp_path):
    custom_config = tmp_path / "invalid.toml"
    custom_config.write_text("invalid = [", encoding="utf-8")

    with ExitStack() as stack:
        mock_logger, _mock_pso = _enter_common_patches(stack, str(custom_config))
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    _fmt, exc = mock_logger.error.call_args.args
    assert "Invalid TOML config file" in str(exc)


def test_plugin_main_with_custom_config_registers_settings(tmp_path):
    custom_config = tmp_path / "custom.toml"
    custom_config.write_text("n_particles = 1\n", encoding="utf-8")
    target_field = (
        OptimizerConfigField(
            name="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            min=10,
            max=100,
            dtype="int",
        ),
    )
    settings = MagicMock()
    default_toml_file = Path("/default/config.toml")

    class DummySettings:
        model_config = {"toml_file": [default_toml_file], "env_prefix": "model_eval_state_"}

    class FakeSimulator:
        data_field = target_field

        def __init__(self, *args, **kwargs):
            pass

    class FakeBenchmark:
        data_field = ()

        def __init__(self, *args, **kwargs):
            pass

    with ExitStack() as stack:
        stack.enter_context(patch.object(config_module, "Settings", DummySettings))
        mock_register_settings = stack.enter_context(patch.object(config_module, "register_settings"))
        stack.enter_context(patch.object(config_module, "get_settings", return_value=settings))
        stack.enter_context(patch("optix.optimizer.register.register_ori_functions"))
        stack.enter_context(patch("optix.plugins.load_general_plugins"))
        stack.enter_context(patch.dict("optix.optimizer.register.simulates", {"vllm": FakeSimulator}))
        stack.enter_context(patch.dict("optix.optimizer.register.benchmarks", {"vllm_benchmark": FakeBenchmark}))
        # Deployment stack validation requires real executables and the deploy
        # config; stub them out for the test environment
        stack.enter_context(patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {})))
        stack.enter_context(patch("optix.deploy_env.emit_runtime_hints"))
        stack.enter_context(patch("optix.deploy_env.validate_deploy_stack"))
        stack.enter_context(patch("optix.optimizer.store.DataStorage"))
        stack.enter_context(patch("optix.optimizer.scheduler.Scheduler"))
        stack.enter_context(patch("optix.optimizer.experience_fine_tunning.FineTune"))
        pso = stack.enter_context(patch("optix.optimizer.optimizer.PSOOptimizer"))
        stack.enter_context(patch.object(sys, "argv", _build_plugin_args(str(custom_config))))

        main()

    mock_register_settings.assert_called_once()
    custom_settings = mock_register_settings.call_args.args[0]()
    assert custom_settings.model_config["toml_file"] == [default_toml_file, custom_config.resolve()]
    assert custom_settings.model_config["extra"] == "allow"
    pso.return_value.run_plugin.assert_called_once()
