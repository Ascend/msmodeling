"""Regression coverage for dispatch_ffn_combine default behavior."""

from unittest.mock import MagicMock, patch

from cli.inference import text_generate as cli_text_generate
from tensor_cast import config as tc_config
from tensor_cast.core import model_builder
from tensor_cast.core.user_config import UserInputConfig
from tests.helpers.cli_runner import run_cli_main, run_module_main


def test_user_input_config_enables_dfc_by_default():
    orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
    try:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True

        assert UserInputConfig().enable_dispatch_ffn_combine is True
        assert tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine is True
    finally:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = orig_dfc


def test_user_input_config_uses_internal_dfc_default():
    orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
    try:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = False

        assert UserInputConfig().enable_dispatch_ffn_combine is False
    finally:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = orig_dfc


def test_cli_main_enables_dfc_by_default_without_cli_flag(monkeypatch):
    captured = {}

    class FakeMetrics:
        def print_info(self):
            captured["printed"] = True

    class FakeModelRunner:
        def __init__(self, user_input):
            captured["user_input"] = user_input

        def run_inference(self, generate_inputs_func):
            captured["generate_inputs_func"] = generate_inputs_func
            return FakeMetrics()

    orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
    try:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
        monkeypatch.setattr(cli_text_generate, "print_logo", lambda: None)
        monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", FakeModelRunner)

        result = run_cli_main(
            cli_text_generate.main,
            ["Qwen/Qwen3-32B", "--num-queries", "1", "--query-length", "1"],
            prog="text_generate",
        )

        assert result.returncode == 0
        assert captured["user_input"].enable_dispatch_ffn_combine is True
        assert captured["printed"] is True
    finally:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = orig_dfc


def test_cli_rejects_dispatch_ffn_combine_flag():
    result = run_module_main(
        "cli.inference.text_generate",
        [
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "1",
            "--enable-dispatch-ffn-combine",
        ],
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --enable-dispatch-ffn-combine" in result.stderr


def test_build_model_applies_user_dfc_config_before_compile():
    captured = {}
    fake_model = MagicMock()
    fake_model.is_vl_model = False
    fake_model_config = MagicMock()
    fake_model_config.parallel_config.pipeline_parallel_size = 1

    def fake_compile(model, **kwargs):
        captured["dfc"] = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
        captured["kwargs"] = kwargs
        return model

    user_input = UserInputConfig(
        device="TEST_DEVICE",
        model_id="Qwen/Qwen3-32B",
        do_compile=True,
        enable_dispatch_ffn_combine=False,
    )
    orig_dfc = tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine
    try:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = True
        with (
            patch.object(model_builder.ConfigResolver, "resolve", return_value=fake_model_config),
            patch.object(model_builder, "TransformerModel", return_value=fake_model),
            patch.object(model_builder, "get_backend", return_value="fake_backend"),
            patch("torch.compile", side_effect=fake_compile),
        ):
            result = model_builder.build_model(user_input)

        assert result is fake_model
        assert captured["dfc"] is False
        assert captured["kwargs"]["backend"] == "fake_backend"
    finally:
        tc_config.compilation.fusion_patterns.enable_dispatch_ffn_combine = orig_dfc
