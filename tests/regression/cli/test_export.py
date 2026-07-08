import pytest
from cli.inference import text_generate


def test_text_generate_help_describes_profiling_interpolation(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["text_generate", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        text_generate.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "with optional interpolation" in normalized_help
    assert "--disable-profiling-interpolation" in normalized_help
    assert "exact and partial profiling matches only" in normalized_help


def test_export_empirical_metrics_requires_profiling(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--export-empirical-metrics",
            "metrics.json",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        text_generate.main()
    assert exc_info.value.code == 2


def test_text_generate_main_propagates_disable_interpolation(monkeypatch):
    captured = {}

    class FakeUserInputConfig:
        @classmethod
        def from_args(cls, args):
            captured["disable_profiling_interpolation"] = args.disable_profiling_interpolation
            return object()

    class FakeMetrics:
        def print_info(self):
            captured["printed"] = True

    class FakeModelRunner:
        def __init__(self, user_input):
            captured["user_input"] = user_input

        def run_inference(self, generate_inputs_func):
            captured["generate_inputs_func"] = generate_inputs_func
            return FakeMetrics()

    monkeypatch.setattr(text_generate, "print_logo", lambda: None)
    monkeypatch.setattr("tensor_cast.core.user_config.UserInputConfig", FakeUserInputConfig)
    monkeypatch.setattr("tensor_cast.core.model_runner.ModelRunner", FakeModelRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--disable-profiling-interpolation",
        ],
    )

    text_generate.main()

    assert captured["disable_profiling_interpolation"] is True
    assert captured["printed"] is True
