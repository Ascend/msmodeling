import pytest
from cli.inference import text_generate


def test_export_empirical_metrics_requires_profiling(monkeypatch):
    monkeypatch.setattr(text_generate, "check_dependencies", lambda: None)
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
