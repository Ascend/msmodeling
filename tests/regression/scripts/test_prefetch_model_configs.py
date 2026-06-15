"""Regression tests for scripts.prefetch_model_configs."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.prefetch_model_configs as prefetch
from tests.helpers.cli_runner import run_module_main


@pytest.fixture(autouse=True)
def _clean_prefetch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "HF_HOME",
        "TORCH_HOME",
        "MODELSCOPE_CACHE",
        "MSMODELING_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-32B", True),
        ("deepseek-ai/DeepSeek-R1", True),
        (" tests/foo", False),
        ("tests/foo", False),
        ("tensor_cast/bar", False),
        ("http://example.com/model", False),
        ("org/model.json", False),
        ("a/model", False),
        ("org/b", False),
        ("org/model with space", False),
        (r"org\model", False),
    ],
)
def test_looks_like_model_id_filters_expected_values(model_id: str, expected: bool) -> None:
    assert prefetch._looks_like_model_id(model_id) is expected


def test_iter_string_values_walks_nested_dicts_and_lists() -> None:
    data = {
        "a": "Qwen/Qwen3-32B",
        "b": ["deepseek-ai/DeepSeek-R1", {"c": "not/a/file.json"}],
        "d": 123,
    }

    assert list(prefetch._iter_string_values(data)) == [
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-R1",
        "not/a/file.json",
    ]


def test_collect_from_python_returns_empty_for_syntax_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text('x = "unterminated', encoding="utf-8")

    assert prefetch._collect_from_python(path, frozenset()) == set()


def test_collect_from_json_returns_empty_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{bad json", encoding="utf-8")

    assert prefetch._collect_from_json(path, frozenset()) == set()


def test_collect_model_ids_discovers_from_python_and_json_and_skips_ignored_paths(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    (scan_dir / "suite").mkdir(parents=True)
    (scan_dir / "tests" / ".ci").mkdir(parents=True)
    (scan_dir / "tests" / "assets" / "cache").mkdir(parents=True)
    (scan_dir / "scripts" / "helpers").mkdir(parents=True)

    (scan_dir / "suite" / "case.py").write_text(
        "\n".join(
            [
                'MODEL_A = "Qwen/Qwen3-32B"',
                'MODEL_B = "deepseek-ai/DeepSeek-R1"',
                'IGNORE_A = "tests/fixture"',
                'IGNORE_B = "org/model.json"',
            ]
        ),
        encoding="utf-8",
    )
    (scan_dir / "suite" / "case.json").write_text(
        json.dumps(
            {
                "models": ["Qwen/Qwen3-32B", {"id": "THUDM/GLM-4-9B"}],
                "nested": {"remote": "deepseek-ai/DeepSeek-R1"},
            }
        ),
        encoding="utf-8",
    )
    (scan_dir / "tests" / ".ci" / "ignored.py").write_text('MODEL = "ignored/Model"', encoding="utf-8")
    (scan_dir / "tests" / "assets" / "cache" / "ignored.json").write_text(
        json.dumps({"model": "ignored/CacheModel"}),
        encoding="utf-8",
    )
    (scan_dir / "scripts" / "helpers" / "ignored.py").write_text(
        'MODEL = "ignored/HelperModel"',
        encoding="utf-8",
    )

    assert prefetch.collect_model_ids(scan_dir) == [
        "Qwen/Qwen3-32B",
        "THUDM/GLM-4-9B",
        "deepseek-ai/DeepSeek-R1",
    ]


def test_env_overrides_activate_sets_and_restores_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "old-hf")
    monkeypatch.setenv("MSMODELING_OFFLINE", "1")

    overrides = prefetch.EnvOverrides(
        hf_home="new-hf",
        torch_home="new-torch",
        modelscope_cache="new-ms",
    )

    with overrides.activate():
        assert prefetch.os.environ["HF_HOME"] == "new-hf"
        assert prefetch.os.environ["TORCH_HOME"] == "new-torch"
        assert prefetch.os.environ["MODELSCOPE_CACHE"] == "new-ms"
        assert prefetch.os.environ["MSMODELING_OFFLINE"] == "0"
        assert prefetch.os.environ["HF_HUB_OFFLINE"] == "0"
        assert prefetch.os.environ["TRANSFORMERS_OFFLINE"] == "0"
        assert prefetch.os.environ["HF_DATASETS_OFFLINE"] == "0"

    assert prefetch.os.environ["HF_HOME"] == "old-hf"
    assert "TORCH_HOME" not in prefetch.os.environ
    assert "MODELSCOPE_CACHE" not in prefetch.os.environ
    assert prefetch.os.environ["MSMODELING_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in prefetch.os.environ
    assert "TRANSFORMERS_OFFLINE" not in prefetch.os.environ
    assert "HF_DATASETS_OFFLINE" not in prefetch.os.environ


def test_prefetch_result_to_dict_serializes_all_fields() -> None:
    result = prefetch.PrefetchResult(model_id="org/model", source="huggingface", success=False, error="boom")

    assert result.to_dict() == {
        "model_id": "org/model",
        "source": "huggingface",
        "success": False,
        "error": "boom",
    }


def test_huggingface_prefetcher_fetch_success_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _from_pretrained(model_id: str, **kwargs: object) -> object:
        calls.append((model_id, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )

    fetcher = prefetch.HuggingFacePrefetcher()
    result = fetcher.fetch("Qwen/Qwen3-32B")

    assert result == prefetch.PrefetchResult(
        model_id="Qwen/Qwen3-32B",
        source="huggingface",
        success=True,
    )
    assert calls == [("Qwen/Qwen3-32B", {})]


def test_huggingface_prefetcher_fetch_retries_with_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _from_pretrained(model_id: str, **kwargs: object) -> object:
        calls.append((model_id, kwargs))
        if len(calls) == 1:
            raise RuntimeError("set trust_remote_code=True")
        return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )

    fetcher = prefetch.HuggingFacePrefetcher()
    result = fetcher.fetch("deepseek-ai/DeepSeek-R1")

    assert result.success is True
    assert calls == [
        ("deepseek-ai/DeepSeek-R1", {}),
        ("deepseek-ai/DeepSeek-R1", {"trust_remote_code": True}),
    ]


@pytest.mark.parametrize(
    ("parameter_name", "expected_key"),
    [
        ("ignore_file_pattern", "ignore_file_pattern"),
        ("ignore_patterns", "ignore_patterns"),
    ],
)
def test_modelscope_build_snapshot_kwargs_supports_both_parameter_names(
    parameter_name: str,
    expected_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _snapshot_download(model_id: str, **kwargs: object) -> str:
        return f"/tmp/{model_id}"

    if parameter_name == "ignore_file_pattern":

        def _snapshot_download(model_id: str, ignore_file_pattern: object = None) -> str:
            return f"/tmp/{model_id}"
    else:

        def _snapshot_download(model_id: str, ignore_patterns: object = None) -> str:
            return f"/tmp/{model_id}"

    monkeypatch.setitem(
        sys.modules,
        "modelscope",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *_a, **_kw: object()),
            snapshot_download=_snapshot_download,
        ),
    )

    fetcher = prefetch.ModelScopePrefetcher()

    assert fetcher._build_snapshot_kwargs("Qwen/Qwen3-32B") == {
        expected_key: prefetch._MODELSCOPE_WEIGHT_IGNORE_PATTERNS,
    }


def test_modelscope_prefetcher_fetch_retries_with_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot_calls: list[tuple[str, dict[str, object]]] = []
    config_calls: list[tuple[str, dict[str, object]]] = []

    def _snapshot_download(model_id: str, ignore_patterns: object = None) -> str:
        snapshot_calls.append((model_id, {"ignore_patterns": ignore_patterns}))
        return f"/cache/{model_id}"

    def _from_pretrained(local_dir: str, **kwargs: object) -> object:
        config_calls.append((local_dir, kwargs))
        if len(config_calls) == 1:
            raise RuntimeError("please enable trust_remote_code")
        return object()

    monkeypatch.setitem(
        sys.modules,
        "modelscope",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained),
            snapshot_download=_snapshot_download,
        ),
    )

    fetcher = prefetch.ModelScopePrefetcher()
    result = fetcher.fetch("THUDM/GLM-4-9B")

    assert result == prefetch.PrefetchResult(
        model_id="THUDM/GLM-4-9B",
        source="modelscope",
        success=True,
    )
    assert snapshot_calls == [
        (
            "THUDM/GLM-4-9B",
            {"ignore_patterns": prefetch._MODELSCOPE_WEIGHT_IGNORE_PATTERNS},
        )
    ]
    assert config_calls == [
        ("/cache/THUDM/GLM-4-9B", {}),
        ("/cache/THUDM/GLM-4-9B", {"trust_remote_code": True}),
    ]


class _FailingPrefetcher:
    def __init__(self, message: str) -> None:
        self.message = message

    def fetch(self, _model_id: str) -> prefetch.PrefetchResult:
        raise RuntimeError(self.message)


class _SuccessfulPrefetcher:
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: list[str] = []

    def fetch(self, model_id: str) -> prefetch.PrefetchResult:
        self.calls.append(model_id)
        return prefetch.PrefetchResult(model_id=model_id, source=self.source, success=True)


def test_try_prefetch_returns_first_successful_result() -> None:
    ok = _SuccessfulPrefetcher("huggingface")

    result = prefetch._try_prefetch(
        "Qwen/Qwen3-32B",
        [_FailingPrefetcher("first failed"), ok],
    )

    assert result == prefetch.PrefetchResult(
        model_id="Qwen/Qwen3-32B",
        source="huggingface",
        success=True,
    )
    assert ok.calls == ["Qwen/Qwen3-32B"]


def test_try_prefetch_returns_unresolved_when_all_prefetchers_fail() -> None:
    result = prefetch._try_prefetch(
        "Qwen/Qwen3-32B",
        [_FailingPrefetcher("first failed"), _FailingPrefetcher("last failed")],
    )

    assert result == prefetch.PrefetchResult(
        model_id="Qwen/Qwen3-32B",
        source="unresolved",
        success=False,
        error="last failed",
    )


def test_prefetch_all_preserves_input_order() -> None:
    ok = _SuccessfulPrefetcher("huggingface")

    results = prefetch._prefetch_all(
        ["Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-R1"],
        [ok],
    )

    assert [item.model_id for item in results] == ["Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-R1"]
    assert ok.calls == ["Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-R1"]


def test_build_prefetchers_skips_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefetch, "HuggingFacePrefetcher", lambda: "hf")

    def _raise_import_error() -> str:
        raise ImportError("modelscope missing")

    monkeypatch.setattr(prefetch, "ModelScopePrefetcher", _raise_import_error)

    assert prefetch._build_prefetchers() == ["hf"]


def test_write_manifest_persists_results_as_json(tmp_path: Path) -> None:
    results = [
        prefetch.PrefetchResult(model_id="Qwen/Qwen3-32B", source="dry-run", success=True),
        prefetch.PrefetchResult(model_id="bad/model", source="unresolved", success=False, error="boom"),
    ]

    manifest_path = prefetch._write_manifest(tmp_path, tmp_path / "scan", results)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["dest_dir"] == str(tmp_path)
    assert manifest["scan_dir"] == str(tmp_path / "scan")
    assert manifest["models"] == [item.to_dict() for item in results]


def test_main_returns_2_when_scan_dir_is_missing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="scripts.prefetch_model_configs"):
        result = run_module_main(
            "scripts.prefetch_model_configs",
            [
                "--scan-dir",
                str(tmp_path / "missing"),
                "--dest-dir",
                str(tmp_path / "cache"),
            ],
        )

    assert result.returncode == 2
    assert f"scan dir not found: {(tmp_path / 'missing').resolve()}" in caplog.text


def test_main_dry_run_writes_manifest_and_returns_zero(tmp_path: Path) -> None:
    scan_dir = tmp_path / "tests"
    scan_dir.mkdir()
    (scan_dir / "case.py").write_text('MODEL = "Qwen/Qwen3-32B"', encoding="utf-8")

    dest_dir = tmp_path / "cache"
    result = run_module_main(
        "scripts.prefetch_model_configs",
        [
            "--scan-dir",
            str(scan_dir),
            "--dest-dir",
            str(dest_dir),
            "--dry-run",
        ],
    )

    manifest = json.loads((dest_dir / "model_config_manifest.json").read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert manifest["models"] == [
        {
            "model_id": "Qwen/Qwen3-32B",
            "source": "dry-run",
            "success": True,
            "error": "",
        }
    ]


def test_main_returns_1_when_no_model_ids_are_discovered(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_dir = tmp_path / "tests"
    scan_dir.mkdir()
    (scan_dir / "case.py").write_text('MODEL = "tests/not-a-model"', encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="scripts.prefetch_model_configs"):
        result = run_module_main(
            "scripts.prefetch_model_configs",
            [
                "--scan-dir",
                str(scan_dir),
                "--dest-dir",
                str(tmp_path / "cache"),
            ],
        )

    assert result.returncode == 1
    assert "No model id discovered from tests scan." in caplog.text


def test_main_returns_1_when_no_prefetchers_are_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scan_dir = tmp_path / "tests"
    scan_dir.mkdir()
    (scan_dir / "case.py").write_text('MODEL = "Qwen/Qwen3-32B"', encoding="utf-8")
    monkeypatch.setattr(prefetch, "_build_prefetchers", list)

    with caplog.at_level(logging.ERROR, logger="scripts.prefetch_model_configs"):
        result = run_module_main(
            "scripts.prefetch_model_configs",
            [
                "--scan-dir",
                str(scan_dir),
                "--dest-dir",
                str(tmp_path / "cache"),
            ],
        )

    assert result.returncode == 1
    assert "Neither transformers nor modelscope is installed." in caplog.text


def test_main_writes_manifest_and_returns_1_when_prefetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_dir = tmp_path / "tests"
    scan_dir.mkdir()
    (scan_dir / "case.py").write_text('MODEL = "Qwen/Qwen3-32B"', encoding="utf-8")
    dest_dir = tmp_path / "cache"

    monkeypatch.setattr(prefetch, "_build_prefetchers", lambda: [object()])
    monkeypatch.setattr(
        prefetch,
        "_prefetch_all",
        lambda model_ids, prefetchers: [
            prefetch.PrefetchResult(model_id=model_ids[0], source="unresolved", success=False, error="boom")
        ],
    )

    result = run_module_main(
        "scripts.prefetch_model_configs",
        [
            "--scan-dir",
            str(scan_dir),
            "--dest-dir",
            str(dest_dir),
        ],
    )

    manifest = json.loads((dest_dir / "model_config_manifest.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert manifest["models"] == [
        {
            "model_id": "Qwen/Qwen3-32B",
            "source": "unresolved",
            "success": False,
            "error": "boom",
        }
    ]
