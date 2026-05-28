#!/usr/bin/env python3
"""Prefetch model config files required by tests into a target cache directory."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SCAN_DIR = REPO_ROOT / "tests"
DEFAULT_DEST_DIR = REPO_ROOT / "tests" / "assets" / "cache"

_MODELSCOPE_WEIGHT_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.npz",
    "*.onnx",
    "*.gguf",
    "*.zip",
    "*.tar",
    "*.tar.gz",
]

_IGNORE_PREFIXES = (
    "tests/",
    "tensor_cast/",
    "serving_cast/",
    "trace/",
    "docs/",
    "./",
    "../",
    "http://",
    "https://",
)
_IGNORE_OWNERS = frozenset({"tests", "tensor_cast", "serving_cast", "trace", "docs", "web_ui"})
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_KNOWN_EXTENSIONS = frozenset({".yaml", ".yml", ".json", ".py", ".md", ".txt", ".csv"})


@dataclass(frozen=True, slots=True)
class PrefetchResult:
    model_id: str
    source: str
    success: bool
    error: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EnvOverrides:
    """Environment variables to set during prefetch operations."""

    hf_home: str
    torch_home: str
    modelscope_cache: str

    @contextlib.contextmanager
    def activate(self) -> Iterator[None]:
        env_keys = (
            "HF_HOME",
            "TORCH_HOME",
            "MODELSCOPE_CACHE",
            "MSMODELING_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        )
        old = {k: os.environ.get(k) for k in env_keys}
        os.environ["HF_HOME"] = self.hf_home
        os.environ["TORCH_HOME"] = self.torch_home
        os.environ["MODELSCOPE_CACHE"] = self.modelscope_cache
        os.environ["MSMODELING_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class ConfigPrefetcher(Protocol):
    """Strategy for downloading and loading a model config from one source."""

    def fetch(self, model_id: str) -> PrefetchResult: ...


class HuggingFacePrefetcher:
    def __init__(self) -> None:
        from transformers import AutoConfig

        self._AutoConfig = AutoConfig

    def fetch(self, model_id: str) -> PrefetchResult:
        try:
            self._AutoConfig.from_pretrained(model_id)
        except Exception as exc:
            if "trust_remote_code" not in str(exc):
                raise
            self._AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return PrefetchResult(model_id=model_id, source="huggingface", success=True)


class ModelScopePrefetcher:
    def __init__(self) -> None:
        import modelscope

        self._AutoConfig = modelscope.AutoConfig
        self._snapshot_download = modelscope.snapshot_download

    def fetch(self, model_id: str) -> PrefetchResult:
        kwargs = self._build_snapshot_kwargs(model_id)
        local_dir = self._snapshot_download(model_id, **kwargs)
        try:
            self._AutoConfig.from_pretrained(local_dir)
        except Exception as exc:
            if "trust_remote_code" not in str(exc):
                raise
            self._AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        return PrefetchResult(model_id=model_id, source="modelscope", success=True)

    def _build_snapshot_kwargs(self, model_id: str) -> dict[str, Any]:
        """Detect which ignore-pattern kwarg the installed modelscope accepts."""
        import inspect

        sig = inspect.signature(self._snapshot_download)
        if "ignore_file_pattern" in sig.parameters:
            return {"ignore_file_pattern": _MODELSCOPE_WEIGHT_IGNORE_PATTERNS}
        return {"ignore_patterns": _MODELSCOPE_WEIGHT_IGNORE_PATTERNS}


def _looks_like_model_id(value: str) -> bool:
    text = value.strip()
    if not text or "/" not in text or "\\" in text or " " in text:
        return False
    if not _MODEL_ID_PATTERN.fullmatch(text):
        return False
    if text.startswith(_IGNORE_PREFIXES):
        return False
    owner, _, name = text.partition("/")
    if len(owner) < 2 or len(name) < 2:
        return False
    if owner in _IGNORE_OWNERS:
        return False
    if not any(ch.isalpha() for ch in owner) or not any(ch.isalpha() for ch in name):
        return False
    if name.endswith(tuple(_KNOWN_EXTENSIONS)):
        return False
    # Reject names with a plausible file extension suffix
    dot_idx = name.rfind(".")
    if dot_idx != -1:
        suffix = name[dot_idx + 1 :]
        if suffix.isalpha() and len(suffix) <= 5:
            return False
    return True


def _iter_string_values(data: Any) -> Iterator[str]:
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _iter_string_values(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_string_values(item)


def _collect_from_python(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = node.value.strip()
            if _looks_like_model_id(candidate):
                found.add(candidate)
    return found


def _collect_from_json(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    found: set[str] = set()
    for value in _iter_string_values(data):
        if _looks_like_model_id(value):
            found.add(value.strip())
    return found


def collect_model_ids(scan_dir: Path) -> list[str]:
    model_ids: set[str] = set()
    for path in scan_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(("tests/.ci/", "tests/assets/cache/", "scripts/helpers/")):
            continue
        if path.suffix == ".py":
            model_ids.update(_collect_from_python(path))
        elif path.suffix == ".json":
            model_ids.update(_collect_from_json(path))
    return sorted(model_ids)


def _build_prefetchers() -> list[ConfigPrefetcher]:
    prefetchers: list[ConfigPrefetcher] = []
    try:
        prefetchers.append(HuggingFacePrefetcher())
    except ImportError:
        pass
    try:
        prefetchers.append(ModelScopePrefetcher())
    except ImportError:
        pass
    return prefetchers


def _prefetch_all(
    model_ids: Sequence[str],
    prefetchers: Sequence[ConfigPrefetcher],
) -> list[PrefetchResult]:
    results: list[PrefetchResult] = []
    for model_id in model_ids:
        result = _try_prefetch(model_id, prefetchers)
        results.append(result)
    return results


def _try_prefetch(
    model_id: str,
    prefetchers: Sequence[ConfigPrefetcher],
) -> PrefetchResult:
    last_error = ""
    for prefetcher in prefetchers:
        try:
            return prefetcher.fetch(model_id)
        except Exception as exc:
            last_error = str(exc)
    return PrefetchResult(
        model_id=model_id,
        source="unresolved",
        success=False,
        error=last_error,
    )


def _write_manifest(dest_dir: Path, scan_dir: Path, results: list[PrefetchResult]) -> Path:
    manifest = {
        "schema_version": 1,
        "scan_dir": str(scan_dir),
        "dest_dir": str(dest_dir),
        "models": [r.to_dict() for r in results],
    }
    manifest_path = dest_dir / "model_config_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest-dir",
        default=str(DEFAULT_DEST_DIR),
        help="Directory used as HF_HOME/TORCH_HOME/MODELSCOPE_CACHE for config prefetch.",
    )
    parser.add_argument(
        "--scan-dir",
        default=str(DEFAULT_SCAN_DIR),
        help="Directory to scan for model ids. Default: tests/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover model ids and write manifest without downloading configs.",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir).expanduser().resolve()
    scan_dir = Path(args.scan_dir).expanduser().resolve()
    if not scan_dir.exists():
        print(f"scan dir not found: {scan_dir}", file=sys.stderr)
        return 2

    dest_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = EnvOverrides(
        hf_home=str(dest_dir),
        torch_home=str(dest_dir),
        modelscope_cache=str(dest_dir),
    )

    with env_overrides.activate():
        model_ids = collect_model_ids(scan_dir)
        if not model_ids:
            print("No model id discovered from tests scan.", file=sys.stderr)
            return 1

        print(f"Discovered {len(model_ids)} model ids.", file=sys.stderr)
        if args.dry_run:
            results = [PrefetchResult(model_id=mid, source="dry-run", success=True) for mid in model_ids]
            for r in results:
                print(f"[DRY-RUN] {r.model_id}", file=sys.stderr)
        else:
            prefetchers = _build_prefetchers()
            if not prefetchers:
                print(
                    "Neither transformers nor modelscope is installed.",
                    file=sys.stderr,
                )
                return 1
            results = _prefetch_all(model_ids, prefetchers)
            for r in results:
                if r.success:
                    print(f"[OK] {r.model_id} ({r.source})", file=sys.stderr)
                else:
                    print(f"[FAIL] {r.model_id}: {r.error}", file=sys.stderr)

    manifest_path = _write_manifest(dest_dir, scan_dir, results)
    print(f"manifest written: {manifest_path}", file=sys.stderr)

    failures = [item for item in results if not item.success]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
