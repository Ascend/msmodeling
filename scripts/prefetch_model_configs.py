#!/usr/bin/env python3
"""Prefetch model config files required by tests into a target cache directory."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
_HUB_ALLOWLIST_REL: Final = Path("tests/assets/hub_model_allowlist.txt")
_MODEL_ASSIGNMENT_NAME_HINTS: Final = ("MODEL", "model_id", "pretrained")
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

        sig = inspect.signature(self._snapshot_download)  # pylint: disable=no-member
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


def _hub_model_allowlist(scan_dir: Path) -> frozenset[str]:
    ids: set[str] = set()
    for base in (scan_dir, REPO_ROOT):
        path = base / _HUB_ALLOWLIST_REL
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.add(stripped)
    return frozenset(ids)


def _is_model_assignment_target(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _MODEL_ASSIGNMENT_NAME_HINTS)


def _candidate_from_literal(value: str, allowlist: frozenset[str]) -> str | None:
    candidate = value.strip()
    if candidate not in allowlist or not _looks_like_model_id(candidate):
        return None
    return candidate


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_from_python(path: Path, allowlist: frozenset[str]) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            literal = _string_literal(node.value)
            if literal is None:
                continue
            candidate = _candidate_from_literal(literal, allowlist)
            if candidate is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_model_assignment_target(target.id):
                    found.add(candidate)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = _candidate_from_literal(node.value, allowlist)
            if candidate is not None:
                found.add(candidate)
    return found


def _collect_from_json(path: Path, allowlist: frozenset[str]) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    found: set[str] = set()
    for value in _iter_string_values(data):
        candidate = _candidate_from_literal(value, allowlist)
        if candidate is not None:
            found.add(candidate)
    return found


def collect_model_ids(scan_dir: Path) -> list[str]:
    allowlist = _hub_model_allowlist(scan_dir)
    if not allowlist:
        return []
    model_ids: set[str] = set()
    for path in scan_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            try:
                rel = path.relative_to(scan_dir).as_posix()
            except ValueError:
                continue
        if rel.startswith(("tests/.ci/", "tests/assets/cache/", "scripts/helpers/")):
            continue
        if path.suffix == ".py":
            model_ids.update(_collect_from_python(path, allowlist))
        elif path.suffix == ".json":
            model_ids.update(_collect_from_json(path, allowlist))
    return sorted(model_ids)


def _iter_string_values(data: Any) -> Iterator[str]:
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _iter_string_values(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_string_values(item)


def _build_prefetchers() -> list[ConfigPrefetcher]:
    prefetchers: list[ConfigPrefetcher] = []
    with contextlib.suppress(ImportError):
        prefetchers.append(HuggingFacePrefetcher())
    with contextlib.suppress(ImportError):
        prefetchers.append(ModelScopePrefetcher())
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

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
        logger.error("scan dir not found: %s", scan_dir)
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
            logger.error("No model id discovered from tests scan.")
            return 1

        logger.info("Discovered %d model ids.", len(model_ids))
        if args.dry_run:
            results = [PrefetchResult(model_id=mid, source="dry-run", success=True) for mid in model_ids]
            for r in results:
                logger.info("[DRY-RUN] %s", r.model_id)
        else:
            prefetchers = _build_prefetchers()
            if not prefetchers:
                logger.error("Neither transformers nor modelscope is installed.")
                return 1
            results = _prefetch_all(model_ids, prefetchers)
            for r in results:
                if r.success:
                    logger.info("[OK] %s (%s)", r.model_id, r.source)
                else:
                    logger.error("[FAIL] %s: %s", r.model_id, r.error)

    manifest_path = _write_manifest(dest_dir, scan_dir, results)
    logger.info("manifest written: %s", manifest_path)

    failures = [item for item in results if not item.success]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
