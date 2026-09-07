"""Tests for json_adapter --check (bundle drift guard, RFC §3.8)."""

from __future__ import annotations

import json
import shutil

import pytest
from services.json_adapter import BUNDLED_MODULES, check_module
from services.schema_registry import generate_form_configs


@pytest.fixture()
def bundle_copy(tmp_path):
    """Generate a fresh bundle and return a writable copy.

    The JSON bundle files are intentionally gitignored (generated at build time),
    so we generate them first, then copy to a temp dir for test isolation.
    """
    # Generate fresh bundle to a staging dir
    staging = tmp_path / "staging"
    generate_form_configs(staging)
    # Copy to a separate dir for test mutation
    dst = tmp_path / "config"
    shutil.copytree(staging, dst)
    return dst


@pytest.mark.parametrize("module_id", BUNDLED_MODULES)
def test_bundle_in_sync(module_id, bundle_copy):
    """The committed bundle matches what the registry generates (CI guard)."""
    assert check_module(module_id, bundle_copy) == []


def test_detects_field_content_drift(bundle_copy):
    p = bundle_copy / "forms" / "text_generate.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for f in d["fields"]:
        if f["id"] == "num-queries":
            f["default"] = 999  # simulated stale bundle
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    diffs = check_module("text_generate", bundle_copy)
    assert any("num-queries" in d and "content differs" in d for d in diffs)


def test_detects_missing_field(bundle_copy):
    p = bundle_copy / "forms" / "video_generate.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["fields"] = [f for f in d["fields"] if f["id"] != "dtype"]
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    diffs = check_module("video_generate", bundle_copy)
    assert any("'dtype'" in d and "missing from bundle" in d for d in diffs)


def test_detects_stale_field(bundle_copy):
    p = bundle_copy / "forms" / "throughput_optimizer.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["fields"].append({"id": "ghost-field", "dataType": "string"})
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    diffs = check_module("throughput_optimizer", bundle_copy)
    assert any("'ghost-field'" in d and "not generated" in d for d in diffs)


def test_detects_missing_bundle_file(bundle_copy):
    (bundle_copy / "forms" / "text_generate.json").unlink()
    diffs = check_module("text_generate", bundle_copy)
    assert any("bundle file missing" in d for d in diffs)


def test_non_bundled_module_reports_unsupported():
    diffs = check_module("image_generate", None)
    assert any("no bundled form JSON" in d for d in diffs)
