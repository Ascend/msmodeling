"""Tests for sig-review review_api — path matching, SIG routing, ownership loading."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("review_api.py")
OWNERSHIP_PATH = Path(__file__).parent.parent / "sig_ownership.json"


def load_module():
    spec = importlib.util.spec_from_file_location("review_api", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# path_matches
# ---------------------------------------------------------------------------


def test_path_matches_directory_prefix():
    mod = load_module()
    assert mod.path_matches("tensor_cast/ops/binary_op.py", "tensor_cast/ops/")
    assert mod.path_matches("tensor_cast/ops/__init__.py", "tensor_cast/ops/")
    assert not mod.path_matches("tensor_cast/ops_extras/x.py", "tensor_cast/ops/")


def test_path_matches_exact_file():
    mod = load_module()
    assert mod.path_matches("tensor_cast/config.py", "tensor_cast/config.py")
    assert mod.path_matches("tensor_cast/config/sub.py", "tensor_cast/config")
    assert not mod.path_matches("tensor_cast/config_v2.py", "tensor_cast/config.py")
    assert not mod.path_matches("tensor_cast/config.py", "tensor_cast/config")


# ---------------------------------------------------------------------------
# route_to_sig — explicit matching
# ---------------------------------------------------------------------------


def _sample_sigs():
    return [
        {
            "name": "模型适配",
            "chair": "alice",
            "reviewers": ["bob"],
            "approver": "carol",
            "paths": [
                "tensor_cast/transformers/builtin_model/",
                "tensor_cast/ops/",
            ],
        },
        {
            "name": "实测算子查询",
            "chair": "dave",
            "reviewers": ["eve"],
            "approver": "frank",
            "paths": [
                "tensor_cast/performance_model/",
            ],
        },
    ]


def test_route_to_sig_explicit_match():
    mod = load_module()
    result = mod.route_to_sig(
        ["tensor_cast/ops/add.py", "tensor_cast/performance_model/emp.py"],
        _sample_sigs(),
    )
    assert "模型适配" in result
    assert "实测算子查询" in result
    assert result["模型适配"]["matched_paths"] == ["tensor_cast/ops/add.py"]
    assert result["实测算子查询"]["matched_paths"] == ["tensor_cast/performance_model/emp.py"]
    assert result["模型适配"]["match_type"] == "explicit"


def test_route_to_sig_longest_prefix_wins():
    mod = load_module()
    sigs = [
        {"name": "outer", "paths": ["tensor_cast/"]},
        {"name": "inner", "paths": ["tensor_cast/ops/"]},
    ]
    result = mod.route_to_sig(["tensor_cast/ops/x.py"], sigs)
    assert "inner" in result
    assert "outer" not in result


def test_route_to_sig_unmatched_files():
    mod = load_module()
    result = mod.route_to_sig(["unknown/path.py"], _sample_sigs())
    assert "_unmatched" in result
    assert result["_unmatched"]["matched_paths"] == ["unknown/path.py"]
    assert result["_unmatched"]["sig"] is None


# ---------------------------------------------------------------------------
# route_to_sig — fallback
# ---------------------------------------------------------------------------


def test_route_to_sig_fallback_match():
    mod = load_module()
    fallback = {"tensor_cast/": "模型适配", "cli/": "模型适配"}
    result = mod.route_to_sig(
        ["tensor_cast/new_module.py"],
        _sample_sigs(),
        fallback_sigs=fallback,
    )
    assert "模型适配" in result
    assert result["模型适配"]["match_type"] == "fallback"


def test_route_to_sig_fallback_longest_prefix_wins():
    mod = load_module()
    sigs = [
        {"name": "A", "paths": ["tensor_cast/ops/"]},
        {"name": "B", "paths": []},
    ]
    fallback = {"tensor_cast/": "B", "tensor_cast/ops/": "A"}
    result = mod.route_to_sig(["tensor_cast/ops/x.py"], sigs, fallback_sigs=fallback)
    # explicit match takes priority over fallback
    assert "A" in result
    assert result["A"]["match_type"] == "explicit"


def test_route_to_sig_no_fallback_when_explicit_matches():
    mod = load_module()
    fallback = {"tensor_cast/": "实测算子查询"}
    result = mod.route_to_sig(
        ["tensor_cast/ops/add.py"],
        _sample_sigs(),
        fallback_sigs=fallback,
    )
    assert "模型适配" in result
    assert "实测算子查询" not in result


# ---------------------------------------------------------------------------
# load_ownership
# ---------------------------------------------------------------------------


def test_load_ownership_reads_valid_json(tmp_path):
    mod = load_module()
    data = {"sigs": [{"name": "test", "paths": ["a/"]}], "fallback_sigs": {}}
    f = tmp_path / "ownership.json"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loaded = mod.load_ownership(str(f))
    assert loaded["sigs"][0]["name"] == "test"


def test_load_ownership_dies_on_missing_file(tmp_path):
    mod = load_module()
    with pytest.raises(SystemExit) as exc_info:
        mod.load_ownership(str(tmp_path / "nonexistent.json"))
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# sig_ownership.json consistency (uses the real config file)
# ---------------------------------------------------------------------------


def test_ownership_json_is_valid():
    data = json.loads(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    assert "sigs" in data
    assert isinstance(data["sigs"], list)
    assert len(data["sigs"]) > 0


def test_ops_routes_to_model_adaptation_sig():
    mod = load_module()
    data = json.loads(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    result = mod.route_to_sig(
        ["tensor_cast/ops/binary_op.py", "tensor_cast/ops/__init__.py"],
        data["sigs"],
        data.get("fallback_sigs", {}),
    )
    assert "模型适配" in result
    assert "实测算子查询" not in result


def test_performance_model_routes_to_empirical_op_query_sig():
    mod = load_module()
    data = json.loads(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    result = mod.route_to_sig(
        ["tensor_cast/performance_model/empirical.py"],
        data["sigs"],
        data.get("fallback_sigs", {}),
    )
    assert "实测算子查询" in result


def test_no_duplicate_paths_across_sigs():
    data = json.loads(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    all_paths = []
    for sig in data["sigs"]:
        all_paths.extend(sig.get("paths", []))
    duplicates = [p for p in all_paths if all_paths.count(p) > 1]
    assert not duplicates, f"Duplicate paths: {set(duplicates)}"
