#!/usr/bin/env python3
"""自检：R0 底座裁决门禁 + 角色分侧写入通道（合成输入，不跑真服务、不碰真机）。

覆盖：
  1. check_base_review.evaluate / main：未闭环 → 退出码 3；裁决闭环 → 退出码 0；无效裁决记录不算闭环；
  2. run_pd_round.check_base_review_gate：run 目录未闭环 → ValueError；无产物 → 仅告警；
     --require-base-review → 缺产物即失败；--allow-unreviewed-base → 跳过；
  3. apply_case_base.build_plan：单角色容器子键 → role 后缀 target_field（不写共享 others）；
     role_split_flags（--max-model-len）→ 双侧各建 MAX_MODEL_LEN_<role>；幂等（已在位不重复建）。

跑法：python3 scripts/test_case_base_gate.py   （或 pytest -q scripts/test_case_base_gate.py）
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_base_review as gate  # noqa: E402
import apply_case_base as writer  # noqa: E402
import run_pd_round as dispatch  # noqa: E402


def _subkey_row():
    return {
        "kind": "container_subkey",
        "key": "--additional-config:recompute_scheduler_enable",
        "roles": ["decode"],
        "case_value": True,
        "status": "missing",
        "category": "needs_review",
        "reason": "案例容器的子键在当前底座与搜索维度里都不存在",
        "action": "人工裁决",
        "role_values": {"decode": True},
        "field_name": "recompute_scheduler_enable",
    }


def _alias_subkey_row():
    """容器内 JSON key 与 JSON_SUBKEY_MAP 字段名不同名的情况（enforce_eager → speculative_enforce_eager）。"""
    return {
        "kind": "container_subkey",
        "key": "--speculative-config:enforce_eager",
        "roles": ["prefill"],
        "case_value": True,
        "status": "missing",
        "category": "auto_adopt",
        "reason": "案例容器的子键在当前底座与搜索维度里都不存在",
        "action": "单角色 → role 后缀 target_field",
        "role_values": {"prefill": True},
        "field_name": "speculative_enforce_eager",
    }


def _flag_row():
    return {
        "kind": "flag",
        "key": "--max-model-len",
        "roles": ["prefill", "decode"],
        "case_value": {"prefill": "16384", "decode": "8192"},
        "status": "adopted",
        "category": "present",
        "reason": "已生效（others）",
        "action": "-",
    }


def _write_audit(directory: Path) -> Path:
    audit = {
        "config": str(directory / "config.toml"),
        "counts": {"adopted": 20, "missing": 1, "range_gap": 0},
        "items": [_subkey_row(), _flag_row()],
        "missing": [_subkey_row()],
        "range_gaps": [],
    }
    path = directory / "base_coverage.after.json"
    path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    return path


def _write_decisions(directory: Path) -> Path:
    payload = {
        "decisions": [
            {
                "kind": "container_subkey",
                "key": "--additional-config:recompute_scheduler_enable",
                "verdict": "adopted",
                "owner": "ssq",
                "reason": "已在 serve_plan.JSON_SUBKEY_MAP 注册并写成 recompute_scheduler_enable_decode",
                "decided_at": "2026-10-25",
            }
        ]
    }
    path = directory / "base_review_decisions.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_gate_blocks_then_passes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_audit(root)
        result = gate.evaluate(root / "base_coverage.after.json", root / "base_review_decisions.json")
        assert result["coverage_present"] is True
        assert len(result["rows"]) == 1 and result["rows"][0]["key"].endswith("recompute_scheduler_enable")
        assert result["problems"] == []
        # 缺裁决记录文件时自动生成模板
        assert (root / "base_review_decisions.json").exists()

        _write_decisions(root)
        closed = gate.evaluate(root / "base_coverage.after.json", root / "base_review_decisions.json")
        assert closed["rows"] == [] and closed["problems"] == []

        # 无效裁决（缺 owner）不算闭环
        (root / "base_review_decisions.json").write_text(
            json.dumps(
                {
                    "decisions": [
                        {
                            "key": "--additional-config:recompute_scheduler_enable",
                            "verdict": "adopted",
                            "reason": "无责任人",
                            "decided_at": "2026-10-25",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        invalid = gate.evaluate(root / "base_coverage.after.json", root / "base_review_decisions.json")
        assert len(invalid["rows"]) == 1 and invalid["problems"], invalid


def test_gate_cli_exit_codes():
    script = SCRIPTS / "check_base_review.py"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_audit(root)
        proc = subprocess.run([sys.executable, str(script), "--run-dir", str(root)], capture_output=True, text=True)
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert "recompute_scheduler_enable" in proc.stdout
        report = root / "BASE_REVIEW_REQUIRED.md"
        assert report.exists() and "禁止开跑" in report.read_text(encoding="utf-8")

        _write_decisions(root)
        proc = subprocess.run([sys.executable, str(script), "--run-dir", str(root)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr

        # 无产物的 run 目录：默认阻断，显式放行才可过
        empty = root / "empty-run"
        empty.mkdir()
        proc = subprocess.run([sys.executable, str(script), "--run-dir", str(empty)], capture_output=True, text=True)
        assert proc.returncode == 3, proc.stdout + proc.stderr
        proc = subprocess.run(
            [sys.executable, str(script), "--run-dir", str(empty), "--allow-unreviewed"], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_dispatch_gate_behaviour():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 无 R0 产物：默认只告警；require 时失败
        dispatch.check_base_review_gate(root)
        try:
            dispatch.check_base_review_gate(root, require=True)
        except ValueError as exc:
            assert "缺少 R0 案例底座产物" in str(exc)
        else:
            raise AssertionError("require_base_review 应当拒绝缺产物的 run")

        # 有产物且未闭环：禁止开跑（报告同时落盘）
        _write_audit(root)
        try:
            dispatch.check_base_review_gate(root)
        except ValueError as exc:
            assert "未闭环" in str(exc)
        else:
            raise AssertionError("未闭环的底座应当被 run 门禁拒绝")
        assert (root / "BASE_REVIEW_REQUIRED.md").exists()

        # 显式放行 / 裁决闭环
        dispatch.check_base_review_gate(root, allow_unreviewed=True)
        _write_decisions(root)
        dispatch.check_base_review_gate(root)


def test_writer_role_channels():
    rules = json.loads((SCRIPTS / "case_base_rules.json").read_text(encoding="utf-8"))
    assert "--max-model-len" in (rules.get("role_split_flags") or []), rules.get("role_split_flags")
    assert "--additional-config" not in (rules.get("review_flags") or []), rules.get("review_flags")

    subkey = _subkey_row()
    alias = _alias_subkey_row()
    coverage = {"items": [subkey, alias, _flag_row()], "missing": [subkey, alias], "range_gaps": [], "counts": {}}
    # JSON_SUBKEY_MAP 的形状是 {字段名: (容器名, 容器内 JSON key)}（见 audit._read_json_subkey_map）
    subkey_map = {
        "recompute_scheduler_enable": ("additional-config", "recompute_scheduler_enable"),
        "speculative_enforce_eager": ("speculative-config", "enforce_eager"),
    }
    actions, manual = writer.build_plan(
        None,
        rules,
        coverage,
        {},
        {"ranges": {}},
        {"owned_flags": set()},
        "vllm",
        "python3",
        "config_writer.py",
        Path("/tmp/config.toml"),
        subkey_map=subkey_map,
    )
    shells = [a["shell"] for a in actions]
    assert any("name=recompute_scheduler_enable_decode" in s for s in shells), shells
    # 字段名必须取 JSON_SUBKEY_MAP 的名字（不是容器内 JSON key）
    assert any("name=speculative_enforce_eager_prefill" in s for s in shells), shells
    assert not any("name=enforce_eager_prefill" in s for s in shells), shells
    assert any("name=MAX_MODEL_LEN_prefill" in s for s in shells), shells
    assert any("name=MAX_MODEL_LEN_decode" in s for s in shells), shells
    # 数值型 role target_field 必须带 min/max（缺省时运行期域=0..100 → 16384 被判越界）
    mml = [a for a in actions if "name=MAX_MODEL_LEN_prefill" in a["shell"]][0]["argv"]
    assert "min=16384" in mml and "max=16384" in mml, mml
    # bool 型不需要 min/max
    ee = [a for a in actions if "name=speculative_enforce_eager_prefill" in a["shell"]][0]["argv"]
    assert not any(tok.startswith(("min=", "max=")) for tok in ee), ee
    # 不得退化成共享 others 容器（否则 P/D 无法分侧）
    assert not any("command.others.additional-config" in s for s in shells), shells
    assert manual == [], manual
    for action in actions:
        argv = action["argv"]
        if "--target-field" in argv:
            pos = argv.index("--target-field")
            assert argv[pos + 1] == "engine=vllm_pd", argv
            assert "config_position=run" in argv, argv
    role_verify = [a["verify"] for a in actions if a["verify"]["type"] == "role_target_field"]
    assert {v["name"] for v in role_verify} == {
        "recompute_scheduler_enable_decode",
        "speculative_enforce_eager_prefill",
        "MAX_MODEL_LEN_prefill",
        "MAX_MODEL_LEN_decode",
    }, role_verify

    # 幂等：字段已在位时不重复生成
    cfg = {"vllm_pd": {"target_field": [{"name": "recompute_scheduler_enable_decode"}]}}
    again, _ = writer.build_plan(
        None,
        rules,
        coverage,
        cfg,
        {"ranges": {}},
        {"owned_flags": set()},
        "vllm",
        "python3",
        "config_writer.py",
        Path("/tmp/config.toml"),
        subkey_map=subkey_map,
    )
    assert not any("recompute_scheduler_enable_decode" in a["shell"] for a in again), again


def test_dry_run_preserves_existing_target_and_frozen_config(tmp_path):
    src = tmp_path / "base.toml"
    target = tmp_path / "target.toml"
    src.write_text("[vllm]\n", encoding="utf-8")
    target.write_text("[vllm]\nmarker = 7\n", encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"config": str(src), "missing": [], "items": [], "range_gaps": []}), encoding="utf-8"
    )

    def call(destination, extra=()):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "apply_case_base.py"),
                "--coverage",
                str(coverage),
                "--copy-from",
                str(src),
                "--config",
                str(destination),
                "--force-copy",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    original = target.read_bytes()
    assert call(target).returncode == 0
    assert target.read_bytes() == original
    assert call(src, ("--apply",)).returncode == 2
    assert src.read_text(encoding="utf-8") == "[vllm]\n"
    absent = tmp_path / "absent.toml"
    assert call(absent).returncode == 0 and not absent.exists()


def test_single_role_flag_not_shared_and_both_equal_can_be_shared(tmp_path):
    rules = json.loads((SCRIPTS / "case_base_rules.json").read_text(encoding="utf-8"))
    for roles in (["prefill"], ["decode"], ["prefill", "decode"]):
        row = {
            "kind": "flag",
            "key": "--enable-prefix-caching",
            "roles": roles,
            "case_value": {role: None for role in roles},
            "category": "auto_adopt",
        }
        actions, manual = writer.build_plan(
            None,
            rules,
            {"missing": [row], "items": [], "range_gaps": []},
            {"vllm_pd": {}},
            {"ranges": {}},
            {"owned_flags": set()},
            "vllm",
            "python",
            "cw",
            tmp_path / "config.toml",
        )
        if len(roles) == 1:
            assert not actions and manual[0]["category"] == "needs_review"
        else:
            assert len(actions) == 1 and not manual


def test_fixed_ranges_never_widen(tmp_path):
    rules = json.loads((SCRIPTS / "case_base_rules.json").read_text(encoding="utf-8"))
    for bounds, constant, expected_action in [
        ((0.85, 0.85), None, False),
        ((0.8, 0.9), 0.85, False),
        ((0.8, 0.9), None, True),
    ]:
        block = {
            "name": "GPU_MEMORY_UTILIZATION_prefill",
            "min": bounds[0],
            "max": bounds[1],
            "value": 0.85,
            "dtype": "float",
        }
        if constant is not None:
            block["constant"] = constant
        row = {"key": "--gpu-memory-utilization", "roles": ["prefill"], "case_value": 0.95}
        cfg = {"vllm_pd": {"target_field": [block]}}
        eff = {"ranges": {(row["key"], "prefill"): {"name": block["name"]}}}
        actions, manual = writer.build_plan(
            None,
            rules,
            {"missing": [], "items": [], "range_gaps": [row]},
            cfg,
            eff,
            {"owned_flags": set()},
            "vllm",
            "python",
            "cw",
            tmp_path / "config.toml",
        )
        assert bool(actions) is expected_action
        assert bool(manual) is not expected_action


def test_pd_role_effective_values_do_not_leak():
    import audit_base_coverage as audit

    cfg = {
        "vllm": {"command": {"others": "--max-model-len 100"}},
        "vllm_pd": {
            "env_prefill": {"TEST_ENV": "p"},
            "target_field": [
                {"name": "MAX_MODEL_LEN_decode", "value": 200, "config_position": "run"},
                {"name": "enable_cpu_binding_prefill", "value": True, "config_position": "run"},
            ],
        },
    }
    result = audit._pd_role_effective(
        cfg, "vllm", {"enable_cpu_binding": ("additional-config", "enable_cpu_binding")}, set(), set()
    )
    assert result["prefill"]["flags"]["--max-model-len"] == "100"
    assert result["decode"]["flags"]["--max-model-len"] == 200
    assert "TEST_ENV" not in result["decode"]["env"]
    assert result["prefill"]["containers"]["--additional-config"]["enable_cpu_binding"] is True
    assert "--additional-config" not in result["decode"]["containers"]


def test_pd_audit_compares_each_role_and_values(tmp_path):
    root = SCRIPTS.parents[3]
    config = tmp_path / "config.toml"
    config.write_text(
        '[vllm.command]\nothers = "--max-model-len 100"\n'
        '[vllm_pd.env_prefill]\nFOO = "p"\n'
        '[[vllm_pd.target_field]]\nname = "MAX_MODEL_LEN_decode"\nconfig_position = "run"\nvalue = 200\n',
        encoding="utf-8",
    )
    case = {
        "case_file": "fixture.md",
        "union": {
            "flags": {
                "--max-model-len": {"roles": ["prefill", "decode"], "values": {"prefill": "100", "decode": "100"}}
            },
            "env": {"FOO": {"roles": ["prefill", "decode"], "values": {"prefill": "p", "decode": "p"}}},
            "containers": {},
            "container_subkeys": {},
        },
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    output = tmp_path / "coverage.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_base_coverage.py"),
            "--config",
            str(config),
            "--case-json",
            str(case_path),
            "--optix-root",
            str(root),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1), proc.stderr
    rows = json.loads(output.read_text(encoding="utf-8"))["items"]

    def status(kind, role):
        return next(row for row in rows if row["kind"] == kind and row["roles"] == [role])["status"]

    assert status("flag", "prefill") == "adopted"
    assert status("flag", "decode") == "missing"
    assert status("env", "prefill") == "adopted"
    assert status("env", "decode") == "missing"


def test_user_fixed_param_never_widened(tmp_path):
    rules = json.loads((SCRIPTS / "case_base_rules.json").read_text(encoding="utf-8"))
    name = "GPU_MEMORY_UTILIZATION_prefill"
    (tmp_path / "pd_workflow.json").write_text(json.dumps({"fixed_params": {name: 0.85}}), encoding="utf-8")
    block = {"name": name, "min": 0.8, "max": 0.9, "value": 0.85, "dtype": "float"}
    row = {"key": "--gpu-memory-utilization", "roles": ["prefill"], "case_value": 0.95}
    actions, manual = writer.build_plan(
        None,
        rules,
        {"missing": [], "items": [], "range_gaps": [row]},
        {"vllm_pd": {"target_field": [block]}},
        {"ranges": {(row["key"], "prefill"): {"name": name}}},
        {"owned_flags": set()},
        "vllm",
        "python",
        "cw",
        tmp_path / "config.toml",
    )
    assert not actions and manual[0]["category"] == "needs_review"


def test_apply_is_atomic_when_action_fails(tmp_path, monkeypatch):
    import pytest

    source = tmp_path / "base.toml"
    target = tmp_path / "target.toml"
    source.write_text("[vllm]\n", encoding="utf-8")
    target.write_text("[vllm]\nmarker = 7\n", encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"config": str(source), "missing": [], "items": [], "range_gaps": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_case_base.py",
            "--coverage",
            str(coverage),
            "--copy-from",
            str(source),
            "--config",
            str(target),
            "--force-copy",
            "--apply",
            "--optix-root",
            str(SCRIPTS.parents[3]),
        ],
    )
    original = target.read_bytes()

    def fail(*args, **kwargs):
        raise SystemExit("simulated action failure")

    monkeypatch.setattr(writer, "run_actions", fail)
    with pytest.raises(SystemExit, match="simulated action failure"):
        writer.main()
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".case-base-*.toml"))


def test_plan_output_hardlink_cannot_overwrite_source(tmp_path):
    import os

    source = tmp_path / "base.toml"
    source.write_text("[vllm]\n", encoding="utf-8")
    alias = tmp_path / "plan.json"
    os.link(source, alias)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"config": str(source), "missing": [], "items": [], "range_gaps": []}), encoding="utf-8"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "apply_case_base.py"),
            "--coverage",
            str(coverage),
            "--copy-from",
            str(source),
            "--config",
            str(tmp_path / "target.toml"),
            "--out",
            str(alias),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert source.read_text(encoding="utf-8") == "[vllm]\n"


def main() -> int:
    checks = [
        test_gate_blocks_then_passes,
        test_gate_cli_exit_codes,
        test_dispatch_gate_behaviour,
        test_writer_role_channels,
    ]
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - 自检脚本，失败信息要全量可见
            failed += 1
            print(f"FAIL  {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_dispatch_missing_artifacts_are_quiet_unless_required(tmp_path, capsys):
    import pytest

    dispatch.check_base_review_gate(tmp_path)
    assert capsys.readouterr().err == ""
    with pytest.raises(ValueError, match="缺少 R0 案例底座产物"):
        dispatch.check_base_review_gate(tmp_path, require=True)
