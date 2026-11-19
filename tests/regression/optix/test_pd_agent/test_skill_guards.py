# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Guards that keep the optimizer from concluding what its measurements cannot show.

* ``metrics_gate.py breakdown`` — fitness degenerates to a throughput-only score
  when the SLO is far looser than the measured baseline.
* ``metrics_gate.py compare`` — a difference below the noise floor is
  ``within_noise``, never a "gain".
* ``check_dimensions.py`` — a run that never moved a dimension is not converged.
* ``config_preflight.py`` — SLO must be declared as user-provided.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents/skills/optix-assistant/scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_script("metrics_gate")
dimensions = load_script("check_dimensions")
exporter = load_script("export_negative_knowledge")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def trial(candidate_id, speed, ttft=21.8, tpot=0.0319, params=None, success_rate=1.0):
    return {
        "candidate_id": candidate_id,
        "params": params or {"MAX_NUM_BATCHED_TOKENS_prefill": 16384},
        "fitness": 0.4 * 5300 / speed + 0.07,
        "performance": {
            "generate_speed": speed,
            "time_to_first_token": ttft,
            "time_per_output_token": tpot,
            "success_rate": success_rate,
        },
        "error": None,
    }


def make_run(tmp_path, rounds):
    """rounds: list of list-of-trials, index i -> round i+1"""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    for index, trials in enumerate(rounds, start=1):
        write_json(run_dir / f"results.round-{index}.json", {"round": index, "trials": trials})
    return run_dir


def test_breakdown_flags_degraded_objective(tmp_path, capsys):
    # 600s TTFT SLO against a 21.8s baseline: the exponential penalty saturates and
    # the latency terms only worth ~1.8% of the score.
    run_dir = make_run(tmp_path, [[trial("r1", 176.0)]])
    rc = gate.main(["breakdown", "--run-dir", str(run_dir), "--ttft-slo", "600", "--tpot-slo", "0.05"])
    out = capsys.readouterr().out
    assert rc == 3
    assert "DEGRADED" in out

    # --json must not become a way to bypass the gate
    rc = gate.main(["breakdown", "--run-dir", str(run_dir), "--ttft-slo", "600", "--tpot-slo", "0.05", "--json"])
    assert rc == 3

    # A tight SLO plus a realistic generation target keeps the latency terms
    # meaningful, so the objective is usable.
    rc = gate.main(
        [
            "breakdown",
            "--run-dir",
            str(run_dir),
            "--ttft-slo",
            "25",
            "--tpot-slo",
            "0.032",
            "--gen-target",
            "176",
        ]
    )
    assert rc == 0


def test_compare_reports_within_noise(tmp_path, capsys):
    # 0.67% apart, repeated measurements spread ~0.5% — not separable.
    arm_a = [trial(f"a{i}", speed) for i, speed in enumerate([176.2, 177.0, 176.4])]
    arm_b = [trial(f"b{i}", speed) for i, speed in enumerate([177.4, 178.1, 177.6])]
    run_dir = make_run(tmp_path, [[t] for t in arm_a + arm_b])
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--arm", "a=a", "--arm", "b=b"])
    assert rc == 3
    assert "within_noise" in capsys.readouterr().out


def test_compare_keeps_real_gain(tmp_path, capsys):
    arm_a = [trial(f"a{i}", speed) for i, speed in enumerate([100.0, 100.5, 100.2])]
    arm_b = [trial(f"b{i}", speed) for i, speed in enumerate([130.0, 130.4, 130.2])]
    run_dir = make_run(tmp_path, [[t] for t in arm_a + arm_b])
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--arm", "a=a", "--arm", "b=b"])
    assert rc == 0
    assert "real_gain" in capsys.readouterr().out


def test_compare_arm_matches_exact_id_first(tmp_path, capsys):
    # ``r3`` must not silently absorb ``r7-repeat-r3`` -- assert on the actual
    # arm grouping (the old load_trials-based assertion could never fail).
    rounds = [[trial("r3", 176.0)], [trial("r7-repeat-r3", 176.5)]]
    run_dir = make_run(tmp_path, rounds)
    gate.main(["compare", "--run-dir", str(run_dir), "--arm", "a=r3", "--arm", "b=r7-repeat-r3", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["arms"]["a"]["rounds"] == [1]
    assert payload["arms"]["b"]["rounds"] == [2]


def test_compare_keeps_arms_with_similar_labels(tmp_path, capsys):
    # Long parameter names used to be truncated into the same label, so two arms
    # silently merged into one. Every distinct params set must stay separate.
    rounds = [
        [trial("r1", 176.0, params={"GPU_MEMORY_UTILIZATION_decode": 0.9})],
        [trial("r2", 176.5, params={"GPU_MEMORY_UTILIZATION_decode": 0.93})],
        [trial("r3", 177.0, params={"GPU_MEMORY_UTILIZATION_decode": 0.94})],
    ]
    run_dir = make_run(tmp_path, rounds)
    gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["arms"]) == 3
    assert sum(1 for name in payload["arms"] if "0.93" in name or "0.94" in name) == 2


def space(*params):
    return {
        "search_space": {
            "parameters": list(params),
            "constants": [],
        }
    }


def param(name, default, low, high):
    return {"name": name, "dtype": "int", "config_position": "run", "default": default, "min": low, "max": high}


def test_check_dimensions_finds_uncovered(tmp_path, capsys):
    rounds = [
        [trial("r1", 176.0, params={"A_prefill": 16384})],
        [trial("r2", 176.4, params={"A_prefill": 32768})],
    ]
    run_dir = make_run(tmp_path, rounds)
    write_json(
        run_dir / "context.json",
        space(param("A_prefill", 8192, 4096, 32768), param("B_decode", 64, 8, 64)),
    )
    assert dimensions.main(["--run-dir", str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "B_decode" in out
    assert "不得宣称收敛" in out

    # Declaring the dimension retired with a reason clears the finding.
    write_json(
        run_dir / "dimension_index.json",
        {"B_decode": {"status": "retired", "reason": "prefill 瓶颈，decode 无作用路径"}},
    )
    assert dimensions.main(["--run-dir", str(run_dir), "--max-trials-per-dim", "10"]) == 0


def test_check_dimensions_flags_pinned_default_mismatch(tmp_path, capsys):
    # Every trial runs 64 while the space claims the default is 128: the dimension
    # was never moved, so "covered" would be a lie.
    rounds = [[trial(f"r{i}", 176.0, params={"A_prefill": 16384, "B_decode": 64})] for i in range(1, 5)]
    run_dir = make_run(tmp_path, rounds)
    write_json(run_dir / "context.json", space(param("A_prefill", 8192, 4096, 32768), param("B_decode", 128, 8, 128)))
    assert dimensions.main(["--run-dir", str(run_dir), "--max-trials-per-dim", "10"]) == 1
    out = capsys.readouterr().out
    assert "pinned_off_default" in out
    assert "与空间 default=128 不一致" in out


def test_check_dimensions_derives_cap_from_budget(tmp_path, capsys):
    # 6 trials of budget over 2 movable dimensions -> 3 trials per dimension,
    # so exploring one dimension 4 times already exceeds its share.
    params = [16384, 32768, 16384, 32768]
    rounds = [[trial(f"r{i}", 176.0, params={"A_prefill": value})] for i, value in enumerate(params, start=1)]
    run_dir = make_run(tmp_path, rounds)
    context = space(param("A_prefill", 8192, 4096, 32768), param("B_decode", 64, 8, 64))
    context["budgets"] = {"max_trials": 6}
    write_json(run_dir / "context.json", context)
    assert dimensions.main(["--run-dir", str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert "单维度上限   : 3" in out
    assert "max_trials=6 / 2 个可动维度均分" in out


def run_preflight(config_path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "config_preflight.py"), "--config", str(config_path), *args],
        capture_output=True,
        text=True,
        cwd=str(config_path.parent),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SCRIPTS), "PYTHONIOENCODING": "utf-8"},
    )


@pytest.fixture()
def minimal_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'optimizer_strategy = "agent"\n'
        "use_request_rate_calibration = false\n"
        "\n[vllm.command]\n"
        'model = "/models/model"\n'
        'served_model_name = "model"\n'
        'host = "127.0.0.1"\n'
        "port = 8000\n"
        "\n[ais_bench.command]\n"
        'models = "/datasets/ais_bench"\n'
        "\n[agent_optimizer]\n"
        'run_id = "run-guard"\n'
        "max_rounds = 4\n"
        "max_trials = 4\n"
        "time_limit_minutes = 60\n",
        encoding="utf-8",
    )
    return path


def test_preflight_rejects_agent_filled_slo(minimal_config):
    result = run_preflight(minimal_config, "--tpot-slo", "0.05", "--run-id", "run-guard")
    assert result.returncode == 1
    assert "slo-source" in result.stdout
    assert "tpot_slo" not in minimal_config.read_text(encoding="utf-8")
    assert not (minimal_config.parent / ".agent_optimizer/runs/run-guard/slo_provenance.json").exists()


def test_preflight_rejects_constant_slo_term(minimal_config):
    run_dir = minimal_config.parent / ".agent_optimizer/runs/run-guard"
    write_json(
        run_dir / "results.round-1.json",
        {"round": 1, "trials": [trial("r1", 176.0, ttft=21.8, tpot=0.0319)]},
    )
    result = run_preflight(
        minimal_config,
        "--tpot-slo",
        "10",
        "--slo-source",
        "user",
        "--slo-note",
        "用户：先看吞吐",
        "--run-id",
        "run-guard",
    )
    assert result.returncode == 1
    # TPOT elasticity = 3.0 * 0.0319 / 10 = 0.01
    assert "灵敏度只有 0.01" in result.stdout


def test_preflight_records_slo_provenance(minimal_config):
    run_dir = minimal_config.parent / ".agent_optimizer/runs/run-guard"
    write_json(
        run_dir / "results.round-1.json",
        {"round": 1, "trials": [trial("r1", 176.0, ttft=21.8, tpot=0.0319)]},
    )
    result = run_preflight(
        minimal_config,
        "--tpot-slo",
        "0.032",
        "--slo-source",
        "user",
        "--slo-note",
        "用户：TPOT 32ms 是硬约束",
        "--run-id",
        "run-guard",
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads((run_dir / "slo_provenance.json").read_text(encoding="utf-8"))
    assert payload["source"] == "user"
    assert "硬约束" in payload["note"]


def test_check_dimensions_init_index_preserves_filled_entries(tmp_path):
    """--init-index 不得覆盖 agent 已填写的 dimension_index（retired/reason 是数据）。"""
    run_dir = tmp_path / ".agent_optimizer/runs/run-dim"
    write_json(
        run_dir / "context.json",
        {
            "search_space": {"parameters": [{"name": "MNBT", "default": 4096, "min": 1024, "max": 16384}]},
            "budgets": {"max_trials": 8},
        },
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {"candidate_id": "r1-a", "params": {"MNBT": 4096}},
                {"candidate_id": "r1-b", "params": {"MNBT": 8192}},
            ],
        },
    )
    write_json(
        run_dir / "results.round-2.json",
        {"round": 2, "trials": [{"candidate_id": "r2-a", "params": {"MNBT": 16384}}]},
    )
    index_path = run_dir / "dimension_index.json"
    write_json(
        index_path,
        {
            "MNBT": {"status": "retired", "reason": "A/B-A 三次推翻，差异在噪声内"},
            "OLD_DIM": {"status": "active", "reason": "历史维度"},
        },
    )
    rc = dimensions.main(["--run-dir", str(run_dir), "--init-index"])
    assert rc == 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["MNBT"]["status"] == "retired"
    assert "噪声" in payload["MNBT"]["reason"]
    assert "OLD_DIM" in payload


def test_check_dimensions_force_index_rebuilds(tmp_path):
    run_dir = tmp_path / ".agent_optimizer/runs/run-dim"
    write_json(
        run_dir / "context.json",
        {
            "search_space": {"parameters": [{"name": "MNBT", "default": 4096, "min": 1024, "max": 16384}]},
            "budgets": {"max_trials": 8},
        },
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {"candidate_id": "r1-a", "params": {"MNBT": 4096}},
                {"candidate_id": "r1-b", "params": {"MNBT": 8192}},
            ],
        },
    )
    write_json(
        run_dir / "results.round-2.json",
        {"round": 2, "trials": [{"candidate_id": "r2-a", "params": {"MNBT": 16384}}]},
    )
    index_path = run_dir / "dimension_index.json"
    write_json(
        index_path,
        {"MNBT": {"status": "retired", "reason": "旧理由"}, "OLD_DIM": {"status": "active", "reason": "历史维度"}},
    )
    rc = dimensions.main(["--run-dir", str(run_dir), "--init-index", "--force-index"])
    assert rc == 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    # 人工 retired 声明仍被尊重（脚本不自动撤销人工判定）
    assert payload["MNBT"]["status"] == "retired"
    # 已从 search_space 移除的维度条目被重建清除
    assert "OLD_DIM" not in payload


def test_export_negative_force_rescues_below_threshold(tmp_path):
    """--force 语义：低于判据的条目仍入库且 confidence: low（修复前恒跳过）。"""
    run_dir = tmp_path / ".agent_optimizer/runs/run-neg"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "对照 r1-b：EP 在该场景明确劣化"}},
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                    "performance": {"generate_speed": 300.0},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 280.0},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    # 6.7% < 10% 缺省阈值：无 --force 全部跳过（exit 2）
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 2, rc
    assert not (knowledge / "cases/negative").exists() or not list((knowledge / "cases/negative").glob("*.md"))
    # --force：入库且 confidence: low
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge), "--force"])
    assert rc == 0, rc
    files = list((knowledge / "cases/negative").glob("*.md"))
    assert len(files) == 1
    assert "confidence: low" in files[0].read_text(encoding="utf-8")


def test_resolve_objective_accepts_zero_penalty(tmp_path):
    # data-analysis.md 记录过显式 ``tpot_penalty = 0.0`` 的真实配置：旧的 ``> 0``
    # 判定会把 0 当成「未配置」而沿用默认 3.0，四分量拆解与真实 fitness 对不上。
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(
        "generate_speed_target = 176.0\ntpot_penalty = 0.0\nttft_penalty = 2.0\n",
        encoding="utf-8",
    )
    objective, used = gate.resolve_objective(run_dir, None)
    assert used.endswith("config.toml")
    assert objective == {"gen_target": 176.0, "ttft_penalty": 2.0, "tpot_penalty": 0.0}


def test_compare_reports_incomparable_single_arm(tmp_path, capsys):
    # Round 0 同配置连测：params 完全相同 → 只有 1 个配置组 → 无法配对。旧行为静默
    # exit 0，调用方会把「无法比较」当成「比较通过」。
    run_dir = make_run(tmp_path, [[trial("r1", 176.0)], [trial("r2", 176.5)]])
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert payload["verdict"] == "incomparable"
    assert payload["pairs"] == []

    rc = gate.main(["compare", "--run-dir", str(run_dir)])
    assert rc == 3
    assert "无法比较" in capsys.readouterr().out


def test_compare_reports_incomparable_zero_median(tmp_path, capsys):
    # --metric tpot 时缺失字段的 trial 被 load_trials 记 0 → 配置组中位数 0 → 配对
    # 被静默跳过。必须显式报「无法比较」而不是 exit 0。
    t1 = trial("r1", 176.0, params={"MNBT": 4096})
    t2 = trial("r2", 176.5, params={"MNBT": 8192})
    del t1["performance"]["time_per_output_token"]
    del t2["performance"]["time_per_output_token"]
    run_dir = make_run(tmp_path, [[t1], [t2]])
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--metric", "tpot"])
    out = capsys.readouterr()
    assert rc == 3
    assert "无法比较" in out.out
    assert "中位数为 0" in out.out


def test_breakdown_marks_zero_success_rate_unusable(tmp_path, capsys):
    # success_rate=0 → succ 惩罚项溢出（total=inf、delay_share=0）。旧行为会把
    # 失败 trial 误判成「低延迟占比」，并在 JSON 里输出非标准的 Infinity。
    good = trial("r1", 176.0)
    bad = trial("r2", 176.0, success_rate=0.0)
    run_dir = make_run(tmp_path, [[good, bad]])
    rc = gate.main(["breakdown", "--run-dir", str(run_dir), "--ttft-slo", "600", "--tpot-slo", "0.05"])
    captured = capsys.readouterr()
    assert rc == 3
    assert "不可用" in captured.err
    assert "unusable" in captured.out

    rc = gate.main(["breakdown", "--run-dir", str(run_dir), "--ttft-slo", "600", "--tpot-slo", "0.05", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3
    by_id = {t["candidate_id"]: t for t in payload["trials"]}
    assert by_id["r1"]["usable"] is True
    assert by_id["r2"]["usable"] is False
    assert by_id["r2"]["recomputed"] is None
    assert by_id["r2"]["delay_share"] is None
    assert by_id["r2"]["shares"] is None


def test_breakdown_all_unusable_raises(tmp_path):
    bad = trial("r1", 176.0, success_rate=0.0)
    run_dir = make_run(tmp_path, [[bad]])
    with pytest.raises(SystemExit):
        gate.main(["breakdown", "--run-dir", str(run_dir), "--ttft-slo", "600", "--tpot-slo", "0.05"])


def test_preflight_warns_when_sensitivity_unevaluated(minimal_config):
    # Round 0 回填 SLO：run 目录还没有成功 trial，灵敏度门禁无法评估。必须显式
    # 提示「未评估」，不能让用户把「未评估」误当成「评估合格」。
    result = run_preflight(
        minimal_config,
        "--tpot-slo",
        "0.032",
        "--slo-source",
        "user",
        "--slo-note",
        "用户：TPOT 32ms 硬约束",
        "--run-id",
        "run-guard",
    )
    assert result.returncode == 0, result.stdout
    assert "灵敏度门禁本轮未评估" in result.stdout + result.stderr


def test_best_of_worst_of_respect_lower_is_better():
    # dovakhiin review: best_of/worst_of 按吞吐方向（越大越好）选对照组，
    # 延迟口径下方向必须翻转，否则"最优"组实际是延迟最差的组。
    lat_groups = {
        "on": [{"performance": {"time_per_output_token": 0.040}}],
        "off": [{"performance": {"time_per_output_token": 0.030}}],
    }
    assert exporter.best_of(lat_groups, "time_per_output_token")[0] == "off"
    assert exporter.worst_of(lat_groups, "time_per_output_token")[0] == "on"

    tp_groups = {
        "on": [{"performance": {"generate_speed": 150.0}}],
        "off": [{"performance": {"generate_speed": 200.0}}],
    }
    assert exporter.best_of(tp_groups, "generate_speed")[0] == "off"
    assert exporter.worst_of(tp_groups, "generate_speed")[0] == "on"


def test_fallback_reference_respects_lower_is_better():
    trials = [
        {"params": {"EP": "on"}, "performance": {"time_per_output_token": 0.050}, "_round": 1},
        {"params": {"EP": "off"}, "performance": {"time_per_output_token": 0.031}, "_round": 1},
    ]
    ref = exporter.fallback_reference(trials, "EP", "time_per_output_token", "on")
    assert ref["performance"]["time_per_output_token"] == 0.031


def test_export_lower_is_better_metric_exports_entry(tmp_path):
    # 延迟口径的负结果导出：方向未翻转时 good/bad 颠倒，幅度为负被阈值过滤，
    # 脚本在延迟口径下导不出任何条目。
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "r1-b 劣化，A/B 对照推翻"}},
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                    "performance": {"generate_speed": 200.0, "time_per_output_token": 0.030},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 200.0, "time_per_output_token": 0.050},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    rc = exporter.main(
        ["--run-dir", str(run_dir), "--knowledge-root", str(knowledge), "--metric", "time_per_output_token"]
    )
    assert rc == 0, rc
    files = list((knowledge / "cases/negative").glob("*.md"))
    assert len(files) == 1
    assert "劣化" in files[0].read_text(encoding="utf-8")


def test_export_fingerprint_prefers_structured_model(tmp_path):
    # 正则回退只提 Qwen/DeepSeek；结构化直读后任意命名都能进指纹。
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "context.json",
        {"model": "/mnt/nfs/weight/MyCustom-7B", "hardware": {"num_nodes": 1, "num_per_node": 8}},
    )
    fp = exporter.extract_fingerprint(run_dir)
    assert fp["model"] == "MyCustom-7B"


def test_noise_floor_reads_band_file_before_context(tmp_path):
    # SKILL.md「PD 分离部署的强制约束」第 4 条：noise_band 写 run 目录独立文件；脚本必须优先读它，
    # context.noise_band 只作历史 run 回退。
    run_dir = tmp_path / "run"
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.02})
    write_json(run_dir / "context.json", {"noise_band": {"generate_speed_sigma": 0.06}})
    band_sigma, band_source, errors, _warnings, _acked = gate.load_noise_band(run_dir, "generate_speed", 0.5)
    assert not errors
    assert band_source.startswith("noise_band.json")
    assert band_sigma == 0.02
    norm = {"params": {"X": 1}, "generate_speed": 176.0}
    sigma, source = gate.noise_floor(
        run_dir,
        [norm],
        lambda t: t["generate_speed"],
        "generate_speed",
        0.5,
        band_sigma=band_sigma,
        band_source=band_source,
    )
    assert source.startswith("noise_band.json")
    assert sigma == 0.02


def test_slo_provenance_both_missing_is_consistent(tmp_path):
    # 双方都未登记 tpot_slo（例如只回填了部分 SLO 键）：不得误报「与登记值不一致」。
    # 函数级单测：--check-only 下 tpot_slo 缺失本就会被 completeness 拦下（exit 1），
    # 本用例验证的是不再额外报「与登记值 None 不一致」。
    preflight = load_script("config_preflight")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "slo_provenance.json", {"source": "user", "note": "用户：先看吞吐"})
    assert preflight.check_slo_provenance_file({}, run_dir) == []


def test_export_group_comparison_without_candidate_anchor(tmp_path):
    # a0c4fca8 曾把「比较」提为独立赋值语句，cand 首次出现前就求值 cand[1]
    # 抛 TypeError——reason 无候选 ID 时组间对照是唯一路径，收尾强制步骤直接崩。
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "开启后吞吐明显下降，A/B 对照推翻"}},
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                    "performance": {"generate_speed": 200.0},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 160.0},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 0, rc
    files = list((knowledge / "cases/negative").glob("*.md"))
    assert len(files) == 1


def test_compare_sample_size_matches_standard_formula(tmp_path, capsys):
    # hypot(σ_a, σ_b) 已含两个配置组的 σ（等方差时=√2σ），标准式 n=(z)²(σ_a²+σ_b²)/Δ²
    # 的系数是 (z)²。旧实现再乘 2 → 需求翻倍（σ=Δ=0.5%、α=0.05/power=0.8 时 32 vs 16）。
    arm_a = [trial(f"a{i}", 176.0) for i in range(3)]
    arm_b = [trial(f"b{i}", 177.0) for i in range(3)]
    run_dir = make_run(tmp_path, [[t] for t in arm_a + arm_b])
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.005})
    gate.main(["compare", "--run-dir", str(run_dir), "--arm", "a=a", "--arm", "b=b", "--delta", "0.005", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairs"][0]["n_required"] == 16


def test_export_fallback_uses_median_not_extremes(tmp_path):
    # 回退对照取极值会把两侧噪声都算进 gap：on 组 176/160/140（中位数 160，
    # 极值 140）、对照池 200/220/240（中位数 220，极值 240）。
    # 极值口径 gap=(240-140)/240=41.7%；中位数口径 (220-160)/220=27.3%。
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "开启后吞吐下降，A/B 对照推翻"}},
    )
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 176.0},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 160.0},
                },
                {
                    "candidate_id": "r1-c",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 140.0},
                },
                {
                    "candidate_id": "r2-a",
                    "params": {"MAX_NUM_BATCHED_TOKENS": 16384},
                    "performance": {"generate_speed": 200.0},
                },
                {
                    "candidate_id": "r2-b",
                    "params": {"MAX_NUM_BATCHED_TOKENS": 16384},
                    "performance": {"generate_speed": 220.0},
                },
                {
                    "candidate_id": "r2-c",
                    "params": {"MAX_NUM_BATCHED_TOKENS": 16384},
                    "performance": {"generate_speed": 240.0},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 0, rc
    files = list((knowledge / "cases/negative").glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "27.3%" in text
    assert "41.7%" not in text
    assert "中位数" in text
    assert "confidence: medium" in text


def test_export_ignores_invalid_noise_sigma(tmp_path):
    # σ<=0 会把 2σ 阈值归零：0.5% 的噪声级差异也入库且标 high（修复前行为）。
    # σ 无效时必须回退 --min-ratio 绝对阈值，confidence 不得为 high。
    dim = {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "开启后吞吐下降，A/B 对照推翻"}}
    results = {
        "round": 1,
        "trials": [
            {
                "candidate_id": "r1-a",
                "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                "performance": {"generate_speed": 200.0},
            },
            {
                "candidate_id": "r1-b",
                "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                "performance": {"generate_speed": 199.0},
            },
        ],
    }
    knowledge = tmp_path / "case-knowledge"

    # σ=0：threshold 不得归零 → 0.5% < 10% 全部跳过（exit 2）
    run_dir = tmp_path / "run-zero"
    write_json(run_dir / "dimension_index.json", dim)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.0})
    write_json(run_dir / "results.round-1.json", results)
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 2, rc
    assert not list((knowledge / "cases/negative").glob("*.md"))

    # σ=-0.5：同上
    run_dir = tmp_path / "run-neg"
    write_json(run_dir / "dimension_index.json", dim)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": -0.5})
    write_json(run_dir / "results.round-1.json", results)
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 2, rc

    # σ="0.5%"：float() 会抛 ValueError——必须告警并走回退路径，不 traceback
    run_dir = tmp_path / "run-str"
    write_json(run_dir / "dimension_index.json", dim)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": "0.5%"})
    write_json(run_dir / "results.round-1.json", results)
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 2, rc


def test_check_only_evaluates_sensitivity(minimal_config):
    # 202103708：check-only 必须补跑灵敏度评估，否则「首轮结果产生后复跑
    # --check-only 补做校验」的提示永远落空（tpot_slo=10 + 实测 0.0319 曾输出 OK）。
    text = minimal_config.read_text(encoding="utf-8")
    # tpot_slo 是顶层键：TOML 顶层键必须位于任何 [table] 头之前，append 会落进 [agent_optimizer]
    minimal_config.write_text("tpot_slo = 10.0\n" + text, encoding="utf-8")
    run_dir = minimal_config.parent / ".agent_optimizer/runs/run-guard"
    write_json(run_dir / "results.round-1.json", {"round": 1, "trials": [trial("r1", 176.0, tpot=0.0319)]})
    result = run_preflight(minimal_config, "--check-only", "--run-id", "run-guard")
    assert result.returncode == 1, result.stdout
    assert "灵敏度只有" in result.stdout


def test_check_only_skips_sensitivity_without_results(minimal_config):
    # Round 0 配置阶段 check-only：无实测基线，灵敏度无法评估——
    # 不得误报（exit 0），只允许提示「本轮未评估」。
    text = minimal_config.read_text(encoding="utf-8")
    # tpot_slo 是顶层键：TOML 顶层键必须位于任何 [table] 头之前，append 会落进 [agent_optimizer]
    minimal_config.write_text("tpot_slo = 10.0\n" + text, encoding="utf-8")
    result = run_preflight(minimal_config, "--check-only", "--run-id", "run-guard")
    assert result.returncode == 0, result.stdout
    assert "灵敏度只有" not in result.stdout


# ---------------------------------------------------------------------------
# P2: noise_band 校验（检视 202107780 / 202107969）
# ---------------------------------------------------------------------------


def test_compare_rejects_invalid_noise_sigma(tmp_path, capsys):
    # σ 校准值格式非法（字符串）时不得静默退回池化估值继续判——退回会把 σ 低估
    # 两个数量级、把 within_noise 翻成 real_gain。
    rounds = [[trial("r1", 200.0, params={"TUNED": "off"})], [trial("r2", 250.0, params={"TUNED": "on"})]]
    run_dir = make_run(tmp_path, rounds)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": "0.5%"})
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "ERROR" in out
    assert "noise_band.json.generate_speed_sigma" in out
    assert "verdict" not in out  # 未产生任何判定（没有静默退回池化）


def test_compare_over_cap_sigma_acks_to_noise_dominant(tmp_path, capsys):
    # σ=50% 超可信上限（5%）：默认拒绝（exit 2 + 提示 --noise-ack-wide）；
    # 人工确认接受后，2σ ≥ 目标效应 → noise_dominant 终止判定（exit 4），不再给补测指令。
    rounds = [[trial("r1", 200.0, params={"TUNED": "off"})], [trial("r2", 250.0, params={"TUNED": "on"})]]
    run_dir = make_run(tmp_path, rounds)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.5})
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "--noise-ack-wide" in out

    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json", "--noise-ack-wide", "用户确认高噪声场景"])
    out = capsys.readouterr().out
    assert rc == 4, out
    payload = json.loads(out)
    assert payload["noise_acked"] is True
    assert payload["pairs"][0]["verdict"] == "noise_dominant"


def test_compare_zero_sigma_continues_with_warning(tmp_path, capsys):
    # σ=0 视为确定性测量：继续判定但打 WARNING（不得静默）。
    rounds = [
        [trial("r1a", 200.0, params={"TUNED": "1"}), trial("r1b", 200.0, params={"TUNED": "1"})],
        [trial("r2a", 250.0, params={"TUNED": "2"}), trial("r2b", 250.0, params={"TUNED": "2"})],
    ]
    run_dir = make_run(tmp_path, rounds)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0})
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "确定性测量" in captured.err
    payload = json.loads(captured.out)
    assert payload["sigma_floor_source"].startswith("noise_band.json")


def test_compare_unmeasurable_when_n_required_exceeds_budget(tmp_path, capsys):
    # σ=4%（cap 内无需 ack）、默认目标效应 1% → n_required≈252（每种配置）：
    # 超过 run 预算 → unmeasurable（exit 4），不再按 required_repetitions 补测；
    # 预算充足时保持 insufficient（非终止态，可继续补测，exit 0）。
    rounds = [[trial("r1", 200.0, params={"TUNED": "off"})], [trial("r2", 250.0, params={"TUNED": "on"})]]
    run_dir = make_run(tmp_path, rounds)
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.04})
    write_json(run_dir / "context.json", {"budgets": {"max_trials": 64}})
    rc = gate.main(["compare", "--run-dir", str(run_dir), "--json"])
    out = capsys.readouterr().out
    assert rc == 4, out
    payload = json.loads(out)
    assert payload["pairs"][0]["verdict"] == "unmeasurable"
    assert payload["pairs"][0]["budget_max"] == 64

    run_dir2 = make_run(tmp_path / "wide", rounds)
    write_json(run_dir2 / "noise_band.json", {"generate_speed_sigma": 0.04})
    write_json(run_dir2 / "context.json", {"budgets": {"max_trials": 2000}})
    rc = gate.main(["compare", "--run-dir", str(run_dir2), "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert payload["pairs"][0]["verdict"] == "insufficient"


def test_export_threshold_is_max_of_2sigma_and_min_ratio(tmp_path):
    # σ=0.4% 时 2σ=0.8% < --min-ratio 10%：6% 的差异不能以 2σ 判据入库（旧实现
    # 会标 high）——阈值取 max(2σ, min_ratio)，跳过 → rc 2 无条目。
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "开启后吞吐下降，A/B 对照推翻"}},
    )
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.004})
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                    "performance": {"generate_speed": 200.0},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 188.0},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 2, rc
    assert not list((knowledge / "cases/negative").glob("*.md"))


def test_export_over_cap_sigma_confidence_low(tmp_path, capsys):
    # σ 超可信上限：判据无效 → 按 --min-ratio 回退；超阈值仍入库但 confidence: low
    # （校准坏了的证据不如无 σ 信息可信）。
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "dimension_index.json",
        {"ENABLE_EXPERT_PARALLEL": {"status": "retired", "reason": "开启后吞吐下降，A/B 对照推翻"}},
    )
    write_json(run_dir / "noise_band.json", {"generate_speed_sigma": 0.5})
    write_json(
        run_dir / "results.round-1.json",
        {
            "round": 1,
            "trials": [
                {
                    "candidate_id": "r1-a",
                    "params": {"ENABLE_EXPERT_PARALLEL": "off"},
                    "performance": {"generate_speed": 200.0},
                },
                {
                    "candidate_id": "r1-b",
                    "params": {"ENABLE_EXPERT_PARALLEL": "on"},
                    "performance": {"generate_speed": 150.0},
                },
            ],
        },
    )
    knowledge = tmp_path / "case-knowledge"
    rc = exporter.main(["--run-dir", str(run_dir), "--knowledge-root", str(knowledge)])
    assert rc == 0, rc
    captured = capsys.readouterr()
    assert "超过可信上限" in captured.err
    files = list((knowledge / "cases/negative").glob("*.md"))
    assert len(files) == 1
    assert "confidence: low" in files[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P3: PD 场景 SLO 按阶段键（检视 202103638）
# ---------------------------------------------------------------------------


PD_BLOCK_FULL = (
    "\n[pd_disagg.prefill]\n"
    'mode = "prefill"\n'
    "ttft_slo = 2.0\n"
    "tpot_slo = 0.05\n"
    "\n[pd_disagg.decode]\n"
    'mode = "decode"\n'
    "ttft_slo = 0.5\n"
    "tpot_slo = 0.05\n"
)


def test_preflight_pd_phase_slo_completeness(minimal_config):
    # PD 配置的 fitness 读 [pd_disagg.<phase>].tpot_slo，顶层键不生效——
    # 只配阶段键必须通过（旧实现报「tpot_slo 未设置」误挡轮次）。
    text = minimal_config.read_text(encoding="utf-8")
    minimal_config.write_text(text + PD_BLOCK_FULL, encoding="utf-8")
    result = run_preflight(minimal_config, "--check-only", "--run-id", "run-guard")
    assert result.returncode == 0, result.stdout
    assert "tpot_slo 未设置" not in result.stdout

    # decode 阶段缺 tpot_slo → 指名阶段键，不再误指顶层
    broken = minimal_config.read_text(encoding="utf-8").replace("ttft_slo = 0.5\ntpot_slo = 0.05", "ttft_slo = 0.5")
    minimal_config.write_text(broken, encoding="utf-8")
    result = run_preflight(minimal_config, "--check-only", "--run-id", "run-guard")
    assert result.returncode == 1, result.stdout
    assert "[pd_disagg.decode].tpot_slo 未设置" in result.stdout


def test_preflight_pd_backfill_writes_phase_keys(minimal_config):
    # PD 配置回填 --tpot-slo 必须落到 [pd_disagg.<phase>] 各阶段（fitness 只消费
    # 阶段键），凭证按阶段登记；顶层不得再写死（校验键与生效键同源）。
    text = minimal_config.read_text(encoding="utf-8")
    pd_block = (
        "\n[pd_disagg.prefill]\n"
        'mode = "prefill"\n'
        "ttft_slo = 2.0\n"
        "\n[pd_disagg.decode]\n"
        'mode = "decode"\n'
        "ttft_slo = 0.5\n"
        "tpot_slo = 0.05\n"
    )
    minimal_config.write_text(text + pd_block, encoding="utf-8")
    result = run_preflight(
        minimal_config,
        "--tpot-slo",
        "0.03",
        "--slo-source",
        "user",
        "--slo-note",
        "用户要求 TPOT 30ms",
        "--run-id",
        "run-guard",
    )
    assert result.returncode == 0, result.stdout

    import tomllib

    data = tomllib.loads(minimal_config.read_text(encoding="utf-8"))
    assert "tpot_slo" not in data
    assert data["pd_disagg"]["prefill"]["tpot_slo"] == 0.03
    assert data["pd_disagg"]["decode"]["tpot_slo"] == 0.03
    prov = json.loads(
        (minimal_config.parent / ".agent_optimizer/runs/run-guard/slo_provenance.json").read_text(encoding="utf-8")
    )
    assert prov["tpot_slo"] == 0.03
    assert prov["tpot_slo_by_phase"]["[pd_disagg.prefill]"] == 0.03
    assert prov["tpot_slo_by_phase"]["[pd_disagg.decode]"] == 0.03


def test_breakdown_resolves_pd_phase_slo(tmp_path, capsys):
    # breakdown 对 PD 配置（SLO 只在 [pd_disagg.decode]）不再报 cannot resolve——
    # 按阶段键解析（decode 优先），SLO 来源行注明阶段键。
    run_dir = make_run(tmp_path, [[trial("r1", 176.0)], [trial("r2", 178.0)]])
    cfg = tmp_path / "pd-config.toml"
    cfg.write_text(
        "generate_speed_target = 176.0\ntpot_penalty = 3.0\n\n[pd_disagg.decode]\ntpot_slo = 0.05\n",
        encoding="utf-8",
    )
    rc = gate.main(["breakdown", "--run-dir", str(run_dir), "--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "pd_disagg.decode" in out
    assert "cannot resolve" not in out
