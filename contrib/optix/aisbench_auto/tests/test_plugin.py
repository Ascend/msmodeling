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
"""Focused tests for the thin command adapter."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from aisbench_auto import register
from aisbench_auto.benchmark import AisBenchAuto
from aisbench_auto.config import AisBenchAutoCommandConfig, AisBenchAutoConfig
from pydantic import ValidationError

import optix.config.config as config_module
from optix.config.config import OptimizerConfigField
from optix.deploy_env import RuntimeContext
from optix.optimizer.errors import BenchmarkResultError


@pytest.fixture
def plugin(tmp_path, monkeypatch, request):
    script = tmp_path / "fake_aisbench.py"
    script.write_text(
        """import json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\nassert args[args.index('--concurrency') + 1] == '64'\nassert args[args.index('--request_rate') + 1] == '0'\nout = Path('out/run/performances/model')\nout.mkdir(parents=True)\n(out / 'result.csv').write_text('Performance Parameters,Average\\nTTFT,120 ms\\nTPOT,2.5 ms\\n')\n(out / 'result.json').write_text(json.dumps({'Total Requests': {'total': 10}, 'Success Requests': {'total': 9}, 'Request Throughput': {'total': '4 req/s'}, 'Output Token Throughput': {'total': '200 token/s'}, 'Concurrency': {'total': 64}, 'Max Concurrency': {'total': 512}}))\n""",
        encoding="utf-8",
    )
    cfg = AisBenchAutoConfig(
        process_name=getattr(request, "param", ""),
        work_path=tmp_path,
        output_path=tmp_path / "out",
        command=AisBenchAutoCommandConfig(
            executable=sys.executable,
            script=str(script),
            others="",
        ),
    )
    monkeypatch.setattr(config_module, "settings", SimpleNamespace(output=tmp_path / "optix" / "result"))
    context = RuntimeContext(False, None, Path(sys.executable))
    env = dict(os.environ, PATH=os.environ["PATH"])
    bench = AisBenchAuto(config=cfg, runtime_ctx=context, deploy_env=env)
    yield bench
    bench.stop()


def params():
    return (
        OptimizerConfigField(name="CONCURRENCY", config_position="env", value=64, dtype="int"),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", value=0, dtype="float"),
    )


def test_others_and_native_aisbench_reader(plugin):
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    assert plugin.command[-4:] == ["--concurrency", "64", "--request_rate", "0"]
    metrics = plugin.get_performance_index()
    assert metrics.time_to_first_token == pytest.approx(0.12)
    assert metrics.time_per_output_token == pytest.approx(0.0025)
    assert metrics.generate_speed == 200
    assert metrics.throughput == 4
    assert metrics.success_rate == 0.9
    assert plugin.get_best_concurrency() == 200


def result_script(plugin, rounds, *, tail="", markers=True):
    """Emulate native tool output, including a marker before each experiment."""
    plugin.config.command.script = str(plugin.work_path / "result_rounds.py")
    Path(plugin.config.command.script).write_text(
        "import json, sys\nfrom pathlib import Path\n"
        f"rounds = {rounds!r}\n"
        "for name, ttft in rounds:\n"
        "    folder = Path('out') / name\n"
        + ("    print(f'Current exp folder: {folder}', flush=True)\n" if markers else "")
        + "    dest = folder / 'performances' / 'model'\n"
        "    dest.mkdir(parents=True, exist_ok=True)\n"
        "    (dest / 'result.csv').write_text(f'Performance Parameters,Average\\nTTFT,{ttft} ms\\nTPOT,2.5 ms\\n')\n"
        "    (dest / 'result.json').write_text(json.dumps({'Total Requests': {'total': 10}, 'Success Requests': {'total': 9}, 'Request Throughput': {'total': '4 req/s'}, 'Output Token Throughput': {'total': '200 token/s'}, 'Concurrency': {'total': 128}, 'Max Concurrency': {'total': 1000}}))\n"
        + tail,
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "others,rounds",
    [
        ("--dataset_type prefix_cache --prefix_test", [("warmup", 999), ("formal", 120)]),
        ("--repeat=3", [("z_first", 999), ("a_second", 500), ("middle_final", 120)]),
        ("--dataset_type=prefix_cache --prefix_test --repeat 3", [("warmup", 999), ("formal", 120)]),
    ],
)
def test_native_multi_round_results_use_final_experiment(plugin, others, rounds):
    plugin.config.command.others = others
    result_script(plugin, rounds)
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    assert plugin.get_performance_index().time_to_first_token == pytest.approx(0.12)
    assert plugin.get_performance_metric("TTFT") == pytest.approx(0.12)
    assert plugin.get_best_concurrency() == 384
    assert len(list(plugin.config.output_path.glob("*/performances/*/*.csv"))) == len(rounds)
    selection = plugin.config.output_path / ".optix_selected" / "selection.json"
    assert rounds[-1][0] in selection.read_text(encoding="utf-8")
    plugin.stop()
    assert plugin.get_performance_index().time_to_first_token == pytest.approx(0.12)


@pytest.mark.parametrize(
    "others,rounds,tail,markers",
    [
        ("--dataset_type prefix_cache --prefix_test", [("warmup", 999)], "", True),
        ("--repeat 3", [("first", 999), ("second", 500)], "", True),
        ("--repeat 3", [("first", 999)], "", False),
        ("--repeat 0", [("final", 120)], "", True),
        ("--repeat invalid", [("final", 120)], "", True),
        ("", [("first", 999), ("second", 500)], "", False),
        ("", [("warmup", 999)], "print('Current exp folder: out/missing')\n", True),
        ("", [("warmup", 999)], "print('Current exp folder: ../outside')\n", True),
        ("", [("warmup", 999)], "sys.exit(1)\n", True),
        ("", [("shared", 999), ("shared", 120)], "", True),
        ("", [("final", 120)], "(dest / 'result.json').unlink()\n", True),
        ("", [("final", 120)], "(dest / 'result.json').write_text('broken')\n", True),
        ("", [("final", 120)], "(dest / 'extra.csv').write_text('ambiguous')\n", True),
        ("", [("final", 120)], "Path('out/.optix_selected').mkdir()\n", True),
        (
            "",
            [("final", 120)],
            "Path('outside').mkdir()\nPath('out/.optix_selected').symlink_to(Path('outside').resolve())\n",
            True,
        ),
        (
            "",
            [("final", 120)],
            "external = Path('external.csv')\nexternal.write_text('outside result')\n"
            "(dest / 'result.csv').unlink()\n(dest / 'result.csv').symlink_to(external.resolve())\n",
            True,
        ),
    ],
)
def test_invalid_final_result_never_falls_back(plugin, others, rounds, tail, markers):
    plugin.config.command.others = others
    result_script(plugin, rounds, tail=tail, markers=markers)
    plugin.run(params())
    plugin.process.wait(timeout=10)
    with pytest.raises((BenchmarkResultError, ValueError)):
        plugin.get_performance_index()


def test_absolute_colored_log_paths_and_mtime_do_not_change_selection(plugin):
    plugin.config.command.others = "--repeat 2"
    result_script(
        plugin,
        [("first round", 999), ("final round", 120)],
        tail="import os\nos.utime(Path('out/first round/performances/model/result.csv'), (9999999999, 9999999999))\n",
    )
    script = Path(plugin.config.command.script)
    script.write_text(
        script.read_text(encoding="utf-8").replace(
            "print(f'Current exp folder: {folder}', flush=True)",
            "print(f'\\x1b[32mCurrent exp folder: {folder.resolve()}\\x1b[0m', flush=True)",
        ),
        encoding="utf-8",
    )
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    assert plugin.get_performance_index().time_to_first_token == pytest.approx(0.12)


def test_result_selection_does_not_read_while_process_is_running(plugin, monkeypatch):
    plugin.before_run(params())
    monkeypatch.setattr(plugin, "process", Mock(poll=Mock(return_value=None)))
    with pytest.raises(BenchmarkResultError, match="not completed successfully"):
        plugin.get_performance_index()
    plugin.process = None


def test_next_candidate_cannot_reuse_selected_result(plugin):
    plugin.config.command.others = "--dataset_type prefix_cache --prefix_test"
    result_script(plugin, [("warmup", 999), ("formal", 120)])
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    assert plugin.get_performance_index().time_to_first_token == pytest.approx(0.12)
    plugin.stop()
    result_script(plugin, [("warmup", 999)])
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    with pytest.raises(BenchmarkResultError):
        plugin.get_performance_index()


@pytest.mark.parametrize("plugin", ["", "ais_bench", "ais_bench,load-worker"], indirect=True)
def test_cleanup_runs_before_each_candidate(plugin, monkeypatch):
    events = Mock()
    events.attach_mock(Mock(), "cleanup")
    events.attach_mock(Mock(wraps=plugin.before_run), "prepare")
    monkeypatch.setattr(plugin, "kill_residual_process", events.cleanup)
    monkeypatch.setattr(plugin, "before_run", events.prepare)
    run_params = params()
    for _ in range(2):
        plugin.run(run_params)
        assert plugin.process.wait(timeout=10) == 0
        plugin.stop()

    expected_calls = []
    for _ in range(2):
        if plugin.config.process_name:
            expected_calls.append(call.cleanup(plugin.config.process_name))
        expected_calls.append(call.prepare(run_params))
    assert events.mock_calls == expected_calls


def test_missing_results_are_not_reused(plugin):
    plugin.before_run(params())
    with pytest.raises(BenchmarkResultError):
        plugin.get_performance_index()


def test_register_entry_point(monkeypatch):
    from optix.optimizer import register as registry

    monkeypatch.setattr(registry, "benchmarks", {})
    monkeypatch.setattr(config_module, "custom_settings_func", None)
    register()
    assert registry.benchmarks["aisbench_auto"] is AisBenchAuto


def test_arbitrary_others_quoted_values_and_next_candidate(plugin):
    plugin.config.command.others = '--dataset "/data/a b.jsonl" --seed $SEED'
    for seed, rate in [(12, 0), (34, 2.5)]:
        plugin.before_run(
            (
                OptimizerConfigField(name="CONCURRENCY", config_position="env", value=64, dtype="int"),
                OptimizerConfigField(name="SEED", config_position="env", value=seed, dtype="int"),
                OptimizerConfigField(name="REQUESTRATE", config_position="env", value=rate),
            )
        )
        assert plugin.command[2:] == [
            "--dataset",
            "/data/a b.jsonl",
            "--seed",
            str(seed),
            "--concurrency",
            "64",
            "--request_rate",
            str(rate),
        ]
        plugin.stop()


@pytest.mark.parametrize(
    "others,run_params,missing",
    [
        ("--seed $SEED", params(), "SEED"),
        ("--z $ZETA --seed $SEED --again $ZETA", params(), "SEED, ZETA"),
        ("--generation '{\"seed\": $SEED}'", params(), "SEED"),
        ("", (), "CONCURRENCY, REQUESTRATE"),
        ("", params()[:1], "REQUESTRATE"),
    ],
)
def test_undefined_parameters_fail_before_cleanup_or_launch(plugin, monkeypatch, others, run_params, missing):
    plugin.config.command.others = others
    marker = plugin.config.output_path / "previous.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("previous result", encoding="utf-8")
    popen = Mock()
    monkeypatch.setattr("optix.optimizer.interfaces.custom_process.subprocess.Popen", popen)

    with pytest.raises(ValueError, match=f"^Undefined benchmark parameters: {missing}$"):
        plugin.run(run_params)

    popen.assert_not_called()
    assert plugin.run_log_fp is None
    assert marker.read_text(encoding="utf-8") == "previous result"


def test_parameter_defaults_overrides_and_missing_next_candidate(plugin):
    plugin.config.command.others = "--seed $SEED"
    plugin.config.target_field = [
        *params(),
        OptimizerConfigField(name="seed", config_position="env", value=7, dtype="int"),
    ]
    plugin.before_run()
    assert plugin.command[plugin.command.index("--seed") + 1] == "7"
    plugin.stop()

    plugin.before_run((OptimizerConfigField(name="SEED", config_position="env", value=9, dtype="int"),))
    assert plugin.command[plugin.command.index("--seed") + 1] == "9"
    plugin.stop()

    plugin.config.target_field = list(params())
    with pytest.raises(ValueError, match="^Undefined benchmark parameters: SEED$"):
        plugin.before_run()


@pytest.mark.parametrize("ancestor", [False, True])
def test_output_directory_guard(plugin, monkeypatch, ancestor):
    plugin.config.output_path = plugin.work_path.parent if ancestor else plugin.work_path
    clean = Mock()
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", clean)
    with pytest.raises(ValueError, match="working directory"):
        plugin.before_run(params())
    clean.assert_not_called()


@pytest.mark.parametrize("relation", ["same", "parent", "child"])
def test_optix_result_directory_is_protected(plugin, monkeypatch, relation):
    protected = Path(config_module.settings.output)
    marker = protected / "history.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("optimizer result", encoding="utf-8")
    plugin.config.output_path = {
        "same": protected,
        "parent": protected.parent,
        "child": protected / "benchmark",
    }[relation]
    clean = Mock()
    popen = Mock()
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", clean)
    monkeypatch.setattr("optix.optimizer.interfaces.custom_process.subprocess.Popen", popen)

    with pytest.raises(ValueError, match="conflicts with a protected directory"):
        plugin.run(params())

    clean.assert_not_called()
    popen.assert_not_called()
    assert marker.read_text(encoding="utf-8") == "optimizer result"


@pytest.mark.parametrize("alias", ["parent_segments", "symlink"])
def test_protected_result_alias_is_resolved(plugin, monkeypatch, alias):
    protected = Path(config_module.settings.output)
    marker = protected / "history.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("optimizer result", encoding="utf-8")
    if alias == "symlink":
        output = plugin.work_path / "result_alias"
        output.symlink_to(protected, target_is_directory=True)
    else:
        output = protected.parent / "unused" / ".." / protected.name
    plugin.config.output_path = output
    clean = Mock()
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", clean)

    with pytest.raises(ValueError, match="conflicts with a protected directory"):
        plugin.before_run(params())

    clean.assert_not_called()
    assert marker.read_text(encoding="utf-8") == "optimizer result"


@pytest.mark.parametrize("sentinel_name", ["cwd", "home"])
def test_common_protected_directory_is_rejected(plugin, monkeypatch, sentinel_name):
    sentinel = plugin.work_path / "guarded"
    sentinel.mkdir()
    if sentinel_name == "cwd":
        monkeypatch.chdir(sentinel)
    else:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: sentinel))
    plugin.config.output_path = sentinel
    clean = Mock()
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", clean)

    with pytest.raises(ValueError, match="conflicts with a protected directory"):
        plugin.before_run(params())

    clean.assert_not_called()
    assert sentinel.is_dir()


@pytest.mark.parametrize("failure", ["silent_return", "partial_cleanup"])
def test_failed_output_cleanup_blocks_launch(plugin, monkeypatch, failure):
    stale = plugin.config.output_path / "previous" / "performances" / "model" / "result.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale result", encoding="utf-8")
    if failure == "silent_return":
        monkeypatch.setattr("aisbench_auto.benchmark.remove_file", Mock())
    else:
        monkeypatch.setattr("optix.optimizer.utils.shutil.rmtree", Mock(side_effect=OSError("cannot remove")))
    popen = Mock()
    monkeypatch.setattr("optix.optimizer.interfaces.custom_process.subprocess.Popen", popen)

    with pytest.raises(ValueError, match="Benchmark output was not cleaned"):
        plugin.run(params())

    popen.assert_not_called()
    assert plugin.run_log_fp is None
    assert stale.read_text(encoding="utf-8") == "stale result"


def test_inaccessible_output_blocks_launch(plugin, monkeypatch):
    output = plugin.config.output_path
    output.mkdir()
    original_exists = Path.exists
    original_iterdir = Path.iterdir

    def iterdir(path):
        if path == output:
            raise PermissionError("cannot inspect benchmark output")
        return original_iterdir(path)

    # Some Python versions suppress stat errors in exists(); inspection must still fail.
    monkeypatch.setattr(Path, "exists", lambda path: False if path == output else original_exists(path))
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", Mock())
    popen = Mock()
    monkeypatch.setattr("optix.optimizer.interfaces.custom_process.subprocess.Popen", popen)

    with pytest.raises(PermissionError, match="cannot inspect benchmark output"):
        plugin.run(params())

    popen.assert_not_called()
    assert plugin.run_log_fp is None


def test_output_file_is_not_deleted(plugin, monkeypatch):
    output = plugin.config.output_path
    output.write_text("important file", encoding="utf-8")
    clean = Mock()
    monkeypatch.setattr("aisbench_auto.benchmark.remove_file", clean)

    with pytest.raises(ValueError, match="Benchmark output is not a directory"):
        plugin.before_run(params())

    clean.assert_not_called()
    assert output.read_text(encoding="utf-8") == "important file"


def test_failed_command_cannot_reuse_previous_results(plugin, tmp_path):
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 0
    assert plugin.get_performance_index().success_rate == 0.9
    plugin.stop()
    failing_script = tmp_path / "fail.py"
    failing_script.write_text("import sys; sys.exit(17)\n", encoding="utf-8")
    plugin.config.command.script = str(failing_script)
    plugin.config.command.others = ""
    plugin.run(params())
    assert plugin.process.wait(timeout=10) == 17
    with pytest.raises(BenchmarkResultError):
        plugin.get_performance_index()


def test_script_is_required():
    with pytest.raises(ValidationError):
        AisBenchAutoCommandConfig()
    with pytest.raises(ValidationError):
        AisBenchAutoCommandConfig(script="")


def test_script_resolved_against_work_path(plugin):
    plugin.config.command.script = "fake_aisbench.py"
    plugin.update_command()
    assert plugin.command[1] == str(plugin.work_path / "fake_aisbench.py")
    plugin.config.command.script = "missing.py"
    with pytest.raises(ValueError, match="Benchmark script not found"):
        plugin.update_command()


@pytest.mark.parametrize("from_settings", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_config_loading_forms(plugin, monkeypatch, from_settings, as_dict):
    config = plugin.config.model_copy(deep=True)
    config.output_path = Path("out")
    config.target_field = list(params())
    supplied = config.model_dump(mode="json") if as_dict else config
    monkeypatch.setattr(
        config_module, "settings", SimpleNamespace(output=plugin.work_path / "optix" / "result", aisbench_auto=supplied)
    )
    bench = AisBenchAuto(
        config=None if from_settings else supplied,
        runtime_ctx=plugin._runtime_ctx,
        deploy_env=plugin.env,
    )
    try:
        assert isinstance(bench.config.command, AisBenchAutoCommandConfig)
        assert isinstance(bench.config.work_path, Path)
        # Resolving runtime paths must not change the shared settings object.
        assert config.output_path == Path("out")
        bench.run()
        assert bench.process.wait(timeout=10) == 0
        assert bench.command[-4:] == ["--concurrency", "64", "--request_rate", "0"]
        assert bench.get_performance_index().success_rate == 0.9
    finally:
        bench.stop()


def test_empty_config_is_validated_instead_of_using_settings(plugin):
    with pytest.raises(ValidationError):
        AisBenchAuto(config={}, runtime_ctx=plugin._runtime_ctx, deploy_env=plugin.env)
