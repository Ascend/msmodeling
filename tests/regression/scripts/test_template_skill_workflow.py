"""Workflow self-test: synthetic captures, real project builder/generator/comparator.

This is orchestration validation, never a native-model coverage claim.
"""

import pytest
from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.inference.template_graph import legacy_capture_argv
from template_engine.config_adapter import config_from_args, user_input_for
from tensor_cast.core.user_config import UserInputConfig
from template_engine.model_parameters import formula_parameters, bind_parameters
from template_generate import _build_features, MODEL_CONSTANTS


import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace
from tensor_cast.model_config import ModelConfig, QuantConfig
from template_engine.graph_codec import ArtifactConfig

import torch


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    import template_engine.registry as registry

    monkeypatch.setattr(registry, '_TEMPLATES_DIR', tmp_path / 'registry')
    monkeypatch.setattr(registry, '_LEGACY_PATH', tmp_path / 'absent-legacy.json.xz')
    registry._MODEL_META_CACHE.clear()
    root = Path(__file__).resolve().parents[3]
    path = root / '.agents/skills/msmodeling-template-workflow/scripts/workflow.py'
    spec = importlib.util.spec_from_file_location('template_skill_workflow', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_plan(workflow, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'num_hidden_layers': 2, 'num_experts': 16, 'max_position_embeddings': 8192}))
    run_dir = tmp_path / 'run'
    workflow.plan(SimpleNamespace(model_id='fixture/model', run_dir=run_dir, model_config=config, scope=None))
    return workflow.run_path(run_dir)


def test_typical_plan_keeps_roles_and_resolved_parallel_axes(workflow, tmp_path):
    root = make_plan(workflow, tmp_path)
    document = workflow.read(root / 'plan.json')
    assert document['native_layers'] == 2 and document['cells']
    assert {c['world'] for c in document['cells']} >= {2, 4, 8, 16, 32, 64, 128, 256}
    assert {s['config']['num_mtp_tokens'] for c in document['cells'] for s in c['samples']} == {0, 1, 2, 3}
    for cell in document['cells']:
        train = {s['key'] for s in cell['samples'] if s['role'] == 'train'}
        targets = {s['key'] for s in cell['samples'] if s['role'] != 'train'}
        assert not train & targets
        if cell['config']['is_decode']:
            batches = set(cell['decode_local_batches'])
            assert len(targets) == len(batches) + 1
            training_sequences = {}
            holdout_batches = set()
            for sample in cell['samples']:
                config = sample['config']
                local = (config['B'] + cell['DP'] - 1) // cell['DP']
                if sample['role'] == 'train':
                    training_sequences.setdefault(local, set()).add(config['S'])
                elif sample['role'] == 'holdout':
                    holdout_batches.add(local)
            assert set(training_sequences) == holdout_batches == batches
            assert all(len(sequences) == 2 for sequences in training_sequences.values())
        else:
            assert len(targets) == 2
        assert cell['world'] == cell['config']['TP'] * cell['DP']


def test_requested_scope_overrides_are_not_silently_dropped(workflow, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'num_hidden_layers': 2, 'max_position_embeddings': 4096}))
    scope = workflow.read(workflow.PROFILE)
    scope['requested'].update(
        phase=['decode'],
        MTP=[1],
        DFC=[False],
        linear_families=['DISABLED'],
        attention_actions=['DISABLED'],
        B=[1, 100],
        S=[128, 4096],
    )
    scope_file = tmp_path / 'scope.json'
    workflow.write(scope_file, scope)
    root = workflow.run_path(tmp_path / 'run')
    workflow.plan(SimpleNamespace(model_id='fixture/dense', run_dir=root, model_config=config, scope=scope_file))
    document = workflow.read(root / 'plan.json')
    assert document['model_kind'] == 'dense' and document['cells']
    for cell in document['cells']:
        assert cell['config']['EP'] == 1
        for sample in cell['samples']:
            c = sample['config']
            assert c['is_decode'] and c['num_mtp_tokens'] == 1
            assert c['quantize_linear_action'] == c['quantize_attention_action'] == 'DISABLED'
            assert 1 <= c['B'] <= 100 and 128 <= c['S'] <= 4096


def test_workflow_rejects_compile_off_scope(workflow, tmp_path):
    scope = workflow.read(workflow.PROFILE)
    scope['requested']['compile'] = False
    scope_file = tmp_path / 'compile-off-scope.json'
    workflow.write(scope_file, scope)
    with pytest.raises(ValueError, match='compile=true'):
        workflow.plan(
            SimpleNamespace(model_id='fixture/model', run_dir=tmp_path / 'off', scope=scope_file, model_config=None)
        )


def test_customer_readme_links_and_project_cli_examples_match_source(workflow):
    from cli.inference import template_graph
    from cli.registry.argparse_adapter import build_argparser
    from cli.registry.modules import get_spec

    doc = workflow.REPO / 'template_engine/README.md'
    text = doc.read_text(encoding='utf-8')
    links = re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text)
    for link in links:
        if not link.startswith(('http:', 'https:', '#')):
            assert (doc.parent / link.split('#')[0]).exists(), link
    from cli.registry.modules.template_graph import LEGACY_SPEC

    parsers = {
        'template_engine.capture_cli': build_argparser(get_spec('template_capture')),
        'template_engine.template_graph': build_argparser(LEGACY_SPEC),
    }
    checked = []
    for block in re.findall(r'```(?:shell|bash)\n(.*?)```', text, re.S):
        for line in block.replace('\\\n', '').splitlines():
            argv = shlex.split(line)
            if len(argv) > 3 and argv[:3] == ['python', '-m', 'cli.inference.template_graph'] and '--help' not in argv:
                build_argparser(get_spec('template_' + argv[3])).parse_args(argv[4:])
                checked.append(argv[2])
                continue
            if len(argv) > 3 and argv[:2] == ['python', '-m'] and argv[2] in parsers and '--help' not in argv:
                arguments = argv[3:]
                if argv[2] == 'template_engine.capture_cli':
                    arguments = template_graph.legacy_capture_argv(arguments)
                parsers[argv[2]].parse_args(arguments)
                checked.append(argv[2])
    assert len(checked) >= 8
    for block in re.findall(r'```json\n(.*?)```', text, re.S):
        document = json.loads(block)
        for target in document:
            assert target['model_id'] and target['do_compile'] is True


def test_budget_resume_rebuild_and_independent_comparison(workflow, tmp_path, monkeypatch):
    from template_engine.cache_io import build_cache_key, save_capture
    from template_engine.capture_ops import CapturedOp
    from template_engine.graph_codec import encode_document

    root = make_plan(workflow, tmp_path)
    # Capture subprocesses are synthetic in this test. Isolate their process
    # inventory from unrelated live model runs; test the real limit separately.
    monkeypatch.setattr('psutil.process_iter', lambda _fields: iter(()))
    original_run = subprocess.run
    calls = []

    def cli(argv, **kwargs):
        if 'cli.inference.template_graph' not in argv or 'capture' not in argv:
            return original_run(argv, **kwargs)

        def flag(name):
            return argv[argv.index(name) + 1]

        batch = int(flag('--num-queries'))
        sequence = int(flag('--query-length')) + int(flag('--context-length'))
        tp, ep, md = int(flag('--tp-size')), int(flag('--ep-size')), int(flag('--moe-dp-size'))
        key = build_cache_key(
            tp,
            ep,
            md,
            batch,
            sequence,
            dcp=int(flag('--dcp-size')),
            num_mtp_tokens=int(flag('--num-mtp-tokens')),
            enable_dispatch_ffn_combine='enable_dispatch_ffn_combine' in argv,
            is_decode='--decode' in argv,
            do_compile=True,
            quantize_linear_action=flag('--quantize-linear-action'),
            quantize_attention_action=flag('--quantize-attention-action'),
        )
        from tensor_cast.model_config import ParallelConfig

        parallel = ParallelConfig(
            world_size=max(tp, ep) * md, tensor_parallel_size=tp, expert_parallel_size=ep, moe_data_parallel_size=md
        )
        local_batch = (batch + parallel.data_parallel_size - 1) // parallel.data_parallel_size
        x = torch.empty((local_batch, sequence, 3), device='meta')
        y = torch.empty_like(x)
        path = save_capture(
            key, [CapturedOp('aten.neg.default', (x,), {}, y)], 1.0, flag('--cache-dir'), model_id=flag('--model-id')
        )
        Path(str(path) + '.meta').write_bytes(
            encode_document(
                dict(
                    model_id=flag('--model-id'),
                    num_hidden_layers=2,
                    do_compile=True,
                    weight_key=f'tp{tp}ep{ep}md{md}',
                    model_config=ModelConfig(
                        parallel_config=parallel,
                        quant_config=QuantConfig(),
                        hf_config=ArtifactConfig(workflow.read(root / 'model-config.json')),
                    ),
                    decode_query_semantics='mtp_plus_1',
                    model_parameters=(('E', 16), ('H', 3)),
                ),
                'model_meta/1',
            )
        )
        calls.append(key)
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(dict(key=key, path=str(path), cached=False)), stderr=''
        )

    monkeypatch.setattr(subprocess, 'run', cli)
    options = dict(run_dir=root, max_cells=1, max_captures=6, max_seconds=120, resume=False)
    assert workflow.run(SimpleNamespace(**options)) == 2
    state = workflow.read(root / 'state.json')
    assert len(calls) == 6
    assert any(r['status'] == 'generated_only' for r in state['comparisons'].values())
    assert state['registration']['status'] == 'not_registered'
    first_token = state['build_token']
    assert workflow.run(SimpleNamespace(**{**options, 'resume': True, 'max_captures': 8})) == 2
    report = workflow.read(root / 'report.json')
    assert report['independent_targets_passed'] == 2
    assert report['registration']['status'] == 'registered' and report['optimizer_template_ready']
    from template_engine.registry import template_path_for

    assert template_path_for('fixture/model', 2, do_compile=True) == report['registration']['path']
    assert len(calls) == 8  # Training and frozen candidate were reused.
    registered_sha = workflow.sha(report['registration']['path'])
    assert workflow.run(SimpleNamespace(**{**options, 'resume': True, 'max_captures': 8, 'no_register': True})) == 2
    assert workflow.read(root / 'report.json')['registration']['status'] == 'disabled'
    assert workflow.sha(report['registration']['path']) == registered_sha
    assert len(calls) == 8
    assert workflow.run(SimpleNamespace(**{**options, 'resume': True, 'max_cells': 2, 'max_captures': 20})) == 2
    state = workflow.read(root / 'state.json')
    assert state['build_token'] != first_token
    assert (root / 'history' / (first_token + '.json')).is_file()
    # Decode now plans an independent S for each sampled rank batch instead
    # of requesting an untrained K. Every target in the first two cells must
    # compare successfully, without leaking any target into training.
    planned = workflow.read(root / 'plan.json')['cells'][:2]
    expected = sum(s['role'] != 'train' for cell in planned for s in cell['samples'])
    assert expected == 6
    assert workflow.read(root / 'report.json')['independent_targets_passed'] == expected
    refused = [row for row in state['comparisons'].values() if row['status'] == 'refused']
    assert not refused
    assert all(
        s['key'] not in state['training_keys']
        for c in workflow.read(root / 'plan.json')['cells']
        for s in c['samples']
        if s['role'] != 'train'
    )
    assert workflow.run(SimpleNamespace(**{**options, 'resume': True, 'max_cells': 1, 'max_captures': 20})) == 2
    assert workflow.read(root / 'state.json')['build_token'] == first_token
    original = root / 'plan.json'
    original.write_text(original.read_text() + ' ')
    with pytest.raises(ValueError, match='Plan changed'):
        workflow.run(SimpleNamespace(**{**options, 'resume': True}))


def test_external_capture_limit_stops_before_acquisition(workflow, tmp_path, monkeypatch):
    root = make_plan(workflow, tmp_path)
    processes = [
        SimpleNamespace(info={'cmdline': ['python', '-m', 'template_engine.capture_cli'], 'pid': 100, 'ppid': 1}),
        SimpleNamespace(info={'cmdline': ['python', '-m', 'template_engine.capture_cli'], 'pid': 101, 'ppid': 100}),
        SimpleNamespace(
            info={'cmdline': ['python', '-m', 'cli.inference.template_graph', 'capture'], 'pid': 200, 'ppid': 1}
        ),
    ]
    monkeypatch.setattr('psutil.process_iter', lambda _fields: iter(processes))
    options = dict(run_dir=root, max_cells=1, max_captures=6, max_seconds=120, resume=False)
    assert workflow.run(SimpleNamespace(**options)) == 2
    report = workflow.read(root / 'report.json')
    assert 'External captures occupy' in report['stop']
    assert not list((root / 'receipts').glob('capture-*.json'))


@pytest.mark.parametrize('decode,mtp,query,context', [(False, 0, 768, 0), (True, 0, 1, 767), (True, 3, 4, 764)])
def test_official_and_template_request_configs_agree(decode, mtp, query, context):
    argv = [
        'zai-org/GLM-5.2',
        '--num-queries',
        '7',
        '--query-length',
        str(query),
        '--context-length',
        str(context),
        '--num-mtp-tokens',
        str(mtp),
        '--tp-size',
        '2',
        '--ep-size',
        '4',
        '--compile',
        '--quantize-linear-action',
        'FP8',
        '--quantize-attention-action',
        'INT8',
        '--cache-dir',
        'unused',
    ]
    if decode:
        argv.append('--decode')
    parser = build_argparser(get_spec('template_capture'))
    args = parse_module_args(get_spec("template_capture"), parser, argv)
    # Official conversion and template adapter must produce the same public data.
    args.num_devices = 4
    official = UserInputConfig.from_args(args)
    adapted = user_input_for(config_from_args(args))
    from dataclasses import replace

    assert adapted == replace(official, log_level=adapted.log_level)
    assert adapted.get_request_info() == official.get_request_info()
    assert adapted.get_request_info().seq_len == 768


def test_legacy_capture_preserves_request_and_quantization_defaults():
    parser = build_argparser(get_spec('template_capture'))
    args = parser.parse_args(
        legacy_capture_argv(
            [
                '--B=3',
                '--S=1024',
                '--tp',
                '2',
                '--ep',
                '4',
                '--decode',
                '--num-mtp-tokens',
                '2',
                '--dfc',
                'true',
                '--compile',
                '--cache-dir',
                'unused',
            ]
        )
    )
    config = config_from_args(args)
    assert config['B'] == 3 and config['S'] == 1024
    assert args.query_length == 3 and args.context_length == 1021
    assert config['quantize_linear_action'] == 'DISABLED'
    assert user_input_for(config).enable_dispatch_ffn_combine
    with pytest.raises(ValueError, match='Missing value'):
        legacy_capture_argv(['--S'])


def test_template_refuses_unencoded_compiler_options_and_missing_identity():
    args = build_argparser(get_spec('template_capture')).parse_args(
        [
            'some/model',
            '--num-queries',
            '1',
            '--query-length',
            '8',
            '--cache-dir',
            'unused',
            '--compilation-config',
            'enable_sequence_parallel',
        ]
    )
    with pytest.raises(ValueError, match='does not yet encode'):
        config_from_args(args)
    with pytest.raises(ValueError, match='model_id is required'):
        user_input_for(dict(B=1, S=8, do_compile=True))


def test_model_features_are_artifact_local_and_cache_isolated():
    original = dict(MODEL_CONSTANTS)
    first = formula_parameters(
        {'hidden_size': 4096, 'vocab_size': 1000, 'n_routed_experts': 256, 'num_experts_per_tok': 6}
    )
    second = formula_parameters({'text_config': {'hidden_size': 2048, 'intermediate_size': 8192, 'vocab_size': 2000}})
    first_config = dict(B=2, S=512, TP=2, EP=4, model_parameters=first)
    second_config = {**first_config, 'model_parameters': second}
    assert _build_features(first_config)['H'] == 4096
    assert _build_features(second_config)['H'] == 2048
    assert _build_features(first_config)['H'] == 4096
    assert 'I' not in _build_features(first_config)
    assert MODEL_CONSTANTS == original
    artifact = {'template_meta': {'model_parameters': first}}
    assert bind_parameters(artifact, {})['model_parameters'] == first
    with pytest.raises(ValueError, match='disagree'):
        bind_parameters(artifact, second_config)


@pytest.mark.parametrize(
    'flags,enabled',
    [([], True), (['--no-template-graph'], False), (['--no-template-graph', '--template-graph-optimization'], True)],
)
def test_optimizer_template_cli_toggle(monkeypatch, flags, enabled):
    from cli.inference.throughput_optimizer import arg_parse

    monkeypatch.setattr(
        'sys.argv',
        [
            'throughput_optimizer',
            '--model-id',
            'fixture/model',
            '--device',
            'TEST_DEVICE',
            '--num-devices',
            '2',
            '--input-length',
            '768',
            '--output-length',
            '1',
            '--tp-sizes',
            '1',
            '--ep-sizes',
            '2',
            *flags,
        ],
    )
    assert arg_parse().template_graph_optimization is enabled


def test_unified_dispatch_reaches_template_commands(monkeypatch, capsys):
    from cli import main

    monkeypatch.setattr('sys.argv', ['msmodeling', 'inference', 'template-graph', 'generate', '--help'])
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code == 0
    assert '--query-length' in capsys.readouterr().out


def test_artifact_parameters_survive_lazy_loading_and_identity_checks(tmp_path):
    import torch
    from template_engine.capture_ops import CapturedOp
    from template_engine.template_builder import build_template, _Capture
    from template_engine.template_graph import load_template_graph, generate, TemplateGraphRefusal
    from template_engine.zero_meta import _save_template_lazy
    from template_engine.compare_ops import compare_op_sequences

    parameters = formula_parameters({'hidden_size': 8, 'intermediate_size': 16, 'vocab_size': 32})
    captures = []
    for batch, sequence in ((1, 8), (1, 16), (2, 8), (2, 16)):
        config = dict(
            B=batch,
            S=sequence,
            TP=1,
            EP=1,
            MOE_DP=1,
            DCP=1,
            num_mtp_tokens=0,
            is_decode=False,
            do_compile=True,
            enable_dispatch_ffn_combine=False,
            quantize_linear_action='DISABLED',
            quantize_attention_action='DISABLED',
            model_parameters=parameters,
        )
        x = torch.empty((batch, sequence, 8), device='meta')
        y = torch.empty_like(x)
        captures.append(_Capture(str((batch, sequence)), config, [CapturedOp('aten.neg.default', (x,), {}, y)], 0.0))
    engine, _ = build_template(captures)
    path = tmp_path / 'graph.json.xz'
    _save_template_lazy(
        {'engine': engine, 'meta': {'num_hidden_layers': 1, 'do_compile': True, 'model_ids': ['fixture/model']}}, path
    )
    restored = load_template_graph(path)
    assert restored['engine']['template_meta']['model_parameters'] == parameters
    result = generate(restored, {**captures[0].config, 'model_id': 'fixture/model'}, return_ops=True)
    assert compare_op_sequences(captures[0].ops, result['ops']).passed
    with pytest.raises(TemplateGraphRefusal, match='model identity'):
        generate(restored, {**captures[0].config, 'model_id': 'another/model'})
    with pytest.raises(ValueError, match='disagree'):
        generate(restored, {**captures[0].config, 'model_parameters': (('H', 64),)})


def test_cli_build_and_legacy_capture_share_public_configuration(tmp_path, monkeypatch, capsys):
    from cli.inference import template_graph as cli

    assert cli.main(['--help']) == 0
    assert 'capture' in capsys.readouterr().out
    with pytest.raises(ValueError, match='Expected template action'):
        cli.main(['unsupported-action'])
    calls = []

    def build(path, **options):
        calls.append((path, options))
        return {}, {'groups': 2}

    monkeypatch.setattr(cli, 'build_template_graph', build)
    args = ['--cache-dir', str(tmp_path), '--template', str(tmp_path / 'template.json.xz'), '--compile-mode', 'on']
    assert cli.main(['build', *args]) == 0
    assert json.loads(capsys.readouterr().out)['groups'] == 2
    assert calls[-1][1] == dict(cache_dir=str(tmp_path), do_compile=True, num_hidden_layers=None)
    assert cli.legacy_graph_main(['--build', *args]) == 0
    assert json.loads(capsys.readouterr().out)['groups'] == 2
    assert calls[0] == calls[1]

    forwarded = []
    monkeypatch.setattr(cli, 'main', lambda argv: forwarded.extend(argv) or 0)
    assert cli.legacy_capture_main(['--B', '3', '--S', '128', '--compile', '--cache-dir', str(tmp_path)]) == 0
    assert forwarded[0] == 'capture'
    spec = get_spec('template_capture')
    parsed = build_argparser(spec).parse_args(forwarded[1:])
    assert parsed.num_queries == 3 and parsed.query_length == 128 and parsed.compile


def test_legacy_query_retains_generation_refusals(tmp_path, monkeypatch, capsys):
    from cli.inference import template_graph as cli

    configs = [{'B': 3, 'S': 128}, {'B': 7, 'S': 128}]
    requests = tmp_path / 'targets.json'
    requests.write_text(json.dumps(configs), encoding='utf-8')
    monkeypatch.setattr(cli, 'load_template_graph', lambda _path: {})

    def generate_target(_graph, config, **_options):
        if config['B'] == 7:
            raise cli.TemplateGraphRefusal(config, 'structure_bucket_missing')
        return dict(
            covered=True, coverage='exact_group', op_count=1, execution_time_s={'analytic': 0.01}, wall_time_s=0
        )

    monkeypatch.setattr(cli, 'generate', generate_target)
    assert cli.legacy_graph_main(['--template', 'fixture.json.xz', '--sweep-json', str(requests), '--json']) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]['covered'] and rows[0]['op_count'] == 1
    assert not rows[1]['covered'] and rows[1]['blocker'] == 'structure_bucket_missing'


def test_diagnostic_cli_outputs_coverage_and_no_capture_resolution(tmp_path, monkeypatch, capsys):
    from cli.inference import template_graph_diagnostics as diagnostic
    from template_engine import zero_meta

    engine = {'groups': {}}
    monkeypatch.setattr(zero_meta, 'load_template_engine', lambda _path: engine)
    monkeypatch.setattr(
        diagnostic, 'describe_coverage', lambda _engine, config: dict(config=config, verdict='interpolated')
    )
    report = tmp_path / 'coverage.json'
    assert (
        diagnostic.coverage_main(
            [
                '--cache-dir',
                str(tmp_path),
                '--template',
                'fixture.json.xz',
                '--sweep',
                'B=3:4,S=128',
                '--out',
                str(report),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(report.read_text(encoding='utf-8'))
    assert result['interpolated'] == 2
    assert [row['config']['B'] for row in result['rows']] == [3, 4]
    assert diagnostic.coverage_main(['--cache-dir', str(tmp_path / 'empty'), '--configs', 'B=3,S=128']) == 2
    assert json.loads(capsys.readouterr().out)['error'] == 'no capture keys found'

    monkeypatch.setattr(diagnostic, 'load_or_build_template', lambda *_args, **_kwargs: (engine, 'fixture.json.xz'))
    calls = []

    def resolve(config, **options):
        calls.append((config, options))
        if config['B'] == 3:
            return dict(
                key='fixture',
                covered=True,
                source='template',
                op_count=1,
                execution_time_s={'analytic': 0.01},
                wall_time_s=0,
            )
        return dict(key='fixture', covered=False, blocker='structure_bucket_missing', wall_time_s=0)

    monkeypatch.setattr(diagnostic, 'resolve_config', resolve)
    assert (
        diagnostic.resolve_main(
            [
                '--cache-dir',
                str(tmp_path),
                '--template',
                'fixture.json.xz',
                '--config',
                'B=3,S=128,compile=true',
                '--config',
                'B=7,S=128,compile=true',
                '--no-capture',
                '--json',
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]['covered'] and rows[0]['op_count'] == 1
    assert rows[1]['blocker'] == 'structure_bucket_missing'
    assert calls[0][0] == dict(B=3, S=128, do_compile=True)
    assert calls[0][1]['allow_capture'] is False
    assert calls[1][1]['allow_capture'] is False


@pytest.mark.parametrize('decode', [False, True])
def test_cli_capture_and_generate_export_the_requested_graph(tmp_path, monkeypatch, capsys, decode):
    from cli.inference import template_graph as cli
    from template_engine.capture_ops import CapturedOp
    from template_engine.graph_codec import decode_capture

    request = [
        '--model-id',
        'fixture/model',
        '--num-queries',
        '3',
        '--query-length',
        '1' if decode else '128',
        '--context-length',
        '127' if decode else '0',
        '--tp-size',
        '1',
        '--ep-size',
        '2',
        '--compile',
        '--quantize-linear-action',
        'DISABLED',
    ]
    if decode:
        request.append('--decode')
    captured = []
    monkeypatch.setattr(cli, 'write_capture', lambda args: captured.append(args) or 0)
    assert cli.main(['capture', *request, '--cache-dir', str(tmp_path)]) == 0
    args = captured[0]
    assert (args.model_id, args.B, args.S, args.decode, args.compile) == ('fixture/model', 3, 128, decode, True)
    assert (args.tp, args.ep, args.moe_dp, args.dcp) == (1, 2, 1, 1)
    assert args.cache_dir == str(tmp_path) and args.quant_linear == 'DISABLED'

    x = torch.empty((3, 128, 8), device='meta')
    ops = [CapturedOp('aten.neg.default', (x,), {}, torch.empty_like(x))]
    forwarded = []
    monkeypatch.setattr(cli, 'load_template_graph', lambda _path: {'fixture': True})

    def expand(graph, config, **options):
        forwarded.append((graph, config, options))
        return {'ops': ops, 'op_count': 1, 'source': 'template_graph'}

    monkeypatch.setattr(cli, 'generate', expand)
    output = tmp_path / 'candidate.json.xz'
    generation = ['generate', *request, '--template', 'fixture.json.xz', '--output', str(output)]
    assert cli.main(generation) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['path'] == str(output.resolve()) and result['op_count'] == 1
    stored, _ = decode_capture(output.read_bytes())
    assert stored[0].func == 'aten.neg.default' and tuple(stored[0].out.shape) == (3, 128, 8)
    graph, config, options = forwarded[0]
    assert graph == {'fixture': True}
    assert (config['model_id'], config['B'], config['S'], config['is_decode']) == ('fixture/model', 3, 128, decode)
    assert options == dict(num_hidden_layers=None, return_ops=True)

    def refused(_graph, config, **_options):
        raise cli.TemplateGraphRefusal(config, 'structure_bucket_missing')

    monkeypatch.setattr(cli, 'generate', refused)
    with pytest.raises(SystemExit) as failure:
        cli.main(generation)
    assert failure.value.code == 2


@pytest.mark.parametrize("token", ["--cache-dir=-cache", "--cache-dir=--tp"])
def test_review_legacy_attached_values_are_not_option_names(token):
    parser = build_argparser(get_spec("template_capture"))
    args = parser.parse_args(legacy_capture_argv(["--B=2", "--S=8", token]))
    assert args.cache_dir == token.split("=", 1)[1]
    args = parser.parse_args(legacy_capture_argv(["--B=2", "--S=8", "--cache-dir", "--tp"]))
    assert args.cache_dir == "--tp"


def test_review_workflow_compares_complete_frozen_config(workflow):
    expected = {"num_hidden_layers": 2, "hidden_size": 8, "num_experts": 16, "max_position_embeddings": 4096}
    assert workflow.model_snapshot_matches(expected, ArtifactConfig(expected))
    for key in ("hidden_size", "num_experts", "max_position_embeddings"):
        assert not workflow.model_snapshot_matches(expected, ArtifactConfig({**expected, key: 32}))
    assert not workflow.model_snapshot_matches(expected, None)
    assert workflow.model_snapshot_matches({'id2label': {'0': 'LABEL_0'}}, {'id2label': {0: 'LABEL_0'}})
    assert not workflow.model_snapshot_matches({'id2label': {'0': 'LABEL_0'}}, {'id2label': {0: 'OTHER'}})


def test_review_workflow_budget_bounds_read_only_computation(workflow):
    import time

    calls = []
    with pytest.raises(workflow.BudgetReached, match="before"):
        workflow.bounded_read_call(lambda: calls.append(True), 0, "generation")
    assert not calls
    start = time.monotonic()
    with pytest.raises(workflow.BudgetReached, match="during"):
        workflow.bounded_read_call(lambda: time.sleep(0.2), 0.01, "comparison")
    assert time.monotonic() - start < 0.15


def test_capture_defaults_keep_official_model_parameters(monkeypatch, tmp_path):
    from cli.inference import template_graph as cli
    from cli.registry.modules.template_graph import DEFAULT_CAPTURE_DIR

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, 'write_capture', lambda args: calls.append(args) or 0)
    assert cli.main(['capture', '--model-id', 'owner/model', '--num-queries', '2', '--query-length', '256']) == 0
    assert calls[0].cache_dir == DEFAULT_CAPTURE_DIR
    assert calls[0].quant_linear == 'W8A8_DYNAMIC'
    assert calls[0].compile is False
    assert calls[0].layers == 0


@pytest.fixture
def default_cli_corpus(tmp_path, monkeypatch):
    from template_engine import registry

    monkeypatch.setattr(registry, '_TEMPLATES_DIR', tmp_path / 'template_engine/templates')
    monkeypatch.setattr(registry, '_LEGACY_PATH', tmp_path / 'absent.json.xz')
    registry._MODEL_META_CACHE.clear()
    from template_engine.cache_io import build_cache_key, save_capture
    from template_engine.capture_ops import CapturedOp
    from template_engine.graph_codec import encode_document
    from cli.registry.modules.template_graph import DEFAULT_CAPTURE_DIR

    monkeypatch.chdir(tmp_path)

    def populate(model='owner/model', compiled=True):
        for batch, sequence in ((2, 256), (3, 256), (2, 512), (3, 512)):
            key = build_cache_key(1, 1, 1, batch, sequence, do_compile=compiled)
            x = torch.empty((batch, sequence, 3), device='meta')
            path = save_capture(
                key,
                [CapturedOp('aten.neg.default', (x,), {}, torch.empty_like(x))],
                0.0,
                DEFAULT_CAPTURE_DIR,
                model_id=model,
            )
            Path(str(path) + '.meta').write_bytes(
                encode_document(dict(model_id=model, num_hidden_layers=2, weight_key='fixture'), 'model_meta/1')
            )
        return path.parent

    return populate


def test_build_defaults_are_discovered_by_optimizer(default_cli_corpus, capsys):
    from cli.inference import template_graph as cli
    from template_engine.registry import registration_path, template_path_for
    from template_engine.template_graph import load_template_graph

    default_cli_corpus()
    assert cli.main(['build']) == 0
    report = json.loads(capsys.readouterr().out)
    path = Path(report['built'])
    assert path == registration_path('owner/model', True)
    assert load_template_graph(path)['meta']['model_ids'] == ['owner/model']
    assert template_path_for('owner/model', 2, do_compile=True) == str(path)


def test_build_rejects_ambiguous_default_corpora(default_cli_corpus, capsys):
    from cli.inference import template_graph as cli

    default_cli_corpus('first/model')
    default_cli_corpus('second/model')
    with pytest.raises(SystemExit) as exc:
        cli.main(['build'])
    assert exc.value.code == 2
    assert 'Multiple model corpora' in capsys.readouterr().err
    assert not Path('template_engine/templates').exists()


def test_build_filters_compile_mode_and_preserves_explicit_output(default_cli_corpus, capsys):
    from cli.inference import template_graph as cli
    from template_engine.template_graph import load_template_graph

    corpus = default_cli_corpus(compiled=True)
    default_cli_corpus(compiled=False)
    with pytest.raises(SystemExit):
        cli.main(['build'])
    assert 'build COMPILE on/off independently' in capsys.readouterr().err
    assert cli.main(['build', '--cache-dir', str(corpus), '--compile-mode', 'on', '--template', 'custom.json.xz']) == 0
    assert load_template_graph('custom.json.xz')['meta']['do_compile'] is True
    assert not Path('template_engine/templates').exists()
