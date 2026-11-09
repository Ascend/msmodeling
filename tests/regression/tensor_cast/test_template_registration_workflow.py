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

"""Validated subset registration, optimizer discovery and per-call fallback."""

import pytest
import torch
from template_engine.capture_ops import CapturedOp
from template_engine.graph_codec import (
    GraphCodecError,
    decode_capture,
    decode_value,
    encode_capture,
    encode_value,
    loads_bytes,
    save_json_xz,
)


from pathlib import Path
from types import SimpleNamespace


from template_engine import registry
from template_engine.template_builder import _Capture, build_fixed_template
from template_engine.template_graph import generate, load_template_graph, TemplateGraphRefusal
from template_engine.workflow_registration import register_validated
from template_engine.zero_meta import _save_template_lazy

MODEL = 'owner/Workflow-Model'


@pytest.mark.parametrize('tp,batch', [(1, 37), (16, 173)])
def test_shipped_glm_decode_matches_native_batched_mla_contract(tp, batch):
    template = Path(__file__).resolve().parents[3] / 'template_engine/templates/glm_5_3_cmp_on.json.xz'
    graph = load_template_graph(template)
    cfg = dict(
        model_id='zai-org/GLM-5.3',
        TP=tp,
        EP=32,
        MOE_DP=1,
        DCP=1,
        B=batch,
        S=5009,
        is_decode=True,
        num_mtp_tokens=0,
        enable_dispatch_ffn_combine=False,
        do_compile=True,
        quantize_linear_action='W8A8_DYNAMIC',
        quantize_attention_action='INT8',
    )
    ops = generate(graph, cfg, return_ops=True)['ops']
    projections = [op for op in ops if str(op.func) == 'tensor_cast.batch_matmul_quant.default']
    assert len(projections) == 78
    local_batch = (batch + 32 // tp - 1) // (32 // tp)
    for op in projections:
        assert tuple(op.args[0].shape) == (64 // tp, local_batch, 192)
        assert tuple(op.args[1].shape) == (64 // tp, 192, 512)
        assert tuple(op.out.shape) == (64 // tp, local_batch, 512)


@pytest.mark.parametrize('has_pixels', [False, True])
def test_supplied_graph_uses_official_memory_and_observer(monkeypatch, has_pixels):
    from tensor_cast.core import model_runner as module

    class Runtime(SimpleNamespace):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    runner = object.__new__(module.ModelRunner)
    runner.user_input = SimpleNamespace(
        dump_input_shapes=False,
        dump_op_bound_results=False,
        reserved_memory_gb=1,
        num_queries=8,
        query_len=1,
        world_size=1,
        chrome_trace=None,
        block_size=128,
    )
    runner.perf_models = []
    runner.device_profile = object()
    runner.total_device_memory_gb = 32
    runner.model_weight_size_gb = 4
    runner.engram_host_resident_size_gb = 0
    runner.engram_host_resident_per_rank_gb = 0
    visual = object()
    runner.model = SimpleNamespace(
        model_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1), hf_config=SimpleNamespace(model_type='fixture')
        ),
        get_weight_size_nested=lambda _: 2 * 1024**3,
    )
    monkeypatch.setattr(module, 'get_visual', lambda _: visual)
    monkeypatch.setattr(module, 'dcp_kv_token_capacity_factor', lambda _: 1)
    monkeypatch.setattr(module, 'kv_cache_excluded_layer_indices', lambda _: {2})
    monkeypatch.setattr(runner, '_ensure_compiled', lambda: pytest.fail('supplied graph must not compile'))
    cache = torch.empty(1024, device='meta', dtype=torch.uint8)
    inputs = dict(
        kv_cache_by_layers={0: cache, 1: cache, 2: torch.empty(512, device='meta')},
        indexer_cache_by_layers={0: cache},
        indexer_scale_cache_by_layers={0: cache},
        compressed_kv_cache_by_source={0: cache},
        compressed_kv_scale_cache_by_source={0: cache},
        kv_cache_per_token=16,
        indexer_cache_per_token=4,
        compressed_kv_cache_per_token=8,
        pixel_values=object() if has_pixels else None,
    )
    runtime = Runtime(
        event_list=[],
        memory_tracker=SimpleNamespace(peak_mem_usage=lambda: 10 * 1024**3),
        total_execution_time_s=lambda: {'analytic': 0.1},
        table_averages=lambda **_: [],
        get_breakdowns=lambda: {},
    )
    monkeypatch.setattr(module, 'Runtime', lambda *_a, **_k: runtime)
    monkeypatch.setattr(module, 'MemoryTracker', lambda _: object())
    provided = []

    def supply(kwargs):
        assert kwargs is inputs
        provided.append('graph')
        return ['supplied-ops']

    observer = []
    metrics = runner.run_inference(
        [SimpleNamespace(concurrency=8, query_len=1, is_decode=True)],
        generate_inputs_func=lambda *_a, **_k: inputs,
        runtime_observer=observer.append,
        graph_provider=supply,
    )
    assert provided == ['graph'] and runtime.op_invoke_infos == ['supplied-ops']
    assert metrics.kv_cache_size_gb == 3 * 1024 / 1024**3
    assert metrics.kv_cache_per_token_gb == 28 / 1024**3
    assert metrics.model_weight_size_gb == (4 if has_pixels else 2)
    assert metrics.device_memory_available_gb == 21
    assert metrics.tps_per_model == {'analytic': 80}
    assert observer == [runtime]


def template_workflow_script():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / '.agents/skills/msmodeling-template-workflow/scripts/workflow.py'
    spec = importlib.util.spec_from_file_location('template_workflow_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_long_request_uses_official_prefill_and_decode_shapes():
    workflow = template_workflow_script()
    assert workflow.request_shape(16384, 1024, False) == (16384, 16384)
    assert workflow.request_shape(16384, 1024, True) == (1, 16897)
    assert workflow.request_shape(16384, 1024, True, 3) == (4, 16900)


def test_skill_plan_covers_requested_rank_batches_and_long_request_without_target_leaks(tmp_path, monkeypatch):
    import json

    workflow = template_workflow_script()
    scope = workflow.read(workflow.PROFILE)
    scope['requested'].update(
        world_size=[32, 32],
        B=[1, 65],
        S=[1, 65536],
        MTP=[0, 1],
        DFC=[False],
        linear_families=['W8A8'],
        attention_actions=['INT8'],
    )
    scope['typical_moe_layouts'] = [[1, 32, 1, 1], [32, 32, 1, 1]]
    scope_file, model_file = tmp_path / 'scope.json', tmp_path / 'config.json'
    scope_file.write_text(json.dumps(scope), encoding='utf8')
    model_file.write_text(
        json.dumps({'num_hidden_layers': 2, 'n_routed_experts': 256, 'max_position_embeddings': 65536}), encoding='utf8'
    )
    monkeypatch.setattr(workflow, 'epoch', lambda: {'head': 'test'})
    run_dir = tmp_path / 'plan'
    workflow.plan(
        SimpleNamespace(
            model_id=MODEL,
            model_config=model_file,
            scope=scope_file,
            run_dir=run_dir,
            input_length=16384,
            output_length=1024,
            decode_batch_coverage='all',
        )
    )
    document = workflow.read(run_dir / 'plan.json')
    assert document['cells']
    assert any(c['config']['TP'] == c['config']['EP'] == 32 for c in document['cells'])
    for cell in document['cells']:
        assert cell['config']['quantize_linear_action'].startswith('W8A8')
        assert cell['config']['quantize_attention_action'] == 'INT8'
        dp = cell['DP']
        samples = cell['samples']
        train = {((s['config']['B'] + dp - 1) // dp, s['config']['S']) for s in samples if s['role'] == 'train'}
        for sample in samples:
            if sample['role'] != 'train':
                assert ((sample['config']['B'] + dp - 1) // dp, sample['config']['S']) not in train
        long_target = next(s for s in samples if s['role'] == 'long_request')
        assert long_target['request_lengths'] == {'input_length': 16384, 'output_length': 1024}
        if cell['config']['is_decode']:
            expected = list(range(1, (65 + dp - 1) // dp + 1))
            assert cell['decode_local_batches'] == expected
            assert not cell['unplanned_decode_local_batches']
            assert all(len({s for k, s in train if k == local}) >= 2 for local in expected)
            assert long_target['config']['S'] == 16897 + cell['config']['num_mtp_tokens']
        else:
            assert cell['config']['num_mtp_tokens'] == 0
            assert long_target['config']['S'] == 16384


@pytest.mark.parametrize('enum_name', ['LinearQuantType', 'AttentionQuantType', 'QuantScheme', 'QuantGranularity'])
def test_quantization_enum_members_survive_artifact_roundtrip(enum_name):
    from tensor_cast import quantize_utils
    from template_engine.graph_codec import decode_document, encode_document

    for member in getattr(quantize_utils, enum_name):
        restored = decode_document(encode_document({'quantization': member}, 'model_meta/1'), 'model_meta/1')
        assert restored['quantization'] is member


def test_glm5_attention_metadata_roundtrip_keeps_exact_export_boundary():
    from tensor_cast.layers.glm5 import Glm5SparseAttention, glm5_uses_indexshare
    from template_engine.graph_codec import GraphCodecError, decode_document, encode_document

    metadata = {'attention_cls': Glm5SparseAttention}
    restored = decode_document(encode_document(metadata, 'model_meta/1'), 'model_meta/1')
    assert restored['attention_cls'] is Glm5SparseAttention
    with pytest.raises(GraphCodecError, match='not an approved export'):
        decode_document(encode_document({'function': glm5_uses_indexshare}, 'model_meta/1'), 'model_meta/1')


def test_w8a8_c8_model_metadata_preserves_quantization_configuration():
    from tensor_cast.model_config import (
        AttentionQuantConfig,
        LinearQuantConfig,
        ModelConfig,
        ParallelConfig,
        QuantConfig,
    )
    from tensor_cast.quantize_utils import AttentionQuantType, LinearQuantType, QuantGranularity, QuantScheme
    from template_engine.graph_codec import decode_document, encode_document

    metadata = {
        'model_id': MODEL,
        'num_hidden_layers': 61,
        'model_config': ModelConfig(
            ParallelConfig(world_size=32, tensor_parallel_size=4, expert_parallel_size=32),
            QuantConfig(
                linear_configs={'projection': LinearQuantConfig(quant_type=LinearQuantType.W8A8)},
                attention_configs={0: AttentionQuantConfig(quant_type=AttentionQuantType.INT8)},
            ),
        ),
    }
    restored = decode_document(encode_document(metadata, 'model_meta/1'), 'model_meta/1')
    quant = restored['model_config'].quant_config
    assert quant.linear_configs['projection'].quant_type is LinearQuantType.W8A8
    assert quant.linear_configs['projection'].weight_quant_scheme is QuantScheme.SYMMETRIC
    assert quant.linear_configs['projection'].dynamic_quant_granularity is QuantGranularity.PER_TENSOR
    assert quant.attention_configs[0].quant_type is AttentionQuantType.INT8
    assert restored['model_config'].parallel_config == metadata['model_config'].parallel_config
    assert restored['num_hidden_layers'] == 61


def test_performance_acceptance_ignores_only_stride():
    from template_engine.compare_ops import compare_performance_ops

    view = torch.empty_strided((3,), (2,), device='meta')
    dense = torch.empty((3,), device='meta')
    expected = [CapturedOp('aten.alias.default', (view,), {}, view)]
    actual = [CapturedOp('aten.alias.default', (dense,), {}, dense)]
    report = compare_performance_ops(expected, actual)
    assert report['passed'] and not report['strict_passed']
    assert report['mismatches']['stride'] > 0
    for tensor in (torch.empty((4,), device='meta'), torch.empty((3,), device='meta', dtype=torch.int64)):
        assert not compare_performance_ops(expected, [CapturedOp('aten.alias.default', (tensor,), {}, tensor)])[
            'passed'
        ]


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, '_TEMPLATES_DIR', tmp_path / 'registry')
    monkeypatch.setattr(registry, '_LEGACY_PATH', tmp_path / 'absent.json.xz')
    registry._MODEL_META_CACHE.clear()
    captures, records = [], []
    for action in ('DISABLED', 'W8A8_DYNAMIC'):
        for b, s in ((2, 256), (2, 1024), (3, 256), (3, 1024), (5, 1024)):
            c = dict(
                model_id=MODEL,
                B=b,
                S=s,
                TP=1,
                EP=2,
                MOE_DP=1,
                DCP=1,
                do_compile=True,
                is_decode=False,
                num_mtp_tokens=0,
                enable_dispatch_ffn_combine=False,
                quantize_linear_action=action,
                quantize_attention_action='DISABLED',
                model_parameters=(('H', 3),),
            )
            x = torch.empty(((b + 1) // 2, s, 3), device='meta')
            ops = [CapturedOp('aten.neg.default', (x,), {}, torch.empty_like(x))]
            view = torch.empty_strided((3,), (2,), device='meta', dtype=torch.int64)
            ops.append(CapturedOp('aten.alias.default', (view,), {}, view))
            key = str((action, b, s))
            captures.append(_Capture(key, c, ops, 0.0))
            path = tmp_path / f'train-{len(records)}.json.xz'
            path.write_bytes(encode_capture(ops, 0.0))
            records.append(dict(config=c, graph=str(path)))
    engine, _ = build_fixed_template(captures)
    graph = dict(engine=engine, meta=dict(model=MODEL, model_ids=[MODEL], num_hidden_layers=2, do_compile=True))
    source = tmp_path / 'source.json.xz'
    _save_template_lazy(graph, source)
    config = {**captures[0].config, 'B': 4, 'S': 768}
    result = generate(graph, config, return_ops=True, num_hidden_layers=2)
    targets = [dict(config=config, status='compared_pass', structure_bucket_id=result['structure_bucket_id'])]
    return source, config, targets, records


def test_subset_registers_and_optimizer_hits_then_falls_back(artifacts, tmp_path, caplog):
    from serving_cast.parallel_runner import ParallelRunner
    from tensor_cast.model_config import ParallelConfig

    source, config, targets, training = artifacts
    report = register_validated(source, MODEL, 2, True, targets, training, tmp_path / 'run')
    assert report['status'] == 'registered' and report['excluded_buckets'] == 1
    assert report['training_anchors_checked'] == 5
    assert registry.template_path_for(MODEL, 2, do_compile=True) == report['path']
    assert registry.template_path_for('other/Workflow-Model', 2, do_compile=True) is None
    assert len(load_template_graph(source)['engine']['groups']) == 2
    assert len(load_template_graph(report['path'])['engine']['groups']) == 1
    installed_ops = generate(load_template_graph(report['path']), config, return_ops=True)['ops']
    assert installed_ops[-1].out.stride() == (2,)
    assert installed_ops[-1].args[0] is installed_ops[-1].out
    again = register_validated(source, MODEL, 2, True, targets, training, tmp_path / 'run')
    assert again['sha256'] == report['sha256'] and again['backup'] is None

    calls = {'native': 0, 'template': 0}

    def native(*_args, **kwargs):
        if 'graph_provider' in kwargs:
            assert callable(kwargs['graph_provider'])
            calls['template'] += 1
            return SimpleNamespace()
        calls['native'] += 1
        return 'native'

    user = SimpleNamespace(
        model_id=MODEL,
        do_compile=True,
        num_mtp_tokens=0,
        quantize_linear_action='DISABLED',
        quantize_attention_action='DISABLED',
        enable_dispatch_ffn_combine=False,
    )
    inner = SimpleNamespace(
        user_input=user,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                parallel_config=ParallelConfig(world_size=2, tensor_parallel_size=1, expert_parallel_size=2)
            )
        ),
        run_inference=native,
    )
    optimizer = object.__new__(ParallelRunner)
    optimizer.args = SimpleNamespace(model_id=MODEL, template_graph_optimization=True)
    optimizer._build_base_model_runner = lambda _user: inner
    runner = optimizer._build_model_runner(user)

    request = SimpleNamespace(concurrency=config['B'], query_len=config['S'], seq_len=config['S'], is_decode=False)
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert runner.run_inference([request]).source_tag == 'template_graph'
    assert '[template HIT]' in caplog.text
    assert 'B=4 S=768 TP=1 EP=2' in caplog.text
    caplog.clear()
    user.quantize_linear_action = 'W4A8_STATIC'
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert runner.run_inference([request]) == 'native'
    assert '[template FALLBACK]' in caplog.text
    assert 'B=4 S=768 TP=1 EP=2' in caplog.text
    assert calls == {'native': 1, 'template': 1}
    inner.model.num_hidden_layers = 3
    caplog.clear()
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert optimizer._build_model_runner(user) is inner
    assert 'scope=runner' in caplog.text
    assert 'reason=no_matching_registered_template' in caplog.text
    optimizer.args.template_graph_optimization = False
    caplog.clear()
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert optimizer._build_model_runner(user) is inner
    assert 'reason=template_optimization_disabled' in caplog.text


@pytest.mark.parametrize(
    'decode,sampler,query,sequence,reason',
    [
        (False, True, 8, 8, 'sampler'),
        (False, False, 8, 16, 'chunked_prefill'),
        (True, False, 1, 16, 'decode_semantics_missing'),
        (True, False, 2, 16, 'decode_query_length_not_covered'),
    ],
)
def test_policy_fallback_is_visible(caplog, decode, sampler, query, sequence, reason):
    from template_engine.runner_bridge import TemplateBackedModelRunner

    inner = SimpleNamespace(
        user_input=SimpleNamespace(model_id=MODEL, do_compile=True),
        model=SimpleNamespace(),
        run_inference=lambda *_args: 'native',
    )
    runner = TemplateBackedModelRunner(inner)
    runner._get_graph = lambda: {
        'meta': {'decode_query_semantics': 'mtp_plus_1'} if reason == 'decode_query_length_not_covered' else {}
    }
    request = SimpleNamespace(concurrency=4, query_len=query, seq_len=sequence, is_decode=decode)
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert runner.run_inference([request], with_sampler=sampler) == 'native'
    assert '[template FALLBACK]' in caplog.text
    assert f'reason={reason}' in caplog.text
    assert f'B=4 S={sequence if decode else query}' in caplog.text


def test_optimizer_bypass_skips_registry_and_workload_cache(monkeypatch, caplog):
    from serving_cast.parallel_runner import ParallelRunner

    def unexpected_lookup(*_args, **_kwargs):
        pytest.fail('Disabled template optimization must not query the registry')

    monkeypatch.setattr(registry, 'template_path_for', unexpected_lookup)
    user = SimpleNamespace(model_id=MODEL)
    calls = []
    inner = SimpleNamespace(user_input=user, run_inference=lambda: calls.append('native'))
    optimizer = object.__new__(ParallelRunner)
    optimizer.args = SimpleNamespace(model_id=MODEL, template_graph_optimization=False)
    optimizer._workload_cache = object()  # Any cache access would fail.
    optimizer._get_model_runnner = lambda _user: inner
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        runner = optimizer._build_model_runner(user)
    assert runner is inner
    runner.run_inference()
    assert calls == ['native']
    assert 'reason=template_optimization_disabled' in caplog.text


def test_failed_readback_restores_previous_registry(artifacts, tmp_path, monkeypatch):
    source, _config, targets, training = artifacts
    installed = registry.register_template(source, MODEL, 2, do_compile=True)
    before = Path(installed).read_bytes()
    monkeypatch.setattr(registry, 'template_path_for', lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match='readback'):
        register_validated(source, MODEL, 2, True, targets, training, tmp_path / 'run')
    assert Path(installed).read_bytes() == before
    assert list((tmp_path / 'run/registration/backups').glob('*.json.xz'))


def test_pending_or_mismatched_bucket_is_not_registered(artifacts, tmp_path):
    source, _config, targets, training = artifacts
    for status in ('generated_only', 'mismatch'):
        report = register_validated(
            source, MODEL, 2, True, [*targets, {**targets[0], 'status': status}], training, tmp_path / status
        )
        assert report['status'] == 'not_registered'
    assert not registry.registration_path(MODEL, True).exists()


def test_registration_rejects_wrong_model_and_depth(artifacts):
    source, *_ = artifacts
    with pytest.raises(ValueError, match='identity'):
        registry.register_template(source, 'other/Workflow-Model', 2, do_compile=True)
    with pytest.raises(ValueError, match='layer'):
        registry.register_template(source, MODEL, 3, do_compile=True)


def test_optimizer_request_preserves_official_dcp_axis():
    from tensor_cast.model_config import ParallelConfig
    from template_engine.runner_bridge import TemplateBackedModelRunner

    parallel = ParallelConfig(
        world_size=4, tensor_parallel_size=2, expert_parallel_size=4, decode_context_parallel_size=2
    )
    user = SimpleNamespace(
        model_id=MODEL,
        do_compile=True,
        num_mtp_tokens=0,
        quantize_linear_action='DISABLED',
        quantize_attention_action='DISABLED',
    )
    inner = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(parallel_config=parallel)), user_input=user
    )
    runner = TemplateBackedModelRunner(inner)
    request = SimpleNamespace(concurrency=4, query_len=1, seq_len=768, is_decode=True)
    assert runner._template_config(request)['DCP'] == 2


def _sample_tree():
    shared = torch.empty_strided((4, 8), (8, 1), dtype=torch.bfloat16)
    other = torch.empty_strided((3,), (1,), dtype=torch.int8)
    op = CapturedOp(
        func="aten.view.default",
        args=[shared, {"size": (2, 4), "scale": other}],
        kwargs={"dtype": torch.float32, "memory": torch.contiguous_format},
        out=[shared, other],
    )
    return op, shared, other


def test_roundtrip_preserves_alias_identity_and_tuples():
    op, shared, other = _sample_tree()
    tensors: list[dict] = []
    ids: dict[int, int] = {}
    encoded = encode_value([op, (1, 2), {"k": shared}], tensors, ids)
    # aliasing: shared appears in args and out - one tensor entry
    assert len(tensors) == 2
    decoded = decode_value(
        encoded,
        [  # placeholder rebuild
            torch.empty_strided(
                tuple(t["s"]),
                tuple(t["st"]),
                dtype={"torch.bfloat16": torch.bfloat16, "torch.int8": torch.int8}[t["d"]],
            )
            for t in tensors
        ],
    )
    new_op = decoded[0]
    assert isinstance(new_op, CapturedOp)
    assert new_op.func == "aten.view.default"
    assert isinstance(decoded[1], tuple) and tuple(decoded[1]) == (1, 2)
    # alias identity: args[0] and out[0] are THE SAME rebuilt object
    assert new_op.args[0] is new_op.out[0]
    assert decoded[2]["k"] is new_op.out[0]
    assert new_op.args[0].shape == torch.Size([4, 8])
    assert new_op.args[0].stride() == torch.Size([8, 1])
    assert new_op.args[0].dtype == torch.bfloat16
    assert new_op.kwargs["dtype"] == torch.float32
    assert new_op.kwargs["memory"] == torch.contiguous_format


def test_roundtrip_preserves_alias_identity_through_capture_payload():
    op, _shared, _other = _sample_tree()
    payload = encode_capture([op, op], 1.5)
    ops, exec_time = decode_capture(payload)
    assert exec_time == 1.5
    assert ops[0].args[0] is ops[0].out[0]
    assert ops[0].args[0] is ops[1].args[0]


def test_unknown_dtype_names_cannot_instantiate_arbitrary_objects():
    with pytest.raises(GraphCodecError):
        decode_value({"__dt__": "torch.system"}, [])
    with pytest.raises(GraphCodecError):
        decode_value({"__mf__": "torch.system"}, [])


def test_registry_metadata_obeys_artifact_size_limits(tmp_path, monkeypatch):
    from template_engine import graph_codec

    artifact = tmp_path / 'oversized.json.xz'
    artifact.write_bytes(
        graph_codec.encode_document({'meta': {'model': 'fixture/model', 'padding': 'x' * 1024}}, 'template_lazy/1')
    )
    registry._MODEL_META_CACHE.clear()
    assert registry._load_meta(artifact)['model'] == 'fixture/model'
    monkeypatch.setattr(graph_codec, 'MAX_UNCOMPRESSED_BYTES', 128)
    registry._MODEL_META_CACHE.clear()
    with pytest.raises(graph_codec.GraphCodecError, match='size'):
        registry._load_meta(artifact)


def test_save_json_xz_atomic_roundtrip(tmp_path):
    payload = {"format": "test/1", "value": [1, 2, 3]}
    target = tmp_path / "artifact.json.xz"
    save_json_xz(payload, target)
    assert loads_bytes(target.read_bytes()) == payload
    # a zero-byte file must not look valid
    empty = tmp_path / "empty.json.xz"
    empty.write_bytes(b"")
    from template_engine.graph_codec import GraphCodecError as _E

    with pytest.raises(_E):
        loads_bytes(empty.read_bytes())


def test_decompression_bomb_is_refused(tmp_path):
    """Review issue #32 PR security follow-up: an xz stream that would
    expand past the uncompressed-size cap is refused BEFORE parsing, and
    truncated streams refuse honestly as well.
    """

    import lzma

    from template_engine import graph_codec

    bomb = lzma.compress(b"\x00" * (8 * 1024 * 1024), preset=9)
    assert len(bomb) < 1024 * 1024  # ~8000x ratio, tiny on disk
    # the cap is lowered in-process so the test stays fast: point it under
    # the bomb's expanded size
    original = graph_codec.MAX_UNCOMPRESSED_BYTES
    graph_codec.MAX_UNCOMPRESSED_BYTES = 1 * 1024 * 1024
    try:
        with pytest.raises(GraphCodecError, match="uncompressed-size cap"):
            loads_bytes(bomb)
    finally:
        graph_codec.MAX_UNCOMPRESSED_BYTES = original
    # truncated stream: magic bytes then garbage cut-off
    truncated = lzma.compress(b"hello world", preset=6)[:20]
    with pytest.raises(GraphCodecError):
        loads_bytes(truncated)


def test_atomic_write_never_leaves_fixed_tmp(tmp_path):
    """Review issue #32 PR security follow-up: writers use mkstemp in the
    destination directory - no fixed ``<target>.tmp`` path that a planted
    symlink could redirect - and the target content lands intact.
    """

    from template_engine.graph_codec import atomic_write_bytes

    target = tmp_path / "out.json.xz"
    atomic_write_bytes(target, b"payload-bytes")
    assert target.read_bytes() == b"payload-bytes"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != target.name]
    assert not leftovers, f"temporary files leaked: {leftovers}"


@pytest.mark.parametrize(
    "payload",
    [
        {"__u__": [1], "name": "x"},
        {"__t__": 0},
        {"__d__": "literal"},
        {"(1, 2)": "x", "(1)": "y", "([1])": "z"},
    ],
)
def test_review_codec_escapes_sentinels_and_keeps_string_keys(payload):
    assert decode_value(encode_value(payload, [], {}), []) == payload


@pytest.mark.parametrize(
    "tag,value",
    [
        ("__dc__", {"cls": "evil_module.Record", "fields": {}}),
        ("__fn__", "evil_module.function"),
        ("__cls__", "evil_module.Class"),
        ("__enum__", "evil_module.Kind.VALUE"),
        ("__fn__", "tensor_cast.quantize_utils.get_torch_dtype_from_quant_type"),
        ("__enum__", "tensor_cast.quantize_utils.QuantScheme.__members__"),
    ],
)
def test_review_codec_refuses_unknown_modules_before_import(tag, value, monkeypatch):
    import importlib

    calls = []
    monkeypatch.setattr(importlib, "import_module", lambda name: calls.append(name))
    with pytest.raises(GraphCodecError):
        decode_value({tag: value}, [])
    assert not calls


def test_review_hf_snapshots_are_data_only_and_decoder_memory_is_capped(monkeypatch):
    import lzma
    from template_engine import graph_codec

    snapshot = decode_value(
        {"__cfg__": {"cls": "transformers.untrusted.CustomConfig", "dict": {"hidden_size": 17}}}, []
    )
    assert snapshot.hidden_size == 17 and snapshot.to_dict() == {"hidden_size": 17}
    compressed = lzma.compress(b'{}', preset=9)
    monkeypatch.setattr(graph_codec, "MAX_DECODER_MEMORY_BYTES", 1024 * 1024)
    with pytest.raises(GraphCodecError, match="memory limit"):
        loads_bytes(compressed)


def test_review_oversized_file_is_rejected_before_body_read(tmp_path, monkeypatch):
    from template_engine import graph_codec

    path = tmp_path / "large.json"
    with path.open("wb") as stream:
        stream.truncate(4096)
    monkeypatch.setattr(graph_codec, "MAX_UNCOMPRESSED_BYTES", 128)
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("unbounded read")))
    with pytest.raises(GraphCodecError, match="input-size"):
        graph_codec.load_json_xz(path)


@pytest.mark.parametrize("identities", [7, True, "owner/Workflow-Model", [7], None, []])
def test_review_malformed_registry_identities_do_not_crash(identities):
    meta = {"model_ids": identities}
    assert registry._model_matches(meta, MODEL) is False


def test_review_registry_requires_explicit_known_depth(tmp_path, monkeypatch):
    from template_engine.graph_codec import encode_document

    folder = tmp_path / "registry"
    folder.mkdir()
    monkeypatch.setattr(registry, "_TEMPLATES_DIR", folder)
    monkeypatch.setattr(registry, "_LEGACY_PATH", tmp_path / "absent")
    path = registry.registration_path(MODEL, True)
    path.write_bytes(encode_document({"meta": {"model_ids": [MODEL], "do_compile": True}}, "template_lazy/1"))
    assert registry.template_path_for(MODEL, 32, do_compile=True) is None
    assert registry.template_path_for(MODEL, do_compile=True) == str(path)


def test_review_cache_namespaces_and_layer_options_are_distinct(tmp_path):
    from template_engine.cache_io import cache_path, save_capture, load_capture, cache_variant_dir, build_cache_key

    key = build_cache_key(1, 1, 1, 2, 8)
    assert cache_path(key, tmp_path, model_id="org/name") != cache_path(key, tmp_path, model_id="org__name")
    for layers, sampler, elapsed in ((0, False, 1), (2, False, 2), (2, True, 3)):
        root = cache_variant_dir(tmp_path, layers, sampler)
        save_capture(key, [], elapsed, root, model_id=MODEL)
        assert load_capture(key, root, model_id=MODEL)[1] == elapsed
    assert load_capture(key, tmp_path, model_id="another/model") is None


def _review_cache_writer(root, elapsed, ready, start, results):
    import time
    from template_engine import cache_io

    original = cache_io.encode_capture

    def slow_encode(ops, seconds):
        time.sleep(0.2)
        return original(ops, seconds)

    cache_io.encode_capture = slow_encode
    ready.put(True)
    start.wait(20)
    cache_io.save_capture("test-key", [], elapsed, root, model_id=MODEL)
    results.put(cache_io.load_capture("test-key", root, model_id=MODEL)[1])


def test_review_simultaneous_capture_writers_keep_the_first_publication(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    ready, start, results = context.Queue(), context.Event(), context.Queue()
    workers = [
        context.Process(target=_review_cache_writer, args=(str(tmp_path), value, ready, start, results))
        for value in (1, 2)
    ]
    for worker in workers:
        worker.start()
    try:
        for _ in workers:
            assert ready.get(timeout=30)
        start.set()
        observed = [results.get(timeout=30) for _ in workers]
        assert observed[0] == observed[1]
    finally:
        for worker in workers:
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            assert worker.exitcode == 0


def test_review_meta_tensor_pool_keeps_sources_alive_for_lazy_iterables():
    import weakref
    import gc
    from template_engine.capture_ops import serialize_op_invoke_infos

    references = []

    def records():
        for n in range(1, 5):
            if references:
                gc.collect()
                assert all(ref() is not None for ref in references)
            tensor = torch.empty((n,), device="meta")
            references.append(weakref.ref(tensor))
            yield CapturedOp("aten.alias.default", (tensor,), {}, tensor)

    serialized = serialize_op_invoke_infos(records())
    assert [op.out.shape[0] for op in serialized] == [1, 2, 3, 4]


def test_review_lazy_group_is_decoded_once_for_concurrent_readers(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from template_engine import zero_meta

    calls = []

    def decode(blob):
        calls.append(blob)
        time.sleep(0.02)
        return {"entries": []}

    monkeypatch.setattr(zero_meta, "_decode_body_blob", decode)
    engine = {"groups": {"g": {}}, "_body_blobs": {"g": b"body"}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: zero_meta.ensure_group_body(engine, "g"), range(8)))
    assert calls == [b"body"] and engine["groups"]["g"] == {"entries": []}


def test_review_zero_meta_and_lookup_keep_distinct_layer_caches(tmp_path, monkeypatch):
    from template_engine import capture_ops
    from template_engine.template_lookup import ensure_ops
    from template_engine.zero_meta import resolve_config

    calls = []

    def capture_stub(*args, **kwargs):
        layer = kwargs["num_hidden_layers_override"]
        calls.append(layer)
        return [], float(layer)

    monkeypatch.setattr(capture_ops, "capture", capture_stub)
    cfg = dict(B=2, S=8, TP=1, EP=1, MOE_DP=1, DCP=1, do_compile=True, model_id=MODEL)
    for depth in (1, 61):
        out = resolve_config(cfg, cache_dir=tmp_path, capture_fn=capture_stub, num_hidden_layers=depth)
        assert out["execution_time_s"]["capture"] == depth
    assert resolve_config(cfg, cache_dir=tmp_path, capture_fn=capture_stub, num_hidden_layers=1)["source"] == "cache"
    for depth in (1, 61):
        out = ensure_ops(2, 8, model_id=MODEL, cache_dir=tmp_path, num_hidden_layers_override=depth, do_compile=True)
        assert out["exec_time_s"] == depth
    assert calls == [1, 61]


def test_review_decode_stamp_requires_every_decode_capture(tmp_path):
    from template_engine.cache_io import build_cache_key, save_capture
    from template_engine.graph_codec import encode_document
    from template_engine.template_graph import build_template_graph

    paths = {}
    for decode in (False, True):
        key = build_cache_key(1, 1, 1, 2, 8, is_decode=decode, do_compile=True)
        x = torch.empty((2, 8), device="meta")
        paths[decode] = save_capture(
            key, [CapturedOp("aten.neg.default", (x,), {}, torch.empty_like(x))], 0.0, tmp_path
        )
    metadata = {
        "model_id": MODEL,
        "weight_key": "fixture",
        "num_hidden_layers": 1,
        "do_compile": True,
        "decode_query_semantics": "mtp_plus_1",
    }
    Path(str(paths[False]) + ".meta").write_bytes(encode_document(metadata, "model_meta/1"))
    graph, _ = build_template_graph(tmp_path / "graph.json.xz", tmp_path, num_hidden_layers=1, do_compile=True)
    assert graph["meta"]["decode_query_semantics"] is None
    Path(str(paths[True]) + ".meta").write_bytes(encode_document(metadata, "model_meta/1"))
    graph, _ = build_template_graph(tmp_path / "graph.json.xz", tmp_path, num_hidden_layers=1, do_compile=True)
    assert graph["meta"]["decode_query_semantics"] == "mtp_plus_1"


@pytest.mark.parametrize("depth", [0, -1, True, 1.5, 80])
def test_review_explicit_depth_cannot_be_silently_ignored(depth):
    with pytest.raises(
        TemplateGraphRefusal,
        match="layer count",
    ):
        generate(
            {"engine": {"groups": {}}, "meta": {"num_hidden_layers": 2}},
            {"B": 2, "S": 8},
            num_hidden_layers=depth,
        )


def test_official_deepseek_metadata_exports_remain_supported():
    from tensor_cast.layers.deepseek_v4 import DeepseekV4SparseAttention, route_deepseek_v4_gate

    for value in (DeepseekV4SparseAttention, route_deepseek_v4_gate):
        assert decode_value(encode_value(value, [], {}), []) is value


@pytest.mark.parametrize('error_type', [ValueError, TypeError, RuntimeError])
def test_runner_does_not_hide_input_or_internal_errors(monkeypatch, error_type):
    from template_engine.runner_bridge import TemplateBackedModelRunner
    from template_engine import template_graph

    def fail(*_args, **_kwargs):
        raise error_type('invalid input or implementation failure')

    def native(*_args, **_kwargs):
        pytest.fail('Input and internal errors must not be converted to a native fallback')

    inner = SimpleNamespace(model=SimpleNamespace(), run_inference=native)
    runner = TemplateBackedModelRunner(inner)
    runner._get_graph = lambda: {'meta': {}}
    runner._template_config = lambda _request: {}
    monkeypatch.setattr(template_graph, 'generate', fail)
    request = SimpleNamespace(is_decode=False, seq_len=8, query_len=8)
    with pytest.raises(error_type, match='invalid input'):
        runner.run_inference([request])


def test_capture_metadata_write_failure_is_reported(monkeypatch, tmp_path):
    from cli.inference import template_graph as cli
    from template_engine import capture_ops, graph_codec

    monkeypatch.setattr(capture_ops, 'capture', lambda *_args, **_kwargs: ([], 0.0))
    original = graph_codec.atomic_write_bytes

    def write(path, blob):
        if str(path).endswith('.meta'):
            raise OSError('metadata destination is not writable')
        return original(path, blob)

    monkeypatch.setattr(graph_codec, 'atomic_write_bytes', write)
    with pytest.raises(OSError, match='metadata destination'):
        cli.main(
            [
                'capture',
                '--model-id',
                MODEL,
                '--num-queries',
                '2',
                '--query-length',
                '8',
                '--compile',
                '--cache-dir',
                str(tmp_path),
            ]
        )


def test_windows_model_paths_use_short_names_and_keep_previous_discovery(tmp_path, monkeypatch):
    from template_engine.graph_codec import encode_document
    import re

    identity = r'C:\models\Qwen3-0.6B'
    monkeypatch.setattr(registry, '_TEMPLATES_DIR', tmp_path)
    monkeypatch.setattr(registry, '_LEGACY_PATH', tmp_path / 'absent.json.xz')
    registry._MODEL_META_CACHE.clear()
    assert registry.slugify_model_id(identity) == 'qwen3_0_6b'
    legacy = re.sub(r'[^0-9a-zA-Z]+', '_', identity).lower().strip('_')
    previous = tmp_path / f'{legacy}_cmp_on.json.xz'
    previous.write_bytes(
        encode_document(
            {'meta': {'model_ids': [identity], 'num_hidden_layers': 28, 'do_compile': True}}, 'template_lazy/1'
        )
    )
    assert registry.template_path_for(identity, 28, do_compile=True) == str(previous)


def test_compact_body_preserves_formulas_tensors_and_container_identity():
    import json
    from template_engine.zero_meta import _encode_body_blob, _decode_body_blob

    tensor = torch.empty_strided((2, 3), (7, 2), dtype=torch.bfloat16, device="meta")
    formula = {"shape": [{"type": "B"}, None], "stride": [None, None]}
    body = {
        "entries": [
            {
                "func": "aten.alias.default",
                "base_args": (tensor,),
                "base_kwargs": {},
                "base_out": tensor,
                "args_f": [formula],
                "out_f": formula,
            },
            {
                "func": "aten.neg.default",
                "base_args": (tensor,),
                "base_kwargs": {"中文": (None, True)},
                "base_out": tensor,
                "out_f": formula,
                "kwargs_f": None,
            },
        ]
    }
    blob = _encode_body_blob(body)
    packed = json.loads(blob)
    assert isinstance(blob, str) and len(packed["formulas"]) == 3
    assert len(packed["entry_fields"]) == 2
    restored = _decode_body_blob(blob)
    before_tensors, after_tensors = [], []
    assert encode_value(body, before_tensors, {}) == encode_value(restored, after_tensors, {})
    assert before_tensors == after_tensors
    first, second = restored["entries"]
    assert first["base_out"] is second["base_args"][0] is second["base_out"]
    assert first["out_f"] is not second["out_f"]
    first["out_f"]["shape"].append(99)
    assert second["out_f"]["shape"] == [{"type": "B"}, None]
    assert body["entries"][0]["out_f"] is formula


@pytest.mark.parametrize("field,index", [("formula", -1), ("formula", True), ("fields", -1), ("fields", 999)])
def test_compact_body_rejects_invalid_table_references(field, index):
    import json
    from template_engine.zero_meta import _encode_body_blob, _decode_body_blob

    packed = json.loads(_encode_body_blob({"entries": [{"func": "aten.neg.default", "out_f": None}]}))
    group = packed['body']['entries']['groups'][0]
    if field == 'formula':
        group['columns'][1] = ['constant', index]
    else:
        group['fields'] = index
    with pytest.raises(ValueError, match="reference"):
        _decode_body_blob(json.dumps(packed))


@pytest.mark.parametrize("version", [1])
def test_compact_body_rejects_retired_formats(version):
    import json
    from template_engine.zero_meta import _decode_body_blob

    with pytest.raises(ValueError, match="unsupported template group body JSON format"):
        _decode_body_blob(json.dumps({"format": f"group_body_json/{version}"}))


def test_compact_body_reads_shipped_v2_format():
    import json
    from template_engine.zero_meta import _encode_body_blob, _decode_body_blob, _unpack_entry_columns

    body = {"entries": [{"func": "aten.neg.default", "out_f": None}]}
    packed = json.loads(_encode_body_blob(body))
    packed['body']['entries'] = _unpack_entry_columns(packed['body']['entries'])
    packed['format'] = 'group_body_json/2'
    assert _decode_body_blob(json.dumps(packed)) == body


def test_disabled_template_skips_configuration_checks(caplog):
    from template_engine.runner_bridge import wrap_registered_runner

    runner = object()
    with caplog.at_level("INFO", logger="template_engine.runner_bridge"):
        assert wrap_registered_runner(runner, enabled=False) is runner
    assert "template_optimization_disabled" in caplog.text


def test_column_storage_restores_order_types_and_empty_containers():
    from template_engine.zero_meta import _pack_entry_columns, _unpack_entry_columns

    fields = [['func', 'value']]
    rows = [
        [0, 'aten.alias.default', {'tuple': [], 'mixed': True, 'id': 7}],
        [0, 'aten.neg.default', None],
        [0, 'aten.alias.default', {'tuple': [], 'mixed': 1, 'id': -5}],
        [0, 'aten.neg.default', {}],
        [0, 'aten.alias.default', {'tuple': [], 'mixed': '1', 'id': 3}],
    ]
    restored = _unpack_entry_columns(_pack_entry_columns(rows, fields))
    assert restored == rows
    assert type(restored[0][2]['mixed']) is bool and type(restored[2][2]['mixed']) is int
    restored[0][2]['tuple'].append(99)
    assert restored[2][2]['tuple'] == []


@pytest.mark.parametrize('index', [-1, True, 100])
def test_column_storage_rejects_invalid_operator_group(index):
    from template_engine.zero_meta import _unpack_entry_columns

    with pytest.raises(ValueError, match='reference'):
        _unpack_entry_columns({'order': [index], 'groups': []})


def test_text_generate_and_optimizer_share_template_switch():
    from cli.inference.text_generate import arg_parse
    from cli.registry import shared
    from cli.registry.modules import get_spec

    for module in ('text_generate', 'throughput_optimizer'):
        assert shared.TEMPLATE_GRAPH_OPTIMIZATION in get_spec(module).fields
    argv = ['--model-id', MODEL, '--num-queries', '4', '--query-length', '768', '--compile']
    assert arg_parse(argv).template_graph_optimization is True
    assert arg_parse([*argv, '--no-template-graph']).template_graph_optimization is False


@pytest.mark.parametrize('enabled', [True, False])
def test_text_generate_main_uses_registered_template_or_explicit_native(artifacts, monkeypatch, enabled):
    from cli.inference import text_generate
    from tensor_cast.model_config import ParallelConfig

    source, cfg, _, _ = artifacts
    registry.register_template(source, MODEL, 2, do_compile=True)
    calls = []
    metrics = SimpleNamespace(print_info=lambda: calls.append('print'))

    class Runner:
        def __init__(self, user, *, defer_compile=False):
            assert defer_compile is enabled
            self.user_input = user
            self.model = SimpleNamespace(
                num_hidden_layers=2,
                model_config=SimpleNamespace(
                    parallel_config=ParallelConfig(
                        world_size=2,
                        tensor_parallel_size=1,
                        expert_parallel_size=2,
                    )
                ),
            )
            self.request_info_default = [
                SimpleNamespace(
                    concurrency=cfg['B'],
                    query_len=cfg['S'],
                    seq_len=cfg['S'],
                    is_decode=False,
                )
            ]

        def run_inference(self, *_args, **kwargs):
            calls.append('template' if 'graph_provider' in kwargs else 'native')
            return metrics

    monkeypatch.setattr('tensor_cast.core.model_runner.ModelRunner', Runner)
    monkeypatch.setattr(text_generate, 'print_logo', lambda: None)
    argv = [
        'text_generate',
        '--model-id',
        MODEL,
        '--num-devices',
        '2',
        '--num-queries',
        str(cfg['B']),
        '--query-length',
        str(cfg['S']),
        '--ep-size',
        '2',
        '--compile',
        '--quantize-linear-action',
        'DISABLED',
    ]
    if not enabled:
        argv.append('--no-template-graph')
    monkeypatch.setattr('sys.argv', argv)
    text_generate.main()
    assert calls == ['template' if enabled else 'native', 'print']


@pytest.mark.parametrize(
    'field,value',
    [('quantize_lmhead', True), ('mlp_tp_size', 2), ('block_size', 256), ('multistream_overlap_shared_expert', True)],
)
def test_unencoded_text_generate_modifiers_use_native_runner(field, value, caplog):
    from template_engine.runner_bridge import wrap_registered_runner

    user = SimpleNamespace(model_id=MODEL, **{field: value})
    runner = SimpleNamespace(user_input=user, model=SimpleNamespace())
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert wrap_registered_runner(runner) is runner
    assert f'unsupported_template_option:{field}' in caplog.text


def test_explicit_world_and_dp_not_encoded_by_template_use_native(monkeypatch, caplog):
    from template_engine.runner_bridge import wrap_registered_runner

    monkeypatch.setattr(registry, 'template_path_for', lambda *_args, **_kwargs: 'registered.json.xz')
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        expert_parallel_size=32,
        moe_data_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=8,
    )
    runner = SimpleNamespace(
        user_input=SimpleNamespace(model_id=MODEL, do_compile=True),
        model=SimpleNamespace(model_config=SimpleNamespace(parallel_config=parallel)),
    )
    with caplog.at_level('INFO', logger='template_engine.runner_bridge'):
        assert wrap_registered_runner(runner) is runner
    assert 'parallel_layout_not_encoded' in caplog.text


@pytest.mark.parametrize('dcp', [1, 4])
def test_runtime_metadata_uses_concrete_lengths_and_preserves_aliases(dcp):
    from template_engine.runtime_bridge import request_op_invoke_infos

    meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2]),
        seq_lens=torch.tensor([101, 129]),
        query_lens=torch.tensor([1, 1]),
        is_dcp_decode=True,
    )
    seq = torch.empty((2,), dtype=torch.long, device='meta')
    mla_seq = seq if dcp == 1 else torch.empty_like(seq)
    query = torch.empty_like(seq)
    starts = torch.empty((3,), dtype=torch.long, device='meta')
    x = torch.empty((2, 4, 8), device='meta')
    ops = [
        CapturedOp('aten.alias.default', (seq,), {}, seq),
        CapturedOp('tensor_cast.dsa_indexer.default', (x, x, x, None, seq, 8, query), {}, x),
        CapturedOp(
            'tensor_cast.mla_sparse_attention.default', (x, x, x, x, x, starts, mla_seq, query, 8, 8, 8, None), {}, x
        ),
    ]
    infos = request_op_invoke_infos(ops, meta, dcp_size=dcp)
    assert infos[0].args[0] is infos[0].out is infos[1].args[4] is meta.seq_lens
    assert infos[2].args[5] is meta.query_start_loc and infos[2].args[7] is meta.query_lens
    assert infos[2].args[6].tolist() == ([101, 129] if dcp == 1 else [26, 33])
    assert ops[0].out.device.type == 'meta'  # The cached template remains reusable.


def test_compact_save_keeps_unopened_lazy_buckets(artifacts, tmp_path):
    from template_engine.zero_meta import ensure_group_body

    source, _, _, _ = artifacts
    graph = load_template_graph(source)
    assert len(graph["engine"]["_body_blobs"]) == 2
    saved = tmp_path / "resaved.json.xz"
    _save_template_lazy(graph, saved)
    restored = load_template_graph(saved)
    assert restored["engine"]["groups"].keys() == graph["engine"]["groups"].keys()
    for key, group in restored["engine"]["groups"].items():
        ensure_group_body(restored["engine"], key)
        tensors, expected_tensors = [], []
        assert encode_value(group, tensors, {}) == encode_value(graph["engine"]["groups"][key], expected_tensors, {})
        assert tensors == expected_tensors


@pytest.mark.parametrize('enabled,registered', [(True, True), (True, False), (False, True)])
def test_runner_factory_defers_only_registered_compile_candidates(monkeypatch, enabled, registered):
    from template_engine.runner_bridge import create_runner
    from tensor_cast.core.user_config import UserInputConfig

    calls = []
    monkeypatch.setattr(registry, 'template_path_for', lambda *_a, **_k: 'template.json.xz' if registered else None)
    monkeypatch.setattr(
        'tensor_cast.core.model_runner.ModelRunner',
        lambda user, **kwargs: calls.append((user, kwargs)) or 'runner',
    )
    user = UserInputConfig(model_id=MODEL, do_compile=True)
    assert create_runner(user, enabled=enabled) == 'runner'
    assert calls == [(user, {'defer_compile': True} if enabled and registered else {})]
    assert user.do_compile is True


def test_deferred_model_preserves_compile_config_and_compiles_on_native_fallback(monkeypatch):
    from tensor_cast.core import model_builder
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.user_config import UserInputConfig

    user = UserInputConfig(model_id=MODEL, do_compile=True)
    model = torch.nn.Identity()
    seen = []

    class Resolver:
        def __init__(self, user_input):
            assert user_input.do_compile is True

        def resolve(self):
            return SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))

    monkeypatch.setattr(model_builder, 'ConfigResolver', Resolver)
    monkeypatch.setattr(model_builder, 'TransformerModel', lambda *_a: model)
    monkeypatch.setattr(model_builder, 'compile_model', lambda value, cfg: seen.append((value, cfg)) or value)
    assert model_builder.build_model(user, defer_compile=True) is model
    assert seen == []
    runner = object.__new__(ModelRunner)
    runner.user_input, runner.model, runner._compile_deferred = user, model, True
    runner._ensure_compiled()
    runner._ensure_compiled()
    assert seen == [(model, user)]
    assert runner._compile_deferred is False


def test_registered_template_graph_shared_across_runners_and_reloaded_on_change(tmp_path, monkeypatch):
    from template_engine.runner_bridge import TemplateBackedModelRunner
    from template_engine import template_graph

    path = tmp_path / 'registered.json.xz'
    path.write_bytes(b'first')
    calls = []
    monkeypatch.setattr(template_graph, 'load_template_graph', lambda p: calls.append(p) or {'version': p.read_bytes()})
    a, b = TemplateBackedModelRunner(object(), str(path)), TemplateBackedModelRunner(object(), str(path))
    assert a._get_graph() is b._get_graph()
    assert len(calls) == 1
    path.write_bytes(b'changed-version')
    c = TemplateBackedModelRunner(object(), str(path))
    assert c._get_graph() == {'version': b'changed-version'}
    assert len(calls) == 2
