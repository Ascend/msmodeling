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

"""Fixed buckets never infer operator growth or membership from a B/S box."""

from types import SimpleNamespace
import torch
import pytest
from template_engine.body_select import partition_by_mode
from template_engine.capture_ops import CapturedOp
from template_engine.compare_ops import compare_op_sequences
from template_engine.model_parameters import formula_parameters
from template_engine.template_builder import _Capture, BuildReport, _derive_formulas_tree, build_template
from template_engine.template_graph import generate, TemplateGraphRefusal
from template_engine.serve_rules import prefill_window_regime, prefill_structure_domain
from template_generate import apply_formula_to_val, eval_formula


from template_engine.structure_buckets import FIXED_POLICY
from template_engine.template_builder import build_fixed_template


def config(batch, sequence, decode=False):
    return dict(
        B=batch,
        S=sequence,
        TP=2,
        EP=4,
        MOE_DP=1,
        DCP=1,
        num_mtp_tokens=1 if decode else 0,
        is_decode=decode,
        do_compile=True,
        enable_dispatch_ffn_combine=False,
        quantize_linear_action='DISABLED',
        quantize_attention_action='DISABLED',
        model_parameters=(('E', 256), ('E_per_tok', 6), ('H', 8)),
    )


def capture(batch, sequence, decode=False, alpha=None):
    c = config(batch, sequence, decode)
    local = (batch + 1) // 2
    x = torch.empty((local * (2 if decode else sequence), 8), device='meta')
    ops = []
    if not decode and local * sequence % 2:
        padded = torch.empty((x.shape[0] + 1, 8), device='meta')
        ops.append(CapturedOp('aten.constant_pad_nd.default', (x, [0, 0, 0, 1]), {}, padded))
        x = padded
    for _ in range(local if decode else 1):
        kwargs = {'lengths': [sequence] * local} if decode else {}
        if alpha is not None:
            kwargs['alpha'] = alpha
        ops.append(CapturedOp('aten.alias.default', (x,), kwargs, x))
    return _Capture(str((batch, sequence, alpha)), c, ops, 0.0)


def build(captures):
    engine, report = build_fixed_template(captures)
    assert engine['template_meta']['structure_policy'] == FIXED_POLICY
    return engine, report


def test_combined_template_binds_formula_parameters_only_in_the_declared_cell():
    from template_engine.model_parameters import bind_parameters
    from template_engine.body_select import discrete_cell

    existing = config(2, 257)
    existing.pop('model_parameters')
    added = {**existing, 'EP': 32}
    parameters = (('E', 384), ('E_per_tok', 6), ('H', 8))
    engine = {
        'template_meta': {
            'parameter_bindings': [
                {'cell': list(discrete_cell(added)), 'model_parameters': parameters, 'structure_policy': FIXED_POLICY}
            ]
        }
    }
    assert bind_parameters(engine, existing) == existing
    result = bind_parameters(engine, added)
    assert result['structure_policy'] == FIXED_POLICY
    assert result['model_parameters'] == parameters
    with pytest.raises(ValueError, match='parameters disagree'):
        bind_parameters(engine, {**added, 'model_parameters': (('H', 9),)})


def test_padding_predicate_selects_fixed_body_and_bucket_ids_are_stable():
    samples = [
        capture(b, s) for b, s in ((2, 257), (2, 515), (5, 257), (5, 515), (3, 257), (3, 515), (5, 258), (9, 258))
    ]
    engine, report = build(samples)
    assert report.failed_formulas == 0 and len(engine['groups']) == 2
    reverse, _ = build(list(reversed(samples)))

    def ids(template):
        return {g['mode_meta']['structure_bucket']['id'] for g in template['groups'].values()}

    assert ids(engine) == ids(reverse)
    for b, s in ((11, 777), (13, 8193)):
        truth = capture(b, s)
        result = generate(
            {'engine': engine, 'meta': {'num_hidden_layers': 1}},
            truth.config,
            num_hidden_layers=1,
            return_ops=True,
        )
        assert result['structure_bucket_id'] in ids(engine)
        assert compare_op_sequences(truth.ops, result['ops']).passed


def test_decode_reuses_known_rank_batch_and_refuses_an_unseen_structure(monkeypatch):
    def reject_fold(_captures):
        raise AssertionError('fixed builds must not invoke the repeat folder')

    monkeypatch.setattr('template_engine.template_builder._detect_fold_cells', reject_fold, raising=False)
    samples = [capture(b, s, True) for b in (2, 3, 5) for s in (257, 515)]
    engine, report = build(samples)
    assert report.failed_formulas == 0 and len(engine['groups']) == 3
    assert not any('program' in g or 'fold' in g for g in engine['groups'].values())
    for b, s in ((1, 777), (4, 4099), (6, 8193)):
        truth = capture(b, s, True)
        result = generate(
            {'engine': engine, 'meta': {'num_hidden_layers': 1}},
            truth.config,
            num_hidden_layers=1,
            return_ops=True,
        )
        assert f'_B{b}_S{s}' in result['key']  # Canonical fitting must not rewrite request/cache identity.
        assert compare_op_sequences(truth.ops, result['ops']).passed
    with pytest.raises(TemplateGraphRefusal, match='structure_bucket_missing'):
        generate(
            {'engine': engine, 'meta': {'num_hidden_layers': 1}},
            config(7, 777, True),
            num_hidden_layers=1,
        )


def test_prefill_equivalent_rank_batch_reuses_single_anchor_without_extrapolating():
    sample = capture(2, 257)
    engine, _ = build([sample])
    template = {'engine': engine, 'meta': {'num_hidden_layers': 1}}
    truth = capture(1, 257)
    result = generate(template, truth.config, return_ops=True)
    assert result['coverage'] == 'exact_group'
    assert '_B1_S257' in result['key']
    assert compare_op_sequences(truth.ops, result['ops']).passed
    from template_generate import generate_ops_from_template

    assert compare_op_sequences(truth.ops, generate_ops_from_template(engine, truth.config)).passed
    for batch, sequence in ((1, 259), (3, 257)):
        with pytest.raises(TemplateGraphRefusal):
            generate(template, config(batch, sequence))


def test_prefill_unseen_sequence_reuses_the_same_rank_batch():
    engine, report = build([capture(2, s) for s in (257, 515)])
    assert report.failed_formulas == 0
    truth = capture(1, 777)
    result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    assert result['coverage'] == 'interpolated'
    assert compare_op_sequences(truth.ops, result['ops']).passed


def test_decode_does_not_fit_across_rank_batches_with_equal_structure():
    def sample(batch, sequence):
        result = capture(batch, sequence, True)
        result.ops = result.ops[:1]
        return result

    engine, report = build([sample(b, s) for b in (2, 4) for s in (257, 515)])
    assert len(engine['groups']) == 2
    assert report.failed_formulas == 0
    for batch in (1, 3):
        truth = sample(batch, 777)
        result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
        assert compare_op_sequences(truth.ops, result['ops']).passed


@pytest.mark.parametrize('sequence', [2, 127, 777, 16897])
def test_decode_index_window_width_matches_official_short_context_rule(sequence):
    from tensor_cast.layers.deepseek_v4 import get_window_topk_idxs, get_compress_topk_idxs
    from template_engine.parameter_features import prefill_cache_features

    parameters = {'WINDOW': 128, 'INDEX_TOPK': 1024, 'COMPRESS_RATIO/128': 128}
    features = prefill_cache_features(2, sequence, 16, 32, 1, 0, parameters, decode=True)
    window = get_window_topk_idxs(128, 1, 1, torch.device('meta'), True)
    compressed = get_compress_topk_idxs(128, 1, 1, torch.device('meta'), sequence)
    assert features['prefill/index/128/window_width'] == window.shape[-1] + compressed.shape[-1]


def test_window_split_does_not_alias_sequence_remainder_to_batch():
    def sample(batch, sequence):
        c = config(batch, sequence)
        c['model_parameters'] = (*c['model_parameters'], ('WINDOW', 128))
        packed = ((batch + 1) // 2) * sequence
        tail = torch.empty((1, packed, 8), device='meta')[:, -128:]
        cutoff = packed % 128
        lengths = [128 - cutoff, cutoff]
        chunks = torch.split(tail, lengths, dim=1)
        return _Capture(
            str((batch, sequence)),
            c,
            [CapturedOp('aten.split_with_sizes.default', (tail, lengths, 1), {}, chunks)],
            0.0,
        )

    engine, report = build([sample(b, s) for b in (2, 3) for s in (2049, 8193)])
    assert report.failed_formulas == 0
    truth = sample(1, 777)
    result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict()


def test_compressor_output_uses_its_schema_instead_of_an_accidental_routing_fit():
    from template_engine.parameter_features import prefill_cache_features

    def sample(batch, sequence):
        c = config(batch, sequence)
        c['EP'] = 32
        parameters = {'E': 384, 'E_per_tok': 6, 'H': 8, 'WINDOW': 128, 'INDEX_TOPK': 1024, 'COMPRESS_RATIO/128': 128}
        c['model_parameters'] = tuple(sorted(parameters.items()))
        local = (batch + 15) // 16
        hidden = torch.empty((1, local * sequence, 8), device='meta')
        blocks = prefill_cache_features(batch, sequence, 2, 32, 1, 0, parameters)['prefill/cache/128/blocks']
        cache = torch.empty((blocks, 128, 8), dtype=torch.int8, device='meta')
        args = (hidden, cache, 128, 8, 2, False, None, torch.full((local,), sequence, dtype=torch.long))
        outputs = torch.ops.tensor_cast.compressor.default(*args)
        return _Capture(
            str((batch, sequence)), c, [CapturedOp('tensor_cast.compressor.default', args, {}, outputs)], 0.0
        )

    engine, _ = build([sample(b, s) for b, s in ((16, 130), (16, 132), (17, 2049), (17, 8193))])
    truth = sample(49, 127)
    assert truth.ops[0].out[0].shape[1] == 3
    result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict()


def test_single_query_compressor_uses_incremental_row_count_in_prefill():
    from template_engine.parameter_features import prefill_cache_features
    from tensor_cast.layers.deepseek_v4 import get_window_topk_idxs

    parameters = {'WINDOW': 128, 'INDEX_TOPK': 1024, 'COMPRESS_RATIO/4': 4, 'COMPRESS_RATIO/128': 128}
    for local, sequence in ((1, 1), (3, 1), (4, 2), (1, 127)):
        hidden = torch.empty((1, local * sequence, 8), device='meta')
        cache = torch.empty((1, 128, 8), device='meta')
        query = torch.full((local,), sequence, dtype=torch.long)
        features = prefill_cache_features(local * 16, sequence, 2, 32, 1, 0, parameters)
        for ratio in (4, 128):
            output, _ = torch.ops.tensor_cast.compressor(hidden, cache, ratio, 8, 2, False, None, query)
            assert features[f'prefill/cache/{ratio}/compressed_tokens'] == output.shape[1]
            window = get_window_topk_idxs(128, 1, local * sequence, torch.device('meta'), sequence < 5)
            assert features[f'prefill/index/{ratio}/window_width'] == window.shape[-1] + max(sequence // ratio, 1)


def test_moe_dispatch_shapes_follow_schema_at_unseen_large_batch():
    from types import SimpleNamespace
    from tensor_cast.layers.moe_layer import FusedMoETensorCast, assign_experts

    def sample(batch, sequence):
        c = config(batch, sequence)
        c.update(TP=4, EP=32)
        c['model_parameters'] = (('E', 384), ('E_per_tok', 6), ('H', 8))
        local = (batch + 7) // 8
        tokens = (local * sequence + 3) // 4
        start, count = assign_experts(384, 32, 0)
        context = SimpleNamespace(
            ep_group=SimpleNamespace(world_size=32, rank_in_group=0),
            num_external_shared_experts=0,
            num_global_experts=384,
            expert_idx_start=start,
            num_local_experts=count,
        )
        incoming, outgoing, *_ = FusedMoETensorCast.get_split_sizes(context, tokens * 6, 6)
        hidden = torch.empty((tokens, 8), device='meta')
        indices = torch.empty((tokens, 6), dtype=torch.long, device='meta')
        routed = torch.ops.tensor_cast.init_routing_v2(hidden, indices, False)
        dispatched = torch.ops.tensor_cast.all_to_all(routed, outgoing, incoming, 0, list(range(32)))
        processed = -dispatched
        combined = torch.ops.tensor_cast.all_to_all(processed, incoming, outgoing, 0, list(range(32)))
        ops = [
            CapturedOp('tensor_cast.init_routing_v2.default', (hidden, indices, False), {}, routed),
            CapturedOp(
                'tensor_cast.all_to_all.default', (routed, outgoing, incoming, 0, list(range(32))), {}, dispatched
            ),
            CapturedOp('aten.neg.default', (dispatched,), {}, processed),
            CapturedOp(
                'tensor_cast.all_to_all.default', (processed, incoming, outgoing, 0, list(range(32))), {}, combined
            ),
        ]
        return _Capture(str((batch, sequence)), c, ops, 0.0)

    engine, _ = build([sample(b, s) for b, s in ((8, 132), (8, 136), (8, 260), (24, 180), (64, 131), (160, 8191))])
    truth = sample(255, 16383)
    generated = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    result = compare_op_sequences(truth.ops, generated['ops'])
    assert result.passed, result.as_dict()


def test_single_packed_token_does_not_freeze_other_single_query_batches():
    def sample(batch, sequence):
        c = config(batch, sequence)
        c.update(TP=16, EP=32)
        c['model_parameters'] = (('E', 384), ('E_per_tok', 6), ('H', 8), ('WINDOW', 128), ('COMPRESS_RATIO/128', 128))
        local = (batch + 1) // 2
        # The real short-query graph can retain a cache-sized view at one
        # packed token (4096 rows), while other requests use packed rows.
        rows = 4096 if local * sequence == 1 else local * sequence
        x = torch.empty((rows, 8), device='meta')
        return _Capture(str((batch, sequence)), c, [CapturedOp('aten.neg.default', (x,), {}, -x)], 0.0)

    engine, _ = build([sample(b, s) for b, s in ((2, 1), (6, 1), (8, 1), (2, 2), (4, 2), (4, 3), (2, 4))])
    truth = sample(3, 1)
    result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict()


def test_singleton_prefill_anchor_accepts_a_dp_alias_seen_in_another_branch():
    def sample(batch, sequence):
        c = config(batch, sequence)
        c.update(TP=2, EP=32)
        c['model_parameters'] = (('E', 384), ('E_per_tok', 6), ('H', 8), ('WINDOW', 128), ('COMPRESS_RATIO/128', 128))
        x = torch.empty((((batch + 15) // 16) * sequence, 8), device='meta')
        return _Capture(str((batch, sequence)), c, [CapturedOp('aten.neg.default', (x,), {}, -x)], 0.0)

    engine, _ = build([sample(32, 1), sample(17, 130)])
    truth = sample(17, 1)
    out = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    comparison = compare_op_sequences(truth.ops, out['ops'])
    assert comparison.passed, comparison.as_dict()
    with pytest.raises(TemplateGraphRefusal):
        generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, sample(33, 1).config)


@pytest.mark.parametrize('tp,batch', [(4, 121), (16, 127)])
def test_sparse_mla_metadata_uses_rank_batch_instead_of_accidental_global_batch_fit(tp, batch):
    def sample(b, s):
        c = config(b, s)
        c.update(TP=tp, EP=32)
        c['model_parameters'] = (('E', 256), ('E_per_tok', 8), ('H', 8), ('INDEX_TOPK', 2048))
        local = (b + 32 // tp - 1) // (32 // tp)
        q = torch.empty((local * s, 2, 8), device='meta')
        projected = torch.empty_like(q)
        absorbed = torch.empty((0, 2, 8), device='meta')
        cache = torch.empty((1, 128, 8), device='meta')
        blocks = torch.empty((local, (s + 127) // 128), dtype=torch.long, device='meta')
        starts = torch.arange(0, (local + 1) * s, s)
        lengths = torch.full((local,), s, dtype=torch.long)
        query = torch.full((local,), s, dtype=torch.long)
        topk = torch.empty((local * s, 1), dtype=torch.long, device='meta')
        scale = torch.empty((1,), device='meta')
        args = (
            q,
            projected,
            absorbed,
            cache,
            blocks,
            starts,
            lengths,
            query,
            8,
            8,
            2048,
            topk,
            scale,
            None,
            scale,
            None,
            scale,
            None,
            scale,
            None,
            None,
            None,
            None,
        )
        output = torch.ops.tensor_cast.mla_sparse_attention_quant(*args)
        return _Capture(
            str((b, s)), c, [CapturedOp('tensor_cast.mla_sparse_attention_quant.default', args, {}, output)], 0.0
        )

    dp = 32 // tp
    engine, _ = build([sample(b, s) for b in (dp, dp + 1) for s in (256, 4096)])
    truth = sample(batch, 777)
    result = generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, truth.config, return_ops=True)
    assert result['ops'][0].args[5].shape[0] == (batch + dp - 1) // dp + 1
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict()


def test_overlapping_predicates_do_not_resolve_by_observed_rectangle():
    samples = [capture(b, s, alpha=1.0) for b in (3, 7) for s in (256, 1024)]
    samples += [capture(b, s, alpha=2.0) for b in (11, 15) for s in (2048, 4096)]
    engine, _ = build(samples)
    with pytest.raises(TemplateGraphRefusal, match='mode_ambiguous'):
        generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, config(5, 768), num_hidden_layers=1)


def test_expansion_must_preserve_the_selected_structure_fingerprint():
    engine, _ = build([capture(b, s) for b in (3, 7) for s in (256, 1024)])
    group = next(iter(engine['groups'].values()))
    group['entries'].append(group['entries'][-1])
    with pytest.raises(TemplateGraphRefusal, match='structure_bucket_changed'):
        generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, config(5, 768), num_hidden_layers=1)


def test_undeclared_structure_domain_is_exact_anchor_only():
    samples = [capture(2, 256), capture(3, 512)]
    for sample in samples:
        sample.config.pop('model_parameters')
    engine, _ = build(samples)
    with pytest.raises(TemplateGraphRefusal, match='structure_bucket_missing'):
        generate({'engine': engine, 'meta': {'num_hidden_layers': 1}}, config(4, 768), num_hidden_layers=1)


def test_constant_observed_cache_pages_keep_the_official_capacity_law():
    from tensor_cast.core.input_generator import _resolve_v4_kv_cache_size

    def cache_capture(batch, sequence):
        c = config(batch, sequence, True)
        c['model_parameters'] = (*c['model_parameters'], ('COMPRESS_RATIO/128', 128), ('WINDOW', 128))
        local = (batch + 1) // 2
        model = SimpleNamespace(
            text_config=SimpleNamespace(sliding_window=128), model_config=SimpleNamespace(hf_config=SimpleNamespace())
        )
        layer = SimpleNamespace(_head_dim=8, compress_ratio=128)
        pages = _resolve_v4_kv_cache_size(
            model, layer, block_size=128, batch_size=local, total_kv_tokens=local * (sequence + 2)
        )[0]
        x = torch.empty((1, local * 2, 8), device='meta')
        cache = torch.empty((pages, 128, 8), device='meta')
        written, compressed = torch.empty_like(cache), torch.empty_like(cache)
        ops = [
            CapturedOp('tensor_cast.scatter_nd_update_mla.default', (x, cache, None, 2, None), {}, written),
            CapturedOp(
                'tensor_cast.compressor.default', (x, written, 128, 8, 64, False, None, None), {}, (x, compressed)
            ),
            CapturedOp('aten.copy_.default', (cache, cache), {}, cache),
        ]
        return _Capture(str((batch, sequence)), c, ops, 0.0)

    samples = [cache_capture(16, s) for s in (257, 515)]
    assert samples[0].ops[0].args[1].shape[0] == samples[1].ops[0].args[1].shape[0] == 9
    engine, _ = build(samples)
    truth = cache_capture(15, 8193)
    assert truth.ops[0].args[1].shape[0] == 12
    result = generate(
        {'engine': engine, 'meta': {'num_hidden_layers': 1}},
        truth.config,
        num_hidden_layers=1,
        return_ops=True,
    )
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict(max_diffs=5)


def test_fixed_artifact_enforces_admission_for_default_low_level_expansion():
    from template_generate import generate_ops_from_template, TemplateExpandError

    engine, _ = build([capture(2, s, True) for s in (257, 515)])
    with pytest.raises(TemplateExpandError, match='structure_bucket_missing'):
        generate_ops_from_template(engine, config(7, 777, True))


def prefill_config(batch, sequence):
    return dict(
        B=batch,
        S=sequence,
        TP=1,
        EP=2,
        MOE_DP=1,
        DCP=1,
        num_mtp_tokens=0,
        is_decode=False,
        do_compile=True,
        enable_dispatch_ffn_combine=False,
        quantize_linear_action='DISABLED',
        quantize_attention_action='DISABLED',
        model_parameters=formula_parameters(
            dict(hidden_size=8, intermediate_size=16, vocab_size=64, n_routed_experts=256, num_experts_per_tok=6)
        ),
    )


def routing_capture(batch, sequence):
    from tensor_cast.layers.moe_layer import FusedMoETensorCast, assign_experts

    start, count = assign_experts(256, 2, 0)
    context = SimpleNamespace(
        ep_group=SimpleNamespace(world_size=2, rank_in_group=0),
        num_external_shared_experts=0,
        num_global_experts=256,
        expert_idx_start=start,
        num_local_experts=count,
    )
    local_batch = (batch + 1) // 2
    splits = FusedMoETensorCast.get_split_sizes(context, local_batch * sequence * 6, 6)
    rows = [sum(rank[i] for rank in splits[3]) for i in range(count)]
    x = torch.empty((sum(rows), 8), device='meta')
    chunks = tuple(torch.empty((n, 8), device='meta') for n in rows)
    ops = [CapturedOp('aten.split_with_sizes.default', (x, rows, 0), {}, chunks)]
    all_outputs = []
    for chunk, size in zip(chunks, rows):
        sliced = torch.empty_like(chunk)
        ops.append(CapturedOp('aten.slice.Tensor', (chunk, 0, 0, size), {}, sliced))
        pieces = tuple(torch.empty((size // 2, 8), device='meta') for _ in range(2))
        ops.append(CapturedOp('aten.split_with_sizes.default', (sliced, [size // 2] * 2, 0), {}, pieces))
        all_outputs.extend(pieces)
    y = torch.empty_like(x)
    ops.append(CapturedOp('aten.cat.default', (all_outputs, 0), {}, y))
    return _Capture(str((batch, sequence)), prefill_config(batch, sequence), ops, 0.0)


def test_prefill_routing_columns_and_consumers_share_their_own_formula():
    captures = [routing_capture(b, s) for b, s in ((2, 256), (2, 1024), (3, 256), (3, 1024), (5, 1024), (2, 8192))]
    engine, report = build_template(captures)
    assert len(engine['groups']) == 1 and report.failed_formulas == 0
    for b, s in ((4, 768), (7, 2048), (11, 777), (17, 8193)):
        truth = routing_capture(b, s)
        actual = generate(
            {'engine': engine, 'meta': {'num_hidden_layers': 1}},
            truth.config,
            num_hidden_layers=1,
            return_ops=True,
        )['ops']
        comparison = compare_op_sequences(truth.ops, actual)
        assert comparison.passed, comparison.as_dict(max_diffs=3)


def test_repeated_request_parameters_extrapolate_past_the_last_observed_index():
    configs = [prefill_config(b, s) for b, s in ((2, 256), (2, 1024), (3, 256), (3, 1024), (5, 1024))]
    for boolean in (False, True):
        values = [[False if boolean else c['S']] * ((c['B'] + 1) // 2) for c in configs]
        formula = _derive_formulas_tree(values[0], values, configs, BuildReport(), 'metadata')
        target = prefill_config(17, 8193)
        assert apply_formula_to_val(values[0], formula, target, strict=True) == [False if boolean else 8193] * 9


def test_parameter_list_lengths_do_not_split_prefill_but_decode_is_unchanged():
    def capture(b, decode):
        c = prefill_config(b, 256)
        c['is_decode'] = decode
        op = CapturedOp(
            'tensor_cast.dsa_indexer.default',
            (),
            {'is_decode_values': [decode] * ((b + 1) // 2), 'query_lens_values': [256] * ((b + 1) // 2)},
            None,
        )
        return _Capture(str(b), c, [op], 0.0)

    assert len(partition_by_mode([capture(2, False), capture(3, False)])) == 1
    assert len(partition_by_mode([capture(2, True), capture(3, True)])) == 2


def test_exact_split_division_refuses_a_nondivisible_dimension():
    formula = dict(type='DOMAIN_CEIL_DIV', feat='B', divisor=2, require_divisible=True)
    assert eval_formula(formula, dict(B=4, S=1)) == 2
    assert eval_formula(formula, dict(B=3, S=1)) is None


def test_official_window_branches_share_one_cell_and_select_outside_training_span():
    def capture(batch, sequence):
        cfg = prefill_config(batch, sequence)
        cfg['model_parameters'] = (
            *((k, v) for k, v in cfg['model_parameters'] if k not in ('E', 'E_per_tok')),
            ('WINDOW', 128),
        )
        x = torch.empty(((batch + 1) // 2, sequence, 8), device='meta')
        regime = prefill_window_regime(cfg, 128)
        ops = [CapturedOp('tensor_cast.scatter_nd_update_mla.default', (x,), {}, x)]
        # Different operator bodies exercise routing, independently of fitting.
        for _ in range({'short': 0, 'aligned': 1, 'wrapped': 2}[regime]):
            ops.append(CapturedOp('aten.alias.default', (x,), {}, x))
        return _Capture(str((batch, sequence)), cfg, ops, 0.0)

    samples = [capture(b, s) for lengths in ((4, 16), (256, 1024), (257, 515)) for b in (2, 3, 5) for s in lengths]
    engine, report = build_template(samples)
    assert len(engine['groups']) == 3 and report.failed_formulas == 0
    for sequence in (1, 127, 128, 129, 255, 512, 777, 8193):
        truth = capture(17, sequence)
        result = generate(
            {'engine': engine, 'meta': {'num_hidden_layers': 1}},
            truth.config,
            num_hidden_layers=1,
            return_ops=True,
        )
        comparison = compare_op_sequences(truth.ops, result['ops'])
        assert comparison.passed, comparison.as_dict(max_diffs=3)
    aligned_only = {
        **engine,
        'groups': {
            key: group
            for key, group in engine['groups'].items()
            if group['mode_meta']['prefill_window_branch']['regimes'] == ['aligned']
        },
    }
    with pytest.raises(TemplateGraphRefusal, match='prefill_structure_branch'):
        generate(
            {'engine': aligned_only, 'meta': {'num_hidden_layers': 1}},
            capture(17, 777).config,
            num_hidden_layers=1,
        )


def test_window_branch_uses_packed_tokens_and_empty_expert_conditions():
    assert prefill_window_regime(prefill_config(2, 64), 128) == 'short'
    assert prefill_window_regime(prefill_config(3, 64), 128) == 'short'
    assert prefill_window_regime(prefill_config(7, 64), 128) == 'aligned'
    assert prefill_window_regime(prefill_config(5, 127), 128) == 'wrapped'
    small = prefill_structure_domain(prefill_config(2, 16), 128)
    assert small == prefill_structure_domain(prefill_config(7, 4), 128)
    assert small['empty_experts']
    assert not prefill_structure_domain(prefill_config(2, 64), 128)['empty_experts']


def test_routing_tokens_follow_official_dp_transform_boundary():
    from template_engine.parameter_features import prefill_routing_features

    parameters = {'E': 256, 'E_per_tok': 8}
    # Native GLM TP2/EP4/MD2: DP4 equals EP4, so init_routing receives
    # all 257 packed tokens. TP2/EP4/MD1 instead enters the TP slice.
    direct = prefill_routing_features(4, 257, 2, 4, 2, parameters)
    transformed = prefill_routing_features(2, 257, 2, 4, 1, parameters)
    assert direct['moe_route/tokens'] == 257
    assert transformed['moe_route/tokens'] == 129


def test_region_reference_survives_different_input_and_output_shapes():
    def capture(batch, sequence):
        x = torch.empty(((batch + 1) // 2 * sequence, 8), device='meta')
        weight = torch.empty((8, 16), device='meta')
        y = torch.empty((x.shape[0], 16), device='meta')
        copied = torch.empty_like(y)
        ops = [
            CapturedOp('tensor_cast._internal_mark_region_begin.default', (x, 101), {}, x),
            CapturedOp('aten.mm.default', (x, weight), {}, y),
            CapturedOp('tensor_cast._internal_mark_region_end.default', (y, 101), {}, y),
            CapturedOp('tensor_cast._internal_copy_region_v2.default', (x, y, 101), {}, copied),
        ]
        return _Capture(str((batch, sequence)), prefill_config(batch, sequence), ops, 0.0)

    samples = [capture(b, s) for b in (2, 3, 5) for s in (257, 515)]
    engine, report = build_template(samples)
    assert report.failed_formulas == 0
    truth = capture(11, 777)
    result = generate(
        {'engine': engine, 'meta': {'num_hidden_layers': 1}},
        truth.config,
        num_hidden_layers=1,
        return_ops=True,
    )
    comparison = compare_op_sequences(truth.ops, result['ops'])
    assert comparison.passed, comparison.as_dict(max_diffs=5)


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), 2**63 - 1])
def test_review_integer_constants_are_not_rounded(value):
    assert eval_formula({"type": "CONST", "value": value}, config(2, 8)) == value


@pytest.mark.parametrize("divisor", [0, -1])
def test_review_invalid_domain_divisors_are_refused(divisor):
    assert eval_formula({"type": "DOMAIN_CEIL_DIV", "feat": "B", "divisor": divisor}, config(8, 8)) is None


def test_review_nested_lookup_and_varying_scalar_refusal():
    from template_engine.template_builder import _failed_to_lookup, TemplateBuildError

    nested = {"B": 2, "model_parameters": {"H": 8}}
    result = _failed_to_lookup({"type": "FAILED"}, [(nested, 3), (nested, 3)])
    assert result["entries"] == [{"match": nested, "value": 3}]
    for first, second in [(True, False), ("a", "b"), (0.1, 0.2), (torch.float16, torch.float32)]:
        with pytest.raises(TemplateBuildError, match="scalar differs"):
            _derive_formulas_tree(first, [first, second], [config(2, 8), config(3, 8)], BuildReport(), "args")


def test_review_region_neutralization_preserves_capture_lists():
    from template_engine.template_builder import _build_entry

    source = torch.empty((2, 3), device="meta")
    args = [source, 123456789]
    op = CapturedOp("tensor_cast._internal_mark_region_begin.default", args, {}, source)
    entry = _build_entry(op, [], [config(2, 3)], BuildReport(), 0)
    assert args[1] == entry["base_args"][1] == 123456789


@pytest.mark.parametrize("shape,dim,expected", [([2, 3], 0, [2, 3]), ([1, 3], 0, [3]), ([2, 1], -1, [2])])
def test_review_squeeze_matches_torch(shape, dim, expected):
    from template_engine.shape_rules import derive_shape_semantically

    tensor = torch.empty(shape, device="meta")
    op = CapturedOp("aten.squeeze.dim", (tensor, dim), {}, None)
    assert derive_shape_semantically(op)["out"]["shape"] == expected


def test_review_symbolic_squeeze_and_slice_maximum_start():
    from template_engine.shape_rules import _rule_squeeze, _Spec, derive_shape_semantically

    op = SimpleNamespace(args=(None, 0), kwargs={})
    assert _rule_squeeze(op, [_Spec(["<rt>", 3], None)]) is None
    source = torch.empty((5,), device="meta")
    sliced = CapturedOp("aten.slice.Tensor", (source, 0, 2**63 - 1, None, 1), {}, None)
    assert derive_shape_semantically(sliced)["out"]["shape"] == [0]


def test_review_feature_fits_require_every_sample_and_closed_ceil_boundary():
    from template_engine.formulas import _multi_fit, _multi_fit_offset, _multi3_fit, _ceil_div_fit

    assert _multi_fit([({"B": 1, "S": 1}, 3), ({"B": 2, "S": 1}, 4), ({"B": 3}, 99)], ("B", "S")) is None
    rows = [({"B": i, "S": i + 1, "TP": i % 2}, i * 3) for i in range(1, 6)]
    assert _multi_fit_offset(rows + [({"B": 9}, 99)], ("B", "S")) is None
    assert _multi3_fit(rows + [({"B": 9}, 99)], ("B", "S", "TP")) is None
    samples = [({"B": b}, v) for b, v in [(4, 1), (5, 2), (8, 2)]]
    assert _ceil_div_fit(samples, [{"B": b} for b in (4, 5, 8)])["divisor"] == 4


def test_review_statistical_beams_contain_distinct_feature_sets():
    from template_engine.stat_fit import _candidate_subsets

    names = list("abcdef")
    columns = {name: [(i + j) ** 2 + (i * j) % 7 for i in range(1, 15)] for j, name in enumerate(names)}
    subsets = _candidate_subsets(names, columns, [i * i * 3 + i % 3 for i in range(1, 15)])
    assert len(subsets) == len({frozenset(subset) for subset in subsets})


@pytest.mark.parametrize("value", ["false", "0", "off", "no", "true", "1"])
def test_review_coverage_boolean_values_are_typed(value):
    from template_engine.coverage import _parse_target_spec

    parsed = _parse_target_spec(f"B=2,S=8,is_decode={value},do_compile={value},dfc={value}")
    expected = value in ("true", "1")
    assert parsed["is_decode"] is parsed["do_compile"] is parsed["enable_dispatch_ffn_combine"] is expected
    with pytest.raises(ValueError, match="boolean"):
        _parse_target_spec("is_decode=maybe")


def test_review_nonexact_axis_extrapolation_and_bounded_diagnostics():
    from template_engine.serve_rules import bs_span_blocker
    from template_engine.compare_ops import first_region_id_args

    assert bs_span_blocker({(1, 128), (1, 256)}, 1, 100000, False) is not None
    a = [CapturedOp("aten.alias.default", ([torch.empty((2,), device="meta") for _ in range(40)],), {}, None)]
    b = [CapturedOp("aten.alias.default", ([torch.empty((3,), device="meta") for _ in range(40)],), {}, None)]
    report = compare_op_sequences(a, b, max_diffs=1)
    assert len(report.diffs) == 1 and report.shape_mismatches == 40
    marker = CapturedOp(
        "tensor_cast._internal_mark_region_begin.default", (torch.empty((1,), device="meta"), 42), {}, None
    )
    assert first_region_id_args(a * 10 + [marker]) == [1]


@pytest.mark.parametrize("nested", [None, 3, "invalid", {"hidden_size": 7}])
def test_review_optional_text_config_is_type_checked(nested):
    expected = (("H", 8),) if nested is None else ((("H", 7),) if isinstance(nested, dict) else None)
    assert formula_parameters({"text_config": nested, "hidden_size": 8}) == expected


def test_review_single_legacy_candidate_obeys_its_predicate():
    from template_engine.body_select import select_body, base_key, discrete_cell, STATUS_NONE

    cfg = config(2, 8)
    key = base_key(discrete_cell(cfg))
    template = {"groups": {key: {"mode_meta": {"discrete_predicate": {"TP": [999]}}}}}
    assert select_body(template["groups"], key, cfg)[2] == STATUS_NONE


@pytest.mark.parametrize('value', [True, 2.5, '2', 0, -1])
def test_invalid_request_axes_are_reported_instead_of_coerced(value):
    from template_engine.cache_io import build_key_and_config

    with pytest.raises(ValueError, match='B|integer|positive'):
        build_key_and_config({**config(2, 257), 'B': value})


@pytest.mark.parametrize('field', ['program', 'fold', 'mtp_structure'])
def test_retired_dynamic_formats_are_rejected(field):
    from template_engine.zero_meta import ensure_group_body

    with pytest.raises(ValueError, match='Unsupported dynamic template format'):
        ensure_group_body({'groups': {'bucket': {'entries': [], field: {}}}}, 'bucket')


def test_unidentifiable_stride_does_not_block_fixed_shape_extrapolation(monkeypatch):
    from template_engine import template_builder

    original = template_builder.derive_value_formula
    strides = [173, 229, 151, 199]

    def derive(samples):
        return {'type': 'FAILED'} if [v for _, v in samples] == strides else original(samples)

    monkeypatch.setattr(template_builder, 'derive_value_formula', derive)
    configs = [{**config(b, s), 'structure_policy': FIXED_POLICY} for b, s in ((2, 8), (2, 16), (4, 8), (4, 16))]
    values = [torch.empty_strided((c['B'], 3), (stride, 1), device='meta') for c, stride in zip(configs, strides)]
    report = BuildReport()
    formulas = _derive_formulas_tree(values[0], values, configs, report, 'tensor')
    output = apply_formula_to_val(values[0], formulas, {**configs[0], 'B': 6, 'S': 32}, strict=True)
    assert output.shape == (6, 3) and report.failed_formulas == 0
    assert formulas['stride'][0]['type'] == 'CONTIGUOUS'

    # Unknown shape dimensions retain the existing exact-anchor requirement.
    shapes = [torch.empty((stride, 3), device='meta') for stride in strides]
    report = BuildReport()
    formulas = _derive_formulas_tree(shapes[0], shapes, configs, report, 'tensor')
    assert formulas['shape'][0]['type'] == 'EXACT_LOOKUP' and report.failed_formulas > 0
