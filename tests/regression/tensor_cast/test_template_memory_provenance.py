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

"""Resident-memory provenance must survive capture -> artifact -> served template.

The template path serves Meta placeholders whose identity cannot be linked to
``named_parameters``/``named_buffers`` or to the cache inventory.  Without the
captured provenance those placeholders have no defining op, so the tracker
charges them as request input while the peak formula already carries the full
weight and cache totals - those bytes are counted twice.  These tests pin each
hop of both tags, from capture to the served graph, and the registration that
keeps each total single-counted.
"""

import json

import pytest
import torch

from template_engine.capture_ops import _MetaTensorPool, serialize_op_invoke_infos
from template_engine.graph_codec import (
    TENSOR_CACHE_ATTR,
    TENSOR_WEIGHT_ATTR,
    decode_capture,
    encode_capture,
    tensor_resident_owner,
)
from template_engine.zero_meta import _decode_body_blob, _encode_body_blob
from tensor_cast.performance_model.memory_tracker import MemoryTracker


def _tagged(shape=(8, 4), dtype=torch.bfloat16, owner='weight'):
    tensor = torch.empty(shape, dtype=dtype, device='meta')
    if owner == 'weight':
        setattr(tensor, TENSOR_WEIGHT_ATTR, True)
    elif owner == 'cache':
        setattr(tensor, TENSOR_CACHE_ATTR, True)
    return tensor


class _Op:
    def __init__(self, func, args, out):
        self.func = func
        self.args = args
        self.kwargs = {}
        self.out = out


def test_capture_codec_round_trip_keeps_resident_provenance_and_aliasing():
    weight = _tagged()
    cache = _tagged((32, 4), dtype=torch.int8, owner='cache')
    state = torch.empty(4, dtype=torch.int64, device='meta')
    ops = [
        _Op('aten.mm.default', (state, weight), torch.empty(8, device='meta')),
        # A second reference to the same weight proves aliasing still holds.
        _Op('aten.mm.default', (state, weight), torch.empty(8, device='meta')),
        _Op('aten.index.Tensor', (cache, [state]), torch.empty(8, device='meta')),
    ]

    restored, exec_time = decode_capture(encode_capture(ops, 0.25))

    assert exec_time == 0.25
    first = restored[0].args[1]
    assert tensor_resident_owner(first) == 'weight'
    assert restored[1].args[1] is first
    assert tensor_resident_owner(restored[1].args[0]) is None
    assert tensor_resident_owner(restored[2].args[0]) == 'cache'


def test_body_blob_round_trip_keeps_weight_rows_and_reads_legacy_blobs():
    weight = _tagged()
    state = torch.empty(4, dtype=torch.int64, device='meta')
    body = {'entries': [{'func': 'aten.mm.default', 'args': [state, weight], 'shape': [8]}]}

    decoded = _decode_body_blob(_encode_body_blob(body))

    args = decoded['entries'][0]['args']
    assert tensor_resident_owner(args[1]) == 'weight'
    assert tensor_resident_owner(args[0]) is None

    # A blob written before provenance existed carries no resident rows and must
    # still decode (serving then falls back to charging placeholders as request
    # input, which only over-estimates).
    legacy = _encode_body_blob({'entries': [{'func': 'aten.mm.default', 'args': [weight]}]})
    payload = json.loads(legacy)
    assert payload.pop('weight_tensors') == [0]
    assert tensor_resident_owner(_decode_body_blob(json.dumps(payload))['entries'][0]['args'][0]) is None


def test_body_blob_round_trip_keeps_cache_rows_separately_from_weights():
    weight = _tagged()
    cache = _tagged((32, 4), dtype=torch.int8, owner='cache')

    payload = json.loads(_encode_body_blob({'entries': [{'func': 'aten.mm.default', 'args': [weight, cache]}]}))
    assert payload['weight_tensors'] == [0]
    assert payload['cache_tensors'] == [1]

    decoded = _decode_body_blob(json.dumps(payload))
    args = decoded['entries'][0]['args']
    assert tensor_resident_owner(args[0]) == 'weight'
    assert tensor_resident_owner(args[1]) == 'cache'


def test_expansion_propagates_the_weight_tag_when_shape_formulas_rebuild():
    from template_generate import rebuild_tensor

    weight = _tagged()
    rebuilt = rebuild_tensor(weight, [16, 4])
    assert tuple(rebuilt.shape) == (16, 4)
    assert tensor_resident_owner(rebuilt) == 'weight'

    cache = _tagged((32, 4), dtype=torch.int8, owner='cache')
    assert tensor_resident_owner(rebuild_tensor(cache, [64, 4])) == 'cache'

    state = torch.empty(4, dtype=torch.int64, device='meta')
    assert tensor_resident_owner(rebuild_tensor(state, [8])) is None


def test_meta_tensor_pool_tags_each_resident_account():
    weight = torch.empty(8, dtype=torch.bfloat16)
    cache = torch.empty(32, dtype=torch.int8)
    state = torch.empty(4, dtype=torch.int64)
    owners = {id(weight): 'weight', id(cache): 'cache'}
    ops = [_Op('aten.mm.default', (state, weight, cache), torch.empty(8))]

    serialized = serialize_op_invoke_infos(ops, resident_owner=lambda tensor: owners.get(id(tensor)))

    args = serialized[0].args
    assert tensor_resident_owner(args[1]) == 'weight'
    assert tensor_resident_owner(args[2]) == 'cache'
    assert tensor_resident_owner(args[0]) is None
    assert args[1].device.type == 'meta'
    assert _MetaTensorPool()._resident_owner is None


def _tracker_with_repeated_graph(weight, state, *, provenance):
    tracker = MemoryTracker(None)
    if provenance:
        tracker.register_compiled_weight_constants([weight])
    for repeat in range(3):
        tracker.record_single_op_invocation(
            _Op(
                torch.ops.aten.mm.default,
                (state, weight),
                torch.empty(8, device='meta'),
            ),
            repeat,
        )
    tracker.analyze()
    return tracker


def test_tagged_graph_weights_are_resident_and_untagged_ones_are_request_input():
    # 8x4 bf16 weight = 64 B, 4x8 int64 state = 256 B; replay repeats both three
    # times.  A registered weight is resident, so it never reaches the request
    # account, while an untagged placeholder is charged as request input.
    weight = _tagged((8, 4))
    state = torch.empty((4, 8), dtype=torch.int64, device='meta')

    tagged = _tracker_with_repeated_graph(weight, state, provenance=True)
    untagged = _tracker_with_repeated_graph(weight, state, provenance=False)

    assert tagged.get_forward_memory_summary().request_input_bytes == 4 * 8 * 8
    assert untagged.get_forward_memory_summary().request_input_bytes == 64 + 4 * 8 * 8


def test_capture_classifier_matches_the_native_resident_registry():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, 4))
    model.register_buffer('scale', torch.empty(4))

    tracker = MemoryTracker(None)
    tracker.register_static_weight(model.weight)
    tracker.register_static_weight(model.scale)

    assert tracker.resident_owner(model.weight) == 'weight'
    assert tracker.resident_owner(model.scale) == 'weight'
    assert tracker.resident_owner(torch.empty(8, 4)) is None
    # A repacked view of the weight shares its storage and stays a weight.
    assert tracker.resident_owner(model.weight.view(4, 8)) == 'weight'


def test_capture_classifier_reports_registered_caches_as_cache():
    """A cache tensor is resident too, but under the cache account.

    The served peak already adds the cache inventory, so a tagged cache
    placeholder must be registered as cache instead of being charged again as
    request input.
    """

    tracker = MemoryTracker(None)
    cache = torch.empty(32, 4, dtype=torch.int8)
    tracker.register_kv_cache(cache)

    assert tracker.resident_owner(cache) == 'cache'
    assert tracker.resident_owner(cache.view(4, 32)) == 'cache'


def test_capture_classifier_follows_the_region_equivalence_registry():
    """A copy the runtime aliases to a weight is still a weight.

    Meta placeholders expose no shared storage, so a repacked copy can only be
    recognised through the Region registry's tensor key.  Missing it leaves the
    placeholder untagged and serving then charges those bytes as request input
    on top of the full weight account.
    """
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, 4))

    tracker = MemoryTracker(None)
    tracker.register_static_weight(model.weight)

    alias = torch.empty(8, 4)
    tracker.region_identity_registry.equivalent_tensor_id_manager.add_equivalent_keys(
        [(id(model.weight), 0), (id(alias), 0)]
    )

    assert tracker.resident_owner(alias) == 'weight'
    assert tracker.resident_owner(torch.empty(8, 4)) is None


def test_resident_key_cache_is_invalidated_by_region_keys_added_later():
    """A later capture on the same tracker must not reuse the first mapping.

    ``_resident_key_owners`` resolves every resident tensor through the region
    equivalence registry, and that registry keeps gaining keys while a forward
    is recorded (``Region._add_equivalent_info`` per captured region, and the
    workload replay's alias registration).  Invalidating only on resident
    registration leaves a mapping built before the region declared its alias, so
    a weight the graph copies is charged as request input on the next capture.
    """
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, 4))

    tracker = MemoryTracker(None)
    tracker.register_static_weight(model.weight)
    manager = tracker.region_identity_registry.equivalent_tensor_id_manager

    alias = torch.empty(8, 4)
    other = torch.empty(8, 4)
    # The alias is the oldest key in its group, so merging the weight into the
    # group moves the root the weight resolves to.
    manager.add_equivalent_keys([(id(alias), 0), (id(other), 0)])
    # Prime the cache before the weight joins that group.
    assert tracker.resident_owner(alias) is None

    manager.add_equivalent_keys([(id(alias), 0), (id(model.weight), 0)])

    assert tracker.resident_owner(alias) == 'weight'
    # The merge is applied to the cached map instead of discarding it: a
    # rebuild per union would cost one pass over every resident, and the
    # registry unions keys for every captured region.
    assert tracker._resident_key_owners_cache is not None


def test_rebinding_the_region_registry_stops_following_the_previous_one():
    """A tracker reused across runtimes must not follow the old registry.

    ``Runtime`` builds its own registry and binds it, so the same tracker can
    serve two runtimes in sequence.  Roots from the previous registry describe a
    state these tensors are no longer resolved against, and letting them keep
    feeding the cached map would answer with another runtime's keys.
    """
    from tensor_cast.performance_model.op_invoke_info import RegionIdentityRegistry

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, 4))
    tracker = MemoryTracker(None)
    tracker.register_static_weight(model.weight)

    previous = tracker.region_identity_registry
    alias = torch.empty(8, 4)
    previous.equivalent_tensor_id_manager.add_equivalent_keys([(id(model.weight), 0), (id(alias), 0)])
    assert tracker.resident_owner(alias) == 'weight'

    tracker.bind_region_identity_registry(RegionIdentityRegistry())

    # The fresh registry knows no aliases yet, so the cached answer must be gone
    # rather than inherited from the previous one.
    assert tracker.resident_owner(alias) is None
    assert previous.equivalent_tensor_id_manager._listeners == []
    tracker.region_identity_registry.equivalent_tensor_id_manager.add_equivalent_keys(
        [(id(model.weight), 0), (id(alias), 0)]
    )
    assert tracker.resident_owner(alias) == 'weight'


def test_a_replay_scope_voids_the_cached_region_resolutions():
    """``scoped_aliases`` restores a different state, so cached roots are void."""
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(8, 4))
    tracker = MemoryTracker(None)
    tracker.register_static_weight(model.weight)
    manager = tracker.region_identity_registry.equivalent_tensor_id_manager

    alias = torch.empty(8, 4)
    manager.add_equivalent_keys([(id(model.weight), 0), (id(alias), 0)])
    assert tracker.resident_owner(alias) == 'weight'
    assert tracker._resident_key_owners_cache is not None

    with manager.scoped_aliases():
        # Entering replaces the visible state, so a map built from the outer one
        # cannot be reused.
        assert tracker._resident_key_owners_cache is None
        inner = torch.empty(8, 4)
        manager.add_equivalent_keys([(id(model.weight), 0), (id(inner), 0)])
        assert tracker.resident_owner(inner) == 'weight'

    # Leaving restores the outer state: the scope's mapping is gone, the outer
    # alias still resolves.
    assert tracker.resident_owner(alias) == 'weight'


def _provenance_config(batch, sequence):
    return {
        'B': batch,
        'S': sequence,
        'TP': 2,
        'EP': 4,
        'MOE_DP': 1,
        'DCP': 1,
        'num_mtp_tokens': 0,
        'is_decode': False,
        'do_compile': True,
        'enable_dispatch_ffn_combine': False,
        'quantize_linear_action': 'DISABLED',
        'quantize_attention_action': 'DISABLED',
        'model_parameters': (('E', 256), ('E_per_tok', 6), ('H', 8)),
    }


def _provenance_capture(batch, sequence):
    from template_engine.capture_ops import CapturedOp
    from template_engine.template_builder import _Capture

    local = (batch + 1) // 2
    hidden = torch.empty((local * sequence, 8), device='meta')
    weight = _tagged((16, 8))
    state = torch.empty((local,), dtype=torch.int64, device='meta')
    out = torch.empty((local * sequence, 16), device='meta')
    ops = [
        CapturedOp('aten.mm.default', (hidden, weight), {}, out),
        CapturedOp('aten.index.Tensor', (out, [state]), {}, out.clone()),
    ]
    return _Capture(str((batch, sequence)), _provenance_config(batch, sequence), ops, 0.0)


def test_built_template_artifact_keeps_weight_provenance_through_serving(tmp_path):
    """Capture tag -> group body blob -> saved lazy artifact -> expanded graph."""

    from template_engine.template_builder import build_fixed_template
    from template_engine.template_graph import generate, load_template_graph
    from template_engine.zero_meta import _save_template_lazy

    captures = [_provenance_capture(8, s) for s in (256, 512)]
    engine, _ = build_fixed_template(captures)
    graph = {
        'engine': engine,
        'meta': {
            'model': 'owner/Provenance-Model',
            'num_hidden_layers': 2,
            'do_compile': True,
        },
    }
    path = tmp_path / 'provenance.json.xz'
    _save_template_lazy(graph, path)

    result = generate(
        load_template_graph(path),
        {**_provenance_config(8, 1024), 'model_id': 'owner/Provenance-Model'},
        return_ops=True,
    )
    expanded = result['ops']

    weights = [arg for op in expanded for arg in op.args if getattr(arg, TENSOR_WEIGHT_ATTR, False)]
    assert weights, 'expanded graph lost the captured weight tag'
    assert {tuple(weight.shape) for weight in weights} == {(16, 8)}
    assert {weight.numel() * weight.element_size() for weight in weights} == {16 * 8 * 2}
    state = [
        item for op in expanded if str(op.func).startswith('aten.index') for item in op.args if isinstance(item, list)
    ]
    assert state and not any(getattr(tensor, TENSOR_WEIGHT_ATTR, False) for tensor in state[0])


def test_capture_requires_a_runtime_with_a_memory_tracker():
    """An untagged capture is worse than a failed one, so refuse it by name.

    ``capture`` serializes the observed runtime with the tracker's
    ``resident_owner`` probe.  Without the probe every weight and KV
    placeholder loses its provenance and is charged as request input on top of
    the peak formula that already counts those bytes.  The failure must say
    that, not surface a bare ``AttributeError`` on ``runtime.memory_tracker``,
    and it must not degrade into that double-counting capture.
    """
    from types import SimpleNamespace

    from template_engine.capture_ops import CaptureSerializationError, _resident_owner_probe

    with pytest.raises(CaptureSerializationError, match='memory tracker'):
        _resident_owner_probe(SimpleNamespace())
    with pytest.raises(CaptureSerializationError, match='memory tracker'):
        _resident_owner_probe(SimpleNamespace(memory_tracker=None))

    def probe(tensor):
        return 'weight'

    runtime = SimpleNamespace(memory_tracker=SimpleNamespace(resident_owner=probe))
    assert _resident_owner_probe(runtime) is probe
