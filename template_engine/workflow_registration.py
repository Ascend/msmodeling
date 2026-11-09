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

"""Install only independently validated fixed buckets into optimizer discovery."""

from .file_lock import destination_lock

from .graph_codec import read_artifact_bytes

import hashlib
from pathlib import Path

from . import registry
from .compare_ops import compare_performance_ops
from .graph_codec import atomic_write_bytes, decode_capture
from .template_graph import generate, load_template_graph, TemplateGraphRefusal
from .zero_meta import _encode_template_lazy, ensure_group_body


def _sha(path):
    return hashlib.sha256(read_artifact_bytes(Path(path))).hexdigest()


def register_validated(
    template_path, model_id, layers, do_compile, targets, training, run_dir, *, read_call=None, lock_timeout=60
):
    """targets carry config/status/bucket; training carries config/graph.

    A partial search domain can be useful: install its verified buckets only,
    and keep the report partial. Pending/failed targets exclude their bucket.
    Native graph mismatches are never converted into a successful registration.
    """
    if read_call is None:

        def read_call(_label, callback):
            return callback()

    graph = read_call("registration template load", lambda: load_template_graph(template_path))
    if graph['engine'].get('template_meta', {}).get('structure_policy') != 'fixed_structure_v1':
        raise ValueError('Automatic registration requires fixed structure buckets')
    passed, blocked = {}, set()
    for target in targets:
        bucket = target.get('structure_bucket_id')
        if bucket:
            if target['status'] == 'compared_pass':
                passed.setdefault(bucket, []).append(target['config'])
            else:
                blocked.add(bucket)
    allowed = set(passed) - blocked
    anchor_counts, rejected = {}, []
    for sample in training:
        try:
            result = read_call(
                'registration anchor expansion',
                lambda: generate(graph, sample['config'], return_ops=True),
            )
        except TemplateGraphRefusal:
            continue
        bucket = result['structure_bucket_id']
        if bucket not in allowed:
            continue
        expected, _ = read_call(
            'registration anchor decode', lambda: decode_capture(read_artifact_bytes(Path(sample['graph'])))
        )
        if not read_call('registration anchor comparison', lambda: compare_performance_ops(expected, result['ops']))[
            'passed'
        ]:
            blocked.add(bucket)
            rejected.append({'bucket': bucket, 'reason': 'training_anchor_mismatch'})
        anchor_counts[bucket] = anchor_counts.get(bucket, 0) + 1
    allowed = {bucket for bucket in allowed - blocked if anchor_counts.get(bucket)}
    if not allowed:
        return {'status': 'not_registered', 'reason': 'no_independently_validated_bucket', 'rejected': rejected}

    # Load an independent graph through the lossless codec. Tensor deepcopy can
    # compact non-dense views and silently change their recorded strides.
    deployment = read_call("registration deployment load", lambda: load_template_graph(template_path))
    engine = deployment['engine']
    groups = engine['groups']
    selected = {key: group for key, group in groups.items() if group['mode_meta']['structure_bucket']['id'] in allowed}
    for key in selected:
        read_call("registration body decode", lambda: ensure_group_body(engine, key))
    engine['groups'] = selected
    engine.pop('_body_blobs', None)
    engine.pop('_body_blob_codec', None)
    engine['template_meta']['group_keys'] = [list(key) for key in selected]
    engine['template_meta']['routing'] = {
        name: [key for key in keys if tuple(key) in selected]
        for name, keys in engine['template_meta'].get('routing', {}).items()
        if any(tuple(key) in selected for key in keys)
    }
    engine['template_meta']['coverage'] = {'failed_formulas': 0}
    deployment.pop('mode_router', None)  # Build diagnostics describe the original corpus.
    deployment['router'] = {
        'structures': [
            s for s in deployment.get('router', {}).get('structures', []) if tuple(s['group_key']) in selected
        ]
    }
    configs = [config for bucket in sorted(allowed) for config in passed[bucket]]
    deployment['meta']['registration'] = {
        'validated_buckets': sorted(allowed),
        'validated_targets': len(configs),
        'scope': 'validated_subset',
    }
    folder = Path(run_dir) / 'registration'
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / 'validated-template.json.xz'
    prepared = read_call("registration artifact encoding", lambda: _encode_template_lazy(deployment))
    atomic_write_bytes(artifact, prepared)
    digest = hashlib.sha256(prepared).hexdigest()
    destination = registry.registration_path(model_id, do_compile)
    with destination_lock(destination, timeout=lock_timeout):
        previous = read_artifact_bytes(destination) if destination.exists() else None
        backup = None
        previous_digest = hashlib.sha256(previous).hexdigest() if previous is not None else None
        if previous_digest is not None and previous_digest != digest:
            backup = folder / 'backups' / (previous_digest + '.json.xz')
            atomic_write_bytes(backup, previous)
        try:
            installed = registry.register_template(artifact, model_id, layers, do_compile=do_compile)
            resolved = registry.template_path_for(model_id, layers, do_compile=do_compile)
            if not resolved or Path(resolved).resolve() != Path(installed).resolve() or _sha(resolved) != digest:
                raise ValueError('Optimizer registry discovery/readback differs from installed artifact')
            registered = read_call("registration readback", lambda: load_template_graph(resolved))
            for config in configs:
                original = read_call(
                    "registration original expansion", lambda: generate(graph, config, return_ops=True)
                )
                installed_result = read_call(
                    "registration installed expansion",
                    lambda: generate(registered, config, return_ops=True),
                )
                if not read_call(
                    'registration installed comparison',
                    lambda: compare_performance_ops(original['ops'], installed_result['ops']),
                )['passed']:
                    raise ValueError('Registry readback changed a validated graph')
        except (ValueError, OSError, RuntimeError, TemplateGraphRefusal):
            if previous is not None:
                atomic_write_bytes(destination, previous)
            elif destination.exists():
                destination.unlink()
            registry._MODEL_META_CACHE.pop(str(destination), None)
            raise
    return {
        'status': 'registered',
        'path': installed,
        'sha256': digest,
        'validated_buckets': sorted(allowed),
        'validated_targets': len(configs),
        'training_anchors_checked': sum(anchor_counts[bucket] for bucket in allowed),
        'excluded_buckets': len(groups) - len(selected),
        'backup': str(backup) if backup else None,
        'discovery_verified': True,
        'rejected': rejected,
    }
