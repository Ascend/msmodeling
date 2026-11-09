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

"""Fixed operator/reference buckets below a discrete configuration cell.

These predicates describe official execution conditions, not a B/S bounding
box learned from observations. Unknown or colliding domains never select a
body by proximity. Historical folded artifacts keep their original reader.
"""

import hashlib
import json
from functools import lru_cache

FIXED_POLICY = 'fixed_structure_v1'


@lru_cache(maxsize=128)
def _data_parallel_size(tp, ep, md, dcp):
    from tensor_cast.model_config import ParallelConfig

    parallel = ParallelConfig(
        world_size=max(tp, ep, dcp) * md,
        tensor_parallel_size=tp,
        expert_parallel_size=ep,
        moe_data_parallel_size=md,
        decode_context_parallel_size=dcp,
    )
    return parallel.data_parallel_size


def parallel_geometry(config):
    dimensions = (int(config.get(name, 1)) for name in ('TP', 'EP', 'MOE_DP', 'DCP'))
    dp = _data_parallel_size(*dimensions)
    local = (int(config['B']) + dp - 1) // dp
    return dp, local


def parameter_config(config):
    """Equivalent global B representative for fixed-bucket decode formulas.

    Official input generation consumes ceil(B/DP) requests on each rank.
    Preserve the caller's identity elsewhere; only formula coordinates use
    this canonical representative of the same per-rank request batch.
    """
    if config.get('structure_policy') == FIXED_POLICY and config.get('is_decode'):
        dp, local = parallel_geometry(config)
        return {**config, 'B': local * dp}
    return config


def prefill_anchor_config(groups, config):
    """Use an observed global B representative of the same Prefill rank batch.

    Keep the stored anchor's formula coordinates: existing formulas may use
    global B. Sequence length and structure admission are checked separately;
    this mapping changes neither the target S nor its local batch.
    """
    if config.get('structure_policy') != FIXED_POLICY or config.get('is_decode'):
        return config
    from .body_select import discrete_cell

    cell = discrete_cell(config)
    _, local = parallel_geometry(config)
    batches = set()
    for group in groups.values():
        if tuple(group.get('discrete_cell', ())) != cell:
            continue
        for member in (group.get('mode_meta') or {}).get('member_configs', []):
            if parallel_geometry(member)[1] == local:
                batches.add(int(member['B']))
    if not batches or int(config['B']) in batches:
        return config
    return {**config, 'B': min(batches)}


def structure_domain(config, window=0):
    parameters = dict(config.get('model_parameters') or ())
    if not parameters:
        return None
    dp, local = parallel_geometry(config)
    if config.get('is_decode'):
        query = int(config.get('num_mtp_tokens', 0)) + 1
        return {'phase': 'decode', 'local_batch': local, 'context_empty': int(config['S']) == query}
    from .serve_rules import prefill_structure_domain

    domain = prefill_structure_domain(config, window)
    domain['phase'] = 'prefill'
    tp, ep = int(config.get('TP', 1)), int(config.get('EP', 1))
    # ParallelMoELayer._dp_transform_enter pads only on the DP != EP path.
    domain['tp_padding'] = bool(ep > 1 and dp != ep and local * int(config['S']) % tp)
    return domain


def bucket_id(cell, fingerprint):
    payload = json.dumps((cell, fingerprint), ensure_ascii=True, separators=(',', ':'), default=str)
    return 'sb_' + hashlib.sha256(payload.encode('utf-8')).hexdigest()


def bucket_matches(group, config):
    bucket = (group.get('mode_meta') or {}).get('structure_bucket')
    if not bucket:
        return True
    domain = structure_domain(config, bucket['window'])
    if domain is None:
        from .body_select import _same_config

        return any(_same_config(member, config) for member in group['mode_meta']['member_configs'])
    from .serve_rules import domain_matches

    return any(domain_matches(expected, domain) for expected in bucket['domains'])
