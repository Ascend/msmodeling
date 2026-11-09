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

"""Scalar features from official routing/cache calculations, without a model forward.

The official helper owns the finite-token hotspot estimate and balanced expert
splits. Using its outputs avoids approximating sqrt/ceil routing loads with
linear B/S fits. No model identifiers or captured target values occur here.
"""

from itertools import groupby
from types import SimpleNamespace


def prefill_routing_features(batch, sequence, tp, ep, moe_dp, parameters):
    if ep <= tp or not parameters.get('E') or not parameters.get('E_per_tok'):
        return {}
    from tensor_cast.layers.moe_layer import FusedMoETensorCast, assign_experts
    from .structure_buckets import parallel_geometry

    dp, local_batch = parallel_geometry(dict(TP=tp, EP=ep, MOE_DP=moe_dp, B=batch))
    # ParallelMoELayer transforms the attention DP domain only when DP !=
    # EP. Otherwise fused MoE receives the original packed tokens directly.
    tokens = local_batch * sequence
    padding = (-tokens) % tp if dp != ep else 0
    if dp != ep:
        tokens = (tokens + tp - 1) // tp
    experts, topk = parameters['E'], parameters['E_per_tok']
    start, count = assign_experts(experts, ep, 0)
    context = SimpleNamespace(
        ep_group=SimpleNamespace(world_size=ep, rank_in_group=0),
        num_external_shared_experts=0,
        num_global_experts=experts,
        expert_idx_start=start,
        num_local_experts=count,
    )
    splits = FusedMoETensorCast.get_split_sizes(context, tokens * topk, topk)
    features = {
        'moe_route/tp_padding': padding,
        'moe_route/tokens': tokens,
        'moe_route/routed_total': tokens * topk,
        'moe_route/input_total': sum(splits[0]),
        'moe_route/output_total': sum(splits[1]),
    }
    for name, values in zip(('input_device', 'output_device', 'input_expert'), splits[:3]):
        for index, value in enumerate(values):
            features[f'moe_route/{name}/{index}'] = value
    if splits[3] is not None:
        merged = [sum(rank[index] for rank in splits[3]) for index in range(count)]
        for index, value in enumerate(merged):
            features[f'moe_route/merged_expert/{index}'] = value
        features['moe_route/merged_total'] = sum(merged)
        runs = [(value, len(list(items))) for value, items in groupby(merged)]
        # Balanced expert splits have at most two consecutive token counts.
        for index in range(2):
            value, length = runs[index] if index < len(runs) else (0, 0)
            features[f'moe_route/run/{index}/tokens'] = value
            features[f'moe_route/run/{index}/experts'] = length
            features[f'moe_route/run/{index}/rows'] = value * length
    return features


def prefill_cache_features(batch, sequence, tp, ep, moe_dp, mtp, parameters, *, decode=False):
    window, limit = parameters.get('WINDOW'), parameters.get('INDEX_TOPK')
    ratios = sorted({value for name, value in parameters.items() if name.startswith('COMPRESS_RATIO/')})
    from tensor_cast.core.input_generator import _resolve_v4_kv_cache_size, _resolve_sparse_attention_indexer_num_blocks
    from tensor_cast.core.user_config import UserInputConfig
    from .structure_buckets import parallel_geometry

    _, local_batch = parallel_geometry(dict(TP=tp, EP=ep, MOE_DP=moe_dp, B=batch))
    block = parameters.get('BLOCK_SIZE', UserInputConfig.block_size)
    model = SimpleNamespace(
        text_config=SimpleNamespace(sliding_window=window), model_config=SimpleNamespace(hf_config=SimpleNamespace())
    )
    features = {'prefill/request_tokens': local_batch * sequence, 'prefill/local_batch': local_batch}
    if window:
        packed = local_batch * sequence
        cutoff = packed % window
        features.update(
            {
                'prefill/window/length': min(packed, window),
                'prefill/window/cutoff': cutoff,
                'prefill/window/first': window - cutoff,
            }
        )
    total = local_batch * (sequence + mtp + 1)
    features['prefill/cache/full_blocks'] = max(1, (total + block - 1) // block)
    features['prefill/cache/sequence_blocks'] = (sequence + block - 1) // block
    if limit:
        features['prefill/index/width'] = min(sequence, limit)
    if not window or not ratios:
        return features
    for ratio in ratios:
        layer = SimpleNamespace(_head_dim=1, compress_ratio=ratio)
        shape = _resolve_v4_kv_cache_size(
            model, layer, block_size=block, batch_size=local_batch, total_kv_tokens=local_batch * (sequence + mtp + 1)
        )
        features[f'prefill/cache/{ratio}/blocks'] = shape[0]
        features[f'prefill/cache/{ratio}/index_blocks'] = _resolve_sparse_attention_indexer_num_blocks(
            True, layer, 1, block, local_batch, local_batch * (sequence + mtp + 1)
        )
        # compressor's concrete query_lens==1 branch returns one row, even
        # when the caller requested Prefill rather than Decode.
        compressed = 1 if sequence == 1 else local_batch * sequence // ratio
        features[f'prefill/cache/{ratio}/compressed_tokens'] = compressed
        features[f'prefill/cache/{ratio}/merged_tokens'] = local_batch * sequence + compressed

        if limit:
            width = min(limit, max(sequence // ratio, 1))
            features[f'prefill/index/{ratio}/width'] = width
            # get_window_topk_idxs keeps the full physical window in Decode,
            # even when the logical context is shorter than that window.
            window_width = window if decode or sequence < 5 else min(local_batch * sequence, window)
            features[f'prefill/index/{ratio}/window_width'] = window_width + width
    return features
