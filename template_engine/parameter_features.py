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
    # Dispatch is a property of the expert group: ``ParallelMoELayer`` emits the
    # balanced per-expert all-to-all whenever ``EP > 1``, including when the
    # expert group is not wider than the TP group.  Gating on ``ep <= tp``
    # dropped the features for exactly those cells (Kimi TP32/EP32), so their
    # dispatch lists and per-expert slice shapes degraded to per-anchor
    # EXACT_LOOKUP and no unseen (B, S) could be served.
    if ep <= 1 or not parameters.get('E') or not parameters.get('E_per_tok'):
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


def _dcp_cache_shard_factor(parameters, tp, dcp) -> int:
    """Per-card KV token shard under Decode Context Parallel.

    Mirrors ``input_generator.dcp_kv_token_capacity_factor`` per cell.  The
    native factor is chosen by KV layout, so the capture records ``KV_LAYOUT``
    (``sparse`` / ``mla`` / ``gqa``) and ``KV_HEADS`` next to the other formula
    parameters:

    * ``sparse``: the cache is window+compress sized and its indexer part is not
      sequence-sharded, so the pool keeps factor 1.
    * ``mla``: the latent cache is not partitioned across TP heads, so a card
      really holds ``1 / dcp`` of every sequence's tokens: factor ``dcp``.
    * ``gqa``: DCP re-partitions KV heads, so the per-card bytes are invariant
      and the only gain is the TP replication ``min(dcp, max(1, tp / h_kv))``;
      with ``h_kv`` unavailable the native path claims no gain (factor 1).

    Artifacts captured before ``KV_LAYOUT`` existed keep the historical
    ``1 if WINDOW else dcp`` rule so already-published templates evaluate
    exactly as they did when they were fit; the ``INDEX_TOPK`` half of the
    heuristic covers the other sparse-attention family (DSA / GLM5.3), whose
    cache is window+index sized in the same way and whose native factor is 1.
    """
    dcp = max(1, int(dcp))
    if dcp <= 1:
        return 1
    layout = parameters.get('KV_LAYOUT')
    if layout == 'sparse':
        return 1
    if layout == 'mla':
        return dcp
    if layout == 'gqa':
        heads = parameters.get('KV_HEADS')
        if isinstance(heads, int) and not isinstance(heads, bool) and heads > 0:
            return min(dcp, max(1, int(tp) // heads))
        return 1
    if parameters.get('WINDOW') or parameters.get('INDEX_TOPK'):
        return 1
    return dcp


def prefill_cache_features(batch, sequence, tp, ep, moe_dp, mtp, parameters, *, decode=False, dcp=1):
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
    if limit and not decode:
        # ``sparse_attn_sharedkv.compress_topk`` is the width of the lightning
        # indexer output consumed by sparse attention: min(INDEX_TOPK, packed
        # prefill tokens). Packed tokens are the rank-local batch times S
        # (tensor-parallel shards repeat the token rows, expert-parallel
        # shards split them), so a TP4/EP32 shard at B17/S256 packs 3 requests
        # and keeps topk 768, while S>=INDEX_TOPK always clamps at INDEX_TOPK.
        # Without this feature the small-token anchors either degrade to
        # per-anchor EXACT_LOOKUP or fit a spurious routing MULTI.
        #
        # It precedes ``prefill/index/width`` (min(S, INDEX_TOPK), the
        # per-token indexer width) because the two agree on every anchor with
        # S>=INDEX_TOPK; the packed-token law is the one that extrapolates
        # correctly to a larger rank-local batch at a small S.
        features['prefill/index/topk'] = min(local_batch * sequence, limit)
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
    # Decode Context Parallel keeps only ``1 / shard`` of every sequence's
    # tokens on a card, so the paged pool it sizes shrinks by that factor and
    # the split still rounds UP because a whole block is the allocation unit
    # (``input_generator.dcp_sharded_num_blocks``).  ``shard`` follows the KV
    # layout exactly like ``dcp_kv_token_capacity_factor``: sparse-attention
    # models keep 1, MLA uses ``dcp``, GQA uses the TP head replication.  The
    # earlier "has WINDOW" proxy only matched the sparse case and sharded GQA
    # cells by ``dcp`` even when their real cache does not gain capacity.
    # Without a DCP-aware key the DCP>1 bodies fit ``full_blocks``, which is
    # only right when the per-rank request batch happens to equal DCP; every
    # other context is off by the shard factor.  Kept after ``full_blocks`` so
    # DCP=1 bodies (where the two agree) keep the historical key.
    shard = _dcp_cache_shard_factor(parameters, tp, dcp)
    features['prefill/cache/paged_blocks'] = max(1, -(-features['prefill/cache/full_blocks'] // shard))
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
