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

"""Model formula parameters read from MS's resolved model configuration.

No model is loaded and no process-global constants are changed during expansion.
Legacy artifacts without parameters retain their historical formula vocabulary.
"""

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _kv_layout(model_config, config):
    """Classify the KV layout the same way ``dcp_kv_token_capacity_factor`` does.

    Under Decode Context Parallel the per-card token capacity factor depends on
    the KV layout, not on the model name: sparse-attention models keep factor 1,
    MLA shards by ``dcp``, and GQA only converts the TP head replication
    ``min(dcp, max(1, tp // h_kv))`` into capacity.  The template side cannot
    recover that classification from (B, S, TP, DCP), so record it next to the
    other formula parameters while the resolved model config is still available.
    Captures written before this field existed keep evaluating with the legacy
    ``WINDOW`` heuristic (see ``parameter_features._dcp_cache_shard_factor``).
    """
    if config.get('model_type') in ('deepseek_v4', 'deepseek_v41', 'deepseek_v41_text'):
        return 'sparse'
    mla_config = getattr(model_config, 'mla_config', None)
    if mla_config is not None:
        requires_indexer_cache = getattr(getattr(mla_config, 'mla_cls', None), 'requires_indexer_cache', None)
        sparse = bool(requires_indexer_cache()) if callable(requires_indexer_cache) else False
        return 'sparse' if sparse else 'mla'
    # Everything else carries a per-head KV layout (GQA / MHA).  The native path
    # claims no DCP gain there unless ``num_key_value_heads`` is available, which
    # is exactly what a missing ``KV_HEADS`` reproduces.
    return 'gqa'


def formula_parameters(model_config, *, block_size=None):
    config = getattr(model_config, 'hf_config', model_config)
    text_config = config.get('text_config') if isinstance(config, Mapping) else getattr(config, 'text_config', None)
    if text_config is not None:
        config = text_config
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        return None
    fields = {
        'H': ('hidden_size',),
        'I': ('intermediate_size',),
        'MI': ('moe_intermediate_size',),
        'V': ('vocab_size',),
        'E': ('n_routed_experts', 'num_local_experts', 'num_experts'),
        'E_per_tok': ('num_experts_per_tok', 'num_experts_per_token'),
        'CR': ('compress_ratio',),
        'INDEX_TOPK': ('topk_limit', 'index_topk'),
        'WINDOW': ('sliding_window',),
    }
    values = {}
    for name, aliases in fields.items():
        for alias in aliases:
            value = config.get(alias)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                values[name] = value
                break
    for ratio in config.get('compress_ratios', []) or []:
        if type(ratio) is int and ratio > 0:
            values[f'COMPRESS_RATIO/{ratio}'] = ratio
    kv_heads = config.get('num_key_value_heads')
    if isinstance(kv_heads, int) and not isinstance(kv_heads, bool) and kv_heads > 0:
        values['KV_HEADS'] = kv_heads
    layout = _kv_layout(model_config, config)
    if layout is not None:
        values['KV_LAYOUT'] = layout
    if block_size is not None:
        values['BLOCK_SIZE'] = int(block_size)
    return tuple(sorted(values.items())) if values else None


def bind_parameters(template, config):
    from .structure_buckets import FIXED_POLICY

    config = dict(config)
    metadata = template.get('template_meta', {})
    if metadata.get('parameter_bindings'):
        from .body_select import discrete_cell

        cell = discrete_cell(config)
        binding = next((item for item in metadata['parameter_bindings'] if tuple(item['cell']) == cell), None)
        if binding is not None:
            metadata = {**metadata, **binding}
    if metadata.get('structure_policy') == FIXED_POLICY:
        config['structure_policy'] = FIXED_POLICY
    parameters = metadata.get('model_parameters')
    if parameters is None:
        return dict(config)
    parameters = tuple(tuple(pair) for pair in parameters)
    supplied = config.get('model_parameters')
    if supplied is not None:
        supplied = {name: value for name, value in (tuple(pair) for pair in supplied)}
        # The template declaration is the contract: every declared parameter
        # must agree.  A supplied superset is allowed because resolution keeps
        # gaining derivable inputs (e.g. ``KV_LAYOUT``) that older artifacts
        # never declared; those extras are dropped below so the evaluation stays
        # exactly what the artifact was fit with.
        disagree = [name for name, value in parameters if name in supplied and supplied[name] != value]
        if disagree:
            raise ValueError('model formula parameters disagree with the template')
        # Dropping the extras silently would also swallow a mistyped parameter
        # name in a template's own binding: the artifact would keep serving the
        # value it was fit with while the caller believes its input took effect.
        # Name them so a typo is visible instead of indistinguishable from the
        # intended "newer resolver, older artifact" case.
        undeclared = sorted(set(supplied) - {name for name, _ in parameters})
        if undeclared:
            logger.warning(
                'Model formula parameters %s were supplied but are not declared by this template; '
                'ignoring them so the artifact keeps evaluating as it was fit.',
                ', '.join(undeclared),
            )
    return {**config, 'model_parameters': parameters}
