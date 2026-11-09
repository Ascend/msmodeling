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

from collections.abc import Mapping


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
    if supplied is not None and tuple(tuple(pair) for pair in supplied) != parameters:
        raise ValueError('model formula parameters disagree with the template')
    return {**config, 'model_parameters': parameters}
