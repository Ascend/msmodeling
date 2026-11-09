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

"""Template-specific fields composed with the official inference parameter definitions."""

from cli.registry.datatypes import ModuleSpec, Param
from cli.registry import shared as S
from cli.registry.modules import text_generate as T

DEFAULT_CAPTURE_DIR = './template_runs/captures'

CACHE_DIR = Param(name='cache-dir', data_type='string', required=True, cli_help='Capture corpus directory.')
TEMPLATE = Param(name='template', data_type='string', required=True, cli_help='Template artifact path.')
COMPILE_MODE = Param(
    name='compile-mode',
    data_type='string',
    default='unknown',
    choices=['on', 'off', 'unknown'],
    cli_help='Filter capture compile provenance.',
)
REQUEST = [
    S.MODEL_ID,
    S.DEVICE,
    S.NUM_DEVICES.override(default=None),
    T.NUM_QUERIES.override(required=True),
    T.QUERY_LENGTH.override(required=True),
    T.CONTEXT_LENGTH,
    T.DECODE,
    T.NUM_MTP_TOKENS,
    S.COMPILE,
    S.COMPILATION_CONFIG,
    S.QUANTIZE_LINEAR_ACTION,
    S.QUANTIZE_ATTENTION_ACTION,
    T.TP_SIZE,
    T.EP_SIZE,
    T.MOE_DP_SIZE,
    T.DCP_SIZE,
    T.NUM_HIDDEN_LAYERS_OVERRIDE,
]
CAPTURE_SPEC = ModuleSpec(
    module_id='template_capture',
    title='Capture a model graph',
    cli_prog='msmodeling inference template-graph capture',
    fields=[
        *REQUEST,
        CACHE_DIR.override(required=False, default=DEFAULT_CAPTURE_DIR),
        Param(name='force', data_type='boolean', default=False, cli_help='Replace an existing capture.'),
        Param(name='sampler', data_type='boolean', default=False, cli_help='Include sampling operators.'),
    ],
)
BUILD_SPEC = ModuleSpec(
    module_id='template_build',
    title='Build a template from captured graphs',
    cli_prog='msmodeling inference template-graph build',
    requires_model_id=False,
    fields=[
        CACHE_DIR.override(required=False, default=DEFAULT_CAPTURE_DIR),
        TEMPLATE.override(
            required=False,
            default=None,
            cli_help='Output path; default: auto-discovered model file in template_engine/templates.',
        ),
        COMPILE_MODE,
        T.NUM_HIDDEN_LAYERS_OVERRIDE,
    ],
)
GENERATE_SPEC = ModuleSpec(
    module_id='template_generate',
    title='Generate a graph from a template',
    cli_prog='msmodeling inference template-graph generate',
    fields=[
        *REQUEST,
        TEMPLATE,
        Param(name='output', data_type='string', default=None, cli_help='Save generated ops as JSON+XZ.'),
    ],
)


# The old module syntax shares the same field objects; only aliases/defaults
# differ, so legacy entry points cannot acquire a second model parameter schema.
LEGACY_SPEC = ModuleSpec(
    module_id='template_legacy',
    title='Template graph compatibility entry',
    cli_prog='template_engine.template_graph',
    requires_model_id=False,
    fields=[
        TEMPLATE.override(required=False, default='template_graph.json.xz'),
        CACHE_DIR.override(required=False, default='compile_ops_cache_v4_serving'),
        COMPILE_MODE,
        T.NUM_HIDDEN_LAYERS_OVERRIDE.override(default=None, cli_aliases=('layers',), cli_dest='layers'),
        Param(name='build', data_type='boolean', default=False, cli_help='Build from the capture corpus.'),
        Param(name='config', data_type='string', default=[], cli_action='append', cli_help='Legacy k=v request.'),
        Param(name='sweep-json', data_type='string', default=None, cli_help='Legacy JSON request list.'),
        Param(name='json', data_type='boolean', default=False, cli_help='Print JSON results.'),
    ],
)


COVERAGE_SPEC = ModuleSpec(
    module_id='template_coverage',
    title='Inspect template coverage',
    requires_model_id=False,
    fields=[
        CACHE_DIR,
        TEMPLATE.override(required=False, default=None),
        Param(name='configs', data_type='string', cli_help='Semicolon-separated legacy request specs.'),
        Param(name='sweep', data_type='string', cli_help='Legacy B/S range spec.'),
        Param(name='save-template', data_type='string', cli_help='Optional template output.'),
        Param(name='out', data_type='string', cli_help='Coverage report output.'),
    ],
)
RESOLVE_SPEC = ModuleSpec(
    module_id='template_resolve',
    title='Resolve a graph with optional capture fallback',
    requires_model_id=False,
    fields=[field for field in LEGACY_SPEC.fields if field.name not in ('build', 'compile-mode', 'cache-dir')]
    + [
        CACHE_DIR,
        Param(name='rebuild-template', data_type='boolean', default=False, cli_help='Rebuild the template.'),
        Param(name='no-capture', data_type='boolean', default=False, cli_help='Disable capture fallback.'),
    ],
)
