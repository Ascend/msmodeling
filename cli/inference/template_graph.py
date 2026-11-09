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

"""Template commands using the official CLI registry and MS request configuration."""

import argparse
import json
import logging
import sys
from pathlib import Path

from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.spec_cli import configure_std_logging
from cli.utils import LOG_FORMAT
from template_engine.capture_ops import MODEL_ID, _parse_bool, write_capture
from template_engine.template_graph import (
    TemplateGraphRefusal,
    build_template_graph,
    generate,
    load_template_graph,
)


def legacy_capture_argv(argv):
    """Translate old flags; all validation/defaults are owned by the registry."""
    raw = list(sys.argv[1:] if argv is None else argv)
    aliases = {
        '--B': '--num-queries',
        '--tp': '--tp-size',
        '--ep': '--ep-size',
        '--moe-dp': '--moe-dp-size',
        '--dcp': '--dcp-size',
        '--quant-linear': '--quantize-linear-action',
        '--quant-attention': '--quantize-attention-action',
        '--layers': '--num-hidden-layers-override',
    }
    result, total, mtp, dfc = [], None, 0, False
    # The parser owns arity; an option's value is never scanned as an option.
    from cli.registry.modules.template_graph import CAPTURE_SPEC
    from cli.registry.argparse_adapter import build_argparser

    parser = build_argparser(CAPTURE_SPEC)
    value_options = {name for name, act in parser._option_string_actions.items() if act.nargs != 0}
    index = 0
    while index < len(raw):
        option, separator, attached = raw[index].partition('=')
        translated = aliases.get(option, option)
        if option in ('--S', '--num-mtp-tokens', '--dfc'):
            if separator:
                value = attached
            else:
                index += 1
                if index >= len(raw):
                    raise ValueError('Missing value for ' + option)
                value = raw[index]
            if option == '--S':
                total = int(value)
            elif option == '--dfc':
                dfc = _parse_bool(value)
            else:
                mtp = int(value)
                result.append(option + '=' + value)
        elif separator:
            result.append(translated + '=' + attached)
        elif translated in value_options:
            index += 1
            if index >= len(raw):
                raise ValueError('Missing value for ' + option)
            result.append(translated + '=' + raw[index])
        else:
            result.append(translated)
        index += 1
    present = {token.partition('=')[0] for token in result}
    if total is not None:
        query = mtp + 1 if '--decode' in result else total
        result.extend(['--query-length', str(query), '--context-length', str(total - query)])
    if '--model-id' not in present and '--model-path' not in present:
        result.extend(['--model-id', MODEL_ID])
    if '--quantize-linear-action' not in present:
        result.extend(['--quantize-linear-action', 'DISABLED'])
    if dfc:
        result.extend(['--compilation-config', 'enable_dispatch_ffn_combine'])
    return result


def legacy_capture_main(argv=None):
    return main(['capture', *legacy_capture_argv(argv)])


def _build(args, layers=None):
    from template_engine.cache_io import cache_variant_dir
    from cli.registry.modules.template_graph import DEFAULT_CAPTURE_DIR

    args.cache_dir = cache_variant_dir(args.cache_dir, layers or 0)
    # The default root can contain multiple model namespaces. Never choose one
    # by directory order or by whichever capture happens to be discovered first.
    if Path(args.cache_dir) == Path(cache_variant_dir(DEFAULT_CAPTURE_DIR, layers or 0)):
        root = Path(args.cache_dir)
        corpora = (
            ([root] if any(root.glob('*.json.xz')) else [])
            + [child for child in root.iterdir() if child.is_dir() and any(child.glob('*.json.xz'))]
            if root.is_dir()
            else []
        )
        if len(corpora) > 1:
            raise ValueError('Multiple model corpora found; specify --cache-dir for the intended model directory')
        if corpora:
            args.cache_dir = corpora[0]
    graph, stats = build_template_graph(
        args.template,
        cache_dir=args.cache_dir,
        do_compile={'on': True, 'off': False, 'unknown': None}[args.compile_mode],
        num_hidden_layers=layers or None,
    )
    if args.template is None:
        from template_engine.registry import registration_path
        from template_engine.zero_meta import _save_template_lazy

        meta = graph['meta']
        models = meta['model_ids']
        if len(models) != 1 or models[0] == 'unknown':
            raise ValueError('Cannot infer one model; use --cache-dir for a single model with capture metadata')
        args.template = str(registration_path(models[0], meta['do_compile']))
        _save_template_lazy(graph, args.template)
    print(json.dumps({'built': args.template, **stats}))
    return 0


def legacy_graph_main(argv: list[str] | None = None) -> int:
    from cli.registry.modules.template_graph import LEGACY_SPEC

    parser = build_argparser(LEGACY_SPEC)
    args = parser.parse_args(argv)
    configure_std_logging(args, log_format=LOG_FORMAT)

    # logs go to stderr so stdout stays reserved for the --json contract
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.build:
        return _build(args, args.layers)

    from template_engine.zero_meta import parse_config_spec

    configs = [parse_config_spec(spec) for spec in args.config]
    if args.sweep_json:
        with open(args.sweep_json, encoding="utf-8") as fh:
            configs.extend(dict(item) for item in json.load(fh))
    if not configs:
        parser.error("no configs given (--config or --sweep-json)")

    tg = load_template_graph(args.template)
    results = []
    logger = logging.getLogger(__name__)
    for config in configs:
        try:
            r = generate(tg, config, num_hidden_layers=args.layers)
            results.append(r)
            logger.info('[%s] op_count=%d, generation=%ss', r['coverage'], r['op_count'], r['wall_time_s'])
        except TemplateGraphRefusal as refusal:
            results.append(
                {
                    "covered": False,
                    "config": refusal.config,
                    "blocker": refusal.blocker,
                    "anchors_required": refusal.anchors,
                }
            )
            logger.info("[refused] %s; anchors: %s", refusal.blocker, refusal.anchors)
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    return 0


def main(argv=None):
    from template_engine.config_adapter import config_from_args
    from template_engine.graph_codec import encode_capture

    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ('coverage', 'resolve'):
        from cli.inference.template_graph_diagnostics import coverage_main, resolve_main

        return {'coverage': coverage_main, 'resolve': resolve_main}[raw[0]](raw[1:])

    if not raw or raw[0] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action', choices=('capture', 'build', 'generate', 'coverage', 'resolve'))
        parser.print_help()
        return 0
    action = raw.pop(0)
    if action not in ('capture', 'build', 'generate'):
        raise ValueError('Expected template action: capture, build or generate')
    spec = get_spec('template_' + action)
    parser = build_argparser(spec)
    args = parse_module_args(spec, parser, raw)
    configure_std_logging(args, log_format=LOG_FORMAT)
    try:
        if action == 'build':
            return _build(args, args.num_hidden_layers_override)
        config = config_from_args(args)
        world = max(args.tp_size, args.ep_size, args.dcp_size) * args.moe_dp_size
        if args.num_devices is not None and args.num_devices != world:
            raise ValueError('This template layout requires num-devices = max(TP,EP,DCP)*MOE-DP')
        if action == 'capture':
            if args.device != 'TEST_DEVICE':
                raise ValueError('Template corpus capture currently requires --device TEST_DEVICE')
            return write_capture(
                argparse.Namespace(
                    B=config['B'],
                    S=config['S'],
                    tp=args.tp_size,
                    ep=args.ep_size,
                    moe_dp=args.moe_dp_size,
                    dcp=args.dcp_size,
                    decode=args.decode,
                    compile=args.compile,
                    num_mtp_tokens=args.num_mtp_tokens,
                    dfc=config['enable_dispatch_ffn_combine'],
                    quant_linear=args.quantize_linear_action,
                    quant_attention=args.quantize_attention_action,
                    model_id=args.model_id,
                    layers=args.num_hidden_layers_override,
                    cache_dir=args.cache_dir,
                    force=args.force,
                    sampler=args.sampler,
                )
            )
        graph = load_template_graph(args.template)
        result = generate(
            graph,
            config,
            num_hidden_layers=args.num_hidden_layers_override or None,
            return_ops=bool(args.output),
        )
        ops = result.pop('ops', None)
        if args.output:
            Path(args.output).write_bytes(encode_capture(ops, 0.0))
            result['path'] = str(Path(args.output).resolve())
        print(json.dumps(result, default=str))
        return 0
    except (ValueError, TemplateGraphRefusal) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
