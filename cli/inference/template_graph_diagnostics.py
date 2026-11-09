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

"""Compatibility diagnostic commands; parameter definitions live in cli.registry."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from cli.registry.argparse_adapter import build_argparser
from cli.registry.modules.template_graph import COVERAGE_SPEC, RESOLVE_SPEC
from cli.spec_cli import configure_std_logging
from cli.utils import LOG_FORMAT
from template_engine.coverage import _parse_target_spec, describe_coverage, parse_sweep
from template_engine.zero_meta import parse_config_spec, load_or_build_template, resolve_config


def coverage_main(argv: list[str] | None = None) -> int:
    from template_engine.template_builder import (
        build_fixed_template,
        discover_cache_keys,
        load_captures_from_cache,
        save_template,
    )

    parser = build_argparser(COVERAGE_SPEC)
    args = parser.parse_args(argv)
    configure_std_logging(args, log_format=LOG_FORMAT)

    if args.template:
        from template_engine.zero_meta import load_template_engine

        template = load_template_engine(args.template)
    else:
        keys = discover_cache_keys(args.cache_dir)
        if not keys:
            logging.getLogger(__name__).error("no capture keys found under %s", args.cache_dir)
            print(json.dumps({"error": "no capture keys found"}))
            return 2
        captures = load_captures_from_cache(keys, args.cache_dir)
        template, _ = build_fixed_template(captures)
        if args.save_template:
            save_template(template, args.save_template)

    configs: list[dict[str, Any]] = []
    if args.configs:
        for chunk in args.configs.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            configs.append(_parse_target_spec(chunk))
    if args.sweep:
        configs.extend(parse_sweep(args.sweep))
    if not configs:
        parser.error("nothing to check: pass --configs and/or --sweep")

    rows = [describe_coverage(template, config) for config in configs]
    payload = {
        "groups": {
            str(key): group_keys
            for key, group_keys in ((k, sorted(v.keys())) for k, v in template.get("groups", {}).items())
        },
        "rows": rows,
        "exact": sum(1 for row in rows if row["verdict"] == "exact_group"),
        "interpolated": sum(1 for row in rows if row["verdict"] == "interpolated"),
        "fallback": sum(1 for row in rows if row["verdict"] == "fallback_group"),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    # data output (the coverage report itself), not a log line - stdout stays
    # the machine channel; diagnostic messages go through logging (issue
    # #20 PR review: no print for logs)
    sys.stdout.write(text + "\n")
    return 0


def resolve_main(argv: list[str] | None = None) -> int:
    parser = build_argparser(RESOLVE_SPEC)
    args = parser.parse_args(argv)
    configure_std_logging(args, log_format=LOG_FORMAT)

    # logs go to stderr so stdout stays reserved for the --json contract
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configs: list[dict[str, Any]] = [parse_config_spec(spec) for spec in args.config]
    if args.sweep_json:
        with open(args.sweep_json, encoding="utf-8") as stream:
            configs.extend(dict(item) for item in json.load(stream))
    if not configs:
        parser.error("no configs given (--config or --sweep-json)")

    template, template_path = load_or_build_template(
        args.template,
        args.cache_dir,
        rebuild=args.rebuild_template,
    )

    results = []
    logger = logging.getLogger(__name__)
    for config in configs:
        result = resolve_config(
            config,
            template=template,
            template_path=template_path,
            cache_dir=args.cache_dir,
            allow_capture=not args.no_capture,
            num_hidden_layers=args.layers,
        )
        results.append(result)
        label = result["key"]
        if result["covered"]:
            timing = next(iter(result["execution_time_s"].items()))
            logger.info(
                "[%s] %s: %d ops, %s=%.3fs, %ss",
                result["source"],
                label,
                result["op_count"],
                timing[0],
                timing[1],
                result["wall_time_s"],
            )
        else:
            logger.info("[refused] %s: blocker=%s (%ss)", label, result.get("blocker"), result["wall_time_s"])

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    return 0
