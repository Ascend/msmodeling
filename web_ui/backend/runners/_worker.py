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

"""Subprocess worker entry point.

Invoked as: python -m runners._worker <module_id> <params.json> <result.json>
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import traceback

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> int:
    """Worker entry point.

    Returns:
        Exit code (0 = success, 1 = error, 2 = usage error).
    """
    if len(sys.argv) != 4:
        print(
            "usage: python -m runners._worker <module_id> <params.json> <result.json>",
            file=sys.stderr,
        )
        return 2

    module_id, params_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    # Runner logs -> stdout so the spawner captures them into the job log.
    # Fallback default matches CLI (spec_cli.add_log_options default="error").
    # Module-specific form defaults (e.g. video_generate default="info") are
    # sent explicitly in the params dict by the API layer.
    level_name = str(params.get("log-level", "error")).upper()
    level = getattr(logging, level_name, logging.ERROR)
    logging.basicConfig(stream=sys.stdout, level=level, format=_LOG_FORMAT)

    # Load custom device profiles (tensor_cast/device_profiles/*.py) for EVERY
    # module. throughput_optimizer + video_generate import the package themselves,
    # but text_generate's ModelRunner path does not — without this, a custom
    # profile would not register for text jobs. Idempotent (Python caches the
    # package). Best-effort: a malformed profile file logs a traceback but does
    # not abort the job (it then either runs on a builtin device or fails with a
    # clear "unknown device" error downstream).
    try:
        import tensor_cast.device_profiles  # noqa: F401
    except Exception:
        traceback.print_exc()

    try:
        module = importlib.import_module(f"runners.{module_id}")
        # Case-dedup metadata injected by the main process (NOT form params);
        # pop them out so execute() sees only the real form values.
        cached_hashes = set(params.pop("_cached_case_hashes", []))
        form_schema_version = params.pop("_form_schema_version", None)
        job_id = params.pop("_job_id", None)
        provided = set(params.pop("_provided", []))
        records, skipped = module.execute(
            params,
            cached_hashes=cached_hashes,
            form_schema_version=form_schema_version,
            job_id=job_id,
            provided=provided,
        )
    except Exception:
        traceback.print_exc()
        return 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "skipped": skipped}, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
