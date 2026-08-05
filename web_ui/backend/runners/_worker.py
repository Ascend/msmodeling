"""Subprocess worker for Phase B.

Invoked by ``runners._subprocess.run_module_subprocess`` as::

    python -m runners._worker <module_id> <params.json> <result.json>

It reads params from ``<params.json>``, calls the module's ``execute(params)``
(which runs the unchanged ``tensor_cast``/``serving_cast`` runner API, prints the
CLI-style logs, and returns ``list[dict]``), and writes that list as JSON to
``<result.json>``. All stdout/stderr is captured by the spawner as the job log;
the structured result comes ONLY from the JSON file (never parsed from logs).

Only ``/web`` files are involved.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import traceback

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> int:
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
    level_name = str(params.get("log_level", "info")).upper()
    level = getattr(logging, level_name, logging.INFO)
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
        records, skipped = module.execute(
            params,
            cached_hashes=cached_hashes,
            form_schema_version=form_schema_version,
            job_id=job_id,
        )
    except Exception:
        traceback.print_exc()
        return 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "skipped": skipped}, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
