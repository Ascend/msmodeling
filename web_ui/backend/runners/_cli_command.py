"""Pure CLI command reconstruction (shared by subprocess spawner and API).

The `build_cli_command_string` function renders the equivalent CLI command for
a job from its module_id and params. This is used:
- In `_subprocess.py` to log the reference command before spawning the worker.
- In the API router to include the command in JobStatusResponse for display.

No heavy imports (subprocess/tempfile/ResultRecord) — this is a pure formatter.
"""

from __future__ import annotations

import shlex
from typing import Any

# module_id -> the real CLI module path (for the reference command string only).
_CLI_MODULE = {
    "text_generate": "cli.inference.text_generate",
    "video_generate": "cli.inference.video_generate",
    "throughput_optimizer": "cli.inference.throughput_optimizer",
}

# Field ids that don't map 1:1 (snake_case -> kebab-case) to a CLI flag name.
_FLAG_OVERRIDES = {}

# Field ids to skip entirely when building the CLI command (removed from CLI or
# never had a corresponding flag; kept for backward-compat with cached jobs).
_SKIP_FIELDS: set[str] = set()

# Single-value CLI fields that may contain commas but should NOT be split into
# space-separated tokens. The CLI's argparse defines these as a single string
# argument (split internally on "," by run_inference). Contrast with nargs="+"
# fields (mtp_acceptance_rate, tp_sizes, ...) which DO need comma-splitting.
_NO_SPLIT_FIELDS = {"cache_step_range", "cache_block_range"}


def build_cli_command_string(module_id: str, params: dict[str, Any]) -> str:
    """Render the equivalent CLI command for the job log (reference only — the
    worker, not this command, is what actually runs). Best-effort formatting.
    """
    cli_mod = _CLI_MODULE.get(module_id, f"cli.inference.{module_id}")
    parts: list[str] = [f"python -m {cli_mod}"]
    model_id = params.get("model_id")
    if model_id:
        parts.append(str(model_id))  # positional
    for key, val in params.items():
        # Skip None, False, empty strings, and fields removed from CLI.
        if key == "model_id" or val is None or val is False or val == "" or key in _SKIP_FIELDS:
            continue
        flag = _FLAG_OVERRIDES.get(key, key.replace("_", "-"))
        if val is True:
            # Boolean switch (e.g. --chrome-trace): emit the flag only, no value.
            parts.append(f"--{flag}")
        elif isinstance(val, list):
            parts.append(f"--{flag}")
            parts.extend(shlex.quote(str(v)) for v in val)
        elif isinstance(val, str) and "," in val and key not in _NO_SPLIT_FIELDS:
            # Comma-separated free-text list for a nargs="+" field (e.g.
            # mtp_acceptance_rate "0.8,0.6"): CLI expects space-separated values.
            parts.append(f"--{flag}")
            parts.extend(shlex.quote(v.strip()) for v in val.split(",") if v.strip())
        else:
            parts.append(f"--{flag}")
            parts.append(shlex.quote(str(val)))
    return " ".join(parts)
