"""Run CLI entrypoints in-process so coverage (and test_map) sees the full path.

Subprocess-based CLI tests are invisible to coverage.py here (no
COVERAGE_PROCESS_START/sitecustomize), so they contribute zero coverage and the
incremental gate never maps them to deep core symbols. Calling ``main()``
in-process keeps the same observable result (returncode + captured streams)
while letting coverage measure the real work.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CliResult:
    """Subset of ``subprocess.CompletedProcess`` used by CLI tests."""

    returncode: int
    stdout: str
    stderr: str


def run_cli_main(main_callable: Callable[[], object], argv: list[str], *, prog: str = "cli") -> CliResult:
    """Invoke ``main_callable`` with ``sys.argv`` set to ``[prog, *argv]``.

    ``SystemExit`` codes and integer return values are mapped to ``returncode``
    the same way ``python -m module`` would report them. Other exceptions are
    captured as ``returncode=1`` with the traceback on stderr, mirroring a
    crashed subprocess instead of propagating into the test.
    """
    out = io.StringIO()
    err = io.StringIO()
    returncode = 0
    saved_argv = sys.argv
    sys.argv = [prog, *argv]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                result = main_callable()
            except SystemExit as exc:
                code = exc.code
                if code is None:
                    returncode = 0
                elif isinstance(code, int):
                    returncode = code
                else:
                    returncode = 1
                    print(code, file=sys.stderr)
            except Exception:
                returncode = 1
                traceback.print_exc(file=sys.stderr)
            else:
                returncode = result if isinstance(result, int) else 0
    finally:
        sys.argv = saved_argv
    return CliResult(returncode=returncode, stdout=out.getvalue(), stderr=err.getvalue())


def run_module_main(module_name: str, argv: list[str]) -> CliResult:
    """Import ``module_name`` and run its ``main()`` in-process."""
    module = importlib.import_module(module_name)
    return run_cli_main(module.main, argv, prog=module_name.rsplit(".", 1)[-1])
