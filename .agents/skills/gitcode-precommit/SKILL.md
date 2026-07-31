---
name: gitcode-precommit
description: Use GitCode CLI `precommit check` to verify a repository's pre-commit configuration and local environment before committing or submitting code. Detects `.pre-commit-config.yaml`, verifies/auto-installs the pre-commit tool, ensures the git hook is initialized, and optionally runs the hooks. Trigger before committing or opening a PR, or when a user asks to check pre-commit readiness. Requires GitCode CLI v0.6.0+.
metadata:
  version: 1.0.0
  source: gitcode-cli/skills@6f0d8b0-adapted
---

# gitcode-precommit

Make sure pre-commit is configured and the local environment can run it before code is committed.

## Command Entry

Use `gitcode` for cross-platform instructions. Linux/macOS may use `gc`; Windows PowerShell should use `gitcode`.

Requires GitCode CLI **v0.6.0+** (the `precommit` command was added in v0.6.0). Confirm with `gitcode version` or `gitcode help precommit check`.

## What It Checks

`gitcode precommit check` runs in the current git repository and:

1. Detects a `.pre-commit-config.yaml` (or `.yml`) in the repo root. No config means nothing to check (exit 0).
2. Verifies the `pre-commit` tool is installed.
3. Verifies the git `pre-commit` hook is initialized (worktree-aware).
4. With `--run`, executes `pre-commit run --all-files`.

When the config is present but the tool or hook is missing, **install and
initialize it** rather than just reporting the gap. The command does this
automatically in an interactive terminal; in a non-interactive (non-TTY)
environment — which is the usual case for an agent — pass `--yes` to authorize
the install/init. Use `--no-install` only when the intent is strictly to
diagnose without changing the environment.

## Check

```bash
# Verify the environment is ready
gitcode precommit check

# Verify and actually run the hooks
gitcode precommit check --run

# Machine-readable result
gitcode precommit check --json
```

## Install / Initialize When Missing

When the repo has a pre-commit config but the tool is not installed or the hook
is not initialized, run the check in install-authorizing mode so it installs
pre-commit and/or runs `pre-commit install` before continuing. In an
interactive terminal `--yes` is not required; in automation it is.

```bash
# Install + initialize if missing, then run the hooks (non-interactive / agent)
gitcode precommit check --run --yes

# Install + initialize if missing, no hook run
gitcode precommit check --yes

# Diagnose only, never modify the environment (explicit opt-out)
gitcode precommit check --no-install --json
```

The `actions_taken` array reports what was done, e.g.
`["installed pre-commit via pipx", "ran pre-commit install"]`.

## Flags

| Flag | Effect |
| --- | --- |
| `--run` | After verifying, run `pre-commit run --all-files`. |
| `--no-install` | Only diagnose; never install the tool or hook. Mutually exclusive with `--yes`. |
| `--yes`, `-y` | Authorize environment changes (install/init) in a non-interactive environment. |
| `--json` | Emit a structured result to stdout. |

## JSON Result

```json
{
  "config_found": true,
  "tool_installed": true,
  "tool_version": "3.7.0",
  "hook_installed": true,
  "actions_taken": ["installed pre-commit via pipx", "ran pre-commit install"],
  "run_result": "passed",
  "run_output": "",
  "ok": true
}
```

- Read `ok` together with `config_found`: `ok: true, config_found: false` means "no config, skipped", not "verified ready".
- `run_output` carries the `pre-commit run` output only when `run_result` is `failed`.

## Exit Codes

- `0` ready, or no config (nothing to check).
- `1` not ready / checks failed / not a git repository / non-interactive change refused.
- `2` usage error (e.g. `--no-install` together with `--yes`).

## Cross-platform Install

When auto-installing, the tool tries available installers in order and falls through on failure: `pipx` → `python3 -m pip install --user` → `python -m pip install --user`, plus the `py` launcher on Windows. If no Python tooling is available it prints platform-specific manual instructions instead of installing Python.

## Rules

- Run from inside the target git repository.
- When the repo has a pre-commit config but the tool/hook is missing, install
  and initialize it: pass `--yes` in non-interactive automation so the command
  is authorized to install pre-commit and run `pre-commit install`. Only use
  `--no-install` when the task is explicitly diagnose-only.
- Use `--json` when a script or agent needs to parse readiness; key off `ok`,
  `run_result`, and `actions_taken`.
- This command checks/sets up the local pre-commit environment; it does not
  replace the repository's own test suite or CI.
