---
name: gitcode-pr-create
description: Guide creation of GitCode pull requests using GitCode CLI with branch checks, diff review, body-file support, fork PR support, and JSON result parsing. Trigger when users want to open, draft, or update a GitCode PR.
metadata:
  version: 1.1.0
  source: gitcode-cli/skills@6f0d8b0-adapted
---

# gitcode-pr-create

Create a GitCode PR with evidence and a clean description.

## Command Entry

Use `gitcode` for cross-platform instructions. Linux/macOS may use `gc`; Windows PowerShell should use `gitcode`.

## Workflow

1. Verify current branch and working tree.
2. Run the pre-commit readiness check (see Pre-commit Check).
3. Confirm the source remote is a personal Fork and the target repository is canonical.
4. Review diff and test evidence.
5. Draft PR body in a file.
6. Create PR with `--body-file` and `--json`.

## Local Checks

```bash
git status --short --branch
git remote -v
git log --oneline --decorate -5
git diff --stat origin/master...HEAD
```

Before a code contribution write, verify that the branch-push source is not the
canonical repository:

```bash
python scripts/ai/resolve_repository_context.py \
  --write --require-fork-source --source-remote <fork-remote> \
  --repo Ascend/msmodeling --json
```

## Pre-commit Check

If the repository configures pre-commit, verify the hooks actually ran on the
code before opening the PR. When the tool or hook is missing, install and
initialize it rather than skipping the check. Requires GitCode CLI v0.6.0+; see
the `gitcode-precommit` skill for full details.

```bash
# Install + initialize if missing, then run the hooks (non-interactive / agent)
gitcode precommit check --run --yes

# Interactive terminal: --yes is not required
gitcode precommit check --run
```

- A `0` exit (or `ok: true`) means there is no config, or the hooks passed.
- A non-zero exit means the hooks failed or the environment is not ready; fix
  that before creating the PR rather than relying on remote CI to catch it.

## PR Body Template

```markdown
## Summary
- ...

## Verification
- [ ] command/result
- [ ] `gitcode precommit check --run` passed (if the repo uses pre-commit)

## Risk
- ...

## Related
- Closes #...
```

## Create

```bash
gitcode pr create -R Ascend/msmodeling --fork <fork-owner>/msmodeling --head feature-branch --base master --title "feat: summary" --body-file pr.md --json
```

## Update Existing PR

```bash
gitcode pr edit 123 -R owner/repo --body-file pr.md --json
gitcode pr edit 123 -R owner/repo --title "new title" --json
gitcode pr ready 123 -R owner/repo --ready --yes --json
```

## Rules

- `pr create` supports `--body-file`; prefer it for multi-line descriptions.
- Use `--json` to capture the created PR URL/number.
- Do not create from `master`.
- The source branch must be pushed to a personal Fork; never create or push a feature branch in `Ascend/msmodeling`.
- Always pass `--fork` when the target is `Ascend/msmodeling`.
- Use SSH for push/clone operations.
- Do not include secrets in the PR body.
- Treat `Ascend/msmodeling` as canonical; resolve the writable source from Git remotes.
- Show and confirm target repository before every remote write.
