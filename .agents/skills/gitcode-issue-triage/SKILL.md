---
name: gitcode-issue-triage
description: Triage GitCode issue queues with GitCode CLI by listing, filtering, labeling, prioritizing, grouping duplicates, and preparing triage comments. Trigger for issue queue cleanup, labeling, prioritization, or backlog grooming.
metadata:
  version: 1.0.0
  source: gitcode-cli/skills@6f0d8b0-adapted
---

# gitcode-issue-triage

Triage a GitCode issue queue.

## Command Entry

Use `gitcode` for cross-platform instructions. Linux/macOS may use `gc`; Windows PowerShell should use `gitcode`.

## Workflow

1. List issues with stable filters.
2. Inspect labels and milestones.
3. Review candidate issues before changing metadata.
4. Apply labels or comments in small batches.
5. Summarize changes and unresolved questions.

## Commands

```bash
gitcode issue list -R owner/repo --state open --limit 50 --json
gitcode issue list -R owner/repo --state all --search "keyword" --json
gitcode issue view 123 -R owner/repo --comments --json
gitcode label list -R owner/repo --json
gitcode milestone list -R owner/repo --json
```

Apply metadata:

```bash
gitcode issue label 123 -R owner/repo --add type/bug,priority-high
gitcode issue label 123 -R owner/repo --remove needs-triage
gitcode issue edit 123 -R owner/repo --milestone 5 --json
gitcode issue comment 123 -R owner/repo --body-file triage-comment.md --json
```

Close or reopen only with clear evidence:

```bash
gitcode issue close 123 -R owner/repo --yes --json
gitcode issue reopen 123 -R owner/repo --yes --json
```

## Triage Categories

- Type: bug, feature, enhancement, docs, question
- Priority: critical, high, medium, low
- Scope: API, CLI, docs, auth, repo, issue, PR, release
- Status: needs-info, ready, blocked, duplicate, wontfix

Use the repository's actual labels where possible.

## Rules

- Do not invent labels without checking `label list`.
- Avoid mass changes without showing a preview.
- Add comments when closing as duplicate or needs-info.
- Keep an audit summary of issue numbers changed.
