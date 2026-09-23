#!/usr/bin/env python3

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

"""Run CI + nightly-related pytest waves for the current msmodeling diff.

Passing both waves means this change should not fail (and thus not interrupt)
the matching subset of a full nightly. It is not a full nightly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PRODUCT_ROOTS = (
    "tensor_cast/",
    "serving_cast/",
    "cli/",
    "optix/",
    "web_ui/",
    "scripts/",
    "tools/",
)

CI_MARKER = "not npu and not nightly and not network"
NIGHTLY_RELATED_MARKER = "not npu and (nightly or benchmark or network)"
CANONICAL_OWNER = "Ascend"
CANONICAL_REPO = "Ascend/msmodeling"


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _git(repo: Path, *args: str) -> str:
    proc = _run(["git", *args], cwd=repo)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def _repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if proc.returncode != 0:
        raise SystemExit("not a git repository")
    return Path(proc.stdout.strip())


def _default_base(repo: Path) -> str:
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if "26.2.0" in branch:
        return "26.2.0"
    return "master"


def _resolve_base_ref(repo: Path, base: str, remote: str | None = None) -> str:
    candidates: list[str] = []
    if remote:
        candidates.append(f"{remote}/{base}")
    candidates.extend((f"upstream/{base}", f"origin/{base}", base))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        proc = _run(["git", "rev-parse", "--verify", candidate], cwd=repo)
        if proc.returncode == 0:
            return candidate
    raise SystemExit(f"cannot resolve base ref {base!r}")


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo)
    if proc.returncode in (0, 1):
        return proc.returncode == 0
    detail = (proc.stderr or proc.stdout).strip()
    raise SystemExit(f"git merge-base --is-ancestor {ancestor} {descendant} failed: {detail}")


def _is_dirty(repo: Path) -> bool:
    return bool(_git(repo, "status", "--porcelain").strip())


def _ahead_behind(repo: Path, base_ref: str) -> tuple[int, int]:
    counts = _git(repo, "rev-list", "--left-right", "--count", f"HEAD...{base_ref}").strip()
    parts = counts.split()
    if len(parts) != 2:
        raise SystemExit(f"unexpected rev-list output for HEAD...{base_ref}: {counts!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"non-integer rev-list output for HEAD...{base_ref}: {counts!r}") from None


def _canonical_remote(repo: Path) -> str:
    """Remote that points at Ascend/msmodeling. A contributor fork is not a base."""
    for remote in ("upstream", "origin"):
        proc = _run(["git", "remote", "get-url", remote], cwd=repo)
        if proc.returncode != 0:
            continue
        if _path_owner(proc.stdout) == CANONICAL_OWNER:
            return remote
    raise SystemExit(
        "no remote points at Ascend/msmodeling; "
        "add an upstream (or origin) remote for the canonical repository before syncing"
    )


def _fetch_canonical(repo: Path, base: str) -> tuple[str, str]:
    """Fetch the canonical base. Does not merge or move HEAD."""
    remote = _canonical_remote(repo)
    fetch = _run(["git", "fetch", remote, base], cwd=repo)
    if fetch.returncode != 0:
        raise SystemExit(f"git fetch {remote} {base} failed: {(fetch.stderr or fetch.stdout).strip()}")
    return remote, _resolve_base_ref(repo, base, remote)


def _sync_canonical(repo: Path, base: str) -> str:
    """Fetch the canonical base and merge it so the local pipeline is not stale."""
    remote, base_ref = _fetch_canonical(repo, base)
    ahead, behind = _ahead_behind(repo, base_ref)
    print(f"sync=fetched {remote}/{base} ahead={ahead} behind={behind}")
    if behind and _is_ancestor(repo, "HEAD", base_ref):
        short = _git(repo, "rev-parse", "--short", "HEAD").strip()
        raise SystemExit(
            f"HEAD ({short}) is already contained in {base_ref}; "
            "check out the change you intend to test (for example the PR head) before syncing"
        )
    if behind == 0:
        print("sync=already up to date with canonical base")
        return base_ref
    if _is_dirty(repo):
        raise SystemExit(
            f"canonical {remote}/{base} is {behind} commit(s) ahead, but the worktree is dirty; "
            "commit or stash first, then rerun so the pipeline is not based on a stale master"
        )
    merge = _run(["git", "merge", "--no-edit", base_ref], cwd=repo)
    if merge.returncode != 0:
        _run(["git", "merge", "--abort"], cwd=repo)
        raise SystemExit(
            f"git merge {base_ref} failed; resolve against latest {remote}/{base} before running the pipeline. "
            f"{(merge.stderr or merge.stdout).strip()}"
        )
    ahead, behind = _ahead_behind(repo, base_ref)
    print(f"sync=merged {base_ref} ahead={ahead} behind={behind}")
    return base_ref


def _changed_files(repo: Path, base_ref: str) -> list[str]:
    merge_base = _git(repo, "merge-base", "HEAD", base_ref).strip()
    committed = _git(repo, "diff", "--name-only", f"{merge_base}...HEAD")
    worktree = _git(repo, "diff", "--name-only", "HEAD")
    staged = _git(repo, "diff", "--name-only", "--cached")
    files = {line.strip() for line in (committed + worktree + staged).splitlines() if line.strip()}
    return sorted(files)


def _is_product(path: str) -> bool:
    return path.endswith(".py") and any(path.startswith(root) for root in PRODUCT_ROOTS)


def _is_test_py(path: str) -> bool:
    if not path.startswith("tests/") or not path.endswith(".py"):
        return False
    return not path.startswith("tests/assets/")


def _find_test_map(repo: Path, base: str) -> Path | None:
    env = os.environ.get("MSMODELING_TEST_MAP_PATH", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path
    candidate = repo / ".msmodeling_cache" / "test_map" / base / "test_map.json"
    if candidate.is_file():
        return candidate
    return None


def _nodes_from_test_map(test_map_path: Path, product_files: list[str]) -> set[str]:
    data = json.loads(test_map_path.read_text())
    nodes = data.get("map", data)
    if not isinstance(nodes, dict):
        return set()
    wanted = set(product_files)
    selected: set[str] = set()
    for node, sources in nodes.items():
        if not isinstance(sources, dict):
            continue
        if wanted.intersection(sources):
            selected.add(node)
    return selected


def _collect(repo: Path, python: str, targets: list[str], *, marker: str | None = None) -> list[str]:
    if not targets:
        return []
    cmd = [python, "-m", "pytest", *targets, "-o", "addopts=", "--collect-only", "-q", "--no-header"]
    if marker:
        cmd.extend(["-m", marker])
    proc = _run(cmd, cwd=repo)
    if proc.returncode not in (0, 5):
        sys.stderr.write(proc.stderr or proc.stdout)
        raise SystemExit(f"pytest collect-only failed (exit {proc.returncode})")
    nodes: list[str] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and stripped.startswith("tests/"):
            nodes.append(stripped)
    return nodes


def _pytest_python(repo: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    venv = repo / ".venv" / "bin" / "python"
    if venv.is_file() or venv.is_symlink():
        return str(venv)
    raise SystemExit("no repo virtualenv at .venv/bin/python; run uv sync or pass --python <venv python>")


def _related_dirs(paths: list[str]) -> list[str]:
    dirs: set[str] = set()
    for path in paths:
        parent = str(Path(path).parent)
        if parent.startswith("tests"):
            dirs.add(parent)
    return sorted(dirs)


def _gitcode_bin() -> str | None:
    found = shutil.which("gitcode")
    if found:
        return found
    for extra in (
        Path.home() / ".npm-global/bin/gitcode",
        Path.home() / ".local/lib/npm-global/bin/gitcode",
    ):
        if extra.is_file():
            return str(extra)
    return None


def _canonical_repo(_repo: Path) -> str:
    """Upstream PR target. Comments never go to a contributor fork."""
    return CANONICAL_REPO


def _branch_name(repo: Path) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()


def _path_owner(url: str) -> str | None:
    """Owner from a gitcode remote URL. ``git@host:owner/repo`` and https both work."""
    cleaned = url.strip()
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    if "gitcode.com/" in cleaned:
        path = cleaned.split("gitcode.com/", 1)[1]
    elif "://" not in cleaned and ":" in cleaned:
        path = cleaned.split(":", 1)[1]
    else:
        return None
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None
    return parts[0]


def _fork_owner(repo: Path) -> str | None:
    """Fork owner from ``fork``, else a non-canonical ``origin``. Never a hardcoded user."""
    for remote in ("fork", "origin"):
        proc = _run(["git", "remote", "get-url", remote], cwd=repo)
        if proc.returncode != 0:
            continue
        owner = _path_owner(proc.stdout)
        if owner and owner != "Ascend":
            return owner
    return None


def _item_head_ref(item: dict) -> str:
    head = item.get("head")
    if isinstance(head, dict):
        ref = str(head.get("ref") or head.get("label") or "")
    else:
        ref = str(head or item.get("head_branch") or "")
    if ":" in ref:
        ref = ref.split(":", 1)[1]
    return ref


def _item_fork_owner(item: dict) -> str:
    head = item.get("head")
    if not isinstance(head, dict):
        return ""
    full_name = str((head.get("repo") or {}).get("full_name") or "")
    if "/" not in full_name:
        return ""
    return full_name.split("/", 1)[0]


def _resolve_pr_number(repo: Path, *, explicit: str | None, canonical: str, head: str) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("GITCODE_PR_NUMBER", "").strip()
    if env:
        return env
    gitcode = _gitcode_bin()
    if not gitcode:
        print("comment=skip gitcode CLI not found", file=sys.stderr)
        return None
    owner = _fork_owner(repo)
    if not owner:
        print("comment=skip cannot determine fork owner", file=sys.stderr)
        return None
    proc = _run(
        [gitcode, "pr", "list", "-R", canonical, "--head", head, "--state", "open", "--json", "--limit", "20"],
        cwd=repo,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        print("comment=skip no open PR for this branch", file=sys.stderr)
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("comment=skip no open PR for this branch", file=sys.stderr)
        return None
    items = data if isinstance(data, list) else data.get("pulls") or data.get("items") or []
    matched: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = _item_head_ref(item)
        if ref and ref != head:
            continue
        if _item_fork_owner(item) != owner:
            continue
        number = item.get("number") or item.get("iid")
        if number is None:
            continue
        text = str(number)
        if text not in matched:
            matched.append(text)
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        print(f"comment=skip ambiguous head {head}: {', '.join(matched)}", file=sys.stderr)
        return None
    print("comment=skip no open PR for this branch", file=sys.stderr)
    return None


def _post_result_comment(repo: Path, *, pr: str, canonical: str, body: str) -> None:
    gitcode = _gitcode_bin()
    if not gitcode:
        print("comment=skip gitcode CLI not found")
        return
    proc = _run(
        [gitcode, "pr", "comment", pr, "-R", canonical, "--body", body, "--no-interactive"],
        cwd=repo,
    )
    if proc.returncode != 0:
        print(f"comment=fail {(proc.stderr or proc.stdout).strip()}")
        return
    print(f"comment=ok https://gitcode.com/{canonical}/merge_requests/{pr}")


def _maybe_comment_result(
    repo: Path,
    *,
    enabled: bool,
    pr_arg: str | None,
    status: str,
    selected: int,
    changed: int,
    base_ref: str,
    extra: str,
) -> None:
    if not enabled:
        return
    canonical = _canonical_repo(repo)
    pr = _resolve_pr_number(repo, explicit=pr_arg, canonical=canonical, head=_branch_name(repo))
    if not pr:
        return
    sha = _git(repo, "rev-parse", "HEAD").strip()
    body = (
        f"本地流水结果：{status}\n"
        f"HEAD `{sha[:12]}` · base `{base_ref}` · 改动文件 {changed} · 选测 {selected} 条"
        f"（CI + nightly/benchmark/network 子集，不是整场 nightly）。\n"
        f"{extra}"
    )
    _post_result_comment(repo, pr=pr, canonical=canonical, body=body)


def _filter_wave(repo: Path, python: str, nodes: list[str], marker: str) -> list[str]:
    """Split the selected set once, so each wave runs only its own nodes."""
    if not nodes:
        return []
    return _collect(repo, python, nodes, marker=marker)


def _run_wave(repo: Path, python: str, nodes: list[str], *, name: str, marker: str) -> int:
    if not nodes:
        print(f"wave={name} selected=0 skip")
        return 0
    cmd = [
        python,
        "-m",
        "pytest",
        *nodes,
        "-o",
        "addopts=",
        "-q",
        "--tb=line",
        "--no-header",
    ]
    print(f"wave={name} marker={marker!r} nodes={len(nodes)} prefiltered=1")
    proc = subprocess.run(cmd, cwd=repo, check=False)
    if proc.returncode == 5:
        print(f"wave={name} exit=5 treated_as_empty")
        return 0
    print(f"wave={name} exit={proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pr", default=None, help="GitCode PR number; default: detect from branch")
    parser.add_argument(
        "--comment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the run, post the result on the GitCode PR (pass or fail)",
    )
    parser.add_argument(
        "--sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fetch the canonical base before selecting tests. "
            "--run also merges it into the worktree (default: on). --dry-run fetches only."
        ),
    )
    args = parser.parse_args()
    if not args.dry_run and not args.run:
        args.dry_run = True

    repo = _repo_root(args.repo)
    base = args.base or _default_base(repo)
    if not args.sync:
        print("sync=skipped")
        base_ref = _resolve_base_ref(repo, base)
    elif args.dry_run and not args.run:
        remote, base_ref = _fetch_canonical(repo, base)
        ahead, behind = _ahead_behind(repo, base_ref)
        print(f"sync=fetched {remote}/{base} ahead={ahead} behind={behind} merge=skipped")
    else:
        base_ref = _sync_canonical(repo, base)
    python = _pytest_python(repo, args.python)
    changed = _changed_files(repo, base_ref)
    test_files = [path for path in changed if _is_test_py(path) and (repo / path).is_file()]
    product_files = [path for path in changed if _is_product(path)]

    test_map = _find_test_map(repo, base)
    mapped: set[str] = set()
    if test_map is not None and product_files:
        mapped = _nodes_from_test_map(test_map, product_files)

    collected = set(_collect(repo, python, test_files))
    mapped_files = sorted({node.split("::", 1)[0] for node in mapped if (repo / node.split("::", 1)[0]).is_file()})
    file_recollect = set(_collect(repo, python, mapped_files)) if mapped_files else set()
    related_dirs = _related_dirs(test_files + mapped_files)
    dir_nightly = set(_collect(repo, python, related_dirs, marker="not npu and (nightly or benchmark or network)"))
    selected = sorted(collected | mapped | file_recollect | dir_nightly)

    print(f"repo={repo}")
    print(f"base={base_ref}")
    print(f"changed_files={len(changed)}")
    print(f"test_map={test_map or 'MISSING'}")
    print(f"selected={len(selected)}")
    print(f"from_changed_tests={len(collected)}")
    print(f"from_test_map={len(mapped)}")
    print(f"from_file_recollect={len(file_recollect)}")
    print(f"from_related_dirs={len(dir_nightly)}")
    print("contract=both waves green => this diff should not fail/interrupt the matching nightly subset")
    if product_files and test_map is None:
        print("warning=product files changed but test_map is missing")
    if not changed:
        raise SystemExit(
            f"no diff against {base_ref}; refusing to report a pass with an empty change set. "
            "Check out the PR head (or the change under test) before syncing."
        )

    if args.dry_run and not args.run:
        for node in selected:
            print(node)
        return 0

    if not selected:
        print("no tests selected")
        _maybe_comment_result(
            repo,
            enabled=args.comment,
            pr_arg=args.pr,
            status="通过（无映射用例）",
            selected=0,
            changed=len(changed),
            base_ref=base_ref,
            extra="相对主仓没有 tests/ 或产品文件，未执行 pytest。",
        )
        return 0

    ci_nodes = _filter_wave(repo, python, selected, CI_MARKER)
    nightly_nodes = _filter_wave(repo, python, selected, NIGHTLY_RELATED_MARKER)
    print(f"wave_ci_nodes={len(ci_nodes)}")
    print(f"wave_nightly_nodes={len(nightly_nodes)}")
    ci_exit = _run_wave(repo, python, ci_nodes, name="ci", marker=CI_MARKER)
    nightly_exit = _run_wave(repo, python, nightly_nodes, name="nightly_related", marker=NIGHTLY_RELATED_MARKER)
    empty_note = []
    if not ci_nodes:
        empty_note.append("ci 波 0 条")
    if not nightly_nodes:
        empty_note.append("nightly_related 波 0 条")
    empty_text = (" " + "；".join(empty_note) + "，记为该波通过。") if empty_note else ""
    if ci_exit == 0 and nightly_exit == 0:
        print("result=PASS both waves; this change should not interrupt nightly on the selected subset")
        _maybe_comment_result(
            repo,
            enabled=args.comment,
            pr_arg=args.pr,
            status="通过",
            selected=len(selected),
            changed=len(changed),
            base_ref=base_ref,
            extra=f"ci + nightly_related 两波均绿。{empty_text}".rstrip(),
        )
        return 0
    print("result=FAIL; nightly can still be interrupted by this diff")
    _maybe_comment_result(
        repo,
        enabled=args.comment,
        pr_arg=args.pr,
        status="失败",
        selected=len(selected),
        changed=len(changed),
        base_ref=base_ref,
        extra=(
            f"ci exit={ci_exit} · nightly_related exit={nightly_exit}。"
            f"{empty_text}这次 diff 仍可能打断 nightly 对应子集。"
        ),
    )
    return ci_exit or nightly_exit


if __name__ == "__main__":
    raise SystemExit(main())
