"""Best-effort GitCode PR comments for non-blocking CI gate warnings."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from scripts.helpers._config import Config
    from scripts.helpers.ci_gate.models import GateError
    from scripts.helpers.common.ast_utils import ShadowWarning

logger = logging.getLogger(__name__)

_GITCODE_API_BASE: Final = "https://api.atomgit.com/api/v5"
_UNSCOPED_MARKER: Final = "<!-- msmodeling-ci-gate:unscoped-python -->"
_ALL_EXEMPT_MARKER: Final = "<!-- msmodeling-ci-gate:all-exempt-tests -->"
_EXEMPTION_DRIFT_MARKER: Final = "<!-- msmodeling-ci-gate:exemption-drift -->"
_SHADOWED_DEFS_MARKER: Final = "<!-- msmodeling-ci-gate:shadowed-defs -->"
_HTTP_TIMEOUT_SEC: Final = 10
_MAX_LISTED_PATHS: Final = 20
_PR_COMMENTS_PER_PAGE: Final = 100


@dataclass(frozen=True, slots=True)
class GitCodeCommentConfig:
    owner: str
    repo: str
    pr_number: int
    pat_token: str


class _PathMarkerPostFn(Protocol):
    def __call__(self, paths: tuple[str, ...], *, config: GitCodeCommentConfig) -> None: ...


_GITCODE_ENV_HINT: Final = "set GITCODE_OWNER, GITCODE_REPO, GITCODE_PR_NUMBER, GITCODE_PAT"


def load_gitcode_comment_config(
    cfg: Config | None = None,
) -> GitCodeCommentConfig | None:
    """Load GitCode comment credentials from Config or env; return None when incomplete."""
    if cfg is not None:
        owner = cfg.gitcode_owner.strip()
        repo = cfg.gitcode_repo.strip()
        pr_number = cfg.gitcode_pr_number
        pat_token = cfg.gitcode_pat.strip()
    else:
        owner = os.environ.get("GITCODE_OWNER", "").strip()
        repo = os.environ.get("GITCODE_REPO", "").strip()
        pr_raw = os.environ.get("GITCODE_PR_NUMBER", "").strip()
        pat_token = os.environ.get("GITCODE_PAT", "").strip()
        pr_number = None
        if pr_raw:
            try:
                pr_number = int(pr_raw)
            except ValueError:
                logger.warning("GITCODE_PR_NUMBER must be an integer, got %r", pr_raw)
                return None
    if not owner or not repo or pr_number is None or not pat_token:
        return None
    return GitCodeCommentConfig(owner=owner, repo=repo, pr_number=pr_number, pat_token=pat_token)


def build_unscoped_python_comment_body(paths: tuple[str, ...]) -> str:
    """Render a concise PR comment body for unscoped Python file changes."""
    lines = [
        _UNSCOPED_MARKER,
        "**CI gate (info):** Python file(s) outside configured source roots are not blocking, but please review scope:",
    ]
    lines.extend(f"- `{path}`" for path in paths[:_MAX_LISTED_PATHS])
    remaining = len(paths) - _MAX_LISTED_PATHS
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def _build_headers(pat_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {pat_token}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Connection": "keep-alive",
        "Origin": "https://gitcode.com",
        "Referer": "https://gitcode.com/",
    }


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, object] | None = None,
) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def _list_pr_comments(config: GitCodeCommentConfig) -> list[dict[str, object]]:
    comments: list[dict[str, object]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "comment_type": "pr_comment",
                "direction": "asc",
                "page": str(page),
                "per_page": str(_PR_COMMENTS_PER_PAGE),
            }
        )
        url = f"{_GITCODE_API_BASE}/repos/{config.owner}/{config.repo}/pulls/{config.pr_number}/comments?{query}"
        raw = _request_json("GET", url, headers=_build_headers(config.pat_token))
        if not isinstance(raw, list):
            break
        page_comments = [item for item in raw if isinstance(item, dict)]
        comments.extend(page_comments)
        if len(page_comments) < _PR_COMMENTS_PER_PAGE:
            break
        page += 1
    return comments


def build_all_exempt_tests_comment_body(paths: tuple[str, ...]) -> str:
    """Render a concise PR comment body for changed test files with only exempt nodes."""
    lines = [
        _ALL_EXEMPT_MARKER,
        "**CI gate (info):** Changed test file(s) contain only exempt test node(s); no pytest was scheduled:",
    ]
    lines.extend(f"- `{path}`" for path in paths[:_MAX_LISTED_PATHS])
    remaining = len(paths) - _MAX_LISTED_PATHS
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def build_exemption_drift_comment_body(errors: tuple[GateError, ...]) -> str:
    """Render a PR comment body for stale gate_policy exemptions."""
    lines = [
        _EXEMPTION_DRIFT_MARKER,
        "**CI gate (blocking):** `tests/.ci/gate_policy.yaml` exemption(s) reference deleted or renamed paths:",
    ]
    for err in errors[:_MAX_LISTED_PATHS]:
        label = f"{err.path}::{err.symbol}" if err.symbol else err.path
        detail = f" — {err.detail}" if err.detail else ""
        lines.append(f"- `{label}`{detail}")
    remaining = len(errors) - _MAX_LISTED_PATHS
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def build_shadowed_defs_comment_body(warnings: tuple[ShadowWarning, ...]) -> str:
    """Render a PR comment body for shadowed duplicate definitions."""
    lines = [
        _SHADOWED_DEFS_MARKER,
        "**CI gate (info):** Duplicate function definitions detected; last definition wins for coverage mapping:",
    ]
    lines.extend(
        f"- `{warning.file}:{warning.line}` `{warning.name}` shadowed by line {warning.shadowed_by_line}"
        for warning in warnings[:_MAX_LISTED_PATHS]
    )
    remaining = len(warnings) - _MAX_LISTED_PATHS
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def _find_marker_comment_id(comments: list[dict[str, object]], marker: str) -> int | None:
    for comment in comments:
        body = comment.get("body")
        if isinstance(body, str) and marker in body:
            note_id = comment.get("id")
            if isinstance(note_id, int):
                return note_id
    return None


def _create_pr_comment(config: GitCodeCommentConfig, body: str) -> None:
    url = f"{_GITCODE_API_BASE}/repos/{config.owner}/{config.repo}/pulls/{config.pr_number}/comments"
    _request_json("POST", url, headers=_build_headers(config.pat_token), payload={"body": body})


def _update_pr_comment(config: GitCodeCommentConfig, note_id: int, body: str) -> None:
    url = f"{_GITCODE_API_BASE}/repos/{config.owner}/{config.repo}/pulls/comments/{note_id}"
    _request_json("PATCH", url, headers=_build_headers(config.pat_token), payload={"body": body})


def post_marker_pr_comment(
    paths: tuple[str, ...],
    *,
    config: GitCodeCommentConfig,
    marker: str,
    body: str,
    action_label: str,
) -> None:
    """Create or update a fixed-marker PR comment. Raises on HTTP/API failure."""
    if not paths:
        return
    comments = _list_pr_comments(config)
    note_id = _find_marker_comment_id(comments, marker)
    if note_id is None:
        _create_pr_comment(config, body)
        logger.info("GitCode PR comment created for %s (%d file(s))", action_label, len(paths))
        return
    _update_pr_comment(config, note_id, body)
    logger.info("GitCode PR comment updated for %s (%d file(s))", action_label, len(paths))


def post_unscoped_python_comment(paths: tuple[str, ...], *, config: GitCodeCommentConfig) -> None:
    """Create or update the unscoped-Python PR comment. Raises on HTTP/API failure."""
    post_marker_pr_comment(
        paths,
        config=config,
        marker=_UNSCOPED_MARKER,
        body=build_unscoped_python_comment_body(paths),
        action_label="unscoped Python",
    )


def post_all_exempt_tests_comment(paths: tuple[str, ...], *, config: GitCodeCommentConfig) -> None:
    """Create or update the all-exempt-tests PR comment. Raises on HTTP/API failure."""
    post_marker_pr_comment(
        paths,
        config=config,
        marker=_ALL_EXEMPT_MARKER,
        body=build_all_exempt_tests_comment_body(paths),
        action_label="all-exempt test files",
    )


def post_exemption_drift_comment(errors: tuple[GateError, ...], *, config: GitCodeCommentConfig) -> None:
    """Create or update the exemption-drift PR comment. Raises on HTTP/API failure."""
    paths = tuple(sorted({err.path for err in errors}))
    post_marker_pr_comment(
        paths,
        config=config,
        marker=_EXEMPTION_DRIFT_MARKER,
        body=build_exemption_drift_comment_body(errors),
        action_label="exemption drift",
    )


def post_shadowed_defs_comment(warnings: tuple[ShadowWarning, ...], *, config: GitCodeCommentConfig) -> None:
    """Create or update the shadowed-defs PR comment. Raises on HTTP/API failure."""
    paths = tuple(sorted({warning.file for warning in warnings}))
    post_marker_pr_comment(
        paths,
        config=config,
        marker=_SHADOWED_DEFS_MARKER,
        body=build_shadowed_defs_comment_body(warnings),
        action_label="shadowed definitions",
    )


def _log_gitcode_comment_not_posted(env_hint: str) -> None:
    """GitCode credentials absent; policy warnings are logged by the caller."""
    logger.debug("GitCode PR comment not posted (%s)", env_hint)


def _try_post_comment(post: Callable[[], None]) -> None:
    try:
        post()
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.warning("GitCode PR comment failed (non-blocking): %s", exc)
    except ValueError:
        logger.exception("GitCode PR comment logic error (non-blocking)")


def _maybe_post_marker_comment(
    paths: tuple[str, ...],
    *,
    cfg: Config | None,
    env_hint: str,
    post_fn: _PathMarkerPostFn,
) -> None:
    if not paths:
        return
    config = load_gitcode_comment_config(cfg)
    if config is None:
        _log_gitcode_comment_not_posted(env_hint)
        return
    _try_post_comment(lambda: post_fn(paths, config=config))


def maybe_post_unscoped_python_comment(paths: tuple[str, ...], *, cfg: Config | None = None) -> None:
    """Best-effort PR comment for unscoped Python changes; never raises."""
    _maybe_post_marker_comment(
        paths,
        cfg=cfg,
        env_hint=_GITCODE_ENV_HINT,
        post_fn=post_unscoped_python_comment,
    )


def maybe_post_all_exempt_tests_comment(paths: tuple[str, ...], *, cfg: Config | None = None) -> None:
    """Best-effort PR comment for all-exempt changed test files; never raises."""
    _maybe_post_marker_comment(
        paths,
        cfg=cfg,
        env_hint=_GITCODE_ENV_HINT,
        post_fn=post_all_exempt_tests_comment,
    )


def maybe_post_exemption_drift_comment(errors: tuple[GateError, ...], *, cfg: Config | None = None) -> None:
    """Best-effort PR comment for stale exemptions; never raises."""
    if not errors:
        return
    config = load_gitcode_comment_config(cfg)
    if config is None:
        _log_gitcode_comment_not_posted(_GITCODE_ENV_HINT)
        return
    _try_post_comment(lambda: post_exemption_drift_comment(errors, config=config))


def maybe_post_shadowed_defs_comment(warnings: tuple[ShadowWarning, ...], *, cfg: Config | None = None) -> None:
    """Best-effort PR comment for shadowed duplicate defs; never raises."""
    if not warnings:
        return
    config = load_gitcode_comment_config(cfg)
    if config is None:
        _log_gitcode_comment_not_posted(_GITCODE_ENV_HINT)
        return
    _try_post_comment(lambda: post_shadowed_defs_comment(warnings, config=config))
