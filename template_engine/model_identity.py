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

"""Template identity: is this template allowed to serve this model?

A template records the model it was induced from in ``meta.model_ids``
(build time, taken from the captures themselves) and both discovery
(:func:`~template_engine.registry.template_path_for`) and serving
(:func:`~template_engine.template_graph.generate`) must answer the same
question.  The rule therefore lives here, once, and both call sites use it.

Two spellings of the same model are equivalent:

* the same text, case-insensitively - the repository id a template was
  built from (``zai-org/GLM-5.3``);
* the model-name slug - the last path segment folded to ``[0-9a-z_]``.

The slug is a *narrow, one-directional* equivalence: the template identity
must be a repository id (or a legacy bare slug) and the request must be an
existing local checkout (``/models/GLM-5.3``).  It does not hold the other
way round.  A template captured from a local directory records only that
directory, which cannot name the repository it came from, so it is served
to that same path and to nothing else - not to ``someone-else/GLM-5.3``
and not to another checkout that happens to share the tail, even when the
recorded directory is absent on this host.  Without a
concrete request checkout the identities must agree as text, which also
keeps two repository ids with the same tail distinct: a template built for
``zai-org/GLM-5.3`` is not served to ``someone-else/GLM-5.3``.

Serving a local path used to fail twice: discovery found the file by slug
and then rejected it on the exact-string identity check, and relaxing only
discovery moved the same rejection into ``generate`` as
``TemplateGraphRefusal('model identity mismatch')``.  Issue #667.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    'identity_matches',
    'is_local_model_path',
    'legacy_template_slug',
    'looks_like_model_path',
    'slugify_model_id',
]

# Captures that never recorded a model id are stored under this sentinel; it
# is 'no identity', not a name to match against a directory called 'unknown'.
_UNKNOWN_IDENTITY = 'unknown'


def slugify_model_id(model_id: Any) -> str:
    """``deepseek-ai/DeepSeek-V4-Pro`` -> ``deepseek_v4_pro``."""

    tail = str(model_id).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", tail).lower().strip("_")


def legacy_template_slug(model_id: Any) -> str:
    """Filename slug used by artifacts registered before :func:`slugify_model_id`.

    Those names folded only ``/``, so a Windows capture such as
    ``C:\\models\\GLM-5.3`` was stored as
    ``c__models_glm_5_3_cmp_on.json.xz`` instead of ``glm_5_3_cmp_on.json.xz``.
    The spelling exists for discovery of those files only; identity matching
    always uses :func:`slugify_model_id`.
    """

    return re.sub(r"[^0-9a-zA-Z]+", "_", str(model_id).rstrip("/").split("/")[-1]).lower().strip("_")


def looks_like_model_path(model_id: Any) -> bool:
    """Whether ``model_id`` is *spelled* as a path, without touching the disk.

    A checkout has to be spelled as one: an absolute path, ``~``/``~user``, an
    explicit ``./`` / ``../``, or a Windows drive path.  A bare ``org/name`` is
    always a repository id - it is also a relative path, so accepting it would
    let the process working directory turn ``zai-org/GLM-5.3`` into a checkout
    and invert both identity rules (the real local checkout would stop
    matching, and another owner's id would start matching).

    This is the *stored* side of a comparison: an artifact captured from a
    local directory records that directory, and it stays a checkout identity
    on a machine where the directory is gone (moved, renamed, artifact copied
    elsewhere).  Requiring the disk there would demote the recorded path to
    "a repository id" and let it serve any same-tailed checkout - exactly the
    cross-source confusion the one-directional rule exists to prevent.  Use
    :func:`is_local_model_path` for a *request*, which does need a checkout.
    """

    if not isinstance(model_id, str):
        return False
    text = model_id.strip()
    if not text:
        return False
    return text.startswith(("/", "~", "./", "../")) or (len(text) >= 3 and text[1] == ":" and text[0].isalpha())


def is_local_model_path(model_id: Any) -> bool:
    """Whether ``model_id`` names an existing local directory.

    A request qualifies only when it is both spelled as a path (see
    :func:`looks_like_model_path`) and present on disk: a typo or a
    not-yet-downloaded path stays a repository id and gets the strict textual
    comparison, so a mistyped ``--model-id`` cannot fall back onto a
    same-tailed template.
    """

    if not looks_like_model_path(model_id):
        return False
    try:
        return Path(model_id.strip()).expanduser().is_dir()
    except (OSError, RuntimeError, ValueError):
        # Embedded NUL, unresolvable ``~user``, unreadable parent - all mean
        # "not a local checkout we can identify by directory name".
        return False


def _without_trailing_separators(text: str) -> str:
    """``/models/GLM-5.3/`` and ``/models/GLM-5.3`` name one checkout."""

    trimmed = text.rstrip("/\\")
    if not trimmed or (len(trimmed) == 2 and trimmed[1] == ":"):
        # Root (``/``) and Windows drive roots (``C:\\``) keep their separator.
        return text
    return trimmed


def _matches_one(stored: str, requested: str) -> bool:
    """One stored identity against one request (see the module docstring)."""

    stored = _without_trailing_separators(stored)
    requested = _without_trailing_separators(requested)
    if stored.casefold() == _UNKNOWN_IDENTITY:
        # 'unknown' means "no recorded identity": it must not match a request
        # that happens to spell the sentinel.
        return False
    if stored.casefold() == requested.casefold():
        return True
    if looks_like_model_path(stored) or not is_local_model_path(requested):
        # A checkout-derived identity cannot name its repository (any
        # same-tailed name could belong to another owner), and without a
        # concrete request checkout there is nothing to compare but text.
        # The stored side is judged by spelling only: the recorded directory
        # may legitimately be absent here (artifact copied from another host),
        # and that must not turn it back into a repository id.
        return False
    stored_slug = slugify_model_id(stored)
    # A tail with no ASCII alphanumerics folds to '', and two empty slugs are
    # not the same model.
    return bool(stored_slug) and stored_slug == slugify_model_id(requested)


def identity_matches(identities: Any, requested: Any) -> bool:
    """Whether a template declaring ``identities`` may serve ``requested``.

    ``identities`` is the template's ``meta.model_ids`` list.  A mixed corpus
    (more than one distinct id) never matches: it was captured from several
    models and carries no single architecture to serve.  Malformed metadata
    matches nothing rather than raising - discovery skips the candidate and
    serving refuses it.
    """

    if not isinstance(identities, (list, tuple)) or not identities:
        return False
    if not isinstance(requested, str) or not requested.strip():
        return False
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        return False
    if len({value.casefold() for value in identities}) != 1:
        return False
    # The gate above proves the stored identities are the same text up to
    # case.  `any` keeps the answer independent of list order: two spellings
    # that differ only by case can still differ in local-path existence.
    stripped = [value.strip() for value in identities]
    return any(_matches_one(stored, requested.strip()) for stored in stripped)
