"""Cases router — per-case CLI log lookup.

GET /api/cases/{case_hash}/log — return a single case's captured CLI output.

Replaces the frontend regex-splitting of ``{job_id}.log`` by ``===== Case i/N
=====``, which broke when separators appeared in case bodies. Each case's output
is captured at run time and stored in ``case_logs`` (keyed by case_hash), so a
case reuse (dedup) fetches its log directly — no re-extraction from the source
job's log.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from db import get_session
from services.capture import read_case_log_file
from services.repositories import CaseLogRepository

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("/{case_hash}/log", response_class=PlainTextResponse)
async def get_case_log(
    case_hash: str,
    session: Annotated[object, Depends(get_session)],  # noqa: ARG001 (session ensures DB is up)
    tail: Annotated[int, Query] = 0,
) -> PlainTextResponse:
    """Return one case's CLI log (DB primary, file fallback).

    ``tail=0`` (default) returns the full log; ``tail=N`` returns the last N
    lines. 404 if no log is stored for ``case_hash``.
    """
    # case_hash is a sha256 hex digest (compute_params_hash); reject anything
    # else to block path traversal in the file fallback (read_case_log_file
    # also defends in depth via a resolve()/is_relative_to() containment check).
    if not re.fullmatch(r"[0-9a-f]{64}", case_hash):
        raise HTTPException(status_code=400, detail="invalid case_hash")
    content = CaseLogRepository().get(case_hash)
    if not content:
        content = read_case_log_file(case_hash)
    if not content:
        raise HTTPException(status_code=404, detail=f"No case log for {case_hash}")
    if tail and tail > 0:
        lines = content.splitlines()
        content = "\n".join(lines[-tail:])
    return PlainTextResponse(content=content)
