"""Error model + status mapping + FastAPI exception handlers.

The error body contract (``contracts/rest-api.md``):
    { "detail": "...", "fieldErrors"?: { "<fieldId>": "<message>" } }

``fieldErrors`` is returned on submit validation failure (422) so the frontend
can map messages onto ``el-form-item``s. Handlers are registered on the app in
``main.create_app``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    """Base app error carrying an HTTP status + optional ``fieldErrors``."""

    def __init__(
        self,
        detail: str,
        *,
        status_code: int = 500,
        field_errors: dict[str, str] | None = None,
    ):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.field_errors = field_errors


class ValidationError(AppError):
    """Submit-time validation failure (422 + fieldErrors)."""

    def __init__(self, detail: str, field_errors: dict[str, str] | None = None):
        super().__init__(detail, status_code=422, field_errors=field_errors)


class ConflictError(AppError):
    """409 — e.g. overriding a builtin device profile / schema hash mismatch."""

    def __init__(self, detail: str):
        super().__init__(detail, status_code=409)


class NotFoundError(AppError):
    """404 — referenced module/job/result does not exist."""

    def __init__(self, detail: str):
        super().__init__(detail, status_code=404)


class SchemaMismatchError(AppError):
    """409 — bundled config hash differs from the stored snapshot for a version."""

    def __init__(self, detail: str):
        super().__init__(detail, status_code=409)


def _body(detail: str, field_errors: dict[str, str] | None = None) -> dict:
    """Build the standard error body; includes ``fieldErrors`` only when present."""
    body: dict[str, object] = {"detail": detail}
    if field_errors:
        body["fieldErrors"] = field_errors
    return body


def register_error_handlers(app: FastAPI) -> None:
    """Wire all error -> HTTP mappings onto ``app``."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        """Map any ``AppError`` to its carried status + body (incl. fieldErrors)."""
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.detail, exc.field_errors),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        """Map FastAPI body-validation errors to 422 with ``fieldErrors`` keyed by
        the pydantic error loc (best-effort; the domain validator is the primary
        source of user-facing field messages).
        """
        # FastAPI's body-validation errors -> 422 with fieldErrors keyed by the
        # pydantic error loc (best-effort; the domain validator is the primary
        # source of user-facing field messages).
        field_errors: dict[str, str] = {}
        for err in exc.errors():
            loc = err.get("loc") or ()
            key = ".".join(str(part) for part in loc if part not in ("body",))
            field_errors.setdefault(key or "body", err.get("msg", "invalid"))
        return JSONResponse(
            status_code=422,
            content=_body("Validation failed", field_errors or None),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        """Map Starlette ``HTTPException``s (e.g. router-raised 404/500) to the
        standard error body so the contract is uniform.
        """
        detail = exc.detail if isinstance(exc.detail, str) else "error"
        return JSONResponse(status_code=exc.status_code, content=_body(detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        """500 fallback for any unhandled exception.

        The full traceback is ALWAYS logged server-side (so 500s are
        debuggable from logs). The client body never leaks internals by
        default — only the exception type is exposed. Set ``MSMODELING_DEBUG=1``
        (for internal/shared deployments) to also include the exception
        message in the response body.
        """
        import logging
        import os

        logging.getLogger("msmodeling.web").exception("Unhandled request exception")
        debug = os.environ.get("MSMODELING_DEBUG", "").lower() in ("1", "true", "yes")
        detail = (
            f"Internal server error: {type(exc).__name__}: {exc}"
            if debug
            else f"Internal server error: {type(exc).__name__}"
        )
        return JSONResponse(status_code=500, content=_body(detail))
