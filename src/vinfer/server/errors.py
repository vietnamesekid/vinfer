"""Every error leaves the server in OpenAI's shape, with its HTTP status:
``{"error": {"message", "type", "param", "code"}}`` (vLLM's ``ErrorResponse``).

Malformed and unsupported requests are 400 ``BadRequestError``; an unknown model is 404; any
other exception is 500 ``InternalServerError``, with a generic message (details go to the log).
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from vinfer.server.protocol import ErrorInfo, ErrorResponse

log = logging.getLogger("vinfer.server")

_TYPES = {400: "BadRequestError", 404: "NotFoundError", 500: "InternalServerError"}


class APIError(Exception):
    def __init__(self, message: str, status: int = 400, param: str | None = None) -> None:
        super().__init__(message)
        self.message, self.status, self.param = message, status, param


def error_body(message: str, status: int, param: str | None = None) -> dict[str, Any]:
    kind = _TYPES.get(status) or HTTPStatus(status).phrase.replace(" ", "")
    info = ErrorInfo(message=message, type=kind, param=param, code=status)
    return ErrorResponse(error=info).model_dump()


def error_response(message: str, status: int, param: str | None = None) -> JSONResponse:
    return JSONResponse(error_body(message, status, param), status_code=status)


def _loc(loc: tuple[Any, ...]) -> str:
    parts = [str(p) for p in loc[1:] if not str(p).startswith(("function-", "tagged-union"))]
    return ".".join(parts) if loc and loc[0] == "body" else ".".join(map(str, loc))


def _validation_message(exc: RequestValidationError) -> tuple[str, str | None]:
    """A message built from the errors only (``str(exc)`` would show server internals)."""
    errors = exc.errors()
    if not errors:
        return "invalid request", None
    lines = []
    for e in errors[:5]:
        where = _loc(tuple(e.get("loc", ())))
        msg = str(e.get("msg", "invalid value")).removeprefix("Value error, ")
        lines.append(f"{where}: {msg}" if where else msg)
    more = f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""
    return "; ".join(lines) + more, _loc(tuple(errors[0].get("loc", ()))) or None


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        message, param = _validation_message(exc)
        return error_response(message, 400, param)

    @app.exception_handler(APIError)
    async def _api(_: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.message, exc.status, exc.param)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return error_response("internal server error", 500)
