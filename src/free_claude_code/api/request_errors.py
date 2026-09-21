"""Shared API request validation and safe error logging."""

from collections.abc import Sequence
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    anthropic_error_payload,
    anthropic_error_type_for_failure,
)
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    extract_upstream_error_detail,
    redact_and_truncate,
    redacted_exception_traceback,
    safe_exception_message,
)
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
)

from .request_ids import get_request_body_snapshot, get_request_id

WireApi = Literal["messages", "responses"]


def require_non_empty_messages(messages: Sequence[object]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def ordinary_application_error_response(
    error: ApplicationError,
    *,
    wire_api: WireApi,
    request_id: str,
) -> JSONResponse:
    """Serialize a deterministic application error without terminal headers."""
    if wire_api == "responses":
        return JSONResponse(
            status_code=error.status_code,
            content=openai_error_payload(
                message=error.message,
                error_type=openai_error_type_for_failure(error.kind),
            ),
        )
    return JSONResponse(
        status_code=error.status_code,
        content=anthropic_error_payload(
            error_type=anthropic_error_type_for_failure(error.kind),
            message=error.message,
            request_id=request_id,
        ),
    )


def http_status_for_unexpected_api_exception(_exc: BaseException) -> int:
    return 500


def log_unexpected_api_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log API failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error(
                "{} request_id={}: {}",
                context,
                request_id,
                safe_exception_message(exc),
            )
        else:
            logger.error("{}: {}", context, safe_exception_message(exc))
        logger.error(redacted_exception_traceback(exc))
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def unexpected_http_exception(
    settings: Settings, exc: Exception, *, context: str
) -> HTTPException:
    log_unexpected_api_exception(settings, exc, context=context)
    return HTTPException(
        status_code=http_status_for_unexpected_api_exception(exc),
        detail=safe_exception_message(exc),
    )


async def read_redacted_request_body(request: Request) -> tuple[str | None, bool]:
    """Return the redacted request body for error logs, plus a truncated flag.

    Prefers the body snapshot observed by the correlation middleware: handlers
    for ``Exception`` run at the outermost server-error boundary with a
    receive-less request, so ``request.body()`` always fails there. Falls back
    to a best-effort direct read for inner handlers whose channel is alive.
    """
    snapshot = get_request_body_snapshot(request)
    if snapshot is not None:
        raw, truncated, complete = snapshot
    else:
        try:
            raw = await request.body()
        except Exception:
            return None, False
        if not raw:
            return None, False
        truncated, complete = False, True
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return None, False
    redacted, capped = redact_and_truncate(text)
    cut_short = capped or truncated or not complete
    if cut_short and "truncated after" not in redacted:
        redacted = (
            f"{redacted}\n... [truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes]"
        )
    return redacted, cut_short


async def log_api_error_forensics(
    request: Request,
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    status_code: int,
) -> None:
    """Log one always-on error line plus a verbose bundle when opted in.

    The always-on line carries only status, method, path, request id, and
    exception type — never exception or body text. The verbose bundle
    (exception message, redacted stack trace, redacted full request body,
    upstream provider detail, query params) is gated on
    ``LOG_API_ERROR_TRACEBACKS`` so error forensics stay out of the default
    log stream.
    """
    request_id = get_request_id(request)
    logger.error(
        "{} status={} method={} path={} request_id={} exc_type={}",
        context,
        status_code,
        request.method,
        request.url.path,
        request_id,
        type(exc).__name__,
    )
    if not settings.log_api_error_tracebacks:
        return
    logger.error(
        "{} detail request_id={}: {}",
        context,
        request_id,
        safe_exception_message(exc),
    )
    logger.error(
        "{} traceback request_id={}:\n{}",
        context,
        request_id,
        redacted_exception_traceback(exc),
    )
    body, truncated = await read_redacted_request_body(request)
    if body is not None:
        logger.error(
            "{} request_body request_id={} truncated={}:\n{}",
            context,
            request_id,
            truncated,
            body,
        )
    if isinstance(exc, Exception):
        detail = extract_upstream_error_detail(exc)
        if (
            detail.status_code is not None
            or detail.body_text is not None
            or detail.cause_chain_text is not None
        ):
            logger.error(
                "{} upstream request_id={} upstream_status={} category={} "
                "body_truncated={} cause={} body={}",
                context,
                request_id,
                detail.status_code,
                detail.category_hint,
                detail.body_truncated,
                detail.cause_chain_text,
                detail.body_text,
            )
    query = dict(request.query_params)
    if query:
        logger.error(
            "{} query request_id={} query={}",
            context,
            request_id,
            query,
        )
