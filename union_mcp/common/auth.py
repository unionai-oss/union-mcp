import inspect
import logging
import os
import typing
from functools import wraps

from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


def extract_authorization_header(ctx: Context) -> tuple[str, str] | None:
    """Extract the Authorization header from the request context."""
    auth_header = ctx.request_context.request.headers.get("authorization")
    if auth_header:
        return "authorization", auth_header
    return None


def extract_cookie_header(ctx: Context) -> tuple[str, str] | None:
    """Extract the Cookie header from the request context."""
    cookie_header = ctx.request_context.request.headers.get("cookie")
    if cookie_header:
        return "cookie", cookie_header
    return None


def get_auth_tuples(ctx: Context) -> list[tuple[str, str]]:
    """Extract auth tuples from the request context."""
    extractors = [extract_authorization_header, extract_cookie_header]
    auth_tuples = []
    for extractor in extractors:
        try:
            result = extractor(ctx)
            if result is not None:
                auth_tuples.append(result)
        except Exception as e:
            logger.warning(f"Header extractor {extractor.__name__} failed: {e}")
    return auth_tuples


def require_auth(func: typing.Callable):
    """Decorator to require authentication for FastMCP handlers.

    Extracts auth headers from the request context and sets them in the
    Flyte context using the auth_metadata() context manager.
    """
    from flyte.remote import auth_metadata

    if os.environ.get("DISABLE_AUTH", "0") == "1":
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract context from kwargs (FastMCP passes context)
            ctx: Context = kwargs.get("ctx")
            auth_tuples = get_auth_tuples(ctx)

            if not auth_tuples:
                raise ValueError("Authentication required: Invalid or missing token")

            with auth_metadata(*auth_tuples):
                return await func(*args, **kwargs)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract context from kwargs (FastMCP passes context)
        ctx: Context = kwargs.get("ctx")
        auth_tuples = get_auth_tuples(ctx)

        if not auth_tuples:
            raise ValueError("Authentication required: Invalid or missing token")

        with auth_metadata(*auth_tuples):
            return func(*args, **kwargs)

    return sync_wrapper
