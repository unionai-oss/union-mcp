import inspect
import os
import typing
from functools import wraps

from mcp.server.fastmcp import Context


def authorize(ctx: Context):
    auth_header = ctx.request_context.request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        auth_token = auth_header[7:]
    else:
        raise ValueError("Authentication required: Invalid or missing token")

    if auth_token != os.environ["UNION_MCP_AUTH_TOKEN"]:
        raise ValueError("Authentication required: Invalid or missing token")


def require_auth(func: typing.Callable):
    """Decorator to require authentication for FastMCP handlers"""

    if os.environ.get("DISABLE_AUTH", "0") == "1":
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from kwargs (FastMCP passes context)
            ctx: Context = kwargs.get("ctx")
            authorize(ctx)
            return await func(*args, **kwargs)

        return wrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract context from kwargs (FastMCP passes context)
        ctx: Context = kwargs.get("ctx")
        authorize(ctx)
        return func(*args, **kwargs)

    return wrapper
