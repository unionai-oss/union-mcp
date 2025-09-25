import os
import typing
from functools import wraps

from mcp.server.fastmcp import Context


def require_auth(func: typing.Callable):
    """Decorator to require authentication for FastMCP handlers"""

    valid_auth_token = os.environ["AUTH_TOKEN"]

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract context from kwargs (FastMCP passes context)
        ctx: Context = kwargs.get('ctx')
        auth_header = ctx.request_context.request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            auth_token = auth_header[7:]
        else:
            raise ValueError("Authentication required: Invalid or missing token")

        if auth_token != valid_auth_token:
            raise ValueError("Authentication required: Invalid or missing token")

        return func(*args, **kwargs)

    return wrapper


def require_auth_async(func: typing.Callable) -> typing.Callable:
    """Decorator to require authentication for FastMCP handlers"""

    valid_auth_token = os.environ["AUTH_TOKEN"]

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract context from kwargs (FastMCP passes context)
        ctx: Context = kwargs.get('ctx')
        auth_header = ctx.request_context.request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            auth_token = auth_header[7:]
        else:
            raise ValueError("Authentication required: Invalid or missing token")

        if auth_token != valid_auth_token:
            raise ValueError("Authentication required: Invalid or missing token")

        return await func(*args, **kwargs)

    return wrapper
