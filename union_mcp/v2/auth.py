"""Authentication utilities for Union MCP v2 server."""

from __future__ import annotations

from flyte._context import internal_ctx


def get_authorization_header() -> str:
    """
    Extract the Authorization header from the Flyte internal context.

    Returns:
        The authorization header if present.
    """
    api_key = ""
    for key, value in internal_ctx().data.metadata:
        if key == "authorization":
            api_key = value
            break

    return api_key
