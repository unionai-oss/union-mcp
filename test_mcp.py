"""Run from the repository root:
uv run examples/snippets/clients/streamable_basic.py
"""

import asyncio
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # Connect to a streamable HTTP server
    auth_key = os.getenv("FLYTE_API_KEY")
    mcp_url = os.getenv("MCP_URL", "https://mcp-v2.apps.demo.hosted.unionai.cloud/mcp")
    async with streamablehttp_client(
        mcp_url,
        headers={
            "Authorization": f"Bearer {auth_key}",
        }
    ) as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")


if __name__ == "__main__":
    asyncio.run(main())
