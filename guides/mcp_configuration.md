# MCP Configuration

Once you've followed the steps in the [deployment guides](./deployment_v1.md) or [deployment guides](./deployment_v2.md),
you can configure your MCP client to use the Union MCP server.

## Use with Cursor IDE

Simply update the `~/.cursor/mcp.json` or `<project>/.cursor/mcp.json` file with
either a local or remote MCP server

<details>
<summary>Local MCP server</summary>

```json
{
  "mcpServers": {
    "Union MCP": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--with-editable",
        "/Users/username/union-mcp[v1]",
        "mcp",
        "run",
        // Use "/Users/username/union-mcp/examples/v2/server.py" for v2
        "/Users/username/union-mcp/examples/server.py"
      ],
      "env": {
        "DISABLE_AUTH": "1",
        "FLYTE_API_KEY": "<FLYTE_API_KEY>"
      }
    }
  }
}
```
</details>

<details>
<summary>Remote MCP server</summary>

Replace the `url` with the URL of the deployed app and `<your-token>` with the authentication token.

```json

{
  "mcpServers": {
    "Union MCP": {
      // Use "https://mcp-v2.apps.union-internal.hosted.unionai.cloud/sse" for v2
      "url": "https://mcp.apps.union-internal.hosted.unionai.cloud/sse",
      "headers": {
        "Authorization": "Bearer <UNION_MCP_AUTH_TOKEN>"
      }
    }
  }
}
```
</details>


## Use with Claude Desktop

First, install [Claude Desktop](https://claude.ai/download).

Then, install the server

```
mcp install -e . examples/server.py
```

This will configure the `claude_desktop_config.json` configuration file located in:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

<details>
<summary>Local configuration</summary>

```json
{
  "mcpServers": {
    "Union MCP": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--with-editable",
        "/Users/username/union-mcp[v1]",
        "mcp",
        "run",
        // Use "/Users/username/union-mcp/examples/v2/server.py" for v2
        "/Users/username/union-mcp/examples/server.py"
      ],
      "env": {
        "DISABLE_AUTH": "1",
        "FLYTE_API_KEY": "<FLYTE_API_KEY>"
      }
    }
  }
}
```

</details>

<details>
<summary>Remote configuration</summary>

Replace the `url` with the URL of the deployed app and `<your-token>` with the authentication token.

```json
{
  "mcpServers": {
    "Union MCP": {
      // Use "https://mcp-v2.apps.union-internal.hosted.unionai.cloud/sse" for v2
      "url": "https://mcp.apps.union-internal.hosted.unionai.cloud/sse",
      "headers": {
        "Authorization": "Bearer <UNION_MCP_AUTH_TOKEN>"
      }
    }
  }
}
```
</details>

> [!NOTE]
> Make sure the `uv` executable is available in `/usr/local/bin`, otherwise
> replace `command` with the full path, e.g. `/Users/username/.local/bin/uv`
