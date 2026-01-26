# MCP Configuration

Once you've followed the steps in the [deployment guides](./deployment_v1.md) or [deployment guides](./deployment_v2.md),
you can configure your MCP client to use the Union MCP server.

## Create a Union api key

Create a Union api key:

```bash
uv run --with flyteplugins-union --with flyte==2.0.0b50 flyte create api-key --name <my-api-key>
```

You'll see something like this:

```
╭──────────────────────────────────────────── API Key Created Successfully ────────────────────────────────────────────╮
│ Client ID: <my-api-key>                                                                                              │
│                                                                                                                      │
│ ⚠️ The following API key will only be shown once. Be sure to keep it safe!                                           │
│                                                                                                                      │
│ Configure your headless CLI by setting the following environment variable:                                           │
│                                                                                                                      │
│ export                                                                                                               │
│ FLYTE_API_KEY="<secret-value>"                                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Make sure to save the `<secret-value>` in a secure location.

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
        "/Users/username/union-mcp/examples/v2/server.py"
      ],
      "env": {
        "DISABLE_AUTH": "1",
        "FLYTE_API_KEY": "<secret-value>"
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
      "url": "https://mcp-v2.apps.demo.hosted.unionai.cloud/sdk/mcp",
      "headers": {
        "Authorization": "Bearer <UNION_DEMO_API_KEY>"
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
        "/Users/username/union-mcp/examples/v2/server.py"
      ],
      "env": {
        "DISABLE_AUTH": "1",
        "FLYTE_API_KEY": "<secret-value>"
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
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp-v2.apps.demo.hosted.unionai.cloud/sdk/mcp",
        "--header",
        "Authorization: Bearer <secret-value>"
      ],
    }
  }
}
```
</details>

> [!NOTE]
> Make sure the `uv` executable is available in `/usr/local/bin`, otherwise
> replace `command` with the full path, e.g. `/Users/username/.local/bin/uv`
