# Union MCP

An MCP server to use Union tasks, workflows, and apps as tools.

## Setup

Clone the repo:

```bash
git clone https://github.com/unionai-oss/union-mcp
```

Install `uv` and the `union-mcp` package:

```bash
pip install uv
uv sync
```

## Running a local server

```bash
union register --project mcp-testing examples/workflows.py
```

```bash
mcp run examples/server.py --transport sse
```

```bash
npx @modelcontextprotocol/inspector
```

## Deploying to Union

```bash
union deploy apps --project mcp-testing app.py union-mcp-test-0
```


## Use with Claude Desktop

First, install [Claude Desktop](https://claude.ai/download).

Then, install the server

```
mcp install -e . examples/server.py
```

This will configure the `claude_desktop_config.json` configuration file located in:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

You should see something like

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
        "/Users/username/union-mcp",
        "mcp",
        "run",
        "/Users/username/union-mcp/examples/server.py"
      ]
    }
  }
}
```

> [!NOTE]
> Make sure the `uv` executable is available in `/usr/local/bin`, otherwise
> replace `command` with the full path, e.g. `/Users/username/.local/bin/uv`
