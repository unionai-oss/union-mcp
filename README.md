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
uv sync --extra v1
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running a local server

```bash
mcp run examples/server.py --transport sse
```

## Deploying to Union

Create a config file with `union create login --host <host>`, then export
environment variables:

```bash
export UNION_CONFIG=...  # path to union config file
export UNION_PROJECT=union-mcp  # use custom project name
export UNION_DOMAIN=development  # use custom domain name
export APP_NAME=union-mcp  # use custom app name
export APP_SUBDOMAIN=mcp  # use custom app subdomain
export APP_PORT=8000  # use custom app port
```

Register dummy workflows for testing (optional):

```bash
union register --project $UNION_PROJECT examples/workflows.py
```

Create a secret for the authentication token:

```bash
union create secret --project $UNION_PROJECT UNION_MCP_AUTH_TOKEN --domain $UNION_DOMAIN --value <your-token>
```

Then deploy the app:

```bash
union deploy apps --project $UNION_PROJECT app_v1.py $APP_NAME
```

This command will output the URL of the deployed app that look something like:

```bash
Image union-mcp-server:2Z9KMKXwJJQaikQXhtgELw found. Skip building.
✨ Deploying Application: union-mcp
🔎 Console URL:
https://union-internal.hosted.unionai.cloud/console/projects/union-mcp/domains/development/apps/union-mcp
[Status] Down: Scaling down from 1 to 0 replicas
[Status] Deploying: Deploying revision: [union-mcp-00006]
[Status] Deploying:
[Status] Pending: TrafficNotMigrated: Traffic is not yet migrated to the latest revision.
[Status] Pending: IngressNotConfigured: Ingress has not yet been reconciled.
[Status] Pending: Uninitialized: Waiting for load balancer to be ready
[Status] Active: Service is ready

🚀 Deployed Endpoint: https://mcp.apps.union-internal.hosted.unionai.cloud
```

## Rotating the authentication token

For security reasons, you may want to rotate the authentication token. To do so, override the existing secret:

```bash
union create secret --project $UNION_PROJECT --domain $UNION_DOMAIN UNION_MCP_AUTH_TOKEN --value <your-new-token>
```

## Use with Cursor IDE

Simply update the `.cursor/mcp.json` file with either a local or remote MCP server

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
        "/Users/username/union-mcp",
        "mcp",
        "run",
        "/Users/username/union-mcp/examples/server.py"
      ]
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
      "url": "https://mcp.apps.union-internal.hosted.unionai.cloud/sse",
      "headers": {
        "Authorization": "Bearer <your-token>"
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
        "/Users/username/union-mcp",
        "mcp",
        "run",
        "/Users/username/union-mcp/examples/server.py"
      ]
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
      "url": "https://mcp.apps.union-internal.hosted.unionai.cloud/sse",
      "headers": {
        "Authorization": "Bearer <your-token>"
      }
    }
  }
}
```
</details>

> [!NOTE]
> Make sure the `uv` executable is available in `/usr/local/bin`, otherwise
> replace `command` with the full path, e.g. `/Users/username/.local/bin/uv`
