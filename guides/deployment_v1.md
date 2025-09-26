# Union v1 MCP deployment

This file contains instructions for deploying the Union v1 MCP server.

## Setup

Clone the repo:

```bash
git clone https://github.com/unionai-oss/union-mcp
```

Install `uv`:

```bash
pip install uv
```

## Running a local server

```bash
DISABLE_AUTH=1 uv run --with '.[v1]' mcp run examples/server.py --transport sse
```

## Deploying to Union

Create a config file:

```bash
uv run --with union union create login --host <host>
```

Then export environment variables:

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
uv run --with union union register --project $UNION_PROJECT examples/workflows.py
```

Create a secret for the authentication token:

```bash
uv run --with union union create secret \
    --project $UNION_PROJECT \
    --domain $UNION_DOMAIN \
    UNION_MCP_AUTH_TOKEN \
    --value <your-token>
```

Then deploy the app:

```bash
uv run --with union union deploy apps --project $UNION_PROJECT app_v1.py $APP_NAME
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
uv run --with union union create secret --project $UNION_PROJECT --domain $UNION_DOMAIN UNION_MCP_AUTH_TOKEN --value <your-new-token>
```
