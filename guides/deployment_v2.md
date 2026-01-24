# Union v2 MCP deployment

This file contains instructions for deploying the Union v2 MCP server.

## Setup

Clone the repo:

```bash
git clone https://github.com/unionai-oss/union-mcp
```

Install `uv` and the `union-mcp` package:

```bash
pip install uv
```

## Running a local server

```bash
DISABLE_AUTH=1 uv run --with '.[v2]' mcp run examples/v2/server.py --transport sse
```

## Deploy app to Union

Create a config file:

```bash
uv run --with union union create login --host <host>
```

Then export environment variables:

```bash
export FLYTE_ORG=union-internal  # use custom org name
export FLYTE_PROJECT=union-mcp  # use custom project name
export FLYTE_DOMAIN=development  # use custom domain name
export APP_NAME=union-mcp-v2  # use custom app name
export APP_SUBDOMAIN=mcp-v2  # use custom app subdomain
export APP_PORT=8000  # use custom app port
```

Create a Flyte v2 config file:

```bash
uv run --with '.[v2]' flyte create config --endpoint $FLYTE_ORG --project $FLYTE_PROJECT --domain $FLYTE_DOMAIN
```

Register dummy workflows for testing (optional):

```bash
uv run --with '.[v2]' flyte deploy --project $FLYTE_PROJECT --domain $FLYTE_DOMAIN examples/v2/workflows.py env
```

Create a secret for the authentication token:

```bash
uv run --with union union create secret \
    --project $FLYTE_PROJECT \
    --domain $FLYTE_DOMAIN \
    UNION_MCP_AUTH_TOKEN \
    --value <your-token>
```

Then deploy the app:

```bash
uv run --with '.[v2]' flyte deploy --project $FLYTE_PROJECT --domain $FLYTE_DOMAIN app_v2.py app
```
