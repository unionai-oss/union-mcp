# Union MCP for v2

This file contains instructions for deploying the Union MCP v2 server.

## Setup

Clone the repo:

```bash
git clone https://github.com/unionai-oss/union-mcp
```

Install `uv` and the `union-mcp` package:

```bash
pip install uv
uv venv .venv-v2 --python 3.12
uv pip install '.[v2]'
```

Activate the virtual environment:

```bash
source .venv-v2/bin/activate
```

## Deploy app to Union

Create a config file with `union create login --host <host>`, then export
environment variables:

```bash
export UNION_CONFIG=...  # path to union config file
export UNION_ORG=union-internal  # use custom org name
export UNION_PROJECT=union-mcp  # use custom project name
export UNION_DOMAIN=development  # use custom domain name
export APP_NAME=union-mcp-v2  # use custom app name
export APP_SUBDOMAIN=mcp-v2-test  # use custom app subdomain
export APP_PORT=8000  # use custom app port
```

Register dummy workflows for testing (optional):

```bash
flyte deploy --project $UNION_PROJECT --domain $UNION_DOMAIN examples/v2/workflows.py env
```

Create a secret for the authentication token:

```bash
union create secret --project $UNION_PROJECT UNION_MCP_AUTH_TOKEN --domain $UNION_DOMAIN --value <your-token>
```

Then deploy the app:

```bash
uv run union deploy apps --project $UNION_PROJECT --domain $UNION_DOMAIN app_v2.py $APP_NAME
```