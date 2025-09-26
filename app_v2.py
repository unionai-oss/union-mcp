# /// script
# requires-python = "==3.12"
# dependencies = ["union"]
# ///

import os
import union
from union.app import App


APP_NAME = os.getenv("APP_NAME", "union-mcp-v2")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "mcp-v2-test")
APP_PORT = int(os.getenv("APP_PORT", 8000))
UNION_ORG = os.getenv("UNION_ORG", "union-internal")


image = (
    union.ImageSpec(
        name="union-mcp-server",
        packages=[
            "uv",
            "union-runtime>=0.1.17",
            "mcp[cli]",
        ],
        builder="union",
    )
    # install flyte separately to avoid obstore version conflict with union-runtime
    .with_commands(["pip install flyte==2.0.0b22"])
)


app = App(
    name=APP_NAME,
    subdomain=APP_SUBDOMAIN,
    port=APP_PORT,
    include=["examples/v2/server.py", "union_mcp"],
    container_image=image,
    args="mcp run examples/v2/server.py --transport sse",
    requests=union.Resources(cpu=2, mem="1Gi"),
    secrets=[
        union.Secret(key="EAGER_API_KEY", env_var="FLYTE_API_KEY"),
        union.Secret(key="UNION_MCP_AUTH_TOKEN", env_var="UNION_MCP_AUTH_TOKEN"),
    ],
    env={"UNION_ORG": UNION_ORG},
    requires_auth=False,
)
