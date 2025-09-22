import os
import union
from union.app import App


APP_NAME = os.getenv("APP_NAME", "union-mcp")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "mcp-testing-test")
APP_PORT = os.getenv("APP_PORT", 8000)


image = union.ImageSpec(
    name="union-mcp-server",
    apt_packages=["git"],
    packages=["uv", "union", "union-runtime>=0.1.17", "mcp[cli]"],
    builder="union",
)


app = App(
    name=APP_NAME,
    subdomain=APP_SUBDOMAIN,
    port=APP_PORT,
    include=["examples/server.py", "union_mcp"],
    container_image=image,
    args="mcp run examples/server.py --transport sse",
    requests=union.Resources(cpu=2, mem="1Gi"),
    secrets=[
        union.Secret(
            key="EAGER_API_KEY",
            env_var="UNION_API_KEY",
        )
    ],
    requires_auth=False,
)
