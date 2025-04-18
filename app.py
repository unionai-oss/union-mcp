import union
from union.app import App


image = union.ImageSpec(
    name="union-mcp-server",
    apt_packages=["git"],
    packages=["uv", "union", "union-runtime>=0.1.17", "mcp[cli]"],
    builder="union",
)


app = App(
    name="union-mcp-test-1",
    port=8000,
    include=["examples/server.py", "union_mcp"],
    container_image=image,
    args="mcp run examples/server.py --transport sse",
    requests=union.Resources(cpu=2, mem="1Gi"),
    secrets=[
        union.Secret(
            key="niels-union-api-key",
            env_var="UNION_API_KEY",
        )
    ],
    requires_auth=False,
)
