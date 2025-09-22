import union
from union.app import App


image = union.ImageSpec(
    name="union-mcp-server",
    apt_packages=["git"],
    packages=["uv", "union", "union-runtime>=0.1.17", "mcp[cli]"],
    builder="union",
)


app = App(
    name="union-mcp-test-0",
    subdomain="mcp-testing-test",
    port=8000,
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
