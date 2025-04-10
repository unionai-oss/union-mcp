import union
from union.app import App


image = union.ImageSpec(
    name="union-mcp-server",
    apt_packages=["git"],
    packages=["uv", "union-runtime>=0.1.17", "mcp[cli]"],
    commands=[
        "git clone https://github.com/unionai-oss/union-mcp",
        "uv pip install ./union-mcp",
    ],
    builder="union",
)


app = App(
    name="union-mcp-test-0",
    port=8000,
    include=["server.py"],
    container_image=image,
    args="mcp run server.py --transport sse",
    requests=union.Resources(cpu=2, mem="1Gi"),
    requires_auth=False,
)
