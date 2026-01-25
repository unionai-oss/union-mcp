# /// script
# requires-python = "==3.12"
# dependencies = ["flyte==2.0.0b49"]
# ///

import os
import logging
import pathlib

import flyte
from flyte.app import AppEnvironment, Domain, Scaling, Link


APP_NAME = os.getenv("APP_NAME", "union-mcp-v2")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "mcp-v2")
APP_PORT = int(os.getenv("APP_PORT", 8000))
FLYTE_ENDPOINT = os.getenv("FLYTE_ENDPOINT", "dns:///demo.hosted.unionai.cloud")
FLYTE_ORG = os.getenv("FLYTE_ORG", "demo")
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "union-mcp")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")


image = (
    flyte.Image.from_debian_base(name="union-mcp-server")
    .with_pip_packages("uv", "mcp[cli]==1.26.0", "starlette")
    # .with_pip_packages("flyte==2.0.0b49")
    .with_apt_packages("git")
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@c985e5fe")
    .with_commands(
        [
            "git clone https://github.com/flyteorg/flyte-sdk.git /root/flyte-sdk --branch main",
            "git clone https://github.com/unionai/unionai-examples.git /root/unionai-examples --branch main",
        ]
    )
)


app = AppEnvironment(
    name=APP_NAME,
    domain=Domain(subdomain=APP_SUBDOMAIN),
    port=APP_PORT,
    include=["examples/v2/server.py", "union_mcp"],
    image=image,
    # args="mcp run examples/v2/server.py --transport streamable-http",
    resources=flyte.Resources(cpu=3, memory="8Gi", disk="100Gi"),
    secrets=[
        flyte.Secret(key="EAGER_API_KEY", as_env_var="FLYTE_API_KEY"),
    ],
    env_vars={
        "FLYTE_ORG": FLYTE_ORG,
        "FLYTE_PROJECT": FLYTE_PROJECT,
        "FLYTE_DOMAIN": FLYTE_DOMAIN,
        "DISABLE_AUTH": "0",
        "LOG_LEVEL": "10",
    },
    requires_auth=False,
    scaling=Scaling(replicas=(1, 3)),
    links=[
        Link(
            path="/mcp",
            title="Streamable HTTP transport endpoint",
            is_relative=True,
        )
    ]
)

@app.server
def server():
    from union_mcp.v2.server import mcp

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    deployments = flyte.deploy(app)
    d = deployments[0]
    print(f"Deployed app: {d.table_repr()}")
