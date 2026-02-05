# /// script
# requires-python = "==3.12"
# dependencies = ["flyte==2.0.0b49"]
# ///

import os
import logging
import pathlib

import flyte
from flyte.app import AppEnvironment, Domain, Scaling, Link

from app_v2_tasks import env as tasks_env


APP_NAME = os.getenv("APP_NAME", "union-mcp-v2")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "mcp-v2")
APP_PORT = int(os.getenv("APP_PORT", 8000))
FLYTE_ENDPOINT = os.getenv("FLYTE_ENDPOINT", "dns:///demo.hosted.unionai.cloud")
FLYTE_ORG = os.getenv("FLYTE_ORG", "demo")
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "union-mcp")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")

# -----------------
# App Environment
# -----------------

image = (
    flyte.Image.from_debian_base(name="union-mcp-server")
    .with_pip_packages("uv", "mcp[cli]==1.26.0", "starlette")
    .with_apt_packages("git", "wget")
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@88cda0d")
    .with_commands(
        [
            "git clone https://github.com/flyteorg/flyte-sdk.git /root/flyte-sdk --branch main",
            "git clone https://github.com/unionai/unionai-examples.git /root/unionai-examples --branch main",
            "wget https://www.union.ai/docs/v2/flyte/_static/public/llms-full.txt -O ./full-docs.txt",
        ]
    )
)


app = AppEnvironment(
    name=APP_NAME,
    domain=Domain(subdomain=APP_SUBDOMAIN),
    port=APP_PORT,
    include=["examples/v2/server.py", "union_mcp"],
    image=image,
    resources=flyte.Resources(cpu=3, memory="8Gi", disk="8Gi"),
    env_vars={
        "APP_TASK_VERSION": "bc00e711ae512f4b858682846fd78d43",
        "DISABLE_AUTH": "0",
    },
    requires_auth=True,
    scaling=Scaling(replicas=(1, 2)),
    links=[
        Link(
            path="/sdk/mcp",
            title="Streamable HTTP transport endpoint",
            is_relative=True,
        ),
        Link(
            path="/health",
            title="Health check endpoint",
            is_relative=True,
        )
    ],
)


@app.server
async def server(): 
    import uvicorn
    from union_mcp.v2.server import app

    server = uvicorn.Server(uvicorn.Config(app, port=8000))
    await server.serve()


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    task_deployments = flyte.deploy(tasks_env)
    print(f"Deployed tasks: {task_deployments[0].table_repr()}")
    deployed_task = task_deployments[0].envs["union_mcp_tasks"].deployed_entities[0].deployed_task
    task_version = deployed_task.task_template.id.version

    remote_app = flyte.with_servecontext(env_vars={
        "APP_TASK_VERSION": task_version,
    }).serve(app)
    print(f"Served app: {remote_app.url}")
