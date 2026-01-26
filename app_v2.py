# /// script
# requires-python = "==3.12"
# dependencies = ["flyte==2.0.0b49"]
# ///

import os
import logging
import pathlib
import subprocess
import uuid

import flyte
from flyte.app import AppEnvironment, Domain, Scaling, Link


APP_NAME = os.getenv("APP_NAME", "union-mcp-v2")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN", "mcp-v2")
APP_PORT = int(os.getenv("APP_PORT", 8000))
FLYTE_ENDPOINT = os.getenv("FLYTE_ENDPOINT", "dns:///demo.hosted.unionai.cloud")
FLYTE_ORG = os.getenv("FLYTE_ORG", "demo")
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "union-mcp")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")


# -----------------
# Tasks Environment
# -----------------

task_env = flyte.TaskEnvironment(
    name="union_mcp_script_runner",
    resources=flyte.Resources(cpu=3, memory="8Gi", disk="10Gi"),
    image=(
        flyte.Image
        .from_debian_base(name="union-mcp-script-runner")
        .with_apt_packages("ca-certificates", "git")
        .with_pip_packages("uv", "flyte==2.0.0b49", "unionai-reuse")
        .with_pip_packages("git+https://github.com/unionai-oss/union-mcp.git@main#egg=union-mcp[v2]")
    ),
    reusable=flyte.ReusePolicy(
        replicas=(1, 2),
        idle_ttl=60,
    ),
)



@task_env.task
async def build_script_image_task(script: str) -> dict:
    """Build the container image for a Flyte script.

    This task writes the script to a temporary file and runs the build process
    using uv to build the container image for the script.

    Args:
        script: The Python script content to build.

    Returns:
        A dict containing stdout, stderr, and returncode from the build process.
    """
    filename = f"__build_script_{str(uuid.uuid4())[:16]}__.py"

    with open(filename, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "--with", "flyte@git+https://github.com/flyteorg/flyte-sdk.git@378af3e0",
                "--prerelease=allow",
                filename,
                "--build",
            ],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    finally:
        os.remove(filename)


@task_env.task
async def run_script_remote_task(script: str) -> dict:
    """Run a Flyte script remotely on the cluster.

    This task writes the script to a temporary file and executes it using uv,
    which triggers the script's remote execution logic.

    Args:
        script: The Python script content to run.

    Returns:
        A dict containing stdout, stderr, and returncode from the run process.
    """
    filename = f"__run_script_{str(uuid.uuid4())[:16]}__.py"

    with open(filename, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            ["uv", "run", "--prerelease=allow", filename],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    finally:
        os.remove(filename)



# -----------------
# App Environment
# -----------------

image = (
    flyte.Image.from_debian_base(name="union-mcp-server")
    .with_pip_packages("uv", "mcp[cli]==1.26.0", "starlette")
    # .with_pip_packages("flyte==2.0.0b49")
    .with_apt_packages("git", "wget")
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@c985e5fe")
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
    resources=flyte.Resources(cpu=8, memory="20Gi", disk="64Gi"),
    secrets=[
        flyte.Secret(key="EAGER_API_KEY", as_env_var="FLYTE_API_KEY"),
    ],
    env_vars={
        "FLYTE_ORG": FLYTE_ORG,
        "FLYTE_PROJECT": FLYTE_PROJECT,
        "FLYTE_DOMAIN": FLYTE_DOMAIN,
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
    depends_on=[task_env],
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
    deployments = flyte.deploy(app)
    d = deployments[0]
    print(f"Deployed app: {d.table_repr()}")
