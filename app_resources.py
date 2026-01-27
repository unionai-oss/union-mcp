import subprocess
import os
import uuid

import flyte

from dataclasses import dataclass
from flyte.remote._common import ToJSONMixin


@dataclass
class ProcessResult(ToJSONMixin):
    stdout: str
    stderr: str
    returncode: int
    

# -----------------
# Tasks Environment
# -----------------

task_env = flyte.TaskEnvironment(
    name="union_mcp_script_runner",
    resources=flyte.Resources(cpu=3, memory="8Gi", disk="100Gi"),
    image=(
        flyte.Image
        .from_debian_base(name="union-mcp-script-runner")
        .with_apt_packages("ca-certificates", "git")
        .with_pip_packages(
            "uv",
            "unionai-reuse",
            "git+https://github.com/flyteorg/flyte-sdk.git@c985e5fe",
            "git+https://github.com/unionai-oss/union-mcp.git@0520716#egg=union-mcp[v2]",
        )
    ),
    # reusable=flyte.ReusePolicy(
    #     replicas=(1, 2),
    #     idle_ttl=60,
    # ),
)



@task_env.task
async def build_script_image_task(script: str, tail: int = 50) -> ProcessResult:
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
        out = ProcessResult(
            stdout="\n".join(proc.stdout.splitlines()[-tail:]),
            stderr="\n".join(proc.stderr.splitlines()[-tail:]),
            returncode=proc.returncode,
        )
        return out
    finally:
        os.remove(filename)


@task_env.task
async def run_script_remote_task(script: str, tail: int = 50) -> ProcessResult:
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
        out = ProcessResult(
            stdout="\n".join(proc.stdout.splitlines()[-tail:]),
            stderr="\n".join(proc.stderr.splitlines()[-tail:]),
            returncode=proc.returncode,
        )
        return out
    finally:
        os.remove(filename)
