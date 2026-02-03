import flyte

import union_mcp.v2.resources as resources


env = flyte.TaskEnvironment(
    name="union_mcp_tasks",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=(
        flyte.Image
        .from_debian_base()
        .with_apt_packages("git")
        .with_pip_packages("unionai-reuse")
        .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@88cda0d")
    ),
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=1,
        idle_ttl=30,
    ),
)

@env.task
async def build_image(script: str) -> resources.RunResult:
    return await resources._build_script_image(script)

@env.task
async def run_task(script: str) -> resources.RunResult:
    return await resources._run_script_remote(script)


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(env)
    print(deployments[0].table_repr())
