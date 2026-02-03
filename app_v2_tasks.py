import flyte

import union_mcp.v2.resources as resources


env = flyte.TaskEnvironment(
    name="union_mcp_tasks",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=(
        flyte.Image
        .from_debian_base()
        .with_apt_packages("ca-certificates", "git")
        .with_pip_packages("unionai-reuse")
        .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@4c5f2d66aa10fc631ce492ef0db36b11dda579ca")
    ),
    reusable=flyte.ReusePolicy(
        replicas=4,
        concurrency=4,
        idle_ttl=30,
    ),
)

@env.task
async def build_image(script: str, tail: int = 50) -> resources.RunResult:
    return await resources.build_script_image_(script, tail=tail)

@env.task
async def run_task(script: str, tail: int = 50) -> resources.RunResult:
    return await resources.run_script_remote_(script, tail=tail)


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(env)
    print(deployments[0].table_repr())
