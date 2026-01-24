import flyte


env = flyte.TaskEnvironment(
    name="oomer_env",
    resources=flyte.Resources(cpu="2", memory="250Mi"),  # ❌ this causes an OOM!
)


@env.task
async def oomer(x: int):
    large_list = [x for _ in range(100_000_000)]
    print(large_list)


@env.task
async def oom_wf(x: int):
    """A workflow that creates a large list and prints it."""
    return await oomer(x)
