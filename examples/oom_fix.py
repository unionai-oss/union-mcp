import union


image = union.ImageSpec(
    name="union-mcp-example-oomer",
    builder="union",
    packages=["union"],
)


@union.task(
    container_image=image,
    requests=union.Resources(cpu="2", mem="250Mi"),  # ❌ this causes an OOM!
)
def oomer(x: int):
    large_list = [x for _ in range(100_000_000)]
    print(large_list)


@union.workflow
def oom_wf(x: int):
    """A workflow that creates a large list and prints it."""
    return oomer(x)
