import union


image = union.ImageSpec(
    name="union-mcp-server",
    builder="union",
    packages=["pandas", "pyarrow", "scikit-learn"],
)


actor = union.ActorEnvironment(
    name="add-multiply",
    ttl_seconds=120,
    container_image=image,
)


@actor.task(cache=True, cache_version="1")
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@actor.task(cache=True, cache_version="1")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@union.workflow
def workflow(a: int, b: int) -> int:
    """A workflow that adds and multiplies two numbers."""
    return multiply(add(a, b), add(a, b))
