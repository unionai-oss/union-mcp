import union


image = union.ImageSpec(
    name="union-mcp-example-workflows",
    builder="union",
    packages=["pandas", "pyarrow", "scikit-learn", "union"],
)


actor = union.ActorEnvironment(
    name="add-multiply",
    ttl_seconds=120,
    container_image=image,
)


@actor.task
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@actor.task
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@actor.task
def append_hello(foo: str) -> str:
    """A task that prints a string and returns a string with 'hello' appended to the input."""
    foo = foo + 123  # ❌ this is a bug!
    return foo


@union.workflow
def workflow(a: int, b: int) -> int:
    """A workflow that adds and multiplies two numbers."""
    return multiply(add(a, b), add(a, b))
