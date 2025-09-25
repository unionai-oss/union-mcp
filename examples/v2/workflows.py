"""Examples of interacting with the Union controlplane via the v2 Remote API.

To deploy the tasks in this script, run it with:
```
python examples/v2/workflows.py
```
"""

import asyncio
import flyte


env = flyte.TaskEnvironment(name="my_env")


@env.task
async def add(a: int, b: int) -> int:
    return a + b


@env.task
async def multiply(a: int, b: int) -> int:
    return a * b


@env.task
async def append_hello(foo: str) -> str:
    """A task that prints a string and returns a string with 'hello' appended to the input."""
    foo = foo + 123  # ❌ this is a bug!
    return foo


@env.task
async def workflow(a: int, b: int) -> int:
    args = await asyncio.gather(add(a, b), add(a, b))
    return await multiply(*args)
