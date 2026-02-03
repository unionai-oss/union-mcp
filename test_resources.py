import flyte
import flyte.remote


async def main():
    flyte.init_from_config()

    task = flyte.remote.Task.get(name="union_mcp_tasks.run_task", auto_version="latest")
    run = flyte.run(task, script="print('hello')")
    await run.wait.aio()
    return run.outputs()[0]


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(main()))
