"""Examples of interacting with the Union controlplane via the v2 Remote API."""

import asyncio
import flyte
import flyte.remote
from flyte.syncify import syncify


def run_task(
    name: str,
    inputs: dict,
    project: str | None = None,
    domain: str | None = None,
    version: str | None = None,
) -> dict:
    task = flyte.remote.Task.get(
        name=name,
        project=project,
        domain=domain,
        version=version,
        auto_version="latest" if version is None else None,
    )
    run = flyte.run(task, **inputs)
    return run.to_dict()


def get_task(
    name: str,
    project: str | None = None,
    domain: str | None = None,
    version: str | None = None,
) -> dict:
    return (
        flyte.remote.Task.get(
            name=name,
            project=project,
            domain=domain,
            version=version,
            auto_version="latest" if version is None else None,
        )
        .fetch()
        .to_dict()
    )


async def get_run_details(_run: flyte.remote.Run) -> dict:
    return await _run.action.details()


def get_run(name: str) -> dict:
    _run = flyte.remote.Run.get(name=name)
    details = asyncio.run(get_run_details(_run))
    return details.to_dict()


def list_tasks(
    project: str | None = None,
    domain: str | None = None,
) -> list[dict]:
    tasks = []
    for task in flyte.remote.Task.listall(project=project, domain=domain):
        tasks.append(get_task(task.name, project=project, domain=domain))
    return tasks


if __name__ == "__main__":
    import os

    api_key = os.environ["FLYTE_API_KEY"]
    project = "union-mcp"
    domain = "development"

    flyte.init(
        api_key=api_key,
        org="union-internal",
        project=project,
        domain=domain,
    )

    # task = get_task(name="my_env.add", project=project, domain=domain)
    # print(task)

    # tasks = list_tasks(project=project, domain=domain)
    # print(tasks)

    # run = run_task(name=task["taskId"]["name"], inputs={"a": 1, "b": 2}, project=project, domain=domain)
    # print(run)
    # fetched_run = get_run(name=run["action"]["id"]["run"]["name"])
    fetched_run = get_run(name="rq9p8tfxbwfjhs5wdmkw")
    print(fetched_run)
