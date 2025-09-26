import flyte.remote


async def run_task(
    name: str,
    inputs: dict,
    project: str | None = None,
    domain: str | None = None,
    version: str | None = None,
) -> flyte.remote.Run:
    task = flyte.remote.Task.get(
        name=name,
        project=project,
        domain=domain,
        version=version,
        auto_version="latest" if version is None else None,
    )
    return flyte.run(task, **inputs)


async def get_task(
    name: str,
    project: str | None = None,
    domain: str | None = None,
    version: str | None = None,
) -> flyte.remote.Task:
    return flyte.remote.Task.get(
        name=name,
        project=project,
        domain=domain,
        version=version,
        auto_version="latest" if version is None else None,
    ).fetch()


async def get_run_details(name: str) -> flyte.remote.Run:
    run = flyte.remote.Run.get(name=name)
    details = await run.action.details()
    return details


async def get_run_io(name: str) -> tuple[flyte.remote.ActionInputs, flyte.remote.ActionOutputs]:
    run: flyte.remote.Run = flyte.remote.Run.get(name=name)
    return run.inputs(), run.outputs()


async def list_tasks(
    project: str | None = None,
    domain: str | None = None,
) -> list[flyte.remote.Task]:
    tasks = []
    for task in flyte.remote.Task.listall(project=project, domain=domain):
        tasks.append(await get_task(task.name, project=project, domain=domain))
    return tasks
