"""Union MCP server."""

import os

import flyte
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings

import union_mcp.v2.resources as resources
from union_mcp.common.auth import require_auth


instructions = """
This MCP server is used to interact with Union v2 resources and services.

For tools that take project and domain arguments, the MCP client needs to provide
them to the MCP tool calls, and if not provided, the client needs to ask the
user for the project and domain that they are trying to access.
"""

# Create an MCP server
mcp = FastMCP(
    name="Union v2 MCP",
    instructions=instructions,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


def _init(project: str, domain: str):
    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=os.environ["FLYTE_ORG"],
        project=project,
        domain=domain,
    )


@mcp.tool()
@require_auth
async def run_task(
    name: str,
    inputs: dict,
    project: str,
    domain: str,
    ctx: Context,
) -> dict:
    ctx.info(f"Running task {name} in project {project} and domain {domain}")
    """Run a task with natural language.

    - Based on the prompt and inputs dictionary, determine the task to run
    - Format the inputs dictionary so that it matches the task function signature
    - Invoke the task
    
    Args:
        project: Project to run the task in.
        domain: Domain to run the task in.
        name: Name of the task to run.
        inputs: A dictionary of inputs to the task.

    Returns:
        A dictionary of outputs from the task.
    """
    # Based on the prompt and inputs dictionary, determine the task
    _init(project, domain)
    return (await resources.run_task(name, inputs, project, domain)).to_dict()


@mcp.tool()
@require_auth
async def get_task(name: str, project: str, domain: str, ctx: Context) -> dict:
    """Get a union task."""
    print(f"Getting task {name} in project {project} and domain {domain}")
    _init(project, domain)
    task = await resources.get_task(name, project, domain)
    return task.to_dict()


@mcp.tool()
@require_auth
async def get_run(name: str, project: str, domain: str, ctx: Context) -> dict:
    """Get personalized union execution."""
    print(f"Getting execution {name} in project {project} and domain {domain}")
    _init(project, domain)
    return (await resources.get_run_details(name)).to_dict()


@mcp.tool()
@require_auth
async def get_run_io(name: str, project: str, domain: str, ctx: Context) -> dict:
    """Get personalized union execution."""
    print(f"Getting execution {name} in project {project} and domain {domain}")
    _init(project, domain)
    inputs, outputs = await resources.get_run_io(name)
    return {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
    }


@mcp.tool()
@require_auth
async def list_tasks(
    project: str,
    domain: str,
    ctx: Context,
) -> list[dict]:
    """List all tasks in a project and domain."""
    _init(project, domain)
    print(f"Listing tasks in project {project} and domain {domain}")
    return [task.to_dict() for task in await resources.list_tasks(project, domain)]


@mcp.tool()
@require_auth
async def list_runs(task_name: str, project: str, domain: str, ctx: Context) -> dict:
    """Get a union task inputs and outputs."""
    print(f"Getting runs of {task_name} in project {project} and domain {domain}")
    _init(project, domain)
    runs = await resources.list_runs(task_name, project, domain)
    return [(await run.action.details()).to_dict() for run in runs]


@mcp.tool()
@require_auth
async def run_script(script: str, project: str, domain: str, ctx: Context) -> dict:
    """Run a task script provided by the user.

    - Based on the script, determine the task to register
    - Format the script so that it matches the task function signature
    - Register the task

    Args:
        script: Script to register the task from.
        project: Project to register the task in.
        domain: Domain to register the task in.
    """
    _init(project, domain)
    run_script_result = await resources.run_script(script, project, domain)
    return run_script_result


@mcp.tool()
@require_auth
async def flyte_script_format(ctx: Context) -> str:
    """Get the template format of a Flyte script."""
    return resources.script_format()


@mcp.tool()
@require_auth
async def flyte_script_example(ctx: Context) -> str:
    """Get a full example of a Flyte script."""
    ctx.info("Getting example Flyte script")
    return resources.script_example()


@mcp.tool()
@require_auth
async def search_flyte_examples(pattern: str, ctx: Context) -> str:
    """Grep the Flyte example repository for a pattern."""
    ctx.info("Getting example Flyte script")
    return resources.search_flyte_examples(pattern)
