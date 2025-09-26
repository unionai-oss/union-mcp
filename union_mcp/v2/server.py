"""Union MCP server."""

import os

import flyte
from mcp.server.fastmcp import FastMCP, Context

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
)

UNION_ORG = os.environ["UNION_ORG"]


def _init(project: str, domain: str):
    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=UNION_ORG,
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
) -> tuple[tuple, str]:
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
    run = await resources.run_task(name, inputs, project, domain)
    run.wait()
    return run.outputs().to_dict(), run.url


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
