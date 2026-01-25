"""Union MCP server."""

import os

import flyte
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings
from contextlib import asynccontextmanager

import union_mcp.v2.resources as resources
from union_mcp.common.auth import require_auth


instructions = """
This MCP server is used to interact with Union v2 resources and services."""


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with type-safe context."""
    # Initialize on startup
    PROJECT_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_PROJECT"
    DOMAIN_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_DOMAIN"

    # Startup: Initialize Flyte with passthrough authentication
    await flyte.init_passthrough.aio(
        project=os.getenv(PROJECT_NAME_ENV_VAR, None),
        domain=os.getenv(DOMAIN_NAME_ENV_VAR, None),
    )
    print("Initialized Flyte passthrough auth")
    yield


# Create an MCP server
mcp = FastMCP(
    name="Union v2 MCP",
    instructions=instructions,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    stateless_http=True,
    json_response=True,
    lifespan=app_lifespan,
)


def _init():
    flyte.init(
        api_key=os.getenv("FLYTE_API_KEY"),
        org=os.getenv("FLYTE_ORG"),
        project=os.getenv("FLYTE_PROJECT"),
        domain=os.getenv("FLYTE_DOMAIN"),
    )


@mcp.tool()
@require_auth
async def run_task(
    name: str,
    inputs: dict,
    ctx: Context,
) -> dict:
    await ctx.info(f"Running task {name}")
    """Run a task with natural language.

    - Based on the prompt and inputs dictionary, determine the task to run
    - Format the inputs dictionary so that it matches the task function signature
    - Invoke the task
    
    Args:
        name: Name of the task to run.
        inputs: A dictionary of inputs to the task.

    Returns:
        A dictionary of outputs from the task.
    """
    # Based on the prompt and inputs dictionary, determine the task
    _init()
    return (await resources.run_task(name, inputs)).to_dict()


@mcp.tool()
@require_auth
async def get_task(
    name: str,
    ctx: Context,
) -> dict:
    """Get a union task."""
    print(f"Getting task {name}")
    _init()
    task = await resources.get_task(name)
    return task.to_dict()


@mcp.tool()
@require_auth
async def get_run(
    name: str,
    ctx: Context,
) -> dict:
    """Get personalized union execution."""
    print(f"Getting execution {name}")
    _init()
    return (await resources.get_run_details(name)).to_dict()


@mcp.tool()
@require_auth
async def get_run_io(
    name: str,
    ctx: Context,
) -> dict:
    """Get personalized union execution."""
    print(f"Getting execution {name}")
    _init()
    inputs, outputs = await resources.get_run_io(name)
    return {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
    }


@mcp.tool()
@require_auth
async def list_tasks(
    ctx: Context,
) -> list[dict]:
    """List all tasks."""
    _init()
    print(f"Listing tasks")
    return [task.to_dict() for task in await resources.list_tasks()]


@mcp.tool()
@require_auth
async def list_runs(
    task_name: str,
    ctx: Context,
) -> dict:
    """Get a union task inputs and outputs."""
    print(f"Getting runs of {task_name}")
    _init()
    runs = await resources.list_runs(task_name)
    return [(await run.action.details()).to_dict() for run in runs]


@mcp.tool()
@require_auth
async def build_script_image(
    script: str,
    ctx: Context,
):
    """Build the image for a script.
    
    Args:
        script: Script to build the image for.

    This tool should be used before invoking run_script_remote. This will asynchonously build the image and return the
    result, which contains the build task url. You can use the build task url to monitor the build progress.

    Once this build task is completed, the agent can invoke run_script_remote to run the script on the remote Flyte cluster.
    """
    _init()
    return await resources.build_script_image(script)


@mcp.tool()
@require_auth
async def run_script_remote(
    script: str,
    ctx: Context,
) -> dict:
    """Run a task script provided by the user on remote Flyte cluster.

    IMPORTANT: Make sure the script is built first using build_script_image tool, which should be called before this
    tool. This will asynchronously build the image and return the result, which contains the build task url. Make sure
    that the build task is completed before running the script. Use the get_run tool to monitor the build task.

    Make sure the script is correctly formatted according to flyte_script_format.
    For a complete example, see flyte_script_example.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.

    Args:
        script: Script to register the task from. Use the flyte_script_format to make sure
        the script is correctly formatted.
    """
    _init()
    return await resources.run_script_remote(script)


@mcp.tool()
@require_auth
async def flyte_script_format(ctx: Context) -> str:
    """Get the template format of a Flyte script.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.
    """
    return resources.script_format()


@mcp.tool()
@require_auth
async def flyte_script_example(ctx: Context) -> str:
    """Get a full example of a Flyte script.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.
    """
    await ctx.info("Getting example Flyte script")
    return resources.script_example()


@mcp.tool()
@require_auth
async def search_flyte_sdk_examples(
    pattern: str,
    ctx: Context,
) -> str:
    """Search the Flyte SDK examples repository for files that match a pattern.
    
    Args:
        pattern: The pattern to search for.

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    await ctx.info("Getting example Flyte SDK example scripts")
    return resources.search_flyte_examples(pattern, "/root/flyte-sdk/examples", top_n=3)


@mcp.tool()
@require_auth
async def search_flyte_docs_examples(
    pattern: str,
    ctx: Context,
) -> str:
    """Search the official Flyte Docs examples repository for files that match a pattern.
    
    Args:
        pattern: The pattern to search for.

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    await ctx.info("Getting example Flyte docs")
    return resources.search_flyte_examples(pattern, "/root/unionai-examples/v2", top_n=3)
