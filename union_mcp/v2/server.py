"""Union MCP server."""

import os
from contextlib import asynccontextmanager

import flyte
from flyte.app.extras import FastAPIPassthroughAuthMiddleware
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware import Middleware
from starlette.responses import Response
from starlette.requests import Request

import union_mcp.v2.resources as resources


instructions = """
This MCP server is used to interact with Union v2 resources and services.

When you need to write a Flyte script to build in run:
- Search for any flyte examples in your local workspace that may be relevant to the user's request
- Use the flyte_script_format tool to get the template format of a MCP-ready Flyte script
- Use the search_flyte_sdk_examples and search_flyte_docs_examples tools to find examples related to the user's request
- Build the script image using the build_script_image tool.
- Run the script using the run_script_remote tool.
- Check the status of the run using the get_run tool.
- Once the run completes, you can use the get_run_io tool to get the inputs and outputs of the run.
"""


# Create an MCP server
mcp = FastMCP(
    name="Union v2 MCP",
    instructions=instructions,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
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
    return (await resources.run_task(name, inputs)).to_dict()


@mcp.tool()
async def get_task(
    name: str,
    ctx: Context,
) -> dict:
    """Get a union task."""
    task = await resources.get_task(name)
    return task.to_dict()


@mcp.tool()
async def get_run(
    name: str,
    ctx: Context,
) -> dict:
    """Get run information.
    
    Use wait_for_run_completion to wait for a long-running task run to complete.
    """
    return (await resources.get_run_details(name)).to_dict()


@mcp.tool()
async def wait_for_run_completion(
    name: str,
    ctx: Context,
) -> dict:
    """Wait for a run to complete.

    Use this tool to wait for a long-running task run to complete. Useful when
    the run completion is needed to continue the conversation so that the agent
    doesn't have to keep polling for the run status.

    Use this for when waiting for build_script_image or run_script_remote runs
    to complete.
    
    Args:
        name: Name of the run to wait for.

    Returns:
        A dictionary of run information.
    """
    await ctx.info(f"Waiting for run {name} to complete")
    return (await resources.wait_for_run_completion(name)).to_dict()


@mcp.tool()
async def get_run_io(
    name: str,
    ctx: Context,
) -> dict:
    """Get personalized union execution."""
    inputs, outputs = await resources.get_run_io(name)
    return {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
    }


@mcp.tool()
async def list_tasks(
    ctx: Context,
) -> list[dict]:
    """List all tasks."""
    return [task.to_dict() for task in await resources.list_tasks()]


@mcp.tool()
async def list_runs(
    task_name: str,
    ctx: Context,
) -> dict:
    """Get a union task inputs and outputs."""
    runs = await resources.list_runs(task_name)
    return [(await run.action.details()).to_dict() for run in runs]


@mcp.tool()
async def remote_build_script_image(
    script: str,
    ctx: Context,
) -> dict:
    """Build the image for a script on the remote Flyte cluster.

    Only use this tool if the user explicitly requests to build a script on the remote Flyte cluster.

    This tool should be used before invoking run_script_remote. This will asynchonously build the image and return the
    result, which contains the build task url. You can use the build task url to monitor the build progress.

    Image builds can take 5-10 minutes and up, so if the image doesn't build within 10-15 seconds, you should pause the thinking
    loop, show the build task url to the user and ask the user to explicitly check the build task status again.

    Once this build task is completed, the agent can invoke run_script_remote to run the script on the remote Flyte cluster.

    Args:
        script: Script to build the image for.

    Returns:
        A dictionary containing the image build run url. Use this run url to
        monitor the build progress.
    """
    return await resources.build_script_image(script)


@mcp.tool()
async def remote_run_script_remote(
    script: str,
    ctx: Context,
) -> dict:
    """Run a task script provided by the user on remote Flyte cluster.

    Only use this tool if the user explicitly requests to run a script on the remote Flyte cluster.

    IMPORTANT: Make sure the script is built first using build_script_image tool, which should be called before this
    tool. This will asynchronously build the image and return the result, which contains the build task url. Make sure
    that the build task is completed before running the script. Use the get_run tool to monitor the build task.

    Make sure the script is correctly formatted according to flyte_script_format.
    For a complete example, see flyte_script_example.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.

    Task runs can take 5-10 minutes and more, so if the task run doesn't complete within 10-15 seconds, you should pause
    the thinking loop, show the task run url to the user and ask the user to explicitly check the task run status again.

    Args:
        script: Script to register the task from. Use the flyte_script_format to make sure
        the script is correctly formatted.

    Returns:
        A dictionary containing the run script url. Use this run url to
        monitor the run script progress.
    """
    return await resources.run_script_remote(script)


@mcp.tool()
async def flyte_script_format(ctx: Context) -> str:
    """Get the template format of a Flyte script.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.
    """
    return resources.script_format()


@mcp.tool()
async def flyte_script_example(ctx: Context) -> str:
    """Get a full example of a Flyte script.
    
    Use search_flyte_sdk_examples and search_flyte_docs_examples to find examples
    that match your needs.
    """
    await ctx.info("Getting example Flyte script")
    return resources.script_example()


@mcp.tool()
async def search_flyte_sdk_examples(
    pattern: str,
    ctx: Context,
    before_context_lines: int = 5,
    after_context_lines: int = 5,
) -> str:
    """Search the Flyte SDK examples repository for files that match a pattern.

    This tool is useful for finding specific examples that are hosted on the Flyte SDK repository, which contains
    examples that include bleeding edge features and new functionality.
    
    Args:
        pattern: The pattern to search for.
        before_context_lines: The number of lines to show before each match.
        after_context_lines: The number of lines to show after each match.

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    await ctx.info("Getting example Flyte SDK example scripts")
    return resources.search_flyte_examples(
        pattern,
        "/root/flyte-sdk/examples",
        top_n=3,
        before_context_lines=before_context_lines,
        after_context_lines=after_context_lines,
    )


@mcp.tool()
async def search_flyte_docs_examples(
    pattern: str,
    ctx: Context,
    before_context_lines: int = 5,
    after_context_lines: int = 5,
) -> str:
    """Search the official Flyte Docs examples repository for files that match a pattern.

    This tool is useful to find specific use cases and examples of how to use the Flyte SDK in a python script.
    
    Args:
        pattern: The pattern to search for.
        before_context_lines: The number of lines to show before each match.
        after_context_lines: The number of lines to show after each match.

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    await ctx.info("Getting example Flyte docs")
    return resources.search_flyte_examples(
        pattern,
        "/root/unionai-examples/v2",
        top_n=3,
        before_context_lines=before_context_lines,
        after_context_lines=after_context_lines,
    )


@mcp.tool()
async def search_full_docs(
    pattern: str,
    ctx: Context,
    before_context_lines: int = 20,
    after_context_lines: int = 20,
) -> str:
    """Search the official Flyte Documentation for "User Guide", "Tutorials", "API reference", "Integrations"

    This tool is useful to find out how to use the Flyte SDK, the semantics and functionality of the different flyte
    constructs, CLI commands, and more.
    
    Args:
        pattern: The pattern to search for.
        before_context_lines: The number of lines to show before each match.
        after_context_lines: The number of lines to show after each match.

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    await ctx.info("Getting example Flyte docs")
    return resources.search_flyte_examples(
        pattern,
        "/root/full-docs.txt",
        before_context_lines=before_context_lines,
        after_context_lines=after_context_lines,
    )


@asynccontextmanager
async def lifespan(app: Starlette):
    PROJECT_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_PROJECT"
    DOMAIN_NAME_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_DOMAIN"

    project = os.getenv("FLYTE_PROJECT", os.getenv(PROJECT_NAME_ENV_VAR, None))
    domain = os.getenv("FLYTE_DOMAIN", os.getenv(DOMAIN_NAME_ENV_VAR, None))

    # Startup: Initialize Flyte with passthrough authentication
    await flyte.init_passthrough.aio(
        project=project,
        domain=domain,
    )
    async with mcp.session_manager.run():
        yield


app = Starlette(
    middleware=[
        Middleware(FastAPIPassthroughAuthMiddleware),
    ],
    routes=[
        Mount("/sdk", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

@app.route("/health")
async def health(request: Request) -> Response:
    return Response(content="OK", media_type="text/plain", status_code=200)
