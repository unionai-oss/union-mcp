"""Union MCP server."""

import os
import typing
from datetime import timedelta
from functools import wraps

from mcp.server.fastmcp import FastMCP, Context

import union_mcp.v1.resources as resources


instructions = """
This MCP server is used to interact with Union resources and services.

For tools that take project and domain arguments, the MCP client needs to provide
them to the MCP tool calls, and if not provided, the client needs to ask the
user for the project and domain that they are trying to access.
"""

# Create an MCP server
mcp = FastMCP(
    name="Union MCP",
    instructions=instructions,
)

VALID_AUTH_TOKEN = os.environ["AUTH_TOKEN"]


def _remote(project: str, domain: str):
    import union

    return union.UnionRemote(
        default_project=project,
        default_domain=domain,
    )

def require_auth(func: typing.Callable):
    """Decorator to require authentication for FastMCP handlers"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract context from kwargs (FastMCP passes context)
        ctx: Context = kwargs.get('ctx')
        auth_header = ctx.request_context.request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            auth_token = auth_header[7:]
        else:
            raise ValueError("Authentication required: Invalid or missing token")

        if auth_token != VALID_AUTH_TOKEN:
            raise ValueError("Authentication required: Invalid or missing token")

        return func(*args, **kwargs)

    return wrapper

@mcp.tool()
@require_auth
def run_task(
    name: str,
    inputs: dict,
    project: str,
    domain: str,
    ctx: Context,
) -> tuple[dict, str]:
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
    remote = _remote(project, domain)
    task = remote.fetch_task(project=project, domain=domain, name=name)
    execution = remote.execute(task, inputs, project=project, domain=domain)
    execution = remote.wait(execution, poll_interval=timedelta(seconds=2))
    outputs = {k: v for k, v in execution.outputs.items() if v is not None}
    url = remote.generate_console_url(execution)
    return outputs, url


@mcp.tool()
@require_auth
def run_workflow(
    name: str,
    inputs: dict,
    project: str,
    domain: str,
    ctx: Context,
) -> tuple[dict, str]:
    """Run a workflow with natural language.

    - Based on the prompt and inputs dictionary, determine the workflow to run
    - Format the inputs dictionary so that it matches the workflow function signature
    - Invoke the workflow

    Args:
        project: Project to run the workflow in.
        domain: Domain to run the workflow in.
        name: Name of the task to run.
        inputs: A dictionary of inputs to the workflow.

    Returns:
        A dictionary of outputs from the workflow.
    """
    print(f"Running workflow {name} in project {project} and domain {domain}")
    remote = _remote(project, domain)
    workflow = remote.fetch_workflow(project=project, domain=domain, name=name)
    execution = remote.execute(workflow, inputs, project=project, domain=domain)
    execution = remote.wait(execution, poll_interval=timedelta(seconds=2))
    outputs = {k: v for k, v in execution.outputs.items() if v is not None}
    url = remote.generate_console_url(execution)
    return outputs, url


@mcp.tool()
@require_auth
def get_task(name: str, project: str, domain: str, ctx: Context) -> str:
    """Get a union task."""
    print(f"Getting task {name} in project {project} and domain {domain}")
    remote = _remote(project, domain)
    task = remote.fetch_task(name=name, project=project, domain=domain)
    return str(task)


@mcp.tool()
@require_auth
def get_execution(name: str, project: str, domain: str, ctx: Context) -> dict:
    """Get personalized union execution."""
    print(f"Getting execution {name} in project {project} and domain {domain}")
    remote = _remote(project, domain)
    execution = remote.fetch_execution(name=name, project=project, domain=domain)
    return resources.proto_to_json(execution.to_flyte_idl())


@mcp.tool()
@require_auth
def list_tasks(
    project: str,
    domain: str,
    ctx: Context,
) -> list[resources.TaskMetadata]:
    """List all tasks in a project and domain."""
    remote = _remote(project, domain)
    print(f"Listing tasks in project {project} and domain {domain}")
    return resources.list_tasks(remote, project, domain)


@mcp.tool()
@require_auth
def list_workflows(
    project: str,
    domain: str,
    ctx: Context,
) -> list[resources.WorkflowMetadata]:
    """List all workflows in a project and domain."""
    remote = _remote(project, domain)
    print(f"Listing workflows in project {project} and domain {domain}")
    return resources.list_workflows(remote, project, domain)
