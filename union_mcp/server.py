"""Union MCP server."""

import union
from mcp.server.fastmcp import FastMCP
import union_mcp.resources as resources
from datetime import timedelta

# Create an MCP server
mcp = FastMCP("Union MCP")
remote = union.UnionRemote()


# Tools
# -----

@mcp.tool()
def run_task(project: str, domain: str, name: str, inputs: dict) -> tuple[dict, str]:
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
    task = remote.fetch_task(project=project, domain=domain, name=name)
    ex = remote.execute(task, inputs, project=project, domain=domain)
    ex = remote.wait(ex, poll_interval=timedelta(seconds=2))
    outputs = {k: v for k, v in ex.outputs.items() if v is not None}
    url = remote.generate_console_url(ex)
    return outputs, url


@mcp.tool()
def get_task(name: str, project: str, domain: str) -> str:
    """Get a personalized union"""
    task = remote.fetch_task(name=name, project=project, domain=domain)
    return str(task)


@mcp.tool()
def list_tasks(
    project: str = remote.default_project,
    domain: str = remote.default_domain,
) -> list[resources.TaskMetadata]:
    """List all tasks in a project and domain."""
    return resources.list_tasks(remote, project, domain)


@mcp.tool()
def list_workflows(
    project: str = remote.default_project,
    domain: str = remote.default_domain,
) -> list[resources.WorkflowMetadata]:
    """List all workflows in a project and domain."""
    return resources.list_workflows(remote, project, domain)
