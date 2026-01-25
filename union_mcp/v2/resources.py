import os
import subprocess
import uuid
import flyte.io  # noqa: F401 - imported to register FileTransformer and DirTransformer with TypeEngine
import flyte.remote


async def run_task(
    name: str,
    inputs: dict,
    version: str | None = None,
) -> flyte.remote.ActionDetails:
    task = flyte.remote.Task.get(
        name=name,
        version=version,
        auto_version="latest" if version is None else None,
    )
    run: flyte.remote.Run = flyte.run(task, **inputs)
    return await run.action.details()


async def get_task(
    name: str,
    version: str | None = None,
) -> flyte.remote.Task:
    return flyte.remote.Task.get(
        name=name,
        version=version,
        auto_version="latest" if version is None else None,
    ).fetch()


async def get_run_details(name: str) -> flyte.remote.ActionDetails:
    run = flyte.remote.Run.get(name=name)
    return await run.action.details()


async def get_run_io(
    name: str,
) -> tuple[flyte.remote.ActionInputs, flyte.remote.ActionOutputs]:
    run: flyte.remote.Run = flyte.remote.Run.get(name=name)
    return run.inputs(), run.outputs()


async def list_tasks() -> list[flyte.remote.Task]:
    tasks = []
    for task in flyte.remote.Task.listall():
        tasks.append(await get_task(task.name))
    return tasks


async def list_runs(task_name: str | None = None) -> list[flyte.remote.Run]:
    runs = []
    for run in flyte.remote.Run.listall(
        task_name=task_name,
        limit=10,
        sort_by=("created_at", "desc"),
    ):
        runs.append(run)
    return runs


async def build_script_image(script: str) -> dict:

    filename = f"__build_script_{str(uuid.uuid4())[:16]}__.py"

    with open(filename, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            ["uv", "run", "--prerelease=allow", filename, "--build"],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    finally:
        os.remove(filename)


async def run_script_remote(script: str) -> dict:
    filename = f"__run_script_{str(uuid.uuid4())[:16]}__.py"

    with open(filename, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            ["uv", "run", "--prerelease=allow", filename],
            capture_output=True,
            env=os.environ,
            text=True,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    finally:
        os.remove(filename)


def search_flyte_examples(
    pattern: str, examples_dir: str, top_n: int = 3,
) -> str:
    """Grep for a pattern in flyte-sdk/examples, return top n files with most matches as markdown.

    Args:
        pattern: The pattern to search for.
        examples_dir: The directory to search in. Defaults to "flyte-sdk/examples".

    Returns:
        A markdown-formatted string containing the contents of the top 3 files with the most matches.
    """
    # Use grep -c to count matches per file
    proc = subprocess.run(
        ["grep", "-r", "-c", pattern, examples_dir],
        capture_output=True,
        text=True,
    )

    if proc.returncode not in (0, 1):  # 1 means no matches found
        return f"Error running grep: {proc.stderr}"

    if not proc.stdout.strip():
        return f"No matches found for pattern: {pattern}"

    # Parse output: each line is "filename:count"
    file_counts: list[tuple[str, int]] = []
    for line in proc.stdout.strip().split("\n"):
        if ":" in line:
            # Handle case where filename might contain colons
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                filepath, count_str = parts
                try:
                    count = int(count_str)
                    if count > 0:  # Only include files with matches
                        file_counts.append((filepath, count))
                except ValueError:
                    continue

    if not file_counts:
        return f"No matches found for pattern: {pattern}"

    # Sort by count descending and take top_n
    file_counts.sort(key=lambda x: x[1], reverse=True)
    top_files = file_counts[:top_n]

    # Build markdown output
    markdown_parts = [f"# Top {len(top_files)} files matching pattern: `{pattern}`\n"]

    for filepath, count in top_files:
        markdown_parts.append(f"## `{filepath}` ({count} matches)\n")

        # Read file contents
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Determine language for syntax highlighting
            ext = os.path.splitext(filepath)[1].lstrip(".")
            lang = ext if ext else "text"

            markdown_parts.append(f"```{lang}\n{content}\n```\n")
        except (IOError, OSError) as e:
            markdown_parts.append(f"*Error reading file: {e}*\n")

    return "\n".join(markdown_parts)


def script_format() -> str:
    return """
```python
# /// script
# dependencies = [
#    "flyte>=2.0.0b49",  # THIS IS IMPORTANT: it makes sure the script can be run on the MCP server
#    <package-name>
#    ...
# ]
# ///

import flyte

# Import other packages as needed
...

# Define the task environment
env = flyte.TaskEnvironment(
    name="<task-env-name>",
    resources=flyte.Resources(cpu=<cpu-count>, memory="<memory-size>", gpu="<gpu-name>:<gpu-count>", disk="<disk-size>"),
    image=flyte.Image.from_uv_script(__file__, name="<image-name>", python_version=(<python-major-version>, <python-minor-version>), pre=True)
)

# Define one or more tasks.
@env.task
async def <task-name>(<task-arguments>) -> <task-return-type>:
    <task-body>

# Define helper functions as needed
async def <helper-function-name>(<helper-function-arguments>) -> <helper-function-return-type>:
    <helper-function-body>

# more tasks
...

@env.task
async def main(<main-arguments>) -> <main-return-type>:  # the main task is the entry point for the script
    <main-body>


if __name__ == "__main__":
    import argparse
    import os

    # THIS IS IMPORTANT: it makes sure the script can be both built and run on the MCP server
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    # THIS IS IMPORTANT: it makes sure the script can be run on the MCP server
    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=os.environ["FLYTE_ORG"],
        project=os.environ["FLYTE_PROJECT"],
        domain=os.environ["FLYTE_DOMAIN"],
        # THIS IS IMPORTANT: image builder needs to be set to remote for the script to run on the MCP server
        image_builder="remote",
    )
    # THIS IS IMPORTANT: the script should be built first, then run
    if args.build:
        flyte.build(env.image)
    else:
        # run the task in remote mode
        run = flyte.with_runcontext(mode="remote").run(main, <main-arguments>)
        print(run.url)
```
""".strip()


def script_example() -> str:
    """Get a full example of a Flyte script."""
    return """
```python
# /// script
# dependencies = [
#    "flyte>=2.0.0b49",  # THIS IS IMPORTANT: it makes sure the script can be run on the MCP server
#    "scikit-learn==1.6.1",
#    "pandas",
#    "pyarrow",
# ]
# ///

import flyte

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


env = flyte.TaskEnvironment(
    name="my_example_script",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(__file__, name="example-image", python_version=(3, 13), pre=True)
)

@env.task
async def create_dataset(n_samples: int = 100) -> pd.DataFrame:
    X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2)
    df = pd.DataFrame(X)
    df["target"] = y
    return df

@env.task
async def train_model(dataset: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(dataset.drop(columns=["target"]), dataset["target"])
    return model


@env.task
async def main() -> RandomForestClassifier:
    dataset = await create_dataset()
    model = await train_model(dataset)
    return model


if __name__ == "__main__":
    import argparse
    import os

    # THIS IS IMPORTANT: it makes sure the script can be both built and run on the MCP server
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    # THIS IS IMPORTANT: it makes sure the script can be run on the MCP server
    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=os.environ["FLYTE_ORG"],
        project=os.environ["FLYTE_PROJECT"],
        domain=os.environ["FLYTE_DOMAIN"],
        # THIS IS IMPORTANT: image builder needs to be set to remote for the script to run on the MCP server
        image_builder="remote",
    )
    # THIS IS IMPORTANT: the script should be built first, then run
    if args.build:
        flyte.build(env.image)
    else:
        # run the task in remote mode
        run = flyte.with_runcontext(mode="remote").run(main)
        print(run.url)
```
""".strip()
