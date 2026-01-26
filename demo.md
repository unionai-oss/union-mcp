# Demo

This demo shows how to use the Union MCP server to run compute- and io-intensive
tools on a Flyte cluster.

## Setup

### Cursor

Add the following to your `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "union-mcp-v2": {
      "url": "https://mcp-v2.apps.demo.hosted.unionai.cloud/sdk/mcp"
      "headers": {
        "Authorization": "Bearer <secret-value>"
      }
    }
  }
}
```

### Claude Code

Add the following to your `~/.claude.json` file:

```json
{
  "mcpServers": {
    "union-mcp-v2": {
      "url": "https://mcp-v2.apps.demo.hosted.unionai.cloud/sdk/mcp"
      "headers": {
        "Authorization": "Bearer <secret-value>"
      }
    }
  }
}
```

### Claude Desktop

Configure the `claude_desktop_config.json` configuration file located in:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Make sure that `npx` is installed and available in your `$PATH`.

```json
{
  "mcpServers": {
    "union-mcp-v2": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp-v2.apps.demo.hosted.unionai.cloud/sdk/mcp",
        "--header",
        "Authorization: Bearer ${AUTH_TOKEN}"
      ],
      "env": {
        "AUTH_TOKEN": "<secret-value>"
      }
    }
  }
}

```

## Prompts

### Toy example

> Create a flyte script that fans out tasks to compute the square of the numbers from 1 to 100_000, where each task handles 1_000 numbers, then sums the squares.


### Visualization

> Create and run a flyte script that downloads the dataset at https://github.com/plotly/datasets/blob/master/timeseries.csv and creates a visualization in plotly and run it remotely. The flyte script should use flyte.report to render a beautiful visualization.

### Model training

> Run a flyte script that performs hyperparameter optimization that uses flyte to parallelize the training runs for training a random forest model on the penguins data. Assess f1 score as the evaluation metric, and visualize the results using flyte.report. Make sure the report style is beautiful.

### GPU Batch inference

> Run a flyte script that embeds the "review" column of the "scikit-learn/imdb" huggingface dataset using the "answerdotai/ModernBERT-base" model on a T4 GPU. Use a driver-worker pattern where the driver is a CPU environment and the worker is a GPU environment. Save the embeddings to a json file using flyte.io.File, and use flyte.report make a pretty visualization of the embeddings, including a preview of text the contents of the first five documents and the distribution of their embeddings.
