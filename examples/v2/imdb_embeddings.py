# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "unionai-reuse",
#    "datasets",
#    "transformers>=4.45.0",
#    "torch",
#    "numpy",
#    "plotly",
#    "pandas",
#    "scikit-learn",
# ]
# ///

"""
IMDB Dataset Embedding Example

This script demonstrates:
1. Loading the IMDB dataset from HuggingFace ("scikit-learn/imdb")
2. Embedding the "review" column using the "answerdotai/ModernBERT-base" model
3. Driver-worker pattern: CPU driver orchestrates, GPU worker computes embeddings
4. Saving embeddings to a JSON file using flyte.io.File
5. Visualizing results using flyte.report:
   - Preview of text contents of first 5 documents
   - Distribution of embeddings (PCA or t-SNE reduced to 2D)
"""

import asyncio
import json
import tempfile
from functools import lru_cache
from typing import List

import flyte
import flyte.report
from flyte.io import File

# Base image with dependencies
base_image = (
    flyte.Image
    .from_uv_script(
        __file__,
        name="imdb-embeddings",
        registry="ghcr.io/flyteorg",
        platform=("linux/amd64",),
        python_version=(3, 11),
        pre=True,
    )
    .with_apt_packages("ca-certificates")
)

# GPU Worker Environment - runs on T4 GPU for embedding computation
gpu_worker_env = flyte.TaskEnvironment(
    name="imdb_gpu_worker",
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
    image=base_image,
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=1,
        idle_ttl=300,
    ),
)

# CPU Driver Environment - orchestrates the workflow
env = flyte.TaskEnvironment(
    name="imdb_cpu_driver",
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    image=base_image,
    depends_on=[gpu_worker_env],
)


@lru_cache(maxsize=1)
def get_model_and_tokenizer(model_name: str = "answerdotai/ModernBERT-base"):
    """Lazily load and cache the model and tokenizer."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model {model_name}...")
    # ModernBERT requires flash_attention_2 by default, but T4 GPUs don't support it
    # Use attn_implementation="eager" to fall back to standard attention
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",  # Disable flash attention for T4 compatibility
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
    else:
        print("Model loaded on CPU")

    model.eval()
    return model, tokenizer


def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings."""
    import torch

    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sentiment_to_int(sentiment) -> int:
    """Convert sentiment string or int to integer label."""
    if isinstance(sentiment, int):
        return sentiment
    if isinstance(sentiment, str):
        return 1 if sentiment.lower() == "positive" else 0
    return int(sentiment)


def sentiment_to_label(sentiment) -> str:
    """Convert sentiment to display label."""
    if isinstance(sentiment, int):
        return "Positive" if sentiment == 1 else "Negative"
    if isinstance(sentiment, str):
        return "Positive" if sentiment.lower() == "positive" else "Negative"
    return "Unknown"


def is_positive(sentiment) -> bool:
    """Check if sentiment is positive."""
    if isinstance(sentiment, int):
        return sentiment == 1
    if isinstance(sentiment, str):
        return sentiment.lower() == "positive"
    return False


@gpu_worker_env.task
async def embed_batch(
    texts: List[str],
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 16,
) -> List[List[float]]:
    """
    Embed a batch of texts using the specified model on GPU.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name
        batch_size: Mini-batch size for processing

    Returns:
        List of embedding vectors (as lists of floats)
    """
    import torch

    model, tokenizer = get_model_and_tokenizer(model_name)
    all_embeddings = []

    # Process in mini-batches
    for i in range(0, len(texts), batch_size):
        mini_batch = texts[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            mini_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)

        # Apply mean pooling
        embeddings = mean_pooling(outputs, encoded["attention_mask"])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list and append
        all_embeddings.extend(embeddings.cpu().tolist())

        print(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    return all_embeddings


@env.task(report=True)
async def embed_imdb_dataset(
    num_samples: int = 100,
    model_name: str = "answerdotai/ModernBERT-base",
    chunk_size: int = 50,
) -> File:
    """
    Main driver task that orchestrates embedding the IMDB dataset.

    Args:
        num_samples: Number of samples to process from the dataset
        model_name: HuggingFace model name for embeddings
        chunk_size: Number of texts per GPU worker task

    Returns:
        File containing the embeddings in JSON format
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datasets import load_dataset
    from sklearn.decomposition import PCA

    # Log initial status
    await flyte.report.log.aio("""
    <h1>IMDB Dataset Embedding Pipeline</h1>
    <p>Loading dataset and computing embeddings using ModernBERT...</p>
    """, do_flush=True)

    # Load the IMDB dataset
    print(f"Loading IMDB dataset (first {num_samples} samples)...")
    dataset = load_dataset("scikit-learn/imdb", split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    reviews = dataset["review"]
    # scikit-learn/imdb uses 'sentiment' column with string values ("positive"/"negative")
    raw_labels = dataset["sentiment"]

    print(f"Loaded {len(reviews)} reviews")
    print(f"Sample sentiment value: {raw_labels[0]} (type: {type(raw_labels[0])})")

    # Log dataset info
    await flyte.report.log.aio(f"""
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Dataset Information</h3>
        <ul>
            <li><strong>Dataset:</strong> scikit-learn/imdb</li>
            <li><strong>Samples:</strong> {len(reviews)}</li>
            <li><strong>Model:</strong> {model_name}</li>
        </ul>
    </div>
    """, do_flush=True)

    # Preview first 5 documents
    await flyte.report.log.aio("""
    <h2>Preview: First 5 Documents</h2>
    <div style="max-height: 400px; overflow-y: auto;">
    """, do_flush=True)

    for i in range(min(5, len(reviews))):
        sentiment_label = sentiment_to_label(raw_labels[i])
        is_pos = is_positive(raw_labels[i])
        preview_text = reviews[i][:500] + "..." if len(reviews[i]) > 500 else reviews[i]
        await flyte.report.log.aio(f"""
        <div style="background: {'#e8f5e9' if is_pos else '#ffebee'}; 
                    padding: 15px; margin: 10px 0; border-radius: 8px; 
                    border-left: 4px solid {'#4caf50' if is_pos else '#f44336'};">
            <strong>Document {i + 1}</strong> - <span style="color: {'#2e7d32' if is_pos else '#c62828'};">{sentiment_label}</span>
            <p style="font-size: 14px; color: #333; margin-top: 10px;">{preview_text}</p>
        </div>
        """, do_flush=True)

    await flyte.report.log.aio("</div>", do_flush=True)

    # Split reviews into chunks for parallel processing
    chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]

    await flyte.report.log.aio(f"""
    <h2>Computing Embeddings</h2>
    <p>Processing {len(chunks)} chunks on GPU workers...</p>
    """, do_flush=True)

    # Process chunks in parallel using GPU workers
    all_embeddings = []
    with flyte.group("parallel-embedding"):
        embedding_tasks = [
            embed_batch(list(chunk), model_name)
            for chunk in chunks
        ]
        chunk_results = await asyncio.gather(*embedding_tasks)

    # Flatten results
    for chunk_embeddings in chunk_results:
        all_embeddings.extend(chunk_embeddings)

    print(f"Generated {len(all_embeddings)} embeddings")

    await flyte.report.log.aio(f"""
    <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Embedding Complete!</h3>
        <p>Generated {len(all_embeddings)} embeddings with dimension {len(all_embeddings[0])}</p>
    </div>
    """, do_flush=True)

    # Visualize embedding distribution
    await flyte.report.log.aio("""
    <h2>Embedding Distribution Visualization</h2>
    <p>Using PCA to reduce embeddings to 2D for visualization...</p>
    """, do_flush=True)

    # Reduce dimensionality for visualization
    embeddings_array = np.array(all_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)

    # Create scatter plot
    df_viz = pd.DataFrame({
        "PC1": embeddings_2d[:, 0],
        "PC2": embeddings_2d[:, 1],
        "Sentiment": [sentiment_to_label(label) for label in raw_labels],
        "Text Preview": [r[:100] + "..." if len(r) > 100 else r for r in reviews],
    })

    fig_scatter = px.scatter(
        df_viz,
        x="PC1",
        y="PC2",
        color="Sentiment",
        color_discrete_map={"Positive": "#4caf50", "Negative": "#f44336"},
        hover_data=["Text Preview"],
        title="IMDB Reviews Embedding Space (PCA)",
    )
    fig_scatter.update_layout(
        width=800,
        height=600,
        template="plotly_white",
    )

    await flyte.report.log.aio(
        fig_scatter.to_html(full_html=False, include_plotlyjs="cdn"),
        do_flush=True
    )

    # Create histogram of embedding norms
    norms = np.linalg.norm(embeddings_array, axis=1)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=norms,
        nbinsx=30,
        marker_color="#2196f3",
        name="Embedding Norms",
    ))
    fig_hist.update_layout(
        title="Distribution of Embedding Norms",
        xaxis_title="L2 Norm",
        yaxis_title="Count",
        width=800,
        height=400,
        template="plotly_white",
    )

    await flyte.report.log.aio(
        fig_hist.to_html(full_html=False, include_plotlyjs="cdn"),
        do_flush=True
    )

    # Create embedding component distribution
    fig_components = go.Figure()
    for i in range(min(5, embeddings_array.shape[1])):
        fig_components.add_trace(go.Box(
            y=embeddings_array[:, i],
            name=f"Dim {i + 1}",
        ))
    fig_components.update_layout(
        title="Distribution of First 5 Embedding Dimensions",
        yaxis_title="Value",
        width=800,
        height=400,
        template="plotly_white",
    )

    await flyte.report.log.aio(
        fig_components.to_html(full_html=False, include_plotlyjs="cdn"),
        do_flush=True
    )

    # Save embeddings to JSON file
    output_data = {
        "model": model_name,
        "num_samples": len(all_embeddings),
        "embedding_dim": len(all_embeddings[0]),
        "embeddings": [
            {
                "index": i,
                "text_preview": reviews[i][:200],
                "label": sentiment_to_int(raw_labels[i]),
                "sentiment": sentiment_to_label(raw_labels[i]),
                "embedding": emb,
            }
            for i, emb in enumerate(all_embeddings)
        ],
    }

    temp_dir = tempfile.mkdtemp()
    output_path = f"{temp_dir}/imdb_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"Saved embeddings to {output_path}")

    await flyte.report.log.aio(f"""
    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h3>Output Saved</h3>
        <p>Embeddings saved to JSON file with the following structure:</p>
        <ul>
            <li><strong>model:</strong> {model_name}</li>
            <li><strong>num_samples:</strong> {len(all_embeddings)}</li>
            <li><strong>embedding_dim:</strong> {len(all_embeddings[0])}</li>
        </ul>
    </div>
    """, do_flush=True)

    return await File.from_local(output_path)


@env.task
async def main(
    num_samples: int = 100,
    model_name: str = "answerdotai/ModernBERT-base",
    chunk_size: int = 50,
) -> File:
    """
    Entry point for the IMDB embedding pipeline.

    Args:
        num_samples: Number of samples to process from the dataset
        model_name: HuggingFace model name for embeddings
        chunk_size: Number of texts per GPU worker task

    Returns:
        File containing the embeddings in JSON format
    """
    return await embed_imdb_dataset(
        num_samples=num_samples,
        model_name=model_name,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    import argparse
    import os

    from flyte.remote import auth_metadata

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    flyte.init_passthrough(
        project=os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT"),
        domain=os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN"),
    )
    with auth_metadata(("authorization", os.environ["FLYTE_PASSTHROUGH_API_KEY"])):    
        if args.build:
            uri = flyte.build(env.image, wait=False)
            print(f"build run url: {uri}")
        else:
            run = flyte.with_runcontext(mode="remote").run(main)
            print(run.url)

    # Run with:
    # uv run --prerelease=allow examples/v2/imdb_embeddings.py
    # uv run --prerelease=allow examples/v2/imdb_embeddings.py --build  # to build image first
