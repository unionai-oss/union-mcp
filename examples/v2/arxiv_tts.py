# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "arxiv>=2.0.0",
#    "transformers>=4.45.0",
#    "torch>=2.0.0",
#    "torchaudio",
#    "soundfile",
#    "accelerate",
#    "einops",
#    "scipy",
#    "plotly",
#    "pandas",
# ]
# ///

"""
ArXiv Article to Audio Generator using VibeVoice

This script demonstrates:
1. Fetching the 5 most recently published articles from arXiv
2. Generating audio files from article text using microsoft/VibeVoice-1.5B
3. Driver-worker pattern: CPU driver orchestrates, GPU worker generates audio
4. Saving audio files using flyte.io.File
5. Visualizing results using flyte.report with text previews and embedded audio
"""

import asyncio
import json
import tempfile
import base64
from functools import lru_cache
from typing import List, Dict, Any
from dataclasses import dataclass

import flyte
import flyte.report
from flyte.io import File

# Base image with dependencies
base_image = (
    flyte.Image
    .from_uv_script(
        __file__,
        name="arxiv-vibevoice-tts",
        registry="ghcr.io/flyteorg",
        platform=("linux/amd64",),
        python_version=(3, 11),
        pre=True,
    )
    .with_apt_packages("ca-certificates", "ffmpeg")
)

# GPU Worker Environment - runs on T4 GPU for TTS generation
gpu_worker_env = flyte.TaskEnvironment(
    name="arxiv_gpu_worker",
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu="T4:1"),
    image=base_image,
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=1,
        idle_ttl=300,
    ),
)

# CPU Driver Environment - orchestrates the workflow
cpu_driver_env = flyte.TaskEnvironment(
    name="arxiv_cpu_driver",
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    image=base_image,
    depends_on=[gpu_worker_env],
)


@dataclass
class ArticleInfo:
    """Dataclass to hold article information."""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published: str
    pdf_url: str


@lru_cache(maxsize=1)
def get_tts_model():
    """Lazily load and cache the VibeVoice TTS model and processor."""
    import torch
    from transformers import AutoModelForTextToWaveform, AutoProcessor

    model_name = "microsoft/VibeVoice-1.5B"
    print(f"Loading VibeVoice model: {model_name}...")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
    else:
        print("Model loaded on CPU")

    return model, processor


def truncate_text_for_tts(text: str, max_chars: int = 500) -> str:
    """Truncate text to a reasonable length for TTS."""
    if len(text) <= max_chars:
        return text
    # Find a sentence boundary near the limit
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated + "..."


@gpu_worker_env.task
async def generate_audio_for_article(
    article_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate audio from article text using VibeVoice on GPU.

    Args:
        article_data: Dictionary containing article information

    Returns:
        Dictionary with article info and base64-encoded audio data
    """
    import torch
    import soundfile as sf
    import io
    import base64

    title = article_data["title"]
    abstract = article_data["abstract"]
    arxiv_id = article_data["arxiv_id"]

    print(f"Generating audio for article: {arxiv_id}")

    # Prepare text for TTS - combine title and truncated abstract
    tts_text = f"Title: {title}. Abstract: {truncate_text_for_tts(abstract, 400)}"

    try:
        model, processor = get_tts_model()

        # Prepare input for the model
        inputs = processor(
            text=tts_text,
            return_tensors="pt",
        )

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate audio
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048)

        # Get audio waveform
        audio_waveform = outputs.cpu().numpy().squeeze()

        # Get sample rate from processor config (typically 24000 for VibeVoice)
        sample_rate = getattr(processor, 'sampling_rate', 24000)

        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_waveform, sample_rate, format='WAV')
        audio_buffer.seek(0)

        # Encode as base64 for transport
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        print(f"Successfully generated audio for {arxiv_id}")

        return {
            **article_data,
            "audio_base64": audio_base64,
            "audio_duration_seconds": len(audio_waveform) / sample_rate,
            "sample_rate": sample_rate,
            "tts_text_used": tts_text,
            "status": "success",
        }

    except Exception as e:
        print(f"Error generating audio for {arxiv_id}: {str(e)}")
        return {
            **article_data,
            "audio_base64": None,
            "audio_duration_seconds": 0,
            "sample_rate": 24000,
            "tts_text_used": tts_text,
            "status": f"error: {str(e)}",
        }


@cpu_driver_env.task(report=True)
async def fetch_and_process_articles(
    num_articles: int = 5,
    search_query: str = "cat:cs.AI",
) -> File:
    """
    Main driver task that orchestrates fetching arXiv articles and generating audio.

    Args:
        num_articles: Number of recent articles to fetch
        search_query: arXiv search query (default: AI papers)

    Returns:
        File containing the results in JSON format
    """
    import arxiv
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    # Log initial status
    await flyte.report.log.aio("""
    <h1>ArXiv Article to Audio Generator</h1>
    <p>Fetching recent articles and generating audio using VibeVoice TTS...</p>
    """, do_flush=True)

    # Fetch articles from arXiv
    print(f"Fetching {num_articles} recent articles from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=num_articles,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    articles = []
    for result in client.results(search):
        articles.append({
            "title": result.title,
            "authors": [author.name for author in result.authors[:3]],  # First 3 authors
            "abstract": result.summary,
            "arxiv_id": result.entry_id.split("/")[-1],
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
        })

    print(f"Fetched {len(articles)} articles")

    # Log article info
    await flyte.report.log.aio(f"""
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Fetched Articles</h3>
        <ul>
            <li><strong>Query:</strong> {search_query}</li>
            <li><strong>Count:</strong> {len(articles)}</li>
            <li><strong>Model:</strong> microsoft/VibeVoice-1.5B</li>
        </ul>
    </div>
    """, do_flush=True)

    # Preview fetched articles
    await flyte.report.log.aio("""
    <h2>Article Previews</h2>
    <div style="max-height: 500px; overflow-y: auto;">
    """, do_flush=True)

    for i, article in enumerate(articles):
        preview_abstract = article["abstract"][:300] + "..." if len(article["abstract"]) > 300 else article["abstract"]
        authors_str = ", ".join(article["authors"])
        await flyte.report.log.aio(f"""
        <div style="background: #e8f5e9; padding: 15px; margin: 10px 0; border-radius: 8px; 
                    border-left: 4px solid #4caf50;">
            <strong>Article {i + 1}: {article['title']}</strong>
            <p style="font-size: 12px; color: #666;">
                <strong>Authors:</strong> {authors_str}<br>
                <strong>arXiv ID:</strong> {article['arxiv_id']}<br>
                <strong>Published:</strong> {article['published']}
            </p>
            <p style="font-size: 14px; color: #333; margin-top: 10px;">{preview_abstract}</p>
        </div>
        """, do_flush=True)

    await flyte.report.log.aio("</div>", do_flush=True)

    # Generate audio for each article using GPU workers
    await flyte.report.log.aio(f"""
    <h2>Generating Audio</h2>
    <p>Processing {len(articles)} articles on GPU workers...</p>
    """, do_flush=True)

    # Process articles in parallel using GPU workers
    results = []
    with flyte.group("parallel-audio-generation"):
        audio_tasks = [
            generate_audio_for_article(article)
            for article in articles
        ]
        results = await asyncio.gather(*audio_tasks)

    print(f"Generated audio for {len(results)} articles")

    # Count successes
    successful = [r for r in results if r["status"] == "success"]
    await flyte.report.log.aio(f"""
    <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Audio Generation Complete!</h3>
        <p>Successfully generated audio for {len(successful)}/{len(results)} articles</p>
    </div>
    """, do_flush=True)

    # Display audio results with embedded players
    await flyte.report.log.aio("""
    <h2>Generated Audio Files</h2>
    """, do_flush=True)

    for i, result in enumerate(results):
        if result["status"] == "success" and result["audio_base64"]:
            # Create embedded audio player
            audio_html = f"""
            <div style="background: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 8px; 
                        border-left: 4px solid #ff9800;">
                <h4>Article {i + 1}: {result['title']}</h4>
                <p style="font-size: 12px; color: #666;">
                    <strong>arXiv ID:</strong> {result['arxiv_id']}<br>
                    <strong>Duration:</strong> {result['audio_duration_seconds']:.2f} seconds<br>
                    <strong>Sample Rate:</strong> {result['sample_rate']} Hz
                </p>
                <p style="font-size: 13px; color: #444; background: #f5f5f5; padding: 10px; border-radius: 4px;">
                    <strong>Text used for TTS:</strong><br>{result['tts_text_used'][:200]}...
                </p>
                <audio controls style="width: 100%; margin-top: 10px;">
                    <source src="data:audio/wav;base64,{result['audio_base64']}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            await flyte.report.log.aio(audio_html, do_flush=True)
        else:
            await flyte.report.log.aio(f"""
            <div style="background: #ffebee; padding: 15px; margin: 10px 0; border-radius: 8px; 
                        border-left: 4px solid #f44336;">
                <h4>Article {i + 1}: {result['title']}</h4>
                <p style="color: #c62828;"><strong>Status:</strong> {result['status']}</p>
            </div>
            """, do_flush=True)

    # Create visualization of audio durations
    if successful:
        df_viz = pd.DataFrame([
            {
                "Article": f"Art. {i+1}",
                "Duration (s)": r["audio_duration_seconds"],
                "Title": r["title"][:30] + "..." if len(r["title"]) > 30 else r["title"],
            }
            for i, r in enumerate(results) if r["status"] == "success"
        ])

        fig = px.bar(
            df_viz,
            x="Article",
            y="Duration (s)",
            color="Duration (s)",
            color_continuous_scale="Viridis",
            title="Audio Duration by Article",
            hover_data=["Title"],
        )
        fig.update_layout(
            width=700,
            height=400,
            template="plotly_white",
        )

        await flyte.report.log.aio("""
        <h2>Audio Duration Visualization</h2>
        """, do_flush=True)
        await flyte.report.log.aio(
            fig.to_html(full_html=False, include_plotlyjs="cdn"),
            do_flush=True
        )

    # Save results to JSON file (excluding large base64 audio for the metadata file)
    output_data = {
        "query": search_query,
        "num_articles": len(articles),
        "generation_timestamp": datetime.now().isoformat(),
        "model": "microsoft/VibeVoice-1.5B",
        "articles": [
            {
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "authors": r["authors"],
                "abstract": r["abstract"],
                "published": r["published"],
                "pdf_url": r["pdf_url"],
                "tts_text_used": r.get("tts_text_used", ""),
                "audio_duration_seconds": r.get("audio_duration_seconds", 0),
                "sample_rate": r.get("sample_rate", 24000),
                "status": r["status"],
                "audio_base64": r.get("audio_base64"),  # Include audio data
            }
            for r in results
        ],
    }

    temp_dir = tempfile.mkdtemp()
    output_path = f"{temp_dir}/arxiv_audio_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved results to {output_path}")

    await flyte.report.log.aio(f"""
    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h3>Output Saved</h3>
        <p>Results saved to JSON file with the following structure:</p>
        <ul>
            <li><strong>Query:</strong> {search_query}</li>
            <li><strong>Articles processed:</strong> {len(results)}</li>
            <li><strong>Successful audio generations:</strong> {len(successful)}</li>
            <li><strong>Model:</strong> microsoft/VibeVoice-1.5B</li>
        </ul>
    </div>
    """, do_flush=True)

    return await File.from_local(output_path)


@cpu_driver_env.task
async def main(
    num_articles: int = 5,
    search_query: str = "cat:cs.AI",
) -> File:
    """
    Entry point for the arXiv article to audio pipeline.

    Args:
        num_articles: Number of recent articles to fetch
        search_query: arXiv search query

    Returns:
        File containing the results in JSON format
    """
    return await fetch_and_process_articles(
        num_articles=num_articles,
        search_query=search_query,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--num-articles", type=int, default=5)
    parser.add_argument("--search-query", type=str, default="cat:cs.AI")
    args = parser.parse_args()

    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=os.environ["FLYTE_ORG"],
        project=os.environ["FLYTE_PROJECT"],
        domain=os.environ["FLYTE_DOMAIN"],
        image_builder="remote",
    )

    if args.build:
        uri = flyte.build(base_image, wait=False)
        print(f"build run url: {uri}")
    else:
        # Run the task in remote mode
        run = flyte.with_runcontext(mode="remote").run(
            main,
            num_articles=args.num_articles,
            search_query=args.search_query,
        )
        print(run.url)
