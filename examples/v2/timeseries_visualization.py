# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "pandas",
#    "plotly",
#    "requests",
# ]
# ///

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="timeseries-visualization",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_uv_script(__file__, name="timeseries-viz-image", python_version=(3, 12), pre=True)
)


@env.task(report=True)
async def download_and_visualize_timeseries() -> str:
    """
    Downloads the timeseries dataset from GitHub and creates a beautiful
    Plotly visualization with multiple charts.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import requests
    from io import StringIO

    # Download the dataset
    url = "https://raw.githubusercontent.com/plotly/datasets/master/timeseries.csv"
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the CSV data
    df = pd.read_csv(StringIO(response.text))
    
    # Global font styling
    font_family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif"
    
    # Log initial report header with clean styling
    await flyte.report.log.aio(f"""
    <style>
        .report-container {{
            font-family: {font_family};
            color: #1a1a2e;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .report-title {{
            font-size: 2rem;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}
        .report-subtitle {{
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }}
        .report-subtitle a {{
            color: #3b82f6;
            text-decoration: none;
        }}
        .report-subtitle a:hover {{
            text-decoration: underline;
        }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #1a1a2e;
            margin-top: 2rem;
            margin-bottom: 1rem;
            letter-spacing: -0.01em;
        }}
        .info-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .info-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }}
        .info-list li:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            color: #64748b;
            font-weight: 500;
        }}
        .info-value {{
            color: #1a1a2e;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 1rem;
            font-size: 0.875rem;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stats-table th {{
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            font-weight: 500;
            padding: 0.875rem 1rem;
            text-align: left;
        }}
        .stats-table th:not(:first-child) {{
            text-align: right;
        }}
        .stats-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .stats-table td:not(:first-child) {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .stats-table tr:hover {{
            background-color: #f1f5f9;
        }}
        .stats-table td:first-child {{
            font-weight: 500;
            color: #475569;
        }}
        .success-card {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1px solid #a7f3d0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }}
        .success-card h3 {{
            color: #065f46;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
        }}
        .success-card p {{
            color: #047857;
            font-size: 0.9rem;
            margin: 0 0 0.75rem 0;
        }}
        .success-card ul {{
            margin: 0;
            padding-left: 1.25rem;
            color: #047857;
            font-size: 0.875rem;
        }}
        .success-card li {{
            padding: 0.25rem 0;
        }}
        hr {{
            border: none;
            height: 1px;
            background: #e2e8f0;
            margin: 1.5rem 0;
        }}
    </style>
    <div class="report-container">
        <h1 class="report-title">Time Series Data Visualization</h1>
        <p class="report-subtitle">Dataset: <a href="https://github.com/plotly/datasets/blob/master/timeseries.csv">plotly/datasets/timeseries.csv</a></p>
        <hr>
    """, do_flush=True)
    
    # Show dataset info
    await flyte.report.log.aio(f"""
        <h2 class="section-title">Dataset Overview</h2>
        <ul class="info-list">
            <li><span class="info-label">Number of rows:</span> <span class="info-value">{len(df)}</span></li>
            <li><span class="info-label">Columns:</span> <span class="info-value">{', '.join(df.columns.tolist())}</span></li>
        </ul>
    """, do_flush=True)
    
    # Create a comprehensive visualization with subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Time Series Values Over Time",
            "Rolling Statistics (7-day Window)",
            "Distribution of Values"
        ),
        vertical_spacing=0.12,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "histogram"}]]
    )
    
    # Get numeric columns (excluding date)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    date_col = df.columns[0]  # Assume first column is date
    
    # Color palette for beautiful visualization
    colors = ['#3b82f6', '#ef4444', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899']
    
    # Plot 1: Line chart of all time series
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                name=col,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Plot 2: Rolling mean and std for each series
    for i, col in enumerate(numeric_cols):
        rolling_mean = df[col].rolling(window=7, min_periods=1).mean()
        rolling_std = df[col].rolling(window=7, min_periods=1).std()
        
        # Rolling mean
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=rolling_mean,
                name=f"{col} (7-day avg)",
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=3),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Confidence band (mean ± std)
        fig.add_trace(
            go.Scatter(
                x=df[date_col].tolist() + df[date_col].tolist()[::-1],
                y=(rolling_mean + rolling_std).tolist() + (rolling_mean - rolling_std).tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name=f"{col} ±1 std",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 3: Histogram distribution
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                hovertemplate=f"<b>{col}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ),
            row=3, col=1
        )
    
    # Update layout with clean font styling
    fig.update_layout(
        title=dict(
            text="<b>Time Series Analysis Dashboard</b>",
            font=dict(size=20, color='#1a1a2e', family=font_family),
            x=0.5,
            xanchor='center'
        ),
        height=1100,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family=font_family, color='#475569')
        ),
        font=dict(family=font_family, color='#1a1a2e'),
        paper_bgcolor='white',
        plot_bgcolor='#fafafa',
        margin=dict(b=120, t=80)
    )
    
    # Update subplot title fonts
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='#1a1a2e', family=font_family)
    
    # Update axes with clean styling
    fig.update_xaxes(
        title_font=dict(size=12, color='#64748b', family=font_family),
        tickfont=dict(size=10, color='#64748b', family=font_family),
        gridcolor='#e2e8f0',
        linecolor='#e2e8f0'
    )
    fig.update_yaxes(
        title_font=dict(size=12, color='#64748b', family=font_family),
        tickfont=dict(size=10, color='#64748b', family=font_family),
        gridcolor='#e2e8f0',
        linecolor='#e2e8f0'
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=3, col=1)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    # Add the visualization to the report
    await flyte.report.log.aio(fig.to_html(full_html=False, include_plotlyjs="cdn"), do_flush=True)
    
    # Add summary statistics table with clean styling
    summary_html = '<h2 class="section-title">Summary Statistics</h2><table class="stats-table">'
    summary_html += "<thead><tr><th>Metric</th>"
    for col in numeric_cols:
        summary_html += f"<th>{col}</th>"
    summary_html += "</tr></thead><tbody>"
    
    stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    stat_names = ['Mean', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max']
    desc = df[numeric_cols].describe()
    
    for stat, name in zip(stats, stat_names):
        summary_html += f"<tr><td>{name}</td>"
        for col in numeric_cols:
            summary_html += f"<td>{desc.loc[stat, col]:.4f}</td>"
        summary_html += "</tr>"
    
    summary_html += "</tbody></table>"
    await flyte.report.log.aio(summary_html, do_flush=True)
    
    # Final completion message with clean styling
    await flyte.report.log.aio("""
        <div class="success-card">
            <h3>Visualization Complete</h3>
            <p>The time series data has been successfully downloaded and visualized with:</p>
            <ul>
                <li>Interactive line charts showing all time series</li>
                <li>Rolling statistics with confidence bands</li>
                <li>Distribution histograms</li>
                <li>Comprehensive summary statistics</li>
            </ul>
        </div>
    </div>
    """, do_flush=True)
    
    return f"Successfully visualized {len(df)} data points from timeseries dataset"


@env.task
async def main() -> str:
    result = await download_and_visualize_timeseries()
    return result


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
