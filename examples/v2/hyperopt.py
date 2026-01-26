# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "pandas",
#    "scikit-learn",
#    "plotly",
#    "numpy",
#    "pyarrow",
#    "pydantic",
# ]
# ///

import asyncio
import random

from pydantic import BaseModel

import pandas as pd

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="rf_hyperopt",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="flyte",
        registry="ghcr.io/flyteorg",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


class HyperParams(BaseModel):
    n_estimators: int
    max_depth: int  # use 0 to represent None
    min_samples_split: int
    min_samples_leaf: int
    max_features: str


class TrialResult(BaseModel):
    params: HyperParams
    f1_score: float
    accuracy: float
    train_f1: float


@env.task
async def load_penguins_data() -> pd.DataFrame:
    """Load and preprocess the penguins dataset."""
    from sklearn.preprocessing import LabelEncoder

    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)
    df = df.dropna()

    le_species = LabelEncoder()
    le_island = LabelEncoder()
    le_sex = LabelEncoder()

    df["species_encoded"] = le_species.fit_transform(df["species"])
    df["island_encoded"] = le_island.fit_transform(df["island"])
    df["sex_encoded"] = le_sex.fit_transform(df["sex"])

    print(f"Loaded {len(df)} samples")
    print(f"Target classes: {list(le_species.classes_)}")

    return df


@env.task
async def run_trial(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: HyperParams,
    trial_id: int,
) -> TrialResult:
    """Run a single hyperparameter trial."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score

    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm",
                    "body_mass_g", "island_encoded", "sex_encoded"]

    X_train = train_df[feature_cols].values
    y_train = train_df["species_encoded"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["species_encoded"].values

    max_depth = params.max_depth if params.max_depth > 0 else None

    model = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")

    print(f"Trial {trial_id}: F1={f1:.4f}, Accuracy={accuracy:.4f}")

    return TrialResult(params=params, f1_score=f1, accuracy=accuracy, train_f1=train_f1)


@env.task(report=True)
async def hyperparameter_optimization(n_trials: int = 20) -> TrialResult:
    """Run hyperparameter optimization and generate a report. Returns the best result."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.model_selection import train_test_split

    df = await load_penguins_data()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["species_encoded"]
    )

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    param_space = {
        "n_estimators": [10, 25, 50, 100, 150, 200],
        "max_depth": [0, 3, 5, 7, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
    }

    random.seed(42)
    trial_params = []
    for i in range(n_trials):
        params = HyperParams(
            n_estimators=random.choice(param_space["n_estimators"]),
            max_depth=random.choice(param_space["max_depth"]),
            min_samples_split=random.choice(param_space["min_samples_split"]),
            min_samples_leaf=random.choice(param_space["min_samples_leaf"]),
            max_features=random.choice(param_space["max_features"]),
        )
        trial_params.append(params)

    trial_tasks = []
    with flyte.group("hyperparameter-trials"):
        for i, params in enumerate(trial_params):
            task = run_trial(train_df, test_df, params, i)
            trial_tasks.append(task)
        results = await asyncio.gather(*trial_tasks)

    results_sorted = sorted(results, key=lambda x: x.f1_score, reverse=True)
    best_result = results_sorted[0]

    print(f"\nBest trial: F1={best_result.f1_score:.4f}")

    f1_scores = [r.f1_score for r in results]
    train_f1s = [r.train_f1 for r in results]
    n_estimators_list = [r.params.n_estimators for r in results]
    max_depths = [r.params.max_depth for r in results]

    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "F1 Score Distribution",
            "F1 Score vs n_estimators",
            "Train vs Test F1 Score",
            "F1 Score vs max_depth"
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    fig.add_trace(
        go.Histogram(x=f1_scores, nbinsx=15, name="F1 Score", marker_color="#1f77b4"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=n_estimators_list, y=f1_scores,
            mode="markers",
            marker=dict(size=10, color=f1_scores, colorscale="Viridis", showscale=True),
            hovertemplate="n_estimators: %{x}<br>F1: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=train_f1s, y=f1_scores,
            mode="markers",
            marker=dict(size=10, color="#2ca02c"),
            hovertemplate="Train F1: %{x:.4f}<br>Test F1: %{y:.4f}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[min(train_f1s), 1], y=[min(train_f1s), 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=max_depths, y=f1_scores,
            mode="markers",
            marker=dict(size=10, color=f1_scores, colorscale="Plasma"),
            hovertemplate="max_depth: %{x}<br>F1: %{y:.4f}<extra></extra>"
        ),
        row=2, col=2
    )

    fig.update_layout(height=700, title_text="Hyperparameter Optimization Results", showlegend=False)
    fig.update_xaxes(title_text="F1 Score", row=1, col=1)
    fig.update_xaxes(title_text="n_estimators", row=1, col=2)
    fig.update_xaxes(title_text="Train F1 Score", row=2, col=1)
    fig.update_xaxes(title_text="max_depth (0=None)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2)
    fig.update_yaxes(title_text="Test F1 Score", row=2, col=1)
    fig.update_yaxes(title_text="F1 Score", row=2, col=2)

    chart_html = fig.to_html(include_plotlyjs=True, full_html=False)

    fig_parallel = go.Figure(data=
        go.Parcoords(
            line=dict(color=f1_scores, colorscale="Viridis", showscale=True, cmin=min(f1_scores), cmax=max(f1_scores)),
            dimensions=[
                dict(label="n_estimators", values=n_estimators_list),
                dict(label="max_depth", values=max_depths),
                dict(label="min_samples_split", values=[r.params.min_samples_split for r in results]),
                dict(label="min_samples_leaf", values=[r.params.min_samples_leaf for r in results]),
                dict(label="F1 Score", values=f1_scores),
            ]
        )
    )
    fig_parallel.update_layout(title="Hyperparameter Parallel Coordinates", height=400)
    parallel_html = fig_parallel.to_html(include_plotlyjs=False, full_html=False)

    # Build results table
    table_rows = ""
    for i, r in enumerate(results_sorted[:10]):
        depth_str = str(r.params.max_depth) if r.params.max_depth > 0 else "None"
        table_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{r.f1_score:.4f}</strong></td>
            <td>{r.accuracy:.4f}</td>
            <td>{r.params.n_estimators}</td>
            <td>{depth_str}</td>
            <td>{r.params.min_samples_split}</td>
            <td>{r.params.min_samples_leaf}</td>
            <td>{r.params.max_features}</td>
        </tr>
        """

    best_depth_str = str(best_result.params.max_depth) if best_result.params.max_depth > 0 else "None"

    # Log to flyte report
    main_tab = flyte.report.get_tab("Results")
    main_tab.log("<h1>Random Forest Hyperparameter Optimization</h1>")
    main_tab.log(f"<p>Dataset: Palmer Penguins | Metric: Weighted F1 Score | Trials: {n_trials}</p>")
    main_tab.log(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Best Configuration</h3>
        <p><strong>F1 Score:</strong> {best_result.f1_score:.4f} | <strong>Accuracy:</strong> {best_result.accuracy:.4f}</p>
        <p><strong>n_estimators:</strong> {best_result.params.n_estimators} | <strong>max_depth:</strong> {best_depth_str} |
           <strong>min_samples_split:</strong> {best_result.params.min_samples_split} | <strong>min_samples_leaf:</strong> {best_result.params.min_samples_leaf}</p>
    </div>
    """)
    main_tab.log(f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
            <div style="font-size: 20px; font-weight: bold; color: #1f77b4;">{np.mean(f1_scores):.4f}</div>
            <div>Mean F1</div>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
            <div style="font-size: 20px; font-weight: bold; color: #1f77b4;">{np.std(f1_scores):.4f}</div>
            <div>Std Dev</div>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1;">
            <div style="font-size: 20px; font-weight: bold; color: #1f77b4;">{np.min(f1_scores):.4f} - {np.max(f1_scores):.4f}</div>
            <div>Range</div>
        </div>
    </div>
    """)
    main_tab.log(chart_html)

    charts_tab = flyte.report.get_tab("Parallel Coords")
    charts_tab.log(parallel_html)

    table_tab = flyte.report.get_tab("Top 10")
    table_tab.log(f"""
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #1f77b4; color: white;">
                <th style="padding: 10px;">Rank</th>
                <th style="padding: 10px;">F1 Score</th>
                <th style="padding: 10px;">Accuracy</th>
                <th style="padding: 10px;">n_estimators</th>
                <th style="padding: 10px;">max_depth</th>
                <th style="padding: 10px;">min_samples_split</th>
                <th style="padding: 10px;">min_samples_leaf</th>
                <th style="padding: 10px;">max_features</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """)

    await flyte.report.flush.aio()

    return best_result


@env.task
async def main() -> TrialResult:
    """Main entry point. Returns the best hyperparameter configuration."""
    best_result = await hyperparameter_optimization(n_trials=20)
    return best_result


if __name__ == "__main__":
    import os

    flyte.init(
        api_key=os.environ["FLYTE_API_KEY"],
        org=os.environ["FLYTE_ORG"],
        project=os.environ["FLYTE_PROJECT"],
        domain=os.environ["FLYTE_DOMAIN"],
        image_builder="remote",
    )
    run = flyte.with_runcontext(mode="remote").run(main)
    print(run.url)

    # Run with:
    # uv run --prerelease=allow examples/v2/hyperopt.py
