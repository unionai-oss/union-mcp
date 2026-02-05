# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "pandas",
#    "numpy",
#    "scikit-learn",
#    "xgboost",
#    "pyarrow",
#    "plotly",
# ]
# ///
"""
Change Classification Training Demo

This script trains an XGBoost classifier for document change classification.
It demonstrates:
- Model training with configurable hyperparameters
- Model evaluation with metrics
- Visualization of results via Flyte reports

Key capabilities demonstrated: VERSIONING, LINEAGE
- Each training run has a unique execution ID
- Click on any artifact to see what produced it (lineage)
"""

import asyncio
from dataclasses import dataclass

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="xgboost_training",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    cache="disable",  # use 'auto' to enable automatic caching
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-classifier",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@dataclass
class TrainingConfig:
    """Hyperparameters for XGBoost training."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass
class EvaluationMetrics:
    """Model evaluation metrics."""
    accuracy: float
    f1_weighted: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float


@env.task
async def generate_training_data(n_samples: int, seed: int) -> flyte.io.DataFrame:
    """Generate synthetic training data for change classification."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(seed)
    
    change_types = [
        "THRESHOLD_INCREASE",
        "THRESHOLD_DECREASE", 
        "DRUG_ADDED",
        "DRUG_REMOVED",
        "PA_ADDED",
        "PA_REMOVED",
    ]
    
    # Generate features that correlate with change types
    records = []
    for _ in range(n_samples):
        change_type = np.random.choice(change_types)
        
        # Create features with some correlation to change type
        if change_type in ["THRESHOLD_INCREASE", "THRESHOLD_DECREASE"]:
            has_threshold = 1
            numeric_present = np.random.choice([0, 1], p=[0.1, 0.9])
        else:
            has_threshold = np.random.choice([0, 1], p=[0.7, 0.3])
            numeric_present = np.random.choice([0, 1], p=[0.5, 0.5])
            
        if change_type in ["DRUG_ADDED", "DRUG_REMOVED"]:
            has_drug = 1
        else:
            has_drug = np.random.choice([0, 1], p=[0.4, 0.6])
            
        if change_type in ["PA_ADDED", "PA_REMOVED"]:
            has_requirement = 1
        else:
            has_requirement = np.random.choice([0, 1], p=[0.6, 0.4])
        
        records.append({
            "change_type": change_type,
            "entity_count": np.random.randint(1, 10),
            "word_count": np.random.randint(50, 500),
            "has_drug_entity": has_drug,
            "has_dosage_entity": np.random.choice([0, 1]),
            "has_threshold_entity": has_threshold,
            "has_condition_entity": np.random.choice([0, 1]),
            "has_requirement_entity": has_requirement,
            "numeric_value_present": numeric_present,
            "date_reference_present": np.random.choice([0, 1]),
        })
    
    df = pd.DataFrame(records)
    print(f"Generated {n_samples} training samples")
    return flyte.io.DataFrame.wrap_df(df)


@env.task
async def prepare_features(data: flyte.io.DataFrame) -> tuple[flyte.io.DataFrame, list[str]]:
    """Prepare features for model training."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    df = await data.open(pd.DataFrame).all()
    
    # Encode target
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["change_type"])
    
    # Create derived features
    df["entity_density"] = df["entity_count"] / df["word_count"]
    df["total_entity_flags"] = (
        df["has_drug_entity"] + 
        df["has_dosage_entity"] + 
        df["has_threshold_entity"] +
        df["has_condition_entity"] +
        df["has_requirement_entity"]
    )
    
    # Store class names for later
    class_names = list(le.classes_)
    
    print(f"Prepared features. Classes: {class_names}")
    return flyte.io.DataFrame.wrap_df(df), class_names


@env.task
async def train_xgboost(
    data: flyte.io.DataFrame,
    config: TrainingConfig,
) -> flyte.io.File:
    """Train XGBoost classifier and save model."""
    import pandas as pd
    import joblib
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    
    df = await data.open(pd.DataFrame).all()
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present", "entity_density",
        "total_entity_flags"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Train model
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=config.random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    
    model.fit(X, y)
    
    # Save model
    model_path = "/tmp/xgboost_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"Model trained with {config.n_estimators} estimators, max_depth={config.max_depth}")
    return flyte.io.File(model_path)


@env.task
async def evaluate_model(
    model_file: flyte.io.File,
    data: flyte.io.DataFrame,
    class_names: list[str],
) -> EvaluationMetrics:
    """Evaluate the trained model."""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    
    df = await data.open(pd.DataFrame).all()
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present", "entity_density",
        "total_entity_flags"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Split for evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load and predict
    model_path = await model_file.download()
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
        f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        precision_weighted=float(precision_score(y_test, y_pred, average="weighted")),
        recall_weighted=float(recall_score(y_test, y_pred, average="weighted")),
    )
    
    print(f"Evaluation: Accuracy={metrics.accuracy:.4f}, F1={metrics.f1_weighted:.4f}")
    return metrics


@env.task(report=True)
async def main(
    n_samples: int = 2000,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> str:
    """
    Train and evaluate a change classification model.
    
    This demonstrates:
    - Versioning: Each run with different hyperparameters creates a new model version
    - Lineage: Click on the model to see the training data and parameters that produced it
    - Determinism: Same inputs → same outputs (set random_state for reproducibility)
    
    Returns a JSON string containing the evaluation metrics.
    """
    import json
    import plotly.graph_objects as go
    
    config = TrainingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    
    # Generate data
    data = await generate_training_data(n_samples, seed)
    
    # Prepare features
    features, class_names = await prepare_features(data)
    
    # Train model
    model = await train_xgboost(features, config)
    
    # Evaluate
    metrics = await evaluate_model(model, features, class_names)
    
    # Create report
    tab = flyte.report.get_tab("Results")
    tab.log("<h1>Change Classification Model Training</h1>")
    tab.log(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Training Configuration</h3>
        <p><strong>Samples:</strong> {n_samples} | <strong>Seed:</strong> {seed}</p>
        <p><strong>n_estimators:</strong> {n_estimators} | <strong>max_depth:</strong> {max_depth} | <strong>learning_rate:</strong> {learning_rate}</p>
    </div>
    """)
    
    # Metrics visualization
    fig = go.Figure(data=[
        go.Bar(
            x=["Accuracy", "F1 (Weighted)", "F1 (Macro)", "Precision", "Recall"],
            y=[metrics.accuracy, metrics.f1_weighted, metrics.f1_macro, metrics.precision_weighted, metrics.recall_weighted],
            marker_color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
        )
    ])
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=400,
    )
    tab.log(fig.to_html(include_plotlyjs=True, full_html=False))
    
    tab.log(f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>Key Metrics</h4>
        <table style="width: 100%;">
            <tr><td><strong>Accuracy:</strong></td><td>{metrics.accuracy:.4f}</td></tr>
            <tr><td><strong>F1 Score (Weighted):</strong></td><td>{metrics.f1_weighted:.4f}</td></tr>
            <tr><td><strong>F1 Score (Macro):</strong></td><td>{metrics.f1_macro:.4f}</td></tr>
            <tr><td><strong>Precision:</strong></td><td>{metrics.precision_weighted:.4f}</td></tr>
            <tr><td><strong>Recall:</strong></td><td>{metrics.recall_weighted:.4f}</td></tr>
        </table>
    </div>
    """)
    
    tab.log(f"<p><strong>Change Types:</strong> {', '.join(class_names)}</p>")
    
    await flyte.report.flush.aio()
    
    # Return as JSON string to avoid dataclass serialization issues
    result = {
        "accuracy": metrics.accuracy,
        "f1_weighted": metrics.f1_weighted,
        "f1_macro": metrics.f1_macro,
        "precision_weighted": metrics.precision_weighted,
        "recall_weighted": metrics.recall_weighted,
        "config": {
            "n_samples": n_samples,
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "class_names": class_names,
    }
    return json.dumps(result, indent=2)


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
            run = flyte.with_runcontext(mode="remote").run(
                main,
                n_samples=2000,
                seed=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            )
            print(run.url)
