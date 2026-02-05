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
Full Training Pipeline Demo

This demonstrates a complete ML pipeline with:
1. Feature engineering (with caching)
2. Model training
3. Evaluation
4. Model registration with version

Key capabilities demonstrated:
- CACHING: Feature engineering skipped on re-run
- VERSIONING: Register model v1.0, v1.1, compare metrics
- LINEAGE: Click on model → see what training data produced it
- RERUNS: Same inputs → same outputs (deterministic)
"""

import asyncio

import flyte
import flyte.io
import flyte.report

# Cached environment for expensive operations
cached_env = flyte.TaskEnvironment(
    name="cached_pipeline",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    cache="disable",  # use 'auto' to enable automatic caching
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-pipeline",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)

# Non-cached environment for final outputs (depends on cached_env)
report_env = flyte.TaskEnvironment(
    name="report_env",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[cached_env],  # Required to call tasks in cached_env
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-pipeline",
        registry="ghcr.io/flyteorg",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)




# =============================================================================
# STEP 1: Feature Engineering (CACHED)
# =============================================================================

@cached_env.task
async def load_raw_data(n_samples: int, seed: int) -> flyte.io.DataFrame:
    """Load raw document data. Cached based on n_samples and seed."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(seed)
    
    change_types = [
        "THRESHOLD_INCREASE", "THRESHOLD_DECREASE", 
        "DRUG_ADDED", "DRUG_REMOVED",
        "PA_ADDED", "PA_REMOVED",
    ]
    
    records = []
    for i in range(n_samples):
        change_type = np.random.choice(change_types)
        
        # Create correlated features
        has_threshold = 1 if change_type.startswith("THRESHOLD") else np.random.choice([0, 1], p=[0.7, 0.3])
        has_drug = 1 if "DRUG" in change_type else np.random.choice([0, 1], p=[0.4, 0.6])
        has_requirement = 1 if "PA" in change_type else np.random.choice([0, 1], p=[0.6, 0.4])
        numeric_present = 1 if change_type.startswith("THRESHOLD") else np.random.choice([0, 1])
        
        records.append({
            "doc_id": f"DOC_{i:05d}",
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
    print(f"[CACHED] Loaded {len(df)} raw documents")
    return flyte.io.DataFrame.wrap_df(df)


@cached_env.task
async def extract_features(raw_data: flyte.io.DataFrame) -> flyte.io.DataFrame:
    """
    Extract ML features from raw data.
    
    THIS IS CACHED - expensive feature extraction only runs once per unique input.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    print("[CACHED] Starting feature extraction...")
    
    df = await raw_data.open(pd.DataFrame).all()
    
    # Encode target
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["change_type"])
    
    # Store class mapping
    df["_classes"] = ",".join(le.classes_)
    
    # Create derived features
    df["entity_density"] = df["entity_count"] / df["word_count"]
    df["total_entity_flags"] = (
        df["has_drug_entity"] + df["has_dosage_entity"] + 
        df["has_threshold_entity"] + df["has_condition_entity"] + 
        df["has_requirement_entity"]
    )
    df["numeric_date_combo"] = df["numeric_value_present"] * df["date_reference_present"]
    
    # Normalize
    for col in ["word_count", "entity_count"]:
        df[f"{col}_norm"] = (df[col] - df[col].mean()) / df[col].std()
    
    print(f"[CACHED] Feature extraction complete. Shape: {df.shape}")
    return flyte.io.DataFrame.wrap_df(df)


# =============================================================================
# STEP 2: Model Training
# =============================================================================

@cached_env.task
async def train_model(
    features: flyte.io.DataFrame,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> flyte.io.File:
    """Train XGBoost model. Cached based on features and hyperparameters."""
    import pandas as pd
    import joblib
    from xgboost import XGBClassifier
    
    print(f"[CACHED] Training with n_estimators={n_estimators}, max_depth={max_depth}")
    
    df = await features.open(pd.DataFrame).all()
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present", "entity_density",
        "total_entity_flags", "numeric_date_combo", "word_count_norm", "entity_count_norm"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    
    model_path = "/tmp/model.joblib"
    joblib.dump(model, model_path)
    
    print(f"[CACHED] Model trained and saved")
    return flyte.io.File(model_path)


# =============================================================================
# STEP 3: Evaluation
# =============================================================================

@cached_env.task
async def evaluate_model(
    model_file: flyte.io.File,
    features: flyte.io.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[float, float, float, float, float, int, int]:
    """Evaluate model performance. Returns (accuracy, f1_weighted, f1_macro, precision, recall, train_samples, test_samples)."""
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    df = await features.open(pd.DataFrame).all()
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present", "entity_density",
        "total_entity_flags", "numeric_date_combo", "word_count_norm", "entity_count_norm"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model_path = await model_file.download()
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    
    print(f"Evaluation: Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}")
    return accuracy, f1_weighted, f1_macro, precision, recall, len(X_train), len(X_test)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@report_env.task(report=True)
async def main(
    n_samples: int = 2000,
    data_seed: int = 42,
    version: str = "v1.0",
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> str:
    """
    Full training pipeline demonstrating:
    
    - CACHING: Re-run with same data params → feature engineering skipped
    - VERSIONING: Change version param to register new model version
    - LINEAGE: Each artifact tracks its inputs (click to see provenance)
    - DETERMINISM: Same inputs → same outputs (fixed seeds)
    
    Example runs to demonstrate capabilities:
    
    1. Initial run: version="v1.0", n_estimators=100
    2. Re-run same params: Feature engineering cached (instant)
    3. New version: version="v1.1", n_estimators=200 → Compare metrics
    """
    import plotly.graph_objects as go
    
    # Step 1: Load data (cached)
    raw_data = await load_raw_data(n_samples, data_seed)
    
    # Step 2: Extract features (cached)
    features = await extract_features(raw_data)
    
    # Step 3: Train model (cached based on features + hyperparams)
    model = await train_model(
        features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )
    
    # Step 4: Evaluate
    accuracy, f1_weighted, f1_macro, precision, recall, train_samples, test_samples = await evaluate_model(
        model, features, test_size=0.2, random_state=42
    )
    
    import json
    
    # Create result as JSON string
    result = {
        "version": version,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }
    
    print(f"Model registered: version={version}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1_weighted:.4f}")
    
    # Generate report
    tab = flyte.report.get_tab("Pipeline Results")
    tab.log(f"<h1>Model Training Pipeline - {version}</h1>")
    
    tab.log(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Model Version: {version}</h3>
        <p><strong>Training Samples:</strong> {train_samples} | <strong>Test Samples:</strong> {test_samples}</p>
        <p><strong>Hyperparameters:</strong> n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}</p>
    </div>
    """)
    
    # Metrics chart
    fig = go.Figure(data=[
        go.Bar(
            x=["Accuracy", "F1 (Weighted)", "F1 (Macro)", "Precision", "Recall"],
            y=[accuracy, f1_weighted, f1_macro, precision, recall],
            marker_color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"],
            text=[f"{v:.3f}" for v in [accuracy, f1_weighted, f1_macro, precision, recall]],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=f"Model Performance - {version}",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=400,
    )
    tab.log(fig.to_html(include_plotlyjs=True, full_html=False))
    
    tab.log("""
    <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #17a2b8;">
        <h4>Demonstrated Capabilities</h4>
        <ul>
            <li><strong>Caching:</strong> Re-run this pipeline → feature engineering is skipped</li>
            <li><strong>Versioning:</strong> Change version param to create v1.1, v1.2, etc.</li>
            <li><strong>Lineage:</strong> Click on any task output to see what produced it</li>
            <li><strong>Determinism:</strong> Same inputs always produce same outputs</li>
        </ul>
    </div>
    """)
    
    await flyte.report.flush.aio()
    
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
            uri = flyte.build(cached_env.image, wait=False)
            print(f"build run url: {uri}")
        else:
            run = flyte.with_runcontext(mode="remote").run(
                main,
                n_samples=2000,
                data_seed=42,
                version="v1.0",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            )
            print(run.url)
