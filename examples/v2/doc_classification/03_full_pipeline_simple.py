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
Simple ML Training Pipeline Demo

A simplified version that demonstrates XGBoost classification.
"""

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="simple_pipeline",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-simple",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@env.task(report=True)
async def main(
    n_samples: int = 2000,
    seed: int = 42,
    version: str = "v1.0",
    n_estimators: int = 100,
    max_depth: int = 6,
) -> str:
    """Simple training pipeline that generates data, trains, and evaluates in one task."""
    import json
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier
    import plotly.graph_objects as go
    
    # Generate synthetic data
    np.random.seed(seed)
    
    change_types = [
        "THRESHOLD_INCREASE", "THRESHOLD_DECREASE", 
        "DRUG_ADDED", "DRUG_REMOVED",
        "PA_ADDED", "PA_REMOVED",
    ]
    
    records = []
    for i in range(n_samples):
        change_type = np.random.choice(change_types)
        has_threshold = 1 if change_type.startswith("THRESHOLD") else np.random.choice([0, 1], p=[0.7, 0.3])
        has_drug = 1 if "DRUG" in change_type else np.random.choice([0, 1], p=[0.4, 0.6])
        has_requirement = 1 if "PA" in change_type else np.random.choice([0, 1], p=[0.6, 0.4])
        numeric_present = 1 if change_type.startswith("THRESHOLD") else np.random.choice([0, 1])
        
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
    print(f"Generated {len(df)} samples")
    
    # Prepare features
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["change_type"])
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Train model
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Create result
    result = {
        "version": version,
        "accuracy": accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    
    # Generate report
    tab = flyte.report.get_tab("Results")
    tab.log(f"<h1>Classification Pipeline - {version}</h1>")
    
    tab.log(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Model: {version}</h3>
        <p><strong>Accuracy:</strong> {accuracy:.4f} | <strong>F1 Score:</strong> {f1:.4f}</p>
        <p><strong>Train:</strong> {len(X_train)} | <strong>Test:</strong> {len(X_test)}</p>
    </div>
    """)
    
    fig = go.Figure(data=[
        go.Bar(x=["Accuracy", "F1 Score"], y=[accuracy, f1], marker_color=["#1f77b4", "#2ca02c"])
    ])
    fig.update_layout(title="Model Metrics", yaxis_range=[0, 1], height=300)
    tab.log(fig.to_html(include_plotlyjs=True, full_html=False))
    
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
            uri = flyte.build(env.image, wait=False)
            print(f"build run url: {uri}")
        else:
            run = flyte.with_runcontext(mode="remote").run(
                main,
                n_samples=2000,
                seed=42,
                version="v1.0",
                n_estimators=100,
                max_depth=6,
            )
            print(run.url)
