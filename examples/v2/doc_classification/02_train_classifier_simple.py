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
Change Classification Training Demo (Simplified)

A simplified version that avoids dataclass serialization issues.
Demonstrates XGBoost classification with metrics and visualization.
"""

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="xgboost_simple",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-classifier-simple",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@env.task(report=True)
async def main(
    n_samples: int = 2000,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    version: str = "v1.0",
) -> str:
    """
    Train and evaluate a change classification model.
    
    This demonstrates:
    - Versioning: Each run with different hyperparameters creates a new model version
    - Lineage: Click on artifacts to see what produced them
    - Determinism: Same inputs → same outputs
    """
    import json
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier
    import plotly.graph_objects as go
    
    # Generate synthetic training data
    np.random.seed(seed)
    
    change_types = [
        "THRESHOLD_INCREASE", "THRESHOLD_DECREASE", 
        "DRUG_ADDED", "DRUG_REMOVED",
        "PA_ADDED", "PA_REMOVED",
    ]
    
    records = []
    for _ in range(n_samples):
        change_type = np.random.choice(change_types)
        
        # Create features correlated with change type
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
    
    # Prepare features
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["change_type"])
    class_names = list(le.classes_)
    
    df["entity_density"] = df["entity_count"] / df["word_count"]
    df["total_entity_flags"] = (
        df["has_drug_entity"] + df["has_dosage_entity"] + 
        df["has_threshold_entity"] + df["has_condition_entity"] + 
        df["has_requirement_entity"]
    )
    
    feature_cols = [
        "entity_count", "word_count", "has_drug_entity", "has_dosage_entity",
        "has_threshold_entity", "has_condition_entity", "has_requirement_entity",
        "numeric_value_present", "date_reference_present", "entity_density",
        "total_entity_flags"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Train model
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    
    print(f"Evaluation: Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}")
    
    # Create report
    tab = flyte.report.get_tab("Results")
    tab.log(f"<h1>Change Classification Model - {version}</h1>")
    tab.log(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Training Configuration</h3>
        <p><strong>Model Version:</strong> {version}</p>
        <p><strong>Samples:</strong> {n_samples} | <strong>Seed:</strong> {seed}</p>
        <p><strong>n_estimators:</strong> {n_estimators} | <strong>max_depth:</strong> {max_depth} | <strong>learning_rate:</strong> {learning_rate}</p>
    </div>
    """)
    
    # Metrics visualization
    fig = go.Figure(data=[
        go.Bar(
            x=["Accuracy", "F1 (Weighted)", "F1 (Macro)", "Precision", "Recall"],
            y=[accuracy, f1_weighted, f1_macro, precision, recall],
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
            <tr><td><strong>Accuracy:</strong></td><td>{accuracy:.4f}</td></tr>
            <tr><td><strong>F1 Score (Weighted):</strong></td><td>{f1_weighted:.4f}</td></tr>
            <tr><td><strong>F1 Score (Macro):</strong></td><td>{f1_macro:.4f}</td></tr>
            <tr><td><strong>Precision:</strong></td><td>{precision:.4f}</td></tr>
            <tr><td><strong>Recall:</strong></td><td>{recall:.4f}</td></tr>
        </table>
    </div>
    """)
    
    tab.log(f"<p><strong>Change Types:</strong> {', '.join(class_names)}</p>")
    
    await flyte.report.flush.aio()
    
    result = {
        "version": version,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
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
                version="v1.0",
            )
            print(run.url)
