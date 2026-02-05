# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
#    "pandas",
#    "numpy",
#    "scikit-learn",
#    "pyarrow",
# ]
# ///
"""
Feature Engineering Demo - Demonstrates Caching

This script shows how Flyte caches expensive feature extraction operations.
When you re-run the workflow with the same inputs, the feature engineering
step is automatically skipped (cache hit).

Key capability demonstrated: CACHING
- Run once: feature engineering executes
- Run again with same inputs: feature engineering is skipped (cached)
"""

import flyte
import flyte.io

# Define cached environment for feature engineering
feature_env = flyte.TaskEnvironment(
    name="feature_engineering",
    resources=flyte.Resources(cpu=2, memory="1Gi"),
    cache="disable",  # use 'auto' to enable automatic caching
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-features",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)

# Non-cached environment for tasks that should always run
# depends_on declares that this environment calls tasks from feature_env
compute_env = flyte.TaskEnvironment(
    name="compute",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[feature_env],  # Required: declare dependency on feature_env
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-features",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@feature_env.task
async def generate_synthetic_documents(n_samples: int, seed: int) -> flyte.io.DataFrame:
    """
    Generate synthetic document data representing policy changes.
    
    This simulates the data structure from document parsing:
    - Entity types: drug, dosage, threshold, condition, requirement
    - Change types: THRESHOLD_INCREASE, THRESHOLD_DECREASE, DRUG_ADDED, etc.
    """
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
    
    risk_levels = {
        "THRESHOLD_INCREASE": "medium",
        "THRESHOLD_DECREASE": "high",
        "DRUG_ADDED": "low",
        "DRUG_REMOVED": "high",
        "PA_ADDED": "high",
        "PA_REMOVED": "low",
    }
    
    # Generate synthetic document features
    data = {
        "doc_id": [f"DOC_{i:05d}" for i in range(n_samples)],
        "change_type": np.random.choice(change_types, n_samples),
        "entity_count": np.random.randint(1, 10, n_samples),
        "word_count": np.random.randint(50, 500, n_samples),
        "has_drug_entity": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "has_dosage_entity": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "has_threshold_entity": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "has_condition_entity": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "has_requirement_entity": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "numeric_value_present": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "date_reference_present": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    }
    
    df = pd.DataFrame(data)
    df["risk_level"] = df["change_type"].map(risk_levels)
    
    print(f"Generated {n_samples} synthetic documents")
    print(f"Change type distribution:\n{df['change_type'].value_counts()}")
    
    return flyte.io.DataFrame.wrap_df(df)


@feature_env.task
async def extract_features(raw_data: flyte.io.DataFrame) -> flyte.io.DataFrame:
    """
    Extract ML-ready features from raw document data.
    
    THIS TASK IS CACHED - if inputs haven't changed, this won't re-run.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    print("Starting feature extraction (this is expensive and should be cached)...")
    
    df = await raw_data.open(pd.DataFrame).all()
    
    # Encode categorical variables
    le_change = LabelEncoder()
    le_risk = LabelEncoder()
    
    df["change_type_encoded"] = le_change.fit_transform(df["change_type"])
    df["risk_level_encoded"] = le_risk.fit_transform(df["risk_level"])
    
    # Create derived features
    df["entity_density"] = df["entity_count"] / df["word_count"]
    df["total_entity_flags"] = (
        df["has_drug_entity"] + 
        df["has_dosage_entity"] + 
        df["has_threshold_entity"] +
        df["has_condition_entity"] +
        df["has_requirement_entity"]
    )
    df["numeric_date_combo"] = df["numeric_value_present"] * df["date_reference_present"]
    
    # Normalize numerical features
    df["word_count_norm"] = (df["word_count"] - df["word_count"].mean()) / df["word_count"].std()
    df["entity_count_norm"] = (df["entity_count"] - df["entity_count"].mean()) / df["entity_count"].std()
    
    print(f"Feature extraction complete. Shape: {df.shape}")
    print(f"Features created: {list(df.columns)}")
    
    return flyte.io.DataFrame.wrap_df(df)


@compute_env.task
async def compute_feature_stats(features: flyte.io.DataFrame) -> dict:
    """Compute statistics on the extracted features (not cached - quick computation)."""
    import pandas as pd
    
    df = await features.open(pd.DataFrame).all()
    
    stats = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "mean_entity_density": float(df["entity_density"].mean()),
        "std_entity_density": float(df["entity_density"].std()),
        "change_type_distribution": df["change_type"].value_counts().to_dict(),
    }
    
    print(f"Feature statistics computed: {stats}")
    return stats


@compute_env.task
async def main(n_samples: int = 1000, seed: int = 42) -> dict:
    """
    Main entry point - demonstrates caching behavior.
    
    Run this twice with the same parameters:
    1. First run: Both generate_synthetic_documents and extract_features execute
    2. Second run: Both tasks return cached results (instant)
    
    Change n_samples or seed: Tasks re-execute (cache miss)
    """
    # Generate synthetic data (cached)
    raw_data = await generate_synthetic_documents(n_samples, seed)
    
    # Extract features (cached)
    features = await extract_features(raw_data)
    
    # Compute stats (not cached, but fast)
    stats = await compute_feature_stats(features)
    
    return stats


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
            uri = flyte.build(feature_env.image, wait=False)
            print(f"build run url: {uri}")
        else:
            run = flyte.with_runcontext(mode="remote").run(main, n_samples=1000, seed=42)
            print(run.url)
