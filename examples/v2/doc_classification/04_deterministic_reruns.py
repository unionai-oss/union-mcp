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
Deterministic Reruns Demo

This script demonstrates that Flyte provides reproducible, deterministic outputs:
- Same inputs → Same outputs (every time)
- Fixed random seeds ensure reproducibility
- Results are verifiable and auditable

Key capability demonstrated: RERUNS / REPRODUCIBILITY
Run this multiple times with the same inputs and verify identical outputs.
"""

from dataclasses import dataclass

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="deterministic_demo",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="doc-deterministic",
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@dataclass
class DeterministicResult:
    """Result that should be identical across reruns with same inputs."""
    input_seed: int
    input_n_samples: int
    data_checksum: str
    feature_checksum: str
    prediction_checksum: str
    sample_predictions: list[int]
    accuracy: float


@env.task
async def generate_data(n_samples: int, seed: int) -> flyte.io.DataFrame:
    """Generate synthetic data with fixed seed for reproducibility."""
    import hashlib
    import numpy as np
    import pandas as pd
    
    np.random.seed(seed)  # Fixed seed = deterministic output
    
    change_types = [
        "THRESHOLD_INCREASE", "THRESHOLD_DECREASE", 
        "DRUG_ADDED", "DRUG_REMOVED",
        "PA_ADDED", "PA_REMOVED",
    ]
    
    data = {
        "change_type": np.random.choice(change_types, n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randint(0, 10, n_samples),
        "feature_4": np.random.choice([0, 1], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Compute checksum to verify determinism
    checksum = hashlib.md5(df.to_json().encode()).hexdigest()[:16]
    print(f"Data generated. Checksum: {checksum}")
    
    return flyte.io.DataFrame.wrap_df(df)


@env.task
async def extract_features(data: flyte.io.DataFrame, seed: int) -> flyte.io.DataFrame:
    """Extract features deterministically."""
    import hashlib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    np.random.seed(seed)
    
    df = await data.open(pd.DataFrame).all()
    
    # Encode target
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["change_type"])
    
    # Create derived features
    df["feature_combo"] = df["feature_1"] * df["feature_2"]
    df["feature_sum"] = df["feature_3"] + df["feature_4"]
    
    checksum = hashlib.md5(df.to_json().encode()).hexdigest()[:16]
    print(f"Features extracted. Checksum: {checksum}")
    
    return flyte.io.DataFrame.wrap_df(df)


@env.task
async def train_and_predict(features: flyte.io.DataFrame, seed: int) -> tuple[list[int], float]:
    """Train model and make predictions deterministically."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    df = await features.open(pd.DataFrame).all()
    
    feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_combo", "feature_sum"]
    X = df[feature_cols]
    y = df["target"]
    
    # Fixed random state for train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Fixed random state for model
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test).tolist()
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Predictions made. First 10: {predictions[:10]}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return predictions, float(accuracy)


@env.task(report=True)
async def main(n_samples: int = 500, seed: int = 42) -> DeterministicResult:
    """
    Demonstrate deterministic behavior.
    
    Run this multiple times with the same n_samples and seed.
    The output checksums and predictions will be IDENTICAL every time.
    
    This proves:
    - Reproducibility: Same inputs → Same outputs
    - Auditability: Results can be verified by re-running
    - Determinism: No random variation between runs
    """
    import hashlib
    import pandas as pd
    
    # Generate data
    data = await generate_data(n_samples, seed)
    
    # Extract features
    features = await extract_features(data, seed)
    
    # Train and predict
    predictions, accuracy = await train_and_predict(features, seed)
    
    # Compute checksums
    df_data = await data.open(pd.DataFrame).all()
    df_features = await features.open(pd.DataFrame).all()
    
    data_checksum = hashlib.md5(df_data.to_json().encode()).hexdigest()[:16]
    feature_checksum = hashlib.md5(df_features.to_json().encode()).hexdigest()[:16]
    prediction_checksum = hashlib.md5(str(predictions).encode()).hexdigest()[:16]
    
    result = DeterministicResult(
        input_seed=seed,
        input_n_samples=n_samples,
        data_checksum=data_checksum,
        feature_checksum=feature_checksum,
        prediction_checksum=prediction_checksum,
        sample_predictions=predictions[:20],
        accuracy=accuracy,
    )
    
    # Generate report
    tab = flyte.report.get_tab("Reproducibility Proof")
    tab.log("<h1>Deterministic Execution Proof</h1>")
    
    tab.log(f"""
    <div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3>Reproducibility Guarantee</h3>
        <p>Running this workflow with the same inputs will <strong>always</strong> produce identical outputs.</p>
        <p><strong>Inputs:</strong> n_samples={n_samples}, seed={seed}</p>
    </div>
    """)
    
    tab.log(f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>Output Checksums (verify these match across runs)</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 10px;"><strong>Data Checksum:</strong></td>
                <td style="padding: 10px; font-family: monospace;">{data_checksum}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 10px;"><strong>Feature Checksum:</strong></td>
                <td style="padding: 10px; font-family: monospace;">{feature_checksum}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 10px;"><strong>Prediction Checksum:</strong></td>
                <td style="padding: 10px; font-family: monospace;">{prediction_checksum}</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><strong>Accuracy:</strong></td>
                <td style="padding: 10px;">{accuracy:.6f}</td>
            </tr>
        </table>
    </div>
    """)
    
    tab.log(f"""
    <div style="background: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>First 20 Predictions (verify these match)</h4>
        <p style="font-family: monospace; word-break: break-all;">{predictions[:20]}</p>
    </div>
    """)
    
    tab.log("""
    <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #17a2b8;">
        <h4>How to Verify Determinism</h4>
        <ol>
            <li>Run this workflow and note the checksums</li>
            <li>Run again with the same inputs (n_samples, seed)</li>
            <li>Compare checksums - they will be <strong>identical</strong></li>
            <li>This proves: Same inputs → Same outputs</li>
        </ol>
    </div>
    """)
    
    await flyte.report.flush.aio()
    
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
            run = flyte.with_runcontext(mode="remote").run(main, n_samples=500, seed=42)
            print(run.url)
