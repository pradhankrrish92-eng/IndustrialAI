from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Artifact not found at {model_path}. Run scripts/train.py first.")

    # 1. Load the trained pipeline
    pipeline = joblib.load(model_path)
    print(f"Loaded model artifact from {model_path}")

    # 2. Simulate new, incoming machine telemetry data
    sample_records = pd.DataFrame(
        [
            {
                # Nominal operating machine
                "type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551.0,
                "torque_nm": 42.8,
                "tool_wear_min": 10.0,
            },
            {
                # High-stress machine (high heat, high torque, high tool wear)
                "type": "L",
                "air_temperature_k": 304.5,
                "process_temperature_k": 313.8,
                "rotational_speed_rpm": 1280.0,
                "torque_nm": 68.2,
                "tool_wear_min": 215.0,
            },
        ]
    )

    # 3. Predict failure classes and probabilities
    predictions = pipeline.predict(sample_records)
    probabilities = pipeline.predict_proba(sample_records)[:, 1]

    # 4. Display results
    print("\n--- Telemetry Predictions ---")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities, strict=True), start=1):
        status = "FAILURE RISK (1)" if pred == 1 else "NORMAL (0)"
        print(f"Machine {i}: {status} | Failure Probability: {prob:.2%}")


if __name__ == "__main__":
    main()
