from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    model_path = Path("artifacts/multilabel_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {model_path}. Run scripts/train_multilabel.py first."
        )

    pipeline = joblib.load(model_path)
    print(f"Loaded multi-label diagnostic pipeline from {model_path}")

    # Simulated incoming telemetry
    sample_records = pd.DataFrame(
        [
            {
                # Nominal machine
                "type": "M",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551.0,
                "torque_nm": 42.8,
                "tool_wear_min": 10.0,
            },
            {
                # High heat dissipation risk
                "type": "L",
                "air_temperature_k": 303.0,
                "process_temperature_k": 313.5,
                "rotational_speed_rpm": 1380.0,
                "torque_nm": 55.0,
                "tool_wear_min": 80.0,
            },
            {
                # Overstrain / Tool wear breakdown risk
                "type": "L",
                "air_temperature_k": 299.0,
                "process_temperature_k": 309.0,
                "rotational_speed_rpm": 1250.0,
                "torque_nm": 70.0,
                "tool_wear_min": 220.0,
            },
        ]
    )

    targets = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]
    predictions = pipeline.predict(sample_records)

    print("\n--- Telemetry Diagnostics ---")
    for i, row in enumerate(predictions, start=1):
        is_failure = bool(row[0])
        status = "⚠️ FAILURE RISK" if is_failure else "✅ NORMAL"

        # Collect flagged failure root causes
        diagnosed_modes = [
            targets[idx].upper() for idx, val in enumerate(row[1:], start=1) if val == 1
        ]
        modes_str = ", ".join(diagnosed_modes) if diagnosed_modes else "None"

        print(f"Machine {i}: {status} | Root Causes: {modes_str}")


if __name__ == "__main__":
    main()
