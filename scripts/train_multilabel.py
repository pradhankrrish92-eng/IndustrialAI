from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from industrial_ai.config import load_config
from industrial_ai.data.loader import load_dataset
from industrial_ai.features import IndustrialFeatureEngineer


def main() -> None:
    cfg = load_config()
    df = load_dataset(cfg)

    # Multi-label targets: general failure + 5 specific failure modes
    target_cols = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]

    # Identify feature columns
    ignore_cols = ["udi", "product_id"] + target_cols
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[target_cols]

    print(f"Features: {feature_cols}")
    print(f"Target failure labels: {target_cols}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Numeric features include original columns plus newly engineered features
    engineered_num_cols = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "power_w",
        "overstrain_proxy",
    ]
    cat_cols = ["type"] if "type" in feature_cols else []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), engineered_num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", IndustrialFeatureEngineer()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            ),
        ]
    )

    print("\nTraining Multi-Output Random Forest with Engineered Features...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n--- Diagnostic Classification Reports per Failure Mode ---")
    for idx, target_name in enumerate(target_cols):
        print(f"\nTarget: {target_name.upper()}")
        print(classification_report(y_test[target_name], y_pred[:, idx], zero_division=0))

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "multilabel_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nMulti-label diagnostic model saved to: {model_path}")


if __name__ == "__main__":
    main()
