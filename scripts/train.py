from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from industrial_ai.config import load_config
from industrial_ai.data.loader import load_dataset


def main() -> None:
    # 1. Load config and validated dataset
    cfg = load_config()
    df = load_dataset(cfg)
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Separate features (X) and target (y)
    # Target is binary machine failure
    target_col = "machine_failure"

    # Exclude identifiers and specific failure mode tags from feature set
    non_feature_cols = [
        "udi",
        "product_id",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]

    # Dynamically select numeric and categorical features present in columns
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    cat_cols = ["type"] if "type" in feature_cols else []
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # 3. Train/Test split (stratified due to imbalanced failure cases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Build preprocessing & modeling pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 5. Train the model
    print("\nTraining Random Forest model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # 7. Save model artifact
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved successfully to: {model_path}")


if __name__ == "__main__":
    main()
