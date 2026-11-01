from pathlib import Path

import joblib
import pandas as pd
import pytest

from industrial_ai.features import IndustrialFeatureEngineer


@pytest.fixture
def model():
    path = Path("artifacts/model.joblib")
    if not path.exists():
        pytest.skip("Model artifact not found. Run scripts/train.py first.")
    return joblib.load(path)


def test_model_prediction_shape_and_types(model):
    sample = pd.DataFrame(
        [
            {
                "type": "L",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 50.0,
            }
        ]
    )
    pred = model.predict(sample)
    proba = model.predict_proba(sample)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_industrial_feature_engineer():
    df = pd.DataFrame(
        [
            {
                "type": "L",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 50.0,
            }
        ]
    )
    engineer = IndustrialFeatureEngineer()
    transformed = engineer.transform(df)

    assert "temp_diff_k" in transformed.columns
    assert transformed["temp_diff_k"].iloc[0] == 10.0
    assert "power_w" in transformed.columns
    assert "overstrain_proxy" in transformed.columns
    assert transformed["overstrain_proxy"].iloc[0] == 2000.0
