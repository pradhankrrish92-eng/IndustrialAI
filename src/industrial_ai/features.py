"""Domain-specific feature engineering for the AI4I 2020 predictive maintenance dataset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IndustrialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates physical and mechanical interaction features from sensor telemetry."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Temperature difference (Heat Dissipation proxy)
        if "process_temperature_k" in df.columns and "air_temperature_k" in df.columns:
            df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]

        # 2. Power proxy in Watts: Power = Torque (N*m) * Angular velocity (rad/s)
        # Angular velocity = (rotational_speed_rpm * 2 * pi) / 60
        if "rotational_speed_rpm" in df.columns and "torque_nm" in df.columns:
            angular_velocity = (df["rotational_speed_rpm"] * 2 * np.pi) / 60.0
            df["power_w"] = df["torque_nm"] * angular_velocity

        # 3. Overstrain proxy: Tool wear multiplied by torque
        if "tool_wear_min" in df.columns and "torque_nm" in df.columns:
            df["overstrain_proxy"] = df["tool_wear_min"] * df["torque_nm"]

        return df
