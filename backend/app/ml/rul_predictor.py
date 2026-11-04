
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "rul_model.pkl"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_names = bundle["feature_names"]
RUL_CAP = bundle["rul_cap"]


def predict_rul(sensor_data: dict):
    df = pd.DataFrame([sensor_data])

    missing = [
        col for col in feature_names
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = df[feature_names]

    prediction = float(model.predict(df)[0])

    prediction = max(0.0, prediction)

    return {
        "predicted_rul_cycles": round(prediction, 2),
        "estimated_rul_days": None,
        "rul_category": (
            "Critical" if prediction <= 20
            else "Warning" if prediction <= 60
            else "Healthy"
        )
    }
