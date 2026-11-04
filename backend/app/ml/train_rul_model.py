import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR.parent / "data" / "nasa_cmapss"

MODEL_DIR = BASE_DIR / "ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"

MODEL_PATH = MODEL_DIR / "rul_model.pkl"
METRICS_PATH = MODEL_DIR / "rul_metrics.json"

RUL_CAP = 125

print("\n==========================================")
print(" NASA C-MAPSS RUL MODEL TRAINING")
print("==========================================\n")


# =====================================================
# 2. DATASET COLUMN NAMES
# =====================================================

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

required_files = [TRAIN_PATH, TEST_PATH, RUL_PATH]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"Required dataset not found: {file}")

train_df = pd.read_csv(
    TRAIN_PATH, sep=r"\s+", header=None, names=columns
)

test_df = pd.read_csv(
    TEST_PATH, sep=r"\s+", header=None, names=columns
)

actual_rul = pd.read_csv(
    RUL_PATH, sep=r"\s+", header=None
).iloc[:, 0].values

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)
print("Training engines:", train_df["engine_id"].nunique())
print("Testing engines:", test_df["engine_id"].nunique())


# =====================================================
# 3. DATA CLEANING
# =====================================================

train_df = train_df.dropna().copy()
test_df = test_df.dropna().copy()

print("\nMissing values handled.")


# =====================================================
# 4. CALCULATE TRAINING RUL
# =====================================================

max_cycles = train_df.groupby("engine_id")["cycle"].max()

train_df["RUL"] = (
    train_df["engine_id"].map(max_cycles) - train_df["cycle"]
)

# Standard RUL target capping
train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)

print("Training RUL calculated.")
print("RUL cap:", RUL_CAP)


# =====================================================
# 5. FEATURE SELECTION
# =====================================================

feature_names = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X = train_df[feature_names]
y = train_df["RUL"]

groups = train_df["engine_id"]

print("\nTotal model features:", len(feature_names))


# =====================================================
# 6. ENGINE-LEVEL TRAIN / VALIDATION SPLIT
# =====================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, validation_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_validation = X.iloc[validation_indices]

y_train = y.iloc[train_indices]
y_validation = y.iloc[validation_indices]

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))


# =====================================================
# 7. TRAIN RANDOM FOREST REGRESSOR
# =====================================================

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Training completed.")


# =====================================================
# 8. VALIDATION EVALUATION
# =====================================================

validation_predictions = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation, validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(y_validation, validation_predictions)
)

print("\n========== VALIDATION RESULTS ==========")
print("MAE:", round(validation_mae, 3))
print("RMSE:", round(validation_rmse, 3))


# =====================================================
# 9. TRAIN FINAL MODEL USING ALL TRAINING DATA
# =====================================================

print("\nTraining final model using all training engines...")

final_model = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)


# =====================================================
# 10. TEST SET RUL EVALUATION
# =====================================================

last_test_rows = (
    test_df.sort_values("cycle")
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
)

if len(last_test_rows) != len(actual_rul):
    raise ValueError(
        "Number of test engines does not match RUL target count."
    )

X_test = last_test_rows[feature_names]

test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(
    actual_rul, test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(actual_rul, test_predictions)
)

# NASA asymmetric scoring function
errors = test_predictions - actual_rul

scores = np.where(
    errors < 0,
    np.exp(-errors / 13) - 1,
    np.exp(errors / 10) - 1
)

nasa_score = float(np.sum(scores))

print("\n========== OFFICIAL TEST RESULTS ==========")
print("Test MAE:", round(test_mae, 3))
print("Test RMSE:", round(test_rmse, 3))
print("NASA Score:", round(nasa_score, 3))


# =====================================================
# 11. SAVE TRAINED MODEL
# =====================================================

model_bundle = {
    "model": final_model,
    "feature_names": feature_names,
    "rul_cap": RUL_CAP,
    "model_name": "RandomForestRegressor",
    "validation_mae": float(validation_mae),
    "validation_rmse": float(validation_rmse),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "nasa_score": nasa_score
}

joblib.dump(model_bundle, MODEL_PATH)

metrics = {
    "model": "RandomForestRegressor",
    "dataset": "NASA C-MAPSS FD001",
    "training_engines": int(train_df["engine_id"].nunique()),
    "testing_engines": int(test_df["engine_id"].nunique()),
    "training_records": int(len(train_df)),
    "features": len(feature_names),
    "rul_cap": RUL_CAP,
    "validation_mae": float(validation_mae),
    "validation_rmse": float(validation_rmse),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "nasa_score": nasa_score
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)


# =====================================================
# 12. SAVE REUSABLE PREDICTION FUNCTION
# =====================================================

predictor_code = '''
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
'''

predictor_path = MODEL_DIR / "rul_predictor.py"
predictor_path.write_text(predictor_code, encoding="utf-8")


# =====================================================
# 13. FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": final_model.feature_importances_
}).sort_values(
    "importance", ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(importance.head(10).to_string(index=False))


# =====================================================
# 14. FINAL OUTPUT
# =====================================================

print("\n==========================================")
print(" RUL MODEL BUILD COMPLETED")
print("==========================================")

print("\nSaved files:")
print(MODEL_PATH)
print(METRICS_PATH)
print(predictor_path)

print("\nModel evaluation:")
print(json.dumps(metrics, indent=4))

