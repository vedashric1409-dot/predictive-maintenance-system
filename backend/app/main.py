from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

from app.schemas import SensorDataInput
from app.ml.predictor import predict_single, predict_batch, metrics
from app.database import (
    initialize_database,
    save_prediction,
    get_prediction_history
)

app = FastAPI(title="Predictive Maintenance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

initialize_database()


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(data: SensorDataInput):
    try:
        sensor_data = data.to_model_input()

        result = predict_single(sensor_data)

        history_id = save_prediction(
            sensor_data,
            result
        )

        result["history_id"] = history_id

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        df = pd.read_csv(file.file)
        df.columns = df.columns.str.strip()

        rename_map = {
            "Air_temperature_K": "Air temperature [K]",
            "Process_temperature_K": "Process temperature [K]",
            "Rotational_speed_rpm": "Rotational speed [rpm]",
            "Torque_Nm": "Torque [Nm]",
            "Tool_wear_min": "Tool wear [min]",
        }

        df = df.rename(columns=rename_map)

        expected_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        missing = [
            col for col in expected_columns
            if col not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        results = predict_batch(df)

        for index, prediction in enumerate(results):
            sensor_data = df.iloc[index][expected_columns].to_dict()

            history_id = save_prediction(
                sensor_data,
                prediction
            )

            prediction["history_id"] = history_id

        return {
            "total_rows": len(results),
            "predictions": results,
            "message": "Predictions saved successfully."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/history")
def prediction_history(limit: int = 100):
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000."
        )

    try:
        records = get_prediction_history(limit)

        return {
            "total_records": len(records),
            "history": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve prediction history: {str(e)}"
        )
