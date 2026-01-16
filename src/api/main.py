from __future__ import annotations

import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow.tracking import MlflowClient

from src.common.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MODEL_STAGE,
)

app = FastAPI(title="AutoGluon + MLflow + FastAPI")

class PredictRequest(BaseModel):
    rows: list[dict] = Field(..., description="List of row dicts matching training feature columns")


class ModelHolder:
    def __init__(self):
        self.model = None
        self.loaded_version = None

    def load_model(self):
        local_path = os.getenv("LOCAL_MODEL_PATH")
        if local_path and os.path.exists(local_path):
            print(f"Loading model from local path: {local_path}")
            self.model = mlflow.pyfunc.load_model(local_path)
            self.loaded_version = "local"
        else:
            self.load_from_registry()

    def load_from_registry(self):
        print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Stage 기반 조회
        try:
            versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MODEL_STAGE])
            if not versions:
                raise RuntimeError(f"No model found in stage={MODEL_STAGE} for {MLFLOW_MODEL_NAME}")

            v = versions[0]
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MODEL_STAGE}"

            # pyfunc 모델 로드
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.loaded_version = v.version
        except Exception as e:
            print(f"Failed to load from registry: {e}")
            raise e

holder = ModelHolder()

@app.on_event("startup")
def startup():
    # 서버 기동 시 모델 로드
    holder.load_model()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MLFLOW_MODEL_NAME,
        "stage": MODEL_STAGE,
        "loaded_version": holder.loaded_version,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if holder.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame(req.rows)
    try:
        preds = holder.model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # preds는 numpy/pandas 계열일 수 있음
    return {"predictions": list(map(lambda x: int(x) if str(x).isdigit() else x, preds))}
