from __future__ import annotations

import os
import shutil
import tempfile
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient

from autogluon.tabular import TabularPredictor
from src.common.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
)
from src.trainer.ag_pyfunc_model import AutoGluonPyFuncModel


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 샘플 데이터(오프라인 내장)로 학습 파이프라인 검증
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    label = "target"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label])

    with mlflow.start_run(run_name="autogluon-train") as run:
        run_id = run.info.run_id

        # AutoGluon 모델 학습 (템플릿: 필요 시 presets/time_limit 변경)
        with tempfile.TemporaryDirectory() as tmpdir:
            ag_dir = os.path.join(tmpdir, "ag_model")
            predictor = TabularPredictor(
                label=label,
                path=ag_dir,
                eval_metric="roc_auc",
            ).fit(
                train_data=train_df,
                presets="medium_quality_faster_train",
                time_limit=120,
            )

            # 평가
            perf = predictor.evaluate(test_df)
            # perf는 dict 형태; 주요 메트릭을 MLflow에 기록
            for k, v in perf.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # 파라미터 기록(예시)
            mlflow.log_param("presets", "medium_quality_faster_train")
            mlflow.log_param("time_limit", 120)
            mlflow.log_param("label", label)

            # (A) AutoGluon 폴더 자체를 아티팩트로 저장 (추후 직접 load 가능)
            mlflow.log_artifacts(ag_dir, artifact_path="autogluon_artifacts")

            # (B) Registry를 쓰기 위해 pyfunc 모델로도 등록
            #     - artifacts로 ag_model_dir 전달
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AutoGluonPyFuncModel(),
                artifacts={"ag_model_dir": ag_dir},
                pip_requirements=[
                    "mlflow==2.15.0",
                    "autogluon.tabular==1.5.0",
                    "pandas==2.2.2",
                ],
            )

            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)

    # 자동으로 최신 버전을 Staging으로 올려두는 예시(선택)
    client = MlflowClient()
    latest_version = result.version
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered model: name={MLFLOW_MODEL_NAME}, version={latest_version}, stage=Staging")


if __name__ == "__main__":
    main()
