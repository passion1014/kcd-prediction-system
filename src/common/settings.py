import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "autogluon-demo")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "AutogluonTabularClassifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
