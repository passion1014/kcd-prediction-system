import os
import subprocess
import time
import mlflow
from mlflow.tracking import MlflowClient

def _wait_for_mlflow(tracking_uri: str, timeout_sec: int = 30) -> None:
    """MLflow 서버가 응답할 때까지 대기"""
    client = MlflowClient(tracking_uri=tracking_uri)
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            client.search_experiments(max_results=1)
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"MLflow not ready: {tracking_uri}. last_err={last_err!r}")

def _wait_for_registered_model(client: MlflowClient, name: str, timeout_sec: int = 60):
    """모델이 Registry에 등록될 때까지 대기"""
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            return client.get_registered_model(name)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise AssertionError(f"Registered model not found: {name}. last_err={last_err!r}")

def test_train_register_smoke():
    # 1. 호스트(테스트 실행 환경)에서 접근할 MLflow URI (localhost:5001)
    tracking_uri_host = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri_host)

    # MLflow 서버 준비 대기(방어 로직)
    _wait_for_mlflow(tracking_uri_host, timeout_sec=30)

    # 2. 트레이너 컨테이너 내부에서 사용할 MLflow URI (docker network 내부 DNS)
    tracking_uri_container = "http://mlflow:5000"

    # 이미지 빌드
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.trainer", "-t", "trainer:local", "."],
        check=True,
    )

    # 3. 학습 컨테이너 실행
    # - --network ml_default: docker-compose 네트워크 사용 (OS 독립적)
    # - -v .../mlruns: 로컬 파일 기반 artifact 공유를 위해 볼륨 마운트 필요
    cwd = os.getcwd()
    subprocess.run(
        [
            "docker", "run", "--rm",
            "--network", "ml_default",
            "-v", f"{cwd}/mlruns:/mlflow/mlruns",
            "-e", f"MLFLOW_TRACKING_URI={tracking_uri_container}",
            "-e", "MLFLOW_EXPERIMENT_NAME=autogluon-demo",
            "-e", "MLFLOW_MODEL_NAME=AutogluonTabularClassifier",
            "trainer:local",
        ],
        check=True,
    )

    # 4. 검증 (Host 관점)
    client = MlflowClient(tracking_uri=tracking_uri_host)
    # 비동기 등록 대기
    model = _wait_for_registered_model(client, "AutogluonTabularClassifier", timeout_sec=60)
    assert model.name == "AutogluonTabularClassifier"
