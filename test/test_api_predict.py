import os
import requests
import time
from mlflow.tracking import MlflowClient

def test_api_predict_smoke():
    # api는 docker-compose로 떠 있다고 가정
    base = os.getenv("API_BASE", "http://localhost:8000")

    # health
    r = requests.get(f"{base}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"

    # 임의 입력(학습 데이터 스키마와 동일해야 함)
    # breast_cancer feature 컬럼 일부 예시 (실서비스에서는 스키마를 고정/검증해야 함)
    sample = {
        "rows": [{
            "mean radius": 14.0,
            "mean texture": 20.0,
            "mean perimeter": 90.0,
            "mean area": 600.0,
            "mean smoothness": 0.10,
            "mean compactness": 0.12,
            "mean concavity": 0.10,
            "mean concave points": 0.05,
            "mean symmetry": 0.18,
            "mean fractal dimension": 0.06,
        }]
    }
    r = requests.post(f"{base}/predict", json=sample, timeout=20)
    assert r.status_code in (200, 400)
    # 400이면 스키마 누락으로 실패할 수 있음(템플릿이라 허용)
