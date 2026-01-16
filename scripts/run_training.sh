#!/bin/bash

# 학습 실행 스크립트
# 개발 서버에서 복잡한 docker compose run 명령어 대신 사용합니다.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===================================================="
echo " KCD 예측 시스템 학습 프로세스 시작"
echo "===================================================="

# 인자 처리
if [ -z "$1" ]; then
    echo "사용법: $0 <데이터_경로_또는_--sample>"
    echo "예: $0 /app/data/train_v1.json"
    echo "예: $0 --sample"
    exit 1
fi

DATA_ARG=$1

# Docker Compose를 통해 trainer 실행
# --rm 옵션으로 실행 후 컨테이너 자동 삭제
if [ "$DATA_ARG" == "--sample" ]; then
    echo -e "\n[실행] 샘플 데이터로 학습을 진행합니다..."
    docker compose run --rm trainer python -m src.trainer.train --sample
else
    echo -e "\n[실행] 데이터 경로($DATA_ARG)로 학습을 진행합니다..."
    docker compose run --rm trainer python -m src.trainer.train --data_path "$DATA_ARG"
fi

echo -e "\n===================================================="
echo " 학습 프로세스 종료"
echo " 결과는 MLflow(http://localhost:5001)에서 확인하세요."
echo "===================================================="
