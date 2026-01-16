#!/bin/bash

# 운영 서버용 패키징 스크립트
# API 서버 이미지와 최종 모델(Artifacts)만 추출하여 배포용 패키지를 만듭니다.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/prod_package"
MODEL_PATH=$1

echo "===================================================="
echo " KCD 예측 시스템 운영 서버(Production) 패키징 시작"
echo "===================================================="

# 1. 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR/prod_model"

# 2. 모델 경로 확인 (인자가 없으면 mlruns에서 최신 모델 검색)
if [ -z "$MODEL_PATH" ]; then
    echo "정보: 모델 경로가 지정되지 않았습니다. 최신 모델을 검색합니다..."
    # 가장 최근에 수정된 mlruns 내 model 폴더 검색
    LATEST_MODEL=$(find "$PROJECT_ROOT/mlruns" -type d -name "model" -printf '%T@ %p\n' | sort -rn | head -1 | cut -f2- -d" ")
    
    if [ -z "$LATEST_MODEL" ]; then
        echo "오류: 학습된 모델을 찾을 수 없습니다. 직접 경로를 지정하세요: $0 <model_path>"
        exit 1
    fi
    MODEL_PATH="$LATEST_MODEL"
fi

echo "사용 모델: $MODEL_PATH"

# 3. Docker API 이미지 빌드
echo -e "\n[1/3] API 서버 이미지 빌드 중..."
docker build -t api:latest -f "$PROJECT_ROOT/Dockerfile.api" "$PROJECT_ROOT"

# 4. 이미지 및 모델 저장
echo -e "\n[2/3] 이미지 및 모델 파일 복사 중..."
docker save -o "$OUTPUT_DIR/kcd_api_image.tar" api:latest
cp -r "$MODEL_PATH"/* "$OUTPUT_DIR/prod_model/"

# 5. 운영 설정 파일 복사
echo -e "\n[3/3] 운영 설정 파일 복사 중..."
cp "$PROJECT_ROOT/docker-compose.prod.yml" "$OUTPUT_DIR/"
cp -r "$PROJECT_ROOT/src" "$OUTPUT_DIR/"

echo -e "\n===================================================="
echo " 운영 서버용 패키징 완료!"
echo " 위치: $OUTPUT_DIR"
echo " USB에 $OUTPUT_DIR 폴더 전체를 복사하여 운영 서버로 옮기세요."
echo "===================================================="
