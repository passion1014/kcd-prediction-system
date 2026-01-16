#!/bin/bash

# USB 패키징 스크립트
# 현재 프로젝트의 Docker 이미지를 빌드하고 .tar 파일로 추출합니다.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/usb_package"

echo "===================================================="
echo " KCD 예측 시스템 USB 패키징 시작"
echo "===================================================="

# 1. 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 2. Docker 이미지 빌드
echo -e "\n[1/3] Docker 이미지 빌드 중..."
docker compose build

# 3. 이미지 .tar 추출
echo -e "\n[2/3] 이미지를 .tar 파일로 저장 중..."
docker save -o "$OUTPUT_DIR/kcd_images.tar" mlflow api trainer

# 4. 필수 설정 파일 복사
echo -e "\n[3/3] 설정 파일 복사 중..."
cp "$PROJECT_ROOT/docker-compose.yml" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/.env" "$OUTPUT_DIR/" 2>/dev/null || echo "경고: .env 파일이 없습니다. 기본값을 사용합니다."
cp -r "$PROJECT_ROOT/src" "$OUTPUT_DIR/"

echo -e "\n===================================================="
echo " 패키징 완료!"
echo " 위치: $OUTPUT_DIR"
echo " USB에 $OUTPUT_DIR 폴더 전체를 복사하세요."
echo "===================================================="
