# KCD 예측 시스템 개발서버(테스트베드) 운영 가이드

이 문서는 USB를 통해 옮겨온 Docker 이미지를 활용하여 개발서버에서 학습 및 테스트를 진행하는 방법을 안내합니다.

## 1. 초기 설정 (USB 데이터 로드)

USB에 담긴 폴더를 개발서버의 작업 디렉토리로 복사한 후 아래 명령어를 실행합니다.

```bash
# Docker 이미지 로드
docker load -i kcd_images.tar
```

## 2. 데이터 연결 설정

개발서버의 실제 데이터가 위치한 경로를 `docker-compose.yml`의 `trainer` 서비스 볼륨에 연결합니다.

```yaml
# docker-compose.yml 예시
  trainer:
    ...
    volumes:
      - ./src:/app/src
      - /개발서버/데이터/절대경로:/app/data  # <--- 이 부분을 수정하세요
```

## 3. 서비스 실행 및 확인

```bash
# MLflow 및 API 서버 기동
docker compose up -d mlflow api

# 서비스 상태 확인
curl http://localhost:58080/health
```

## 4. 학습 실행 (In-Container)

개발서버에 있는 데이터를 사용하여 학습을 실행하는 방법은 두 가지입니다.

### 방법 A: 전용 스크립트 사용 (추천 - 간단한 실행)
제공된 `run_training.sh` 스크립트를 사용하면 복잡한 도커 명령어 없이 한 줄로 학습을 시작할 수 있습니다.

```bash
# 샘플 데이터 학습
./scripts/run_training.sh --sample

# 실제 데이터 학습 (컨테이너 내부 경로 기준)
./scripts/run_training.sh /app/data/train_v1.json
```

### 방법 B: 대화형 작업 (코드 수정 및 디버깅)
컨테이너 내부 쉘에 직접 접속하여 여러 번 반복 작업할 때 유용합니다.

```bash
# 1. 학습기(trainer) 컨테이너의 쉘로 접속
docker compose run --name my_work trainer bash

# 2. 컨테이너 내부에서 수동으로 실행
python -m src.trainer.train --data_path /app/data/train_v1.json

# 3. 소스 코드 수정 및 재실행
# 호스트에서 코드를 고치면 컨테이너 안에서 즉시 python 명령어로 테스트 가능합니다.
```

## 5. 모델 업데이트 후 반영

학습이 완료되어 MLflow에 새 모델이 등록되었다면 API 서버를 재시작하여 반영합니다.

```bash
docker compose restart api
```
