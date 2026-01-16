# KCD 코드 예측 시스템 (MLOps Template)

> **목적**: 의료 텍스트에서 KCD(한국표준질병분류) 코드를 자동으로 예측하는 시스템
>
> **기반**: AutoGluon + MLflow + FastAPI 템플릿 위에 NER 및 KCD 예측 모듈 추가

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           입력 텍스트                                    │
│  "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하여 골절 진단"            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        NER 모델 (src/ner/)                              │
│  KoELECTRA 기반 Token Classification                                    │
│  출력: BODY(무릎), SIDE(좌측), DIS_MAIN(골절), TIME(3일전), ...          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Feature 결합                                       │
│  NER Feature + Meta Feature (나이, 성별, 진료과목 등)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     KCD 예측 모델 (src/kcd/)                             │
│  KoELECTRA 기반 Sequence Classification                                  │
│  출력: S82.0 (무릎뼈의 골절)                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
ml/
├── src/
│   ├── ner/                    # NER (개체명 인식) 모듈
│   │   ├── __init__.py
│   │   ├── tags.py             # BIO 태그 정의 (17개)
│   │   ├── data_format.py      # 학습 데이터 포맷
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── model.py            # NER 모델 클래스
│   │   ├── train.py            # 학습 스크립트
│   │   └── inference.py        # 추론 스크립트
│   │
│   ├── kcd/                    # KCD 예측 모듈
│   │   ├── __init__.py
│   │   ├── kcd_dictionary.py   # KCD 코드 사전
│   │   ├── data_format.py      # 입력 Feature 정의
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── model.py            # KCD 예측 모델
│   │   ├── train.py            # 학습 스크립트
│   │   └── pipeline.py         # 전체 파이프라인
│   │
│   ├── api/                    # FastAPI 서버
│   │   └── main.py
│   ├── common/                 # 공통 설정
│   │   └── settings.py
│   └── trainer/                # AutoGluon 학습 (MLflow 연동)
│       ├── train.py
│       └── ag_pyfunc_model.py
│
├── data/
│   ├── ner/                    # NER 학습 데이터
│   │   └── sample_data.json
│   └── kcd/                    # KCD 학습 데이터
│       └── sample_data.json
│
├── ner_output/                 # 학습된 NER 모델
├── kcd_output/                 # 학습된 KCD 모델
├── test/                       # 테스트 코드
├── docs/                       # 문서
│   └── code/history/           # 개발 히스토리
│
├── docker-compose.yml          # MLflow + API 컨테이너
├── Dockerfile.api
├── Dockerfile.trainer
├── Dockerfile.mlflow
└── requirements.*.txt
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install torch transformers tqdm scikit-learn pandas

# 작업 디렉토리 이동
cd /Users/passion1014/project/axlrator/ml
```

### 2. NER 모델 학습

```bash
# 샘플 데이터로 테스트 학습
python -m src.ner.train --sample --epochs 3 --output_dir ./ner_output

# 실제 데이터로 학습
python -m src.ner.train --data_path data/ner/train.json --epochs 10 --output_dir ./ner_output
```

### 3. KCD 예측 모델 학습

```bash
# 샘플 데이터로 테스트 학습
python -m src.kcd.train --sample --epochs 5 --output_dir ./kcd_output

# 실제 데이터로 학습
python -m src.kcd.train --data_path data/kcd/train.json --epochs 10 --output_dir ./kcd_output
```

### 4. 파이프라인 사용

```python
from src.kcd.pipeline import KCDPredictionPipeline

# 파이프라인 생성
pipeline = KCDPredictionPipeline(
    ner_model_path="./ner_output",
    kcd_model_path="./kcd_output"
)

# 예측
result = pipeline.predict(
    text="환자가 좌측 무릎 골절로 수술을 받았습니다.",
    age=45,
    gender="M",
    department="정형외과"
)

print(f"예측 KCD: {result.top_prediction['code']} - {result.top_prediction['name']}")
```

---

## NER 모듈 상세

### 태그 체계

| 구분 | 태그 | 설명 | 예시 |
|-----|------|------|------|
| 핵심 | `BODY` | 신체부위 | 무릎, 어깨, 복부 |
| 핵심 | `SIDE` | 좌/우/양측 | 좌측, 우측, 양쪽 |
| 핵심 | `DIS_MAIN` | 주진단/병명 | 골절, 위염, 당뇨 |
| 핵심 | `SYMPTOM` | 증상 | 통증, 호흡곤란 |
| 맥락 | `CAUSE` | 사고원인 | 낙상, 교통사고 |
| 맥락 | `TIME` | 시점/기간 | 3일전, 급성 |
| 맥락 | `TEST` | 검사/영상 | X-ray, CT, MRI |
| 맥락 | `TREATMENT` | 치료/수술 | 수술, 약물치료 |

### 데이터 포맷

```json
{
  "id": "sample_001",
  "text": "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다.",
  "entities": [
    {"start": 4, "end": 7, "label": "TIME", "text": "3일전"},
    {"start": 14, "end": 16, "label": "SIDE", "text": "좌측"},
    {"start": 17, "end": 19, "label": "BODY", "text": "무릎"},
    {"start": 21, "end": 23, "label": "SYMPTOM", "text": "통증"}
  ]
}
```

### CLI 명령어

```bash
# 태그 체계 확인
python -m src.ner.tags

# 데이터 포맷 테스트
python -m src.ner.data_format

# 학습
python -m src.ner.train --data_path <파일> --epochs 10 --batch_size 16

# 추론
python -m src.ner.inference --model_path ./ner_output --text "텍스트"

# Feature 추출 모드
python -m src.ner.inference --model_path ./ner_output --text "텍스트" --extract_features
```

---

## KCD 모듈 상세

### KCD 코드 사전

- 57개 샘플 코드 포함 (확장 가능)
- 대분류/중분류/소분류/세분류 계층 구조 지원

```python
from src.kcd import get_kcd_dictionary

kcd = get_kcd_dictionary()

# 검색
results = kcd.search_by_name("골절")

# 계층 구조 조회
hierarchy = kcd.get_hierarchy("S82")
```

### 입력 Feature 구조

```python
from src.kcd import NERFeatures, MetaFeatures

# NER에서 추출한 Feature
ner_features = NERFeatures(
    body=["무릎"],
    side=["좌측"],
    dis_main=["골절"],
    symptom=["통증"]
)

# 환자 메타 정보
meta_features = MetaFeatures(
    age=45,
    gender="M",
    reception_route="응급",
    department="정형외과"
)
```

### CLI 명령어

```bash
# KCD 사전 테스트
python -m src.kcd.kcd_dictionary

# 데이터 포맷 테스트
python -m src.kcd.data_format

# 학습
python -m src.kcd.train --data_path <파일> --epochs 10 --batch_size 16
```

---

## Docker 기반 실행 (MLflow + API)

### 1. 서비스 기동

```bash
# 환경 파일 준비
cp .env.example .env

# MLflow + API 기동
docker compose up -d --build
```

### 2. AutoGluon 모델 학습 (기존 템플릿)

```bash
# 트레이너 이미지 빌드
docker build -f Dockerfile.trainer -t trainer:local .

# 학습 실행
docker run --rm --network ml_default \
  -v $(pwd)/mlruns:/mlflow/mlruns \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  trainer:local
```

### 3. 서비스 확인

```bash
# MLflow UI
open http://localhost:5001

# API Health Check
curl http://localhost:58080/health
```

---

## 폐쇄망/USB 이동 배포 가이드

인터넷이 제한된 환경(폐쇄망)에서 USB를 통해 이미지를 옮겨 배포하는 방법입니다.

### 1. 개발 서버 (테스트베드용)
개발 서버에서 직접 학습을 수행하고 코드를 수정하며 테스트하는 환경입니다.

- **패키징 (로컬)**: `./scripts/package_for_usb.sh` 실행 (이미지 3종 + 소스 코드 포함)
- **배포 (개발 서버)**:
  1. `docker load -i kcd_images.tar`로 이미지 로드
  2. `docker compose up -d mlflow api`로 서비스 기동
  3. `./scripts/run_training.sh <데이터경로>` 로 학습 수행
- **상세 가이드**: [README_DEV_SERVER.md](scripts/README_DEV_SERVER.md)

### 2. 운영 서버 (서비스 전용)
학습 도구 없이 최적화된 모델만 가지고 독립적으로 추론 서비스를 제공하는 환경입니다.

- **패키징 (개발 서버)**: `./scripts/package_for_prod.sh` 실행 (API 이미지 + 최종 모델 폴더 포함)
- **배포 (운영 서버)**:
  1. `docker load -i kcd_api_image.tar`로 이미지 로드
  2. `docker compose -f docker-compose.prod.yml up -d`로 기동
  3. 58080포트를 통해 즉시 추론 서비스 시작

---

## 개발 히스토리

- `docs/code/history/session_work_summary_20250109.md` - 세션 작업 내용
- `docs/code/history/kcd_project_directory_structure.md` - 디렉토리 구조
- `docs/code/history/kcd_system_test_guide.md` - 테스트 가이드

---

## 향후 개발 계획

| 항목 | 상태 | 설명 |
|-----|------|------|
| NER 모델 | ✅ 완료 | 8개 엔티티 태그, KoELECTRA 기반 |
| KCD 예측 모델 | ✅ 완료 | Sequence Classification |
| 파이프라인 통합 | ✅ 완료 | NER → KCD End-to-end |
| 실제 학습 데이터 | ❌ 필요 | 라벨링된 의료 텍스트 |
| KCD 사전 확장 | ⚠️ 샘플만 | 전체 KCD 코드 DB 연동 |
| KURE-V1 | ❌ 미구현 | 대/중/소/세 계층적 분류 |
| API 서버 통합 | ❌ 미구현 | FastAPI 엔드포인트 추가 |

---

## 참고

### 필수 학습 데이터 규모 (권장)

```
- NER 학습 데이터: 1,000개 이상의 라벨링된 문장
- KCD 학습 데이터: 각 코드당 100개 이상의 샘플
```

### 기술 스택

- **언어 모델**: KoELECTRA (monologg/koelectra-base-v3-discriminator)
- **ML 프레임워크**: PyTorch, Transformers
- **MLOps**: MLflow
- **API**: FastAPI
- **컨테이너**: Docker, Docker Compose

---

## 기존 템플릿 (AutoGluon + MLflow + FastAPI)

아래는 원래 템플릿의 핵심 구성입니다. KCD 예측 시스템과 별개로 사용 가능합니다.

* **(1) Trainer**: AutoGluon으로 학습 → MLflow에 실험/메트릭 기록 + 모델 등록(Registry)
* **(2) Serving**: FastAPI가 MLflow Registry에서 Production 모델을 로드하여 `/predict` 제공
* **Docker 기반**: `docker compose up`로 MLflow 서버 + API 기동

> **설계 포인트**: AutoGluon "폴더 모델"은 MLflow pyfunc로 래핑하여 Registry/Stage(Production) 사용을 가능하게 함

### 환경 설정

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=autogluon-demo
MLFLOW_MODEL_NAME=AutogluonTabularClassifier
MODEL_STAGE=Production
```

### 실행 순서

1. `docker compose up -d --build` - MLflow + API 기동
2. Trainer 실행 - 모델 학습 및 Registry 등록 (Staging)
3. MLflow UI에서 Production 승격
4. `docker compose restart api` - Production 모델 로드
5. `curl http://localhost:58080/health` - 확인

---

## 라이선스

내부 프로젝트용
