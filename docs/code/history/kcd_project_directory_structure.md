# KCD 코드 예측 시스템 - 디렉토리 구조 및 파일 구성

> 작성일: 2025-01-09

## 프로젝트 디렉토리 구조

```
/Users/passion1014/project/axlrator/ml/
│
├── src/                          # 소스 코드
│   ├── ner/                      # NER (개체명 인식) 모듈
│   ├── kcd/                      # KCD (질병분류코드) 예측 모듈
│   ├── api/                      # FastAPI 서버 (기존)
│   ├── common/                   # 공통 설정 (기존)
│   └── trainer/                  # AutoGluon 학습 (기존)
│
├── data/                         # 데이터 파일
│   ├── ner/                      # NER 학습 데이터
│   └── kcd/                      # KCD 학습 데이터
│
├── ner_output/                   # 학습된 NER 모델
├── kcd_output/                   # 학습된 KCD 모델
├── test/                         # 테스트 코드 (기존)
└── Docker 관련 파일들             # (기존)
```

---

## 1. NER 모듈 (`src/ner/`)

의료 텍스트에서 **개체명(Entity)**을 추출하는 모듈

```
src/ner/
├── __init__.py        # 모듈 export
├── tags.py            # 태그 정의
├── data_format.py     # 데이터 포맷
├── dataset.py         # PyTorch Dataset
├── model.py           # 모델 클래스
├── train.py           # 학습 스크립트
└── inference.py       # 추론 스크립트
```

### 파일별 역할

| 파일 | 주요 클래스/함수 | 역할 |
|-----|----------------|------|
| **tags.py** | `TAGS`, `label2id`, `id2label` | 8개 엔티티의 BIO 태그 정의 |
| **data_format.py** | `Entity`, `NERSample`, `NERDataset` | 학습 데이터 구조 (JSON 형식) |
| **dataset.py** | `NERTokenDataset` | span → BIO 태그 변환, PyTorch Dataset |
| **model.py** | `NERModel`, `NERModelConfig` | KoELECTRA 기반 Token Classification |
| **train.py** | `main()` | CLI 학습 스크립트 |
| **inference.py** | `main()` | CLI 추론 스크립트 |

### 태그 체계 (tags.py)

```python
# 핵심라벨
BODY        # 신체부위 (무릎, 어깨, 복부)
SIDE        # 좌/우/양측
DIS_MAIN    # 주진단/병명 (골절, 위염)
SYMPTOM     # 증상 (통증, 호흡곤란)

# 맥락라벨
CAUSE       # 사고원인 (낙상, 교통사고)
TIME        # 시점/기간 (급성, 3일전)
TEST        # 검사/영상 (X-ray, CT)
TREATMENT   # 치료/수술 (수술, 약물치료)
```

---

## 2. KCD 모듈 (`src/kcd/`)

NER Feature + 메타정보를 바탕으로 **KCD 코드를 예측**하는 모듈

```
src/kcd/
├── __init__.py        # 모듈 export
├── kcd_dictionary.py  # KCD 코드 사전
├── data_format.py     # 데이터 포맷
├── dataset.py         # PyTorch Dataset
├── model.py           # 모델 클래스
├── train.py           # 학습 스크립트
└── pipeline.py        # 전체 파이프라인
```

### 파일별 역할

| 파일 | 주요 클래스/함수 | 역할 |
|-----|----------------|------|
| **kcd_dictionary.py** | `KCDCode`, `KCDDictionary` | KCD 코드 사전 (57개 샘플), 계층 구조 |
| **data_format.py** | `NERFeatures`, `MetaFeatures`, `KCDPredictionSample` | 입력 Feature 구조 정의 |
| **dataset.py** | `KCDClassificationDataset` | PyTorch Dataset for 분류 |
| **model.py** | `KCDPredictionModel`, `KCDModelConfig` | KoELECTRA 기반 Sequence Classification |
| **train.py** | `main()` | CLI 학습 스크립트 |
| **pipeline.py** | `KCDPredictionPipeline` | NER → KCD 예측 통합 파이프라인 |

### 입력 Feature 구조 (data_format.py)

```python
# NER에서 추출한 Feature
NERFeatures:
    body: ["무릎"]
    side: ["좌측"]
    dis_main: ["골절"]
    symptom: ["통증"]
    ...

# 환자 메타 정보
MetaFeatures:
    age: 45
    gender: "M"
    reception_route: "응급"
    department: "정형외과"
    edi_code: ""
    has_edi: False
```

---

## 3. 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                        입력 텍스트                           │
│  "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다."    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     NER 모델 (src/ner/)                      │
│  - model.py: NERModel.predict()                             │
│  - 출력: {"BODY": ["무릎"], "SIDE": ["좌측"], ...}           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature 결합                                │
│  - NER Feature + Meta Feature (나이, 성별, 진료과목 등)       │
│  - data_format.py: NERFeatures + MetaFeatures               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  KCD 예측 모델 (src/kcd/)                    │
│  - model.py: KCDPredictionModel.predict()                   │
│  - 출력: [{"code": "S82.0", "name": "무릎뼈의 골절", ...}]   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     최종 KCD 코드                            │
│  S82.0 - 무릎뼈의 골절                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 사용 예시

### NER 모델 단독 사용

```python
from src.ner import NERModel, load_model

# 모델 로드
model = load_model("./ner_output")

# 예측
entities = model.predict("환자가 좌측 무릎에 통증이 있습니다.")
# [{"label": "SIDE", "text": "좌측"}, {"label": "BODY", "text": "무릎"}, ...]

# Feature 추출
features = model.extract_features("환자가 좌측 무릎에 통증이 있습니다.")
# {"BODY": ["무릎"], "SIDE": ["좌측"], "SYMPTOM": ["통증"], ...}
```

### KCD 예측 모델 단독 사용

```python
from src.kcd import KCDPredictionModel, load_model, NERFeatures, MetaFeatures

# 모델 로드
model = load_model("./kcd_output")

# 예측
results = model.predict(
    text="환자가 좌측 무릎 골절로 수술을 받았습니다.",
    ner_features=NERFeatures(body=["무릎"], side=["좌측"], dis_main=["골절"]),
    meta_features=MetaFeatures(age=45, gender="M", department="정형외과"),
)
# [{"code": "S82.0", "name": "무릎뼈의 골절", "score": 0.85}, ...]
```

### 전체 파이프라인 사용

```python
from src.kcd import KCDPredictionPipeline

# 파이프라인 생성
pipeline = KCDPredictionPipeline(
    ner_model_path="./ner_output",
    kcd_model_path="./kcd_output"
)

# 예측 (NER 자동 추출)
result = pipeline.predict(
    text="환자가 좌측 무릎 골절로 수술을 받았습니다.",
    age=45,
    gender="M",
    department="정형외과"
)

print(result.top_prediction)  # {"code": "S82.0", "name": "무릎뼈의 골절", ...}
```

---

## 5. CLI 명령어

```bash
# NER 학습
python -m src.ner.train --data_path data/ner/train.json --epochs 10 --output_dir ./ner_output

# NER 추론
python -m src.ner.inference --model_path ./ner_output --text "텍스트"

# KCD 학습
python -m src.kcd.train --data_path data/kcd/train.json --epochs 10 --output_dir ./kcd_output
```
