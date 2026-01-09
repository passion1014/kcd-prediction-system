# KCD 코드 예측 시스템 - 테스트 가이드

> 작성일: 2025-01-09

---

## 1. 환경 설정

### 필수 패키지

```bash
pip install torch transformers tqdm scikit-learn
```

### 작업 디렉토리

```bash
cd /Users/passion1014/project/axlrator/ml
```

---

## 2. 개별 모듈 테스트

### 2.1 NER 태그 체계 테스트

```bash
python -m src.ner.tags
```

**예상 출력:**
```
============================================================
NER 태그 체계 (BIO Scheme)
============================================================

[핵심라벨 - Core Labels]
  BODY: 신체부위
  SIDE: 좌/우/양측
  DIS_MAIN: 주진단/병명
  SYMPTOM: 증상

[맥락라벨 - Context Labels]
  CAUSE: 사고원인
  TIME: 시점/기간
  TEST: 검사/영상
  TREATMENT: 치료/수술/처치

총 태그 수: 17개
```

### 2.2 NER 데이터 포맷 테스트

```bash
python -m src.ner.data_format
```

**예상 출력:**
```
============================================================
NER 데이터 포맷 예시
============================================================

[검증 완료] 모든 샘플 정상

[라벨별 통계]
  BODY: 4개
  DIS_MAIN: 4개
  ...
```

### 2.3 KCD 사전 테스트

```bash
python -m src.kcd.kcd_dictionary
```

**예상 출력:**
```
============================================================
KCD 코드 사전 테스트
============================================================

총 코드 수: 57개

[검색: 골절]
  S02: 두개골 및 안면골의 골절
  S22: 늑골, 흉골 및 흉추의 골절
  ...
```

### 2.4 KCD 데이터 포맷 테스트

```bash
python -m src.kcd.data_format
```

**예상 출력:**
```
============================================================
KCD 예측 데이터 포맷 테스트
============================================================

데이터셋 크기: 5개
KCD 코드 종류: 5개
...
```

---

## 3. 모델 학습 테스트

### 3.1 NER 모델 학습 (샘플 데이터)

```bash
python -m src.ner.train --sample --epochs 2 --batch_size 2 --output_dir ./ner_output
```

**예상 출력:**
```
============================================================
NER 모델 학습
============================================================

[설정]
  모델: monologg/koelectra-base-v3-discriminator
  ...

[학습 시작]
Epoch 1/2: 100%|██████████| ...
Epoch 1: Train Loss = 2.7915, Eval Loss = 2.6843
모델이 './ner_output'에 저장되었습니다.
...
```

### 3.2 KCD 모델 학습 (샘플 데이터)

```bash
python -m src.kcd.train --sample --epochs 2 --batch_size 2 --output_dir ./kcd_output
```

**예상 출력:**
```
============================================================
KCD 예측 모델 학습
============================================================

[데이터 로드]
  샘플 데이터 사용
  전체 샘플: 5개
  KCD 코드 종류: 5개
  ...

[학습 시작]
Epoch 1/2: 100%|██████████| ...
모델이 './kcd_output'에 저장되었습니다.
```

---

## 4. 모델 추론 테스트

### 4.1 NER 모델 추론

```bash
python -m src.ner.inference --model_path ./ner_output --text "환자가 좌측 무릎에 통증이 있습니다."
```

### 4.2 NER Feature 추출 모드

```bash
python -m src.ner.inference --model_path ./ner_output --text "환자가 좌측 무릎에 통증이 있습니다." --extract_features
```

---

## 5. 전체 시스템 통합 테스트

### 5.1 Python 스크립트로 테스트

```python
# test_full_system.py

print("=" * 70)
print("KCD 코드 예측 시스템 테스트")
print("=" * 70)

# 1. NER 모델 테스트
print("\n[1] NER 모델 테스트")
print("-" * 50)

from src.ner.model import load_model as load_ner_model

ner_model = load_ner_model("./ner_output")
print(f"  모델 로드: 성공")
print(f"  디바이스: {ner_model.device}")

test_text = "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다."
print(f"\n  입력: {test_text}")
features = ner_model.extract_features(test_text)
print(f"  추출된 Feature: {features}")

# 2. KCD 사전 테스트
print("\n\n[2] KCD 사전 테스트")
print("-" * 50)

from src.kcd.kcd_dictionary import get_kcd_dictionary

kcd_dict = get_kcd_dictionary()
print(f"  총 코드 수: {len(kcd_dict)}개")

search_results = kcd_dict.search_by_name("골절")
print(f"\n  '골절' 검색 결과:")
for r in search_results[:3]:
    print(f"    - {r.code}: {r.name}")

# 3. KCD 예측 모델 테스트
print("\n\n[3] KCD 예측 모델 테스트")
print("-" * 50)

from src.kcd.model import load_model as load_kcd_model
from src.kcd.data_format import NERFeatures, MetaFeatures

kcd_model = load_kcd_model("./kcd_output")
print(f"  모델 로드: 성공")
print(f"  라벨 수: {len(kcd_model.label2id)}개")

results = kcd_model.predict(
    text="환자가 좌측 무릎 골절로 수술을 받았습니다.",
    ner_features=NERFeatures(body=["무릎"], side=["좌측"], dis_main=["골절"]),
    meta_features=MetaFeatures(age=45, gender="M", department="정형외과"),
)

print(f"\n  예측 결과:")
for r in results:
    print(f"    - {r['code']}: {r['name']} ({r['score']:.2%})")

# 4. 전체 파이프라인 테스트
print("\n\n[4] 전체 파이프라인 테스트")
print("-" * 50)

from src.kcd.pipeline import KCDPredictionPipeline

pipeline = KCDPredictionPipeline(
    ner_model_path="./ner_output",
    kcd_model_path="./kcd_output"
)
print("  파이프라인 생성: 성공")

test_inputs = [
    {"text": "감기 증상으로 콧물과 기침이 심합니다.", "age": 25, "gender": "M", "department": "내과"},
    {"text": "당뇨병으로 혈당 조절이 필요합니다.", "age": 60, "gender": "F", "department": "내분비내과"},
]

for inp in test_inputs:
    print(f"\n  입력: {inp['text']}")
    result = pipeline.predict(
        text=inp["text"],
        age=inp["age"],
        gender=inp["gender"],
        department=inp["department"],
    )
    print(f"  예측: {result.top_prediction['code']} - {result.top_prediction['name']}")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)
```

### 5.2 실행 방법

```bash
python test_full_system.py
```

---

## 6. 빠른 테스트 (One-liner)

### NER 모듈 전체 테스트

```bash
python -m src.ner.tags && python -m src.ner.data_format && python -m src.ner.dataset
```

### KCD 모듈 전체 테스트

```bash
python -m src.kcd.kcd_dictionary && python -m src.kcd.data_format && python -m src.kcd.dataset
```

---

## 7. 파이프라인 인터랙티브 테스트

```python
# Python 인터프리터에서 실행
from src.kcd.pipeline import KCDPredictionPipeline

# 파이프라인 생성
pipeline = KCDPredictionPipeline(
    ner_model_path="./ner_output",
    kcd_model_path="./kcd_output"
)

# 테스트 1: 골절
result = pipeline.predict(
    text="환자가 계단에서 넘어져 오른쪽 발목이 부었습니다.",
    age=35,
    gender="F",
    department="정형외과"
)
print(result)

# 테스트 2: 감기
result = pipeline.predict(
    text="3일 전부터 기침과 콧물이 나고 열이 납니다.",
    age=8,
    gender="M",
    department="소아청소년과"
)
print(result)

# 테스트 3: 위염
result = pipeline.predict(
    text="속이 쓰리고 소화가 안됩니다. 내시경 검사 필요.",
    age=45,
    gender="M",
    department="소화기내과"
)
print(result)
```

---

## 8. 예상되는 문제 및 해결

### 문제 1: 모델 로드 실패

```
ValueError: labels.json not found
```

**해결:** 모델 학습을 먼저 실행

```bash
python -m src.kcd.train --sample --epochs 2 --output_dir ./kcd_output
```

### 문제 2: CUDA 관련 오류

```
RuntimeError: CUDA out of memory
```

**해결:** CPU 모드로 실행 (자동으로 CPU 사용)

### 문제 3: 정확도가 낮음

**원인:** 샘플 데이터(3~5개)로만 학습됨

**해결:** 실제 학습 데이터 확보 필요 (최소 1,000개 이상 권장)

---

## 9. 테스트 체크리스트

| 항목 | 명령어 | 예상 결과 |
|-----|-------|----------|
| NER 태그 | `python -m src.ner.tags` | 17개 태그 출력 |
| NER 데이터 | `python -m src.ner.data_format` | 검증 완료 |
| NER 학습 | `python -m src.ner.train --sample` | 모델 저장됨 |
| KCD 사전 | `python -m src.kcd.kcd_dictionary` | 57개 코드 |
| KCD 데이터 | `python -m src.kcd.data_format` | 5개 샘플 |
| KCD 학습 | `python -m src.kcd.train --sample` | 모델 저장됨 |
| 파이프라인 | Python 스크립트 | End-to-end 동작 |
