# 세션 작업 내용 요약

> 작성일: 2025-01-09

---

## 작업 목표

아키텍처 이미지 기반 **KCD(한국표준질병분류) 코드 예측 시스템** 개발

```
사고내용 → NER 모델 → Feature 추출 → KCD 예측 모델 → 최종 KCD 코드
```

---

## 1단계: NER 모델 개발

| 작업 | 설명 |
|-----|------|
| 태그 체계 설계 | 8개 엔티티 (BODY, SIDE, DIS_MAIN, SYMPTOM, CAUSE, TIME, TEST, TREATMENT) |
| 데이터 포맷 정의 | JSON 기반 span 어노테이션 형식 |
| PyTorch Dataset | span → BIO 태그 변환 |
| 모델 구현 | KoELECTRA 기반 Token Classification |
| 학습/추론 스크립트 | Custom training loop 구현 |

---

## 2단계: KCD 예측 모델 개발

| 작업 | 설명 |
|-----|------|
| KCD 사전 구현 | 57개 샘플 코드, 계층 구조 지원 |
| 데이터 포맷 정의 | NER Feature + Meta Feature 결합 |
| PyTorch Dataset | 텍스트 분류용 Dataset |
| 모델 구현 | KoELECTRA 기반 Sequence Classification |
| 파이프라인 통합 | NER → KCD 예측 End-to-end |

---

## 생성된 파일 목록

### `src/ner/` (NER 모듈) - 7개 파일 생성

| 파일 | 설명 |
|-----|------|
| `__init__.py` | 모듈 초기화 및 export |
| `tags.py` | BIO 태그 정의 (17개 태그) |
| `data_format.py` | 학습 데이터 포맷 (Entity, NERSample, NERDataset) |
| `dataset.py` | PyTorch Dataset (NERTokenDataset) |
| `model.py` | NER 모델 클래스 (NERModel, NERModelConfig) |
| `train.py` | 학습 스크립트 |
| `inference.py` | 추론 스크립트 |

### `src/kcd/` (KCD 모듈) - 7개 파일 생성

| 파일 | 설명 |
|-----|------|
| `__init__.py` | 모듈 초기화 및 export |
| `kcd_dictionary.py` | KCD 코드 사전 (KCDCode, KCDDictionary) |
| `data_format.py` | 학습 데이터 포맷 (NERFeatures, MetaFeatures, KCDPredictionSample) |
| `dataset.py` | PyTorch Dataset (KCDClassificationDataset) |
| `model.py` | KCD 예측 모델 클래스 (KCDPredictionModel) |
| `train.py` | 학습 스크립트 |
| `pipeline.py` | 전체 파이프라인 (KCDPredictionPipeline) |

### `data/` (샘플 데이터) - 2개 파일 생성

| 파일 | 설명 |
|-----|------|
| `data/ner/sample_data.json` | NER 샘플 데이터 (3개) |
| `data/kcd/sample_data.json` | KCD 샘플 데이터 (5개) |

### 학습된 모델 - 2개 디렉토리 생성

| 디렉토리 | 설명 |
|-----|------|
| `ner_output/` | 학습된 NER 모델 (~449MB) |
| `kcd_output/` | 학습된 KCD 모델 (~452MB) |

---

## 수정된 파일

기존 파일 수정 없음 (모두 새로 생성)

---

## 전체 파일 구조

```
/Users/passion1014/project/axlrator/ml/
├── src/
│   ├── ner/                    # [NEW] NER 모듈
│   │   ├── __init__.py
│   │   ├── tags.py
│   │   ├── data_format.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── kcd/                    # [NEW] KCD 모듈
│   │   ├── __init__.py
│   │   ├── kcd_dictionary.py
│   │   ├── data_format.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── pipeline.py
│   ├── api/                    # (기존)
│   ├── common/                 # (기존)
│   └── trainer/                # (기존)
├── data/
│   ├── ner/
│   │   └── sample_data.json    # [NEW]
│   └── kcd/
│       └── sample_data.json    # [NEW]
├── ner_output/                 # [NEW] 학습된 NER 모델
├── kcd_output/                 # [NEW] 학습된 KCD 모델
└── docs/
    └── code/
        └── history/            # [NEW] 문서
```

---

## 사용법 요약

```bash
# NER 학습
python -m src.ner.train --sample --epochs 3

# NER 추론
python -m src.ner.inference --model_path ./ner_output --text "텍스트"

# KCD 학습
python -m src.kcd.train --sample --epochs 5

# 전체 파이프라인 사용
from src.kcd.pipeline import KCDPredictionPipeline
pipeline = KCDPredictionPipeline(ner_model_path="./ner_output", kcd_model_path="./kcd_output")
result = pipeline.predict(text="...", age=45, gender="M", department="정형외과")
```

---

## 테스트 결과

| 항목 | 상태 | 결과 |
|-----|------|------|
| NER 모델 로드 | ✅ 성공 | 모델 정상 작동 |
| NER 추출 | ⚠️ 제한적 | 학습 데이터 부족으로 정확도 낮음 |
| KCD 사전 | ✅ 성공 | 57개 코드 검색 가능 |
| KCD 예측 모델 | ✅ 성공 | 5개 라벨 분류 가능 |
| 전체 파이프라인 | ✅ 성공 | End-to-end 동작 확인 |

---

## 현재 한계점

- **NER 모델**: 샘플 3개로 학습 → 엔티티 추출 부정확
- **KCD 모델**: 샘플 5개로 학습 → 분류 정확도 낮음
- 모든 예측이 비슷한 확률(~22%)로 나옴

---

## 실제 사용을 위해 필요한 것

| 항목 | 상태 | 설명 |
|-----|------|------|
| **라벨링 데이터** | ❌ 필요 | 실제 의료 텍스트 + NER 어노테이션 |
| **KCD 코드 사전** | ⚠️ 샘플만 | 전체 KCD 코드 DB 연동 필요 |
| **학습 데이터** | ❌ 필요 | 텍스트 → KCD 코드 매핑 데이터 |
| **KURE-V1** | ❌ 미구현 | 대/중/소/세 계층적 분류 |

```
최소 권장 데이터 규모:
- NER 학습 데이터: 1,000개 이상의 라벨링된 문장
- KCD 학습 데이터: 각 코드당 100개 이상의 샘플
```

---

## 다음 단계 제안

1. **실제 학습 데이터 확보** - NER 라벨링 + KCD 코드 매핑
2. **KCD 사전 확장** - 전체 KCD 코드 DB 연동
3. **KURE-V1 구현** - 계층적 분류 모델
4. **API 서버 구축** - FastAPI로 서비스화
