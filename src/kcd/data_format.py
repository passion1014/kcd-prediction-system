"""
KCD 예측 모델 학습 데이터 포맷

입력 Feature:
1. NER Feature: NER 모델에서 추출한 개체명 (BODY, SIDE, DIS_MAIN, SYMPTOM, CAUSE, TIME, TEST, TREATMENT)
2. Meta Feature: 환자 메타 정보 (나이, 성별, 접수경로, 진료과목)
3. EDI 여부: EDI 코드 존재 여부 및 매핑 정보
4. 원본 텍스트: 사고내용 텍스트

출력:
- KCD 코드 (예: S82.1)
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum


class Gender(Enum):
    """성별"""
    MALE = "M"
    FEMALE = "F"
    UNKNOWN = "U"


class ReceptionRoute(Enum):
    """접수경로"""
    OUTPATIENT = "외래"      # 외래
    EMERGENCY = "응급"       # 응급실
    INPATIENT = "입원"       # 입원
    REFERRAL = "의뢰"        # 타원 의뢰
    UNKNOWN = "기타"


@dataclass
class NERFeatures:
    """NER 모델에서 추출한 Feature"""
    body: list[str] = field(default_factory=list)        # 신체부위
    side: list[str] = field(default_factory=list)        # 좌/우/양측
    dis_main: list[str] = field(default_factory=list)    # 주진단/병명
    symptom: list[str] = field(default_factory=list)     # 증상
    cause: list[str] = field(default_factory=list)       # 사고원인
    time: list[str] = field(default_factory=list)        # 시점/기간
    test: list[str] = field(default_factory=list)        # 검사/영상
    treatment: list[str] = field(default_factory=list)   # 치료/수술

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NERFeatures":
        return cls(
            body=data.get("body", data.get("BODY", [])),
            side=data.get("side", data.get("SIDE", [])),
            dis_main=data.get("dis_main", data.get("DIS_MAIN", [])),
            symptom=data.get("symptom", data.get("SYMPTOM", [])),
            cause=data.get("cause", data.get("CAUSE", [])),
            time=data.get("time", data.get("TIME", [])),
            test=data.get("test", data.get("TEST", [])),
            treatment=data.get("treatment", data.get("TREATMENT", [])),
        )

    def to_text(self) -> str:
        """Feature를 텍스트로 변환 (모델 입력용)"""
        parts = []
        if self.body:
            parts.append(f"[부위] {', '.join(self.body)}")
        if self.side:
            parts.append(f"[방향] {', '.join(self.side)}")
        if self.dis_main:
            parts.append(f"[진단] {', '.join(self.dis_main)}")
        if self.symptom:
            parts.append(f"[증상] {', '.join(self.symptom)}")
        if self.cause:
            parts.append(f"[원인] {', '.join(self.cause)}")
        if self.time:
            parts.append(f"[시간] {', '.join(self.time)}")
        if self.test:
            parts.append(f"[검사] {', '.join(self.test)}")
        if self.treatment:
            parts.append(f"[치료] {', '.join(self.treatment)}")
        return " ".join(parts)

    def is_empty(self) -> bool:
        """모든 필드가 비어있는지 확인"""
        return not any([
            self.body, self.side, self.dis_main, self.symptom,
            self.cause, self.time, self.test, self.treatment
        ])


@dataclass
class MetaFeatures:
    """환자 메타 정보"""
    age: Optional[int] = None                    # 나이
    gender: str = "U"                            # 성별 (M/F/U)
    reception_route: str = "기타"                # 접수경로
    department: str = ""                         # 진료과목
    edi_code: str = ""                          # EDI 코드 (있는 경우)
    has_edi: bool = False                       # EDI 존재 여부

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MetaFeatures":
        return cls(
            age=data.get("age"),
            gender=data.get("gender", "U"),
            reception_route=data.get("reception_route", "기타"),
            department=data.get("department", ""),
            edi_code=data.get("edi_code", ""),
            has_edi=data.get("has_edi", False),
        )

    def to_text(self) -> str:
        """메타 정보를 텍스트로 변환 (모델 입력용)"""
        parts = []
        if self.age is not None:
            parts.append(f"[나이] {self.age}세")
        if self.gender and self.gender != "U":
            gender_str = "남성" if self.gender == "M" else "여성"
            parts.append(f"[성별] {gender_str}")
        if self.reception_route and self.reception_route != "기타":
            parts.append(f"[접수] {self.reception_route}")
        if self.department:
            parts.append(f"[과목] {self.department}")
        if self.has_edi and self.edi_code:
            parts.append(f"[EDI] {self.edi_code}")
        return " ".join(parts)


@dataclass
class KCDPredictionSample:
    """KCD 예측 학습 샘플"""
    id: str                                      # 샘플 ID
    text: str                                    # 원본 사고내용 텍스트
    ner_features: NERFeatures                    # NER 추출 Feature
    meta_features: MetaFeatures                  # 메타 정보
    kcd_code: str                               # 정답 KCD 코드
    kcd_name: str = ""                          # KCD 코드명 (참고용)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "ner_features": self.ner_features.to_dict(),
            "meta_features": self.meta_features.to_dict(),
            "kcd_code": self.kcd_code,
            "kcd_name": self.kcd_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KCDPredictionSample":
        return cls(
            id=data["id"],
            text=data["text"],
            ner_features=NERFeatures.from_dict(data.get("ner_features", {})),
            meta_features=MetaFeatures.from_dict(data.get("meta_features", {})),
            kcd_code=data["kcd_code"],
            kcd_name=data.get("kcd_name", ""),
        )

    def get_model_input(self) -> str:
        """
        모델 입력용 통합 텍스트 생성

        형식: [원본텍스트] [NER Feature] [Meta Feature]
        """
        parts = [self.text]

        ner_text = self.ner_features.to_text()
        if ner_text:
            parts.append(ner_text)

        meta_text = self.meta_features.to_text()
        if meta_text:
            parts.append(meta_text)

        return " ".join(parts)


@dataclass
class KCDPredictionDataset:
    """KCD 예측 데이터셋"""
    samples: list[KCDPredictionSample] = field(default_factory=list)
    version: str = "1.0"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "samples": [s.to_dict() for s in self.samples]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KCDPredictionDataset":
        samples = [KCDPredictionSample.from_dict(s) for s in data.get("samples", [])]
        return cls(
            samples=samples,
            version=data.get("version", "1.0"),
            description=data.get("description", "")
        )

    def save_json(self, path: str | Path):
        """JSON 파일로 저장"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "KCDPredictionDataset":
        """JSON 파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_kcd_codes(self) -> list[str]:
        """데이터셋의 모든 KCD 코드 목록"""
        return list(set(s.kcd_code for s in self.samples))

    def get_label_distribution(self) -> dict[str, int]:
        """KCD 코드별 분포"""
        dist = {}
        for s in self.samples:
            dist[s.kcd_code] = dist.get(s.kcd_code, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def __len__(self) -> int:
        return len(self.samples)


def create_sample_dataset() -> KCDPredictionDataset:
    """샘플 데이터셋 생성 (테스트/문서화용)"""
    samples = [
        KCDPredictionSample(
            id="kcd_001",
            text="환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하여 X-ray 검사 결과 골절로 진단받았습니다.",
            ner_features=NERFeatures(
                body=["무릎"],
                side=["좌측"],
                dis_main=["골절"],
                symptom=["통증"],
                cause=["넘어지면서"],
                time=["3일전"],
                test=["X-ray"],
            ),
            meta_features=MetaFeatures(
                age=45,
                gender="M",
                reception_route="응급",
                department="정형외과",
            ),
            kcd_code="S82.0",
            kcd_name="무릎뼈의 골절",
        ),
        KCDPredictionSample(
            id="kcd_002",
            text="급성 위염으로 인한 복부 통증을 호소하며 내시경 검사 후 약물 치료를 시행하였습니다.",
            ner_features=NERFeatures(
                body=["복부"],
                dis_main=["위염"],
                symptom=["통증"],
                time=["급성"],
                test=["내시경"],
                treatment=["약물 치료"],
            ),
            meta_features=MetaFeatures(
                age=52,
                gender="F",
                reception_route="외래",
                department="소화기내과",
            ),
            kcd_code="K29.1",
            kcd_name="기타 급성 위염",
        ),
        KCDPredictionSample(
            id="kcd_003",
            text="교통사고로 인해 우측 어깨 탈구 진단, CT 촬영 후 수술 예정입니다.",
            ner_features=NERFeatures(
                body=["어깨"],
                side=["우측"],
                dis_main=["탈구"],
                cause=["교통사고"],
                test=["CT"],
                treatment=["수술"],
            ),
            meta_features=MetaFeatures(
                age=28,
                gender="M",
                reception_route="응급",
                department="정형외과",
            ),
            kcd_code="S43.0",
            kcd_name="어깨관절의 탈구",
        ),
        KCDPredictionSample(
            id="kcd_004",
            text="당뇨병 환자로 혈당 조절 불량으로 내원. 인슐린 치료 조절 필요.",
            ner_features=NERFeatures(
                dis_main=["당뇨병"],
                symptom=["혈당 조절 불량"],
                treatment=["인슐린 치료"],
            ),
            meta_features=MetaFeatures(
                age=65,
                gender="M",
                reception_route="외래",
                department="내분비내과",
            ),
            kcd_code="E11.9",
            kcd_name="합병증이 없는 제2형 당뇨병",
        ),
        KCDPredictionSample(
            id="kcd_005",
            text="감기 증상으로 콧물, 기침 호소. 급성 비인두염 진단.",
            ner_features=NERFeatures(
                dis_main=["비인두염"],
                symptom=["콧물", "기침"],
                time=["급성"],
            ),
            meta_features=MetaFeatures(
                age=8,
                gender="F",
                reception_route="외래",
                department="소아청소년과",
            ),
            kcd_code="J00",
            kcd_name="급성 비인두염(감기)",
        ),
    ]

    return KCDPredictionDataset(
        samples=samples,
        version="1.0",
        description="KCD 코드 예측을 위한 샘플 데이터셋"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("KCD 예측 데이터 포맷 테스트")
    print("=" * 60)

    # 샘플 데이터셋 생성
    dataset = create_sample_dataset()

    print(f"\n데이터셋 크기: {len(dataset)}개")
    print(f"KCD 코드 종류: {len(dataset.get_kcd_codes())}개")

    # 라벨 분포
    print("\n[KCD 코드 분포]")
    for code, count in dataset.get_label_distribution().items():
        print(f"  {code}: {count}개")

    # 샘플 출력
    print("\n[샘플 데이터]")
    sample = dataset.samples[0]
    print(f"ID: {sample.id}")
    print(f"원본 텍스트: {sample.text}")
    print(f"NER Features: {sample.ner_features.to_text()}")
    print(f"Meta Features: {sample.meta_features.to_text()}")
    print(f"KCD 코드: {sample.kcd_code} ({sample.kcd_name})")
    print(f"\n모델 입력:")
    print(f"  {sample.get_model_input()}")

    # JSON 출력
    print("\n[JSON 포맷]")
    print(json.dumps(sample.to_dict(), ensure_ascii=False, indent=2))
