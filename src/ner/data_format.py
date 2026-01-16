"""
NER 학습 데이터 포맷 정의

지원 포맷:
1. JSON (span-based): Doccano, Label Studio 등 라벨링 도구 호환
2. JSONL (line-by-line): 대용량 데이터 처리에 적합
3. BIO Tagged: 토큰 단위 BIO 태그 형식
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class Entity:
    """개체명 엔티티"""
    start: int          # 시작 위치 (character offset)
    end: int            # 끝 위치 (character offset, exclusive)
    label: str          # 엔티티 라벨 (예: "BODY", "DIS_MAIN")
    text: str           # 엔티티 텍스트

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class NERSample:
    """NER 학습 샘플"""
    id: str                         # 샘플 ID
    text: str                       # 원본 텍스트
    entities: list[Entity] = field(default_factory=list)  # 엔티티 리스트
    meta: dict = field(default_factory=dict)  # 메타 정보 (선택)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NERSample":
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        return cls(
            id=data["id"],
            text=data["text"],
            entities=entities,
            meta=data.get("meta", {})
        )

    def validate(self) -> list[str]:
        """데이터 유효성 검증"""
        errors = []

        for i, entity in enumerate(self.entities):
            # 범위 검증
            if entity.start < 0 or entity.end > len(self.text):
                errors.append(f"Entity {i}: 범위 오류 (start={entity.start}, end={entity.end}, text_len={len(self.text)})")

            if entity.start >= entity.end:
                errors.append(f"Entity {i}: start >= end")

            # 텍스트 일치 검증
            actual_text = self.text[entity.start:entity.end]
            if actual_text != entity.text:
                errors.append(f"Entity {i}: 텍스트 불일치 (expected='{entity.text}', actual='{actual_text}')")

        return errors


@dataclass
class NERDataset:
    """NER 데이터셋"""
    samples: list[NERSample] = field(default_factory=list)
    version: str = "1.0"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "samples": [s.to_dict() for s in self.samples]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NERDataset":
        samples = [NERSample.from_dict(s) for s in data.get("samples", [])]
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
    def load_json(cls, path: str | Path, use_morphemes: bool = False) -> "NERDataset":
        """JSON 파일에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = cls.from_dict(data)
        
        if use_morphemes:
            dataset = dataset.apply_morpheme_spacing()
        return dataset

    def save_jsonl(self, path: str | Path):
        """JSONL 파일로 저장 (대용량 데이터용)"""
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path, use_morphemes: bool = False) -> "NERDataset":
        """JSONL 파일에서 로드"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(NERSample.from_dict(json.loads(line)))
        dataset = cls(samples=samples)
        
        if use_morphemes:
            dataset = dataset.apply_morpheme_spacing()
        return dataset

    def apply_morpheme_spacing(self) -> "NERDataset":
        """전체 데이터셋에 형태소 단위 띄어쓰기 및 오프셋 재계산 적용"""
        from src.common.nlp_utils import get_analyzer
        analyzer = get_analyzer()
        new_samples = []

        for sample in self.samples:
            original_text = sample.text
            morpheme_data = analyzer.get_morpheme_offsets(original_text)
            
            # 새로운 텍스트 생성
            processed_text = " ".join([m[0] for m in morpheme_data])
            
            # 오프셋 매핑 테이블
            offset_map = {}
            current_new_idx = 0
            for m_text, m_start, m_end in morpheme_data:
                for original_idx in range(m_start, m_end):
                    offset_map[original_idx] = current_new_idx + (original_idx - m_start)
                current_new_idx += len(m_text) + 1 # 공백 포함

            new_entities = []
            for entity in sample.entities:
                new_start = offset_map.get(entity.start)
                if new_start is None:
                    for i in range(entity.start, entity.end):
                        if i in offset_map:
                            new_start = offset_map[i]
                            break
                
                new_end = offset_map.get(entity.end - 1, -1) + 1
                
                if new_start is not None and new_end > new_start:
                    new_entities.append(Entity(
                        start=new_start,
                        end=new_end,
                        label=entity.label,
                        text=processed_text[new_start:new_end]
                    ))
            
            new_samples.append(NERSample(
                id=sample.id,
                text=processed_text,
                entities=new_entities,
                meta=sample.meta
            ))
            
        return NERDataset(samples=new_samples, version=self.version, description=self.description)

    def validate_all(self) -> dict[str, list[str]]:
        """전체 데이터셋 검증"""
        results = {}
        for sample in self.samples:
            errors = sample.validate()
            if errors:
                results[sample.id] = errors
        return results

    def get_label_stats(self) -> dict[str, int]:
        """라벨별 통계"""
        stats = {}
        for sample in self.samples:
            for entity in sample.entities:
                stats[entity.label] = stats.get(entity.label, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))

    def __len__(self) -> int:
        return len(self.samples)


def create_sample_dataset() -> NERDataset:
    """
    샘플 데이터셋 생성 (테스트/문서화용)
    
    학습 과정 (Step 1): 
    - data/ner/sample_data.json 파일을 로드하여 NERDataset 객체를 생성합니다.
    - 이 데이터는 모델의 입력 형식을 검증하고 최소한의 학습 기능을 테스트하는 데 사용됩니다.
    """
    data_path = Path(__file__).resolve().parents[2] / "data" / "ner" / "sample_data.json"

    if not data_path.exists():
        raise FileNotFoundError(f"샘플 데이터 파일을 찾을 수 없습니다: {data_path}")

    try:
        return NERDataset.load_json(data_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"샘플 데이터 파일 파싱에 실패했습니다: {data_path}") from exc


if __name__ == "__main__":
    # 샘플 데이터셋 생성 및 출력
    dataset = create_sample_dataset()

    print("=" * 60)
    print("NER 데이터 포맷 예시")
    print("=" * 60)

    # 검증
    errors = dataset.validate_all()
    if errors:
        print("\n[검증 오류]")
        for sample_id, errs in errors.items():
            print(f"  {sample_id}: {errs}")
    else:
        print("\n[검증 완료] 모든 샘플 정상")

    # 통계
    print("\n[라벨별 통계]")
    for label, count in dataset.get_label_stats().items():
        print(f"  {label}: {count}개")

    # 샘플 출력
    print("\n[샘플 데이터]")
    for sample in dataset.samples[:1]:
        print(f"\nID: {sample.id}")
        print(f"텍스트: {sample.text}")
        print("엔티티:")
        for e in sample.entities:
            print(f"  - [{e.start}:{e.end}] {e.label}: '{e.text}'")

    # JSON 포맷 출력
    print("\n[JSON 포맷]")
    print(json.dumps(dataset.samples[0].to_dict(), ensure_ascii=False, indent=2))
