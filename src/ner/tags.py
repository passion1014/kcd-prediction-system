"""
NER 태그 정의 (BIO Scheme)

핵심라벨 (Core Labels):
- BODY: 신체부위 (예: 대장, 손, 무릎)
- SIDE: 좌/우/양측 (예: 좌측, 우측, 양측)
- DIS_MAIN: 주진단/병명 (예: 골절, 용종, 당뇨병)
- SYMPTOM: 증상 (예: 통증, 호흡곤란, 부종)

맥락라벨 (Context Labels):
- CAUSE: 사고원인 (예: 낙상, 교통사고, 추락)
- TIME: 시점/기간, 급성/만성 (예: 3일전, 급성, 만성)
- TEST: 검사/영상 (예: X-ray, CT, MRI, 혈액검사)
- TREATMENT: 치료/수술/처치 (예: 절제술, 봉합, 투약)
"""

# 핵심라벨 (Core Labels)
CORE_LABELS = [
    "BODY",      # 신체부위
    "SIDE",      # 좌/우/양측
    "DIS_MAIN",  # 주진단/병명
    "SYMPTOM",   # 증상
]

# 맥락라벨 (Context Labels)
CONTEXT_LABELS = [
    "CAUSE",     # 사고원인
    "TIME",      # 시점/기간, 급성/만성
    "TEST",      # 검사/영상
    "TREATMENT", # 치료/수술/처치
]

# 전체 엔티티 라벨 (BIO 제외)
ENTITY_LABELS = CORE_LABELS + CONTEXT_LABELS

# BIO 태그 생성
def generate_bio_tags(entity_labels: list[str]) -> list[str]:
    """
    엔티티 라벨로부터 BIO 태그 리스트 생성

    Args:
        entity_labels: 엔티티 라벨 리스트

    Returns:
        BIO 태그 리스트 (O 포함)
    """
    tags = ["O"]  # Outside 태그
    for label in entity_labels:
        tags.append(f"B-{label}")  # Begin
        tags.append(f"I-{label}")  # Inside
    return tags

# 전체 BIO 태그 리스트
TAGS = generate_bio_tags(ENTITY_LABELS)

# 태그 <-> ID 매핑
label2id = {tag: i for i, tag in enumerate(TAGS)}
id2label = {i: tag for i, tag in enumerate(TAGS)}

# 태그 개수
NUM_TAGS = len(TAGS)


# 라벨별 설명 (UI/문서화용)
LABEL_DESCRIPTIONS = {
    "BODY": {
        "name_ko": "신체부위",
        "examples": ["대장", "손", "무릎", "허리", "목", "어깨", "발목"],
        "description": "인체의 특정 부위를 나타내는 표현"
    },
    "SIDE": {
        "name_ko": "좌/우/양측",
        "examples": ["좌측", "우측", "양측", "왼쪽", "오른쪽", "양쪽"],
        "description": "신체 부위의 방향을 나타내는 표현"
    },
    "DIS_MAIN": {
        "name_ko": "주진단/병명",
        "examples": ["골절", "용종", "당뇨병", "고혈압", "폐렴", "위염"],
        "description": "주요 진단명 또는 질병명"
    },
    "SYMPTOM": {
        "name_ko": "증상",
        "examples": ["통증", "호흡곤란", "부종", "발열", "기침", "두통"],
        "description": "환자가 호소하는 증상"
    },
    "CAUSE": {
        "name_ko": "사고원인",
        "examples": ["낙상", "교통사고", "추락", "충돌", "미끄러짐"],
        "description": "상해나 질병의 원인"
    },
    "TIME": {
        "name_ko": "시점/기간",
        "examples": ["3일전", "급성", "만성", "1주일간", "어제", "최근"],
        "description": "증상 발생 시점이나 기간, 급/만성 여부"
    },
    "TEST": {
        "name_ko": "검사/영상",
        "examples": ["X-ray", "CT", "MRI", "혈액검사", "초음파", "내시경"],
        "description": "진단을 위한 검사나 영상 촬영"
    },
    "TREATMENT": {
        "name_ko": "치료/수술/처치",
        "examples": ["절제술", "봉합", "투약", "물리치료", "수술", "주사"],
        "description": "치료, 수술, 의료 처치"
    },
}


def get_label_info(label: str) -> dict:
    """
    라벨에 대한 상세 정보 반환

    Args:
        label: 엔티티 라벨 (예: "BODY", "DIS_MAIN")

    Returns:
        라벨 정보 딕셔너리
    """
    return LABEL_DESCRIPTIONS.get(label, {})


def print_tag_summary():
    """태그 체계 요약 출력"""
    print("=" * 60)
    print("NER 태그 체계 (BIO Scheme)")
    print("=" * 60)

    print("\n[핵심라벨 - Core Labels]")
    for label in CORE_LABELS:
        info = LABEL_DESCRIPTIONS[label]
        print(f"  {label}: {info['name_ko']}")
        print(f"    예시: {', '.join(info['examples'][:3])}")

    print("\n[맥락라벨 - Context Labels]")
    for label in CONTEXT_LABELS:
        info = LABEL_DESCRIPTIONS[label]
        print(f"  {label}: {info['name_ko']}")
        print(f"    예시: {', '.join(info['examples'][:3])}")

    print(f"\n총 태그 수: {NUM_TAGS}개")
    print(f"태그 목록: {TAGS}")


if __name__ == "__main__":
    print_tag_summary()
