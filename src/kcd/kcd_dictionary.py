"""
KCD (한국표준질병사인분류) 코드 사전

KCD 코드 체계:
- 대분류: 알파벳 기준 (A-Z)
- 중분류: 알파벳 + 숫자 2자리 (예: A00-A09)
- 소분류: 알파벳 + 숫자 2자리 (예: A00)
- 세분류: 소분류 + 점 + 숫자 (예: A00.0)

예시:
- S82: 아래다리의 골절 (소분류)
- S82.0: 무릎뼈의 골절 (세분류)
- S82.1: 경골 근위부의 골절 (세분류)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class KCDCode:
    """KCD 코드 정보"""
    code: str                    # KCD 코드 (예: S82.0)
    name: str                    # 한글명 (예: 무릎뼈의 골절)
    name_en: str = ""            # 영문명 (선택)
    category: str = ""           # 대분류 (예: S00-T98 손상, 중독)
    subcategory: str = ""        # 중분류 (예: S80-S89 무릎 및 아래다리 손상)
    parent_code: str = ""        # 상위 코드 (예: S82)

    @property
    def level(self) -> str:
        """코드 레벨 반환 (대/중/소/세)"""
        if "." in self.code:
            return "세분류"
        elif len(self.code) == 3:
            return "소분류"
        elif "-" in self.code:
            return "중분류"
        else:
            return "대분류"

    @property
    def base_code(self) -> str:
        """세분류의 경우 소분류 코드 반환"""
        if "." in self.code:
            return self.code.split(".")[0]
        return self.code

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category,
            "subcategory": self.subcategory,
            "parent_code": self.parent_code,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KCDCode":
        return cls(
            code=data["code"],
            name=data["name"],
            name_en=data.get("name_en", ""),
            category=data.get("category", ""),
            subcategory=data.get("subcategory", ""),
            parent_code=data.get("parent_code", ""),
        )


class KCDDictionary:
    """KCD 코드 사전"""

    # 대분류 정의 (ICD-10 기반)
    CATEGORIES = {
        "A00-B99": "특정 감염성 및 기생충성 질환",
        "C00-D48": "신생물",
        "D50-D89": "혈액 및 조혈기관의 질환과 면역메커니즘을 침범하는 특정 장애",
        "E00-E90": "내분비, 영양 및 대사 질환",
        "F00-F99": "정신 및 행동 장애",
        "G00-G99": "신경계통의 질환",
        "H00-H59": "눈 및 눈 부속기의 질환",
        "H60-H95": "귀 및 유돌의 질환",
        "I00-I99": "순환계통의 질환",
        "J00-J99": "호흡계통의 질환",
        "K00-K93": "소화계통의 질환",
        "L00-L99": "피부 및 피하조직의 질환",
        "M00-M99": "근골격계통 및 결합조직의 질환",
        "N00-N99": "비뇨생식계통의 질환",
        "O00-O99": "임신, 출산 및 산후기",
        "P00-P96": "출생전후기에 기원한 특정 병태",
        "Q00-Q99": "선천 기형, 변형 및 염색체 이상",
        "R00-R99": "달리 분류되지 않은 증상, 징후와 임상 및 검사의 이상소견",
        "S00-T98": "손상, 중독 및 외인에 의한 특정 기타 결과",
        "V01-Y98": "질병이환 및 사망의 외인",
        "Z00-Z99": "건강상태 및 보건서비스 접촉에 영향을 주는 요인",
    }

    def __init__(self):
        self.codes: dict[str, KCDCode] = {}
        self._load_sample_codes()

    def _load_sample_codes(self):
        """샘플 KCD 코드 로드 (실제 구현시 DB 또는 파일에서 로드)"""
        sample_codes = [
            # 호흡기 질환 (J00-J99)
            KCDCode("J00", "급성 비인두염(감기)", category="J00-J99", subcategory="J00-J06"),
            KCDCode("J00.0", "급성 비인두염", parent_code="J00", category="J00-J99"),
            KCDCode("J01", "급성 부비동염", category="J00-J99", subcategory="J00-J06"),
            KCDCode("J03", "급성 편도염", category="J00-J99", subcategory="J00-J06"),
            KCDCode("J06", "다발성 및 상세불명 부위의 급성 상기도 감염", category="J00-J99"),
            KCDCode("J18", "상세불명 병원체의 폐렴", category="J00-J99", subcategory="J09-J18"),
            KCDCode("J20", "급성 기관지염", category="J00-J99", subcategory="J20-J22"),
            KCDCode("J45", "천식", category="J00-J99", subcategory="J40-J47"),

            # 소화기 질환 (K00-K93)
            KCDCode("K21", "위-식도 역류병", category="K00-K93", subcategory="K20-K31"),
            KCDCode("K25", "위궤양", category="K00-K93", subcategory="K20-K31"),
            KCDCode("K29", "위염 및 십이지장염", category="K00-K93", subcategory="K20-K31"),
            KCDCode("K29.0", "급성 출혈성 위염", parent_code="K29", category="K00-K93"),
            KCDCode("K29.1", "기타 급성 위염", parent_code="K29", category="K00-K93"),
            KCDCode("K29.7", "상세불명의 위염", parent_code="K29", category="K00-K93"),
            KCDCode("K35", "급성 충수염", category="K00-K93", subcategory="K35-K38"),
            KCDCode("K40", "서혜 헤르니아", category="K00-K93", subcategory="K40-K46"),
            KCDCode("K63", "장의 기타 질환", category="K00-K93", subcategory="K55-K64"),
            KCDCode("K63.5", "용종", parent_code="K63", category="K00-K93"),

            # 내분비/대사 질환 (E00-E90)
            KCDCode("E10", "제1형 당뇨병", category="E00-E90", subcategory="E10-E14"),
            KCDCode("E11", "제2형 당뇨병", category="E00-E90", subcategory="E10-E14"),
            KCDCode("E11.0", "혼수를 동반한 제2형 당뇨병", parent_code="E11", category="E00-E90"),
            KCDCode("E11.9", "합병증이 없는 제2형 당뇨병", parent_code="E11", category="E00-E90"),
            KCDCode("E78", "지단백질 대사장애 및 기타 지질혈증", category="E00-E90"),

            # 근골격계 질환 (M00-M99)
            KCDCode("M17", "무릎관절증", category="M00-M99", subcategory="M15-M19"),
            KCDCode("M17.0", "원발성 양측성 무릎관절증", parent_code="M17", category="M00-M99"),
            KCDCode("M17.1", "기타 원발성 무릎관절증", parent_code="M17", category="M00-M99"),
            KCDCode("M23", "무릎의 내부 이상", category="M00-M99", subcategory="M20-M25"),
            KCDCode("M54", "등통증", category="M00-M99", subcategory="M50-M54"),
            KCDCode("M54.5", "요통", parent_code="M54", category="M00-M99"),
            KCDCode("M79", "달리 분류되지 않은 연조직 장애", category="M00-M99"),
            KCDCode("M79.3", "지방층염", parent_code="M79", category="M00-M99"),

            # 손상 (S00-T98)
            KCDCode("S00", "머리의 표재성 손상", category="S00-T98", subcategory="S00-S09"),
            KCDCode("S02", "두개골 및 안면골의 골절", category="S00-T98", subcategory="S00-S09"),
            KCDCode("S06", "두개내 손상", category="S00-T98", subcategory="S00-S09"),
            KCDCode("S22", "늑골, 흉골 및 흉추의 골절", category="S00-T98", subcategory="S20-S29"),
            KCDCode("S32", "요추 및 골반의 골절", category="S00-T98", subcategory="S30-S39"),
            KCDCode("S42", "어깨 및 위팔의 골절", category="S00-T98", subcategory="S40-S49"),
            KCDCode("S42.0", "쇄골의 골절", parent_code="S42", category="S00-T98"),
            KCDCode("S52", "아래팔의 골절", category="S00-T98", subcategory="S50-S59"),
            KCDCode("S52.5", "하부 요골의 골절", parent_code="S52", category="S00-T98"),
            KCDCode("S62", "손목 및 손 부위의 골절", category="S00-T98", subcategory="S60-S69"),
            KCDCode("S72", "대퇴골의 골절", category="S00-T98", subcategory="S70-S79"),
            KCDCode("S72.0", "대퇴골 목의 골절", parent_code="S72", category="S00-T98"),
            KCDCode("S82", "아래다리의 골절", category="S00-T98", subcategory="S80-S89"),
            KCDCode("S82.0", "무릎뼈의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.1", "경골 근위부의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.2", "경골 골간의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.3", "경골 원위부의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.4", "비골 단독의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.5", "내측 복사뼈의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S82.6", "외측 복사뼈의 골절", parent_code="S82", category="S00-T98"),
            KCDCode("S83", "무릎의 관절 및 인대의 탈구, 염좌 및 긴장", category="S00-T98", subcategory="S80-S89"),
            KCDCode("S83.0", "무릎뼈의 탈구", parent_code="S83", category="S00-T98"),
            KCDCode("S83.2", "반월상 연골의 현재 파열", parent_code="S83", category="S00-T98"),
            KCDCode("S92", "발의 골절", category="S00-T98", subcategory="S90-S99"),
            KCDCode("S93", "발목 부위의 관절 및 인대의 탈구, 염좌 및 긴장", category="S00-T98"),
            KCDCode("S93.4", "발목의 염좌 및 긴장", parent_code="S93", category="S00-T98"),
        ]

        for code in sample_codes:
            self.codes[code.code] = code

    def get_code(self, code: str) -> Optional[KCDCode]:
        """코드로 KCD 정보 조회"""
        return self.codes.get(code)

    def search_by_name(self, keyword: str) -> list[KCDCode]:
        """이름으로 검색"""
        keyword = keyword.lower()
        results = []
        for code in self.codes.values():
            if keyword in code.name.lower() or keyword in code.name_en.lower():
                results.append(code)
        return results

    def get_children(self, parent_code: str) -> list[KCDCode]:
        """하위 코드 조회"""
        results = []
        for code in self.codes.values():
            if code.parent_code == parent_code:
                results.append(code)
        return results

    def get_by_category(self, category: str) -> list[KCDCode]:
        """대분류별 코드 조회"""
        results = []
        for code in self.codes.values():
            if code.category == category:
                results.append(code)
        return results

    def get_all_codes(self) -> list[str]:
        """전체 코드 목록 반환"""
        return list(self.codes.keys())

    def get_hierarchy(self, code: str) -> dict:
        """코드의 계층 구조 반환"""
        kcd = self.get_code(code)
        if not kcd:
            return {}

        hierarchy = {
            "code": kcd.code,
            "name": kcd.name,
            "level": kcd.level,
            "category": kcd.category,
            "subcategory": kcd.subcategory,
        }

        # 상위 코드 추가
        if kcd.parent_code:
            parent = self.get_code(kcd.parent_code)
            if parent:
                hierarchy["parent"] = {
                    "code": parent.code,
                    "name": parent.name,
                }

        # 하위 코드 추가
        children = self.get_children(code)
        if children:
            hierarchy["children"] = [
                {"code": c.code, "name": c.name}
                for c in children
            ]

        return hierarchy

    def add_code(self, code: KCDCode):
        """코드 추가"""
        self.codes[code.code] = code

    def save_json(self, path: str):
        """JSON으로 저장"""
        data = {
            "categories": self.CATEGORIES,
            "codes": [code.to_dict() for code in self.codes.values()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "KCDDictionary":
        """JSON에서 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dictionary = cls()
        dictionary.codes = {}
        for code_data in data.get("codes", []):
            code = KCDCode.from_dict(code_data)
            dictionary.codes[code.code] = code

        return dictionary

    def __len__(self) -> int:
        return len(self.codes)

    def __contains__(self, code: str) -> bool:
        return code in self.codes


def extract_code_components(code: str) -> dict:
    """
    KCD 코드의 구성요소 추출

    Args:
        code: KCD 코드 (예: S82.1)

    Returns:
        구성요소 딕셔너리
    """
    result = {
        "original": code,
        "alpha": "",      # 알파벳 부분
        "numeric": "",    # 숫자 부분
        "sub": "",        # 세분류 부분
        "level": "",
    }

    # 패턴: 알파벳(1-2) + 숫자(2) + (. + 숫자)?
    match = re.match(r"([A-Z]+)(\d{2})(?:\.(\d+))?", code.upper())
    if match:
        result["alpha"] = match.group(1)
        result["numeric"] = match.group(2)
        result["sub"] = match.group(3) or ""

        if result["sub"]:
            result["level"] = "세분류"
        else:
            result["level"] = "소분류"

    return result


# 전역 KCD 사전 인스턴스
_kcd_dictionary: Optional[KCDDictionary] = None

def get_kcd_dictionary() -> KCDDictionary:
    """전역 KCD 사전 인스턴스 반환"""
    global _kcd_dictionary
    if _kcd_dictionary is None:
        _kcd_dictionary = KCDDictionary()
    return _kcd_dictionary


if __name__ == "__main__":
    print("=" * 60)
    print("KCD 코드 사전 테스트")
    print("=" * 60)

    kcd = get_kcd_dictionary()
    print(f"\n총 코드 수: {len(kcd)}개")

    # 검색 테스트
    print("\n[검색: 골절]")
    results = kcd.search_by_name("골절")
    for r in results[:5]:
        print(f"  {r.code}: {r.name}")

    # 계층 구조 테스트
    print("\n[계층 구조: S82]")
    hierarchy = kcd.get_hierarchy("S82")
    print(f"  코드: {hierarchy['code']}")
    print(f"  이름: {hierarchy['name']}")
    print(f"  대분류: {hierarchy['category']}")
    if "children" in hierarchy:
        print(f"  하위 코드:")
        for child in hierarchy["children"][:3]:
            print(f"    - {child['code']}: {child['name']}")

    # 코드 분석 테스트
    print("\n[코드 분석: S82.1]")
    components = extract_code_components("S82.1")
    print(f"  원본: {components['original']}")
    print(f"  알파벳: {components['alpha']}")
    print(f"  숫자: {components['numeric']}")
    print(f"  세분류: {components['sub']}")
    print(f"  레벨: {components['level']}")
