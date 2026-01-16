
import MeCab
import os
from pathlib import Path

class MorphemeAnalyzer:
    """MeCab을 사용한 한국어 형태소 분석기 유틸리티"""
    
    def __init__(self):
        # macOS Homebrew 경로 설정
        mecabrc = "/opt/homebrew/etc/mecabrc"
        dicdir = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"
        
        args = f"-r {mecabrc} -d {dicdir}"
        try:
            self.tagger = MeCab.Tagger(args)
        except Exception as e:
            print(f"MeCab 초기화 실패: {e}. 기본 설정으로 시도합니다.")
            self.tagger = MeCab.Tagger()

    def split_morphemes(self, text: str) -> str:
        """문장을 형태소 단위로 띄어쓰기된 문자열로 변환"""
        if not text:
            return ""
        
        parsed = self.tagger.parse(text)
        morphemes = []
        for line in parsed.split("\n"):
            if not line or line == "EOS":
                break
            parts = line.split("\t")
            if len(parts) >= 1:
                morphemes.append(parts[0])
        
        return " ".join(morphemes)

    def get_morpheme_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """
        형태소와 그에 해당하는 원본 텍스트의 시작/끝 위치 반환
        예: "무릎에" -> [("무릎", 0, 2), ("에", 2, 3)]
        """
        if not text:
            return []
            
        parsed = self.tagger.parse(text)
        results = []
        current_idx = 0
        
        for line in parsed.split("\n"):
            if not line or line == "EOS":
                break
            morpheme = line.split("\t")[0]
            
            # 원본 텍스트에서 해당 형태소의 위치 찾기 (공백 무시)
            start = text.find(morpheme, current_idx)
            if start != -1:
                end = start + len(morpheme)
                results.append((morpheme, start, end))
                current_idx = end
            else:
                # 찾지 못한 경우 (특수문자 등) - 길이를 통해 유추
                results.append((morpheme, current_idx, current_idx + len(morpheme)))
                current_idx += len(morpheme)
                
        return results

# 싱글톤 인스턴스
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = MorphemeAnalyzer()
    return _analyzer

def space_morphemes(text: str) -> str:
    """형태소 단위로 띄어쓰기 수행 (편의 함수)"""
    return get_analyzer().split_morphemes(text)

if __name__ == "__main__":
    test_text = "환자가좌측무릎에통증이발생하였습니다."
    print(f"원본: {test_text}")
    print(f"결과: {space_morphemes(test_text)}")
    print(f"오프셋: {get_analyzer().get_morpheme_offsets(test_text)}")
