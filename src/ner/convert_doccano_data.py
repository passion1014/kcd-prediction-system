
import json
from pathlib import Path
from src.ner.data_format import Entity, NERSample, NERDataset
from src.ner.tags import ENTITY_LABELS
from src.common.nlp_utils import get_analyzer

def normalize_label(label: str) -> str:
    """라벨 명칭 정규화 (예: DIS-MAIN -> DIS_MAIN)"""
    normalized = label.replace("-", "_").upper()
    
    # 정의된 라벨 리스트에 없는 경우 가장 유사한 것으로 매핑하거나 경고
    if normalized not in ENTITY_LABELS:
        # 특별히 처리해야 할 매핑이 있다면 여기에 추가
        if normalized == "DIS_MAIN":
            return "DIS_MAIN"
    return normalized

def convert_doccano_jsonl(input_path: Path, output_path: Path, use_morphemes: bool = True):
    """
    admin.jsonl 포맷을 프로젝트 표준 NERSample 포맷으로 변환
    use_morphemes가 True이면 형태소 단위로 띄어쓰기를 적용하고 위치를 재계산합니다.
    """
    samples = []
    analyzer = get_analyzer()
    
    print(f"변환 시작 (형태소 분석기 적용={use_morphemes}): {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                original_text = data["text"]
                sample_id = str(data.get("id", line_idx))
                
                # 형태소 분석기 적용 및 위치 매핑
                if use_morphemes:
                    morpheme_data = analyzer.get_morpheme_offsets(original_text)
                    # 띄어쓰기된 새로운 텍스트 생성
                    processed_text = " ".join([m[0] for m in morpheme_data])
                    
                    # 원본 오프셋 -> 새로운 텍스트 오프셋 매핑 테이블 생성
                    offset_map = {}
                    current_new_idx = 0
                    for m_text, m_start, m_end in morpheme_data:
                        for original_idx in range(m_start, m_end):
                            offset_map[original_idx] = current_new_idx + (original_idx - m_start)
                        current_new_idx += len(m_text) + 1 # 단어 사이 공백 포함
                else:
                    processed_text = original_text
                
                entities = []
                raw_labels = data.get("label", [])
                
                for start, end, label_name in raw_labels:
                    norm_label = normalize_label(label_name)
                    if norm_label not in ENTITY_LABELS:
                        continue
                        
                    if use_morphemes:
                        # 새로운 텍스트상의 위치 찾기
                        # 시작점이 매핑에 없는 경우 가장 가까운 다음 지점 찾기
                        new_start = offset_map.get(start)
                        if new_start is None:
                            # 원본 텍스트의 공백 등으로 인해 매핑이 없는 경우 보정
                            for i in range(start, end):
                                if i in offset_map:
                                    new_start = offset_map[i]
                                    break
                        
                        # 끝점 보정 (마지막 글자의 다음 인덱스)
                        new_end = offset_map.get(end - 1, -1) + 1
                        
                        if new_start is not None and new_end > new_start:
                            entity_text = processed_text[new_start:new_end]
                            entities.append(Entity(
                                start=new_start,
                                end=new_end,
                                label=norm_label,
                                text=entity_text
                            ))
                    else:
                        entity_text = original_text[start:end]
                        entities.append(Entity(
                            start=start,
                            end=end,
                            label=norm_label,
                            text=entity_text
                        ))
                
                samples.append(NERSample(
                    id=sample_id,
                    text=processed_text,
                    entities=entities,
                    meta=data.get("meta", {})
                ))
                
            except Exception as e:
                print(f"  [오류] 라인 {line_idx+1} 처리 실패: {e}")

    dataset = NERDataset(samples=samples)
    dataset.save_jsonl(output_path)
    print(f"변환 완료! 저장 경로: {output_path}")


if __name__ == "__main__":
    # 경로 설정
    base_dir = Path(__file__).resolve().parents[2]
    input_file = base_dir / "data" / "ner" / "admin.jsonl"
    output_file = base_dir / "data" / "ner" / "admin_processed.jsonl"
    
    # 변환 실행
    if input_file.exists():
        convert_doccano_jsonl(input_file, output_file)
    else:
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_file}")
