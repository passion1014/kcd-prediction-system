"""
NER 추론 스크립트

사용법:
    # 단일 텍스트 추론
    python -m src.ner.inference --model_path models/ner --text "환자가 좌측 무릎 골절로 수술을 받았습니다."

    # 파일 입력 추론
    python -m src.ner.inference --model_path models/ner --input_file data/test.txt --output_file results.json
"""

import argparse
import json
from pathlib import Path

from src.ner.model import NERModel, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="NER 추론")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="학습된 모델 경로",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="추론할 텍스트 (단일)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="입력 파일 경로 (한 줄에 하나의 텍스트)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (JSON)",
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Feature 형식으로 추출 (KCD 예측 모델 입력용)",
    )

    return parser.parse_args()


def format_entities(entities: list[dict]) -> str:
    """엔티티 리스트를 보기 좋게 포맷팅"""
    if not entities:
        return "(없음)"

    lines = []
    for e in entities:
        lines.append(f"  [{e['label']:12}] {e['text']}")
    return "\n".join(lines)


def format_features(features: dict) -> str:
    """Feature 딕셔너리를 보기 좋게 포맷팅"""
    lines = []
    for label, values in features.items():
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    return "\n".join(lines) if lines else "(추출된 Feature 없음)"


def main():
    args = parse_args()

    print("=" * 60)
    print("NER 추론")
    print("=" * 60)

    # 모델 로드
    print(f"\n[모델 로드] {args.model_path}")
    model = load_model(args.model_path)
    print(f"  디바이스: {model.device}")

    results = []

    # 단일 텍스트 추론
    if args.text:
        print(f"\n[입력 텍스트]")
        print(f"  {args.text}")

        if args.extract_features:
            features = model.extract_features(args.text)
            print(f"\n[추출된 Feature]")
            print(format_features(features))
            results.append({
                "text": args.text,
                "features": features,
            })
        else:
            entities = model.predict(args.text)
            print(f"\n[추출된 엔티티]")
            print(format_entities(entities))
            results.append({
                "text": args.text,
                "entities": entities,
            })

    # 파일 입력 추론
    elif args.input_file:
        print(f"\n[입력 파일] {args.input_file}")

        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"  텍스트 수: {len(texts)}개")

        for i, text in enumerate(texts):
            print(f"\n--- [{i+1}/{len(texts)}] ---")
            print(f"입력: {text}")

            if args.extract_features:
                features = model.extract_features(text)
                print(f"Feature:\n{format_features(features)}")
                results.append({
                    "text": text,
                    "features": features,
                })
            else:
                entities = model.predict(text)
                print(f"엔티티:\n{format_entities(entities)}")
                results.append({
                    "text": text,
                    "entities": entities,
                })

    else:
        # 대화형 모드
        print("\n[대화형 모드] 텍스트를 입력하세요 (종료: quit)")

        while True:
            try:
                text = input("\n> ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    break
                if not text:
                    continue

                if args.extract_features:
                    features = model.extract_features(text)
                    print(format_features(features))
                else:
                    entities = model.predict(text)
                    print(format_entities(entities))

            except KeyboardInterrupt:
                print("\n종료")
                break

    # 결과 저장
    if args.output_file and results:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[결과 저장] {args.output_file}")


if __name__ == "__main__":
    main()
