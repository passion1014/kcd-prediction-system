"""
NER 모델 학습 스크립트

사용법:
    # 샘플 데이터로 학습 (테스트용)
    python -m src.ner.train --sample

    # 실제 데이터로 학습
    python -m src.ner.train --data_path data/ner_train.json --output_dir models/ner
"""

import argparse
from pathlib import Path

from src.ner.model import NERModel, NERModelConfig
from src.ner.dataset import NERTokenDataset
from src.ner.data_format import create_sample_dataset, NERDataset


def parse_args():
    parser = argparse.ArgumentParser(description="NER 모델 학습")

    # 데이터 관련
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="학습 데이터 파일 경로 (JSON 또는 JSONL)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="샘플 데이터로 학습 (테스트용)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본값: 0.8)",
    )

    # 모델 관련
    parser.add_argument(
        "--model_name",
        type=str,
        default="monologg/koelectra-base-v3-discriminator",
        help="사전학습 모델 이름",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="최대 시퀀스 길이 (기본값: 128)",
    )

    # 학습 관련
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="학습 에포크 수 (기본값: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 크기 (기본값: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="학습률 (기본값: 5e-5)",
    )

    # 출력 관련
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ner_output",
        help="모델 저장 경로",
    )

    return parser.parse_args()


def main():
    """
    NER 모델 학습 메인 프로세스
    
    학습 과정 (Step 1 & 3):
    1. --sample 옵션 사용 시 sample_data.json을 로드합니다.
    2. NERModelConfig를 통해 하이퍼파라미터(에포크, 배치 크기 등)를 설정합니다.
    3. 로드된 데이터를 학습용과 검증용으로 분할합니다.
    """
    args = parse_args()

    print("=" * 60)
    print("NER 모델 학습")
    print("=" * 60)

    # 설정 생성
    config = NERModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
    )

    print(f"\n[설정]")
    print(f"  모델: {config.model_name}")
    print(f"  최대 길이: {config.max_length}")
    print(f"  배치 크기: {config.batch_size}")
    print(f"  에포크: {config.num_epochs}")
    print(f"  학습률: {config.learning_rate}")
    print(f"  출력 경로: {config.output_dir}")

    # 모델 초기화
    print("\n[모델 초기화]")
    model = NERModel(config=config)
    print(f"  디바이스: {model.device}")

    # 데이터 로드
    print("\n[데이터 로드]")
    if args.sample:
        print("  샘플 데이터 사용")
        dataset = create_sample_dataset()
        samples = dataset.samples

        # 간단히 train/val 분할
        split_idx = max(1, int(len(samples) * args.train_ratio))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:] if split_idx < len(samples) else []

    elif args.data_path:
        print(f"  데이터 파일: {args.data_path}")
        path = Path(args.data_path)

        if path.suffix == ".jsonl":
            dataset = NERDataset.load_jsonl(path)
        else:
            dataset = NERDataset.load_json(path)

        # 데이터 검증
        errors = dataset.validate_all()
        if errors:
            print(f"  경고: {len(errors)}개 샘플에 오류 발견")
            for sample_id, errs in list(errors.items())[:3]:
                print(f"    {sample_id}: {errs[0]}")

        # 분할
        import random
        samples = dataset.samples.copy()
        random.shuffle(samples)
        split_idx = int(len(samples) * args.train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

    else:
        print("  오류: --data_path 또는 --sample 옵션을 지정하세요.")
        return

    print(f"  학습 샘플: {len(train_samples)}개")
    print(f"  검증 샘플: {len(val_samples)}개")

    # 라벨 통계
    all_samples = train_samples + val_samples
    label_stats = {}
    for sample in all_samples:
        for entity in sample.entities:
            label_stats[entity.label] = label_stats.get(entity.label, 0) + 1

    print(f"\n[라벨 분포]")
    for label, count in sorted(label_stats.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}개")

    # Dataset 생성
    print("\n[Dataset 생성]")
    train_dataset = NERTokenDataset(
        samples=train_samples,
        tokenizer=model.tokenizer,
        max_length=config.max_length,
    )

    val_dataset = None
    if val_samples:
        val_dataset = NERTokenDataset(
            samples=val_samples,
            tokenizer=model.tokenizer,
            max_length=config.max_length,
        )

    print(f"  학습 데이터셋: {len(train_dataset)}개")
    if val_dataset:
        print(f"  검증 데이터셋: {len(val_dataset)}개")

    # 학습
    print("\n[학습 시작]")
    result = model.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=config.output_dir,
    )

    print("\n[학습 완료]")
    print(f"  모델 저장 위치: {config.output_dir}")

    # 추론 테스트
    print("\n[추론 테스트]")
    test_texts = [
        "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다.",
        "급성 위염으로 복부 통증을 호소하여 내시경 검사를 시행하였습니다.",
    ]

    for text in test_texts:
        print(f"\n  입력: {text}")
        entities = model.predict(text)
        if entities:
            for e in entities:
                print(f"    - [{e['label']}] {e['text']}")
        else:
            print("    (추출된 엔티티 없음)")


if __name__ == "__main__":
    main()
