"""
KCD 예측 모델 학습 스크립트

사용법:
    # 샘플 데이터로 학습 (테스트용)
    python -m src.kcd.train --sample

    # 실제 데이터로 학습
    python -m src.kcd.train --data_path data/kcd_train.json --output_dir models/kcd
"""

import argparse
from pathlib import Path

from src.kcd.model import KCDPredictionModel, KCDModelConfig
from src.kcd.dataset import KCDClassificationDataset, create_label_mappings, split_dataset
from src.kcd.data_format import create_sample_dataset, KCDPredictionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="KCD 예측 모델 학습")

    # 데이터 관련
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="학습 데이터 파일 경로 (JSON)",
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
        default=256,
        help="최대 시퀀스 길이 (기본값: 256)",
    )

    # 학습 관련
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="학습 에포크 수 (기본값: 5)",
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
        default=2e-5,
        help="학습률 (기본값: 2e-5)",
    )

    # 출력 관련
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kcd_output",
        help="모델 저장 경로",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("KCD 예측 모델 학습")
    print("=" * 60)

    # 데이터 로드
    print("\n[데이터 로드]")
    if args.sample:
        print("  샘플 데이터 사용")
        dataset = create_sample_dataset()
        samples = dataset.samples
    elif args.data_path:
        print(f"  데이터 파일: {args.data_path}")
        dataset = KCDPredictionDataset.load_json(args.data_path)
        samples = dataset.samples
    else:
        print("  오류: --data_path 또는 --sample 옵션을 지정하세요.")
        return

    print(f"  전체 샘플: {len(samples)}개")

    # 라벨 매핑 생성
    label2id, id2label = create_label_mappings(samples)
    print(f"  KCD 코드 종류: {len(label2id)}개")
    print(f"  코드 목록: {list(label2id.keys())}")

    # 데이터 분할
    train_samples, val_samples = split_dataset(samples, args.train_ratio)
    print(f"  학습 샘플: {len(train_samples)}개")
    print(f"  검증 샘플: {len(val_samples)}개")

    # 설정 생성
    config = KCDModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        num_labels=len(label2id),
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
    model = KCDPredictionModel(
        config=config,
        label2id=label2id,
        id2label=id2label,
    )
    print(f"  디바이스: {model.device}")

    # Dataset 생성
    print("\n[Dataset 생성]")
    train_dataset = KCDClassificationDataset(
        samples=train_samples,
        tokenizer=model.tokenizer,
        label2id=label2id,
        max_length=config.max_length,
    )

    val_dataset = None
    if val_samples:
        val_dataset = KCDClassificationDataset(
            samples=val_samples,
            tokenizer=model.tokenizer,
            label2id=label2id,
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
    test_samples = samples[:3]  # 처음 3개 샘플로 테스트

    for sample in test_samples:
        print(f"\n  입력: {sample.text[:50]}...")
        predictions = model.predict_from_sample(sample, top_k=3)
        print(f"  정답: {sample.kcd_code} ({sample.kcd_name})")
        print(f"  예측:")
        for pred in predictions:
            marker = "✓" if pred["code"] == sample.kcd_code else " "
            print(f"    {marker} {pred['code']}: {pred['name']} ({pred['score']:.4f})")


if __name__ == "__main__":
    main()
