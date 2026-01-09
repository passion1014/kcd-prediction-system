"""
KCD 예측 모델 학습용 PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Optional
import random

from src.kcd.data_format import KCDPredictionSample, KCDPredictionDataset


class KCDClassificationDataset(Dataset):
    """
    KCD 분류를 위한 PyTorch Dataset

    텍스트 + NER Feature + Meta Feature → KCD 코드 분류
    """

    def __init__(
        self,
        samples: list[KCDPredictionSample],
        tokenizer: PreTrainedTokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
    ):
        """
        Args:
            samples: KCDPredictionSample 리스트
            tokenizer: HuggingFace 토크나이저
            label2id: KCD 코드 -> ID 매핑
            max_length: 최대 시퀀스 길이
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 모델 입력 텍스트 생성
        input_text = sample.get_model_input()

        # 토큰화
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 라벨 (KCD 코드 -> ID)
        label_id = self.label2id.get(sample.kcd_code, 0)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

    def get_sample(self, idx: int) -> KCDPredictionSample:
        """원본 샘플 반환"""
        return self.samples[idx]


def create_label_mappings(samples: list[KCDPredictionSample]) -> tuple[dict, dict]:
    """
    샘플에서 라벨 매핑 생성

    Args:
        samples: 샘플 리스트

    Returns:
        (label2id, id2label) 튜플
    """
    labels = sorted(set(s.kcd_code for s in samples))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def split_dataset(
    samples: list[KCDPredictionSample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[KCDPredictionSample], list[KCDPredictionSample]]:
    """
    데이터셋을 train/val로 분할

    Args:
        samples: 전체 샘플 리스트
        train_ratio: 학습 데이터 비율
        seed: 랜덤 시드

    Returns:
        (train_samples, val_samples) 튜플
    """
    random.seed(seed)
    samples_copy = samples.copy()
    random.shuffle(samples_copy)

    split_idx = int(len(samples_copy) * train_ratio)
    return samples_copy[:split_idx], samples_copy[split_idx:]


def create_datasets_from_file(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[KCDClassificationDataset, KCDClassificationDataset, dict, dict]:
    """
    파일에서 데이터셋 생성

    Args:
        data_path: JSON 파일 경로
        tokenizer: HuggingFace 토크나이저
        max_length: 최대 시퀀스 길이
        train_ratio: 학습 데이터 비율
        seed: 랜덤 시드

    Returns:
        (train_dataset, val_dataset, label2id, id2label) 튜플
    """
    # 데이터 로드
    dataset = KCDPredictionDataset.load_json(data_path)

    # 라벨 매핑 생성 (전체 데이터 기준)
    label2id, id2label = create_label_mappings(dataset.samples)

    # 분할
    train_samples, val_samples = split_dataset(dataset.samples, train_ratio, seed)

    # Dataset 생성
    train_dataset = KCDClassificationDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    val_dataset = KCDClassificationDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    return train_dataset, val_dataset, label2id, id2label


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from src.kcd.data_format import create_sample_dataset

    print("=" * 60)
    print("KCD Dataset 테스트")
    print("=" * 60)

    # 토크나이저 로드
    MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 샘플 데이터셋
    sample_data = create_sample_dataset()

    # 라벨 매핑 생성
    label2id, id2label = create_label_mappings(sample_data.samples)
    print(f"\n라벨 수: {len(label2id)}개")
    print(f"라벨 목록: {list(label2id.keys())}")

    # PyTorch Dataset 생성
    dataset = KCDClassificationDataset(
        samples=sample_data.samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=128,
    )

    print(f"\n데이터셋 크기: {len(dataset)}")

    # 첫 번째 샘플 확인
    item = dataset[0]
    print(f"\n[첫 번째 샘플]")
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"attention_mask shape: {item['attention_mask'].shape}")
    print(f"label: {item['labels'].item()} ({id2label[item['labels'].item()]})")

    # 원본 샘플
    sample = dataset.get_sample(0)
    print(f"\n원본 텍스트: {sample.text}")
    print(f"모델 입력: {sample.get_model_input()[:100]}...")
