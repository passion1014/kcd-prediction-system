"""
NER 학습용 PyTorch Dataset

Span 기반 데이터를 토큰 단위 BIO 태그로 변환하여 학습에 사용
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Optional

from src.ner.tags import label2id, TAGS
from src.ner.data_format import NERSample, NERDataset


class NERTokenDataset(Dataset):
    """
    NER 학습용 PyTorch Dataset

    Span 기반 어노테이션을 토큰 단위 BIO 태그로 변환
    """

    def __init__(
        self,
        samples: list[NERSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        label2id: dict = label2id,
    ):
        """
        Args:
            samples: NERSample 리스트
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
            label2id: 라벨 -> ID 매핑
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.processed_data = self._preprocess_all()

    def _preprocess_all(self) -> list[dict]:
        """모든 샘플 전처리"""
        processed = []
        for sample in self.samples:
            try:
                item = self._preprocess_sample(sample)
                if item is not None:
                    processed.append(item)
            except Exception as e:
                print(f"Warning: Failed to process sample {sample.id}: {e}")
        return processed

    def _preprocess_sample(self, sample: NERSample) -> Optional[dict]:
        """
        단일 샘플 전처리: span -> BIO 토큰 태그 변환

        학습 과정 (Step 2):
        - 원본 텍스트를 토크나이저를 통해 토큰 단위로 쪼갭니다.
        - 각 토큰의 위치(offset)와 엔티티의 위치(start, end)를 비교합니다.
        - 시작 토큰에는 'B-', 내부 토큰에는 'I-', 무관한 토큰에는 'O' 태그를 부여합니다.
        - 특수 토큰(CLS, SEP 등)은 손실 계산에서 제외하도록 -100으로 설정합니다.

        Args:
            sample: NERSample 객체

        Returns:
            전처리된 데이터 딕셔너리
        """
        text = sample.text

        # 토큰화 (offset_mapping 포함)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # offset_mapping: [(start, end), ...] - 각 토큰의 원본 텍스트 위치
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # 각 토큰에 대한 BIO 라벨 할당
        labels = []
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            # 특수 토큰 (CLS, SEP, PAD) 또는 빈 토큰
            if token_start == 0 and token_end == 0:
                labels.append(-100)  # CrossEntropyLoss에서 무시됨
                continue

            # 현재 토큰과 겹치는 엔티티 찾기
            label = "O"
            for entity in sample.entities:
                if token_start >= entity.start and token_end <= entity.end:
                    # 토큰이 엔티티 범위 내에 있음
                    if token_start == entity.start:
                        label = f"B-{entity.label}"
                    else:
                        label = f"I-{entity.label}"
                    break
                elif token_start < entity.end and token_end > entity.start:
                    # 부분적으로 겹침 - 엔티티 시작 부분인지 확인
                    if token_start <= entity.start:
                        label = f"B-{entity.label}"
                    else:
                        label = f"I-{entity.label}"
                    break

            labels.append(self.label2id.get(label, self.label2id["O"]))

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample.id,
        }

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> dict:
        item = self.processed_data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }

    def get_sample_id(self, idx: int) -> str:
        """인덱스에 해당하는 샘플 ID 반환"""
        return self.processed_data[idx]["sample_id"]


def create_datasets_from_file(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[NERTokenDataset, NERTokenDataset]:
    """
    파일에서 데이터셋 생성 및 train/val 분할

    Args:
        data_path: JSON 또는 JSONL 파일 경로
        tokenizer: HuggingFace 토크나이저
        max_length: 최대 시퀀스 길이
        train_ratio: 학습 데이터 비율
        seed: 랜덤 시드

    Returns:
        (train_dataset, val_dataset) 튜플
    """
    import random
    from pathlib import Path

    # 데이터 로드
    path = Path(data_path)
    if path.suffix == ".jsonl":
        dataset = NERDataset.load_jsonl(path)
    else:
        dataset = NERDataset.load_json(path)

    # 셔플 및 분할
    samples = dataset.samples.copy()
    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = NERTokenDataset(train_samples, tokenizer, max_length)
    val_dataset = NERTokenDataset(val_samples, tokenizer, max_length)

    return train_dataset, val_dataset


if __name__ == "__main__":
    # 테스트
    from transformers import AutoTokenizer
    from src.ner.data_format import create_sample_dataset

    print("=" * 60)
    print("NER Dataset 테스트")
    print("=" * 60)

    # 토크나이저 로드
    MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 샘플 데이터셋
    sample_dataset = create_sample_dataset()

    # PyTorch Dataset 생성
    ner_dataset = NERTokenDataset(
        samples=sample_dataset.samples,
        tokenizer=tokenizer,
        max_length=64,
    )

    print(f"\n데이터셋 크기: {len(ner_dataset)}")

    # 첫 번째 샘플 확인
    item = ner_dataset[0]
    print(f"\n[첫 번째 샘플]")
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"attention_mask shape: {item['attention_mask'].shape}")
    print(f"labels shape: {item['labels'].shape}")

    # 토큰과 라벨 출력
    tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
    labels = item["labels"].tolist()

    print("\n토큰 | 라벨:")
    for token, label_id in zip(tokens[:20], labels[:20]):
        if label_id == -100:
            label_str = "[IGN]"
        else:
            label_str = TAGS[label_id]
        print(f"  {token:15} | {label_str}")
