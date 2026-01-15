"""
NER 모델 정의 및 학습/추론 기능

KoELECTRA 기반 Token Classification 모델
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from src.ner.tags import TAGS, label2id, id2label, NUM_TAGS, ENTITY_LABELS


# 기본 모델 설정
DEFAULT_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
DEFAULT_MAX_LENGTH = 128


@dataclass
class NERModelConfig:
    """NER 모델 설정"""
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    num_labels: int = NUM_TAGS
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: str = "./ner_output"

    def save(self, path: str):
        """설정을 JSON으로 저장"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NERModelConfig":
        """JSON에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # NERModelConfig에 정의된 필드만 사용
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class NERModel:
    """
    NER 모델 클래스

    학습, 저장, 로드, 추론 기능 제공
    """

    def __init__(
        self,
        config: Optional[NERModelConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            config: 모델 설정 (새 모델 생성 시)
            model_path: 저장된 모델 경로 (로드 시)
        """
        if model_path:
            self._load_from_path(model_path)
        else:
            self.config = config or NERModelConfig()
            self._initialize_model()

    def _initialize_model(self):
        """모델 및 토크나이저 초기화"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_from_path(self, model_path: str):
        """저장된 모델 로드"""
        path = Path(model_path)

        # 설정 로드
        config_path = path / "config.json"
        if config_path.exists():
            self.config = NERModelConfig.load(str(config_path))
        else:
            self.config = NERModelConfig()

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: Optional[str] = None,
    ):
        """
        모델 학습 (Custom Training Loop)

        학습 과정 (Step 4):
        - DataLoader를 통해 데이터를 배치 단위로 가져옵니다.
        - Optimizer(AdamW)와 Scheduler(Linear Warmup)를 설정합니다.
        - 매 배치마다 Forward pass를 수행하여 Loss(CrossEntropy)를 계산합니다.
        - Backward pass를 통해 가중치를 업데이트하고 학습을 진행합니다.
        - 매 에포크가 끝날 때마다 검증 데이터로 성능을 평가하고 최적의 모델을 저장합니다.

        Args:
            train_dataset: 학습 데이터셋 (NERTokenDataset)
            eval_dataset: 평가 데이터셋 (선택)
            output_dir: 출력 디렉토리

        Returns:
            학습 결과 딕셔너리
        """
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Optimizer 및 Scheduler 설정
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(self.config.warmup_steps, total_steps // 10),
            num_training_steps=total_steps,
        )

        # 학습
        self.model.train()
        best_eval_loss = float("inf")
        training_history = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in progress_bar:
                # 배치 데이터를 디바이스로 이동
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = epoch_loss / len(train_loader)
            epoch_result = {"epoch": epoch + 1, "train_loss": avg_train_loss}

            # 평가
            if eval_loader:
                eval_loss = self._evaluate(eval_loader)
                epoch_result["eval_loss"] = eval_loss
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {eval_loss:.4f}")

                # Best 모델 저장
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.save(output_dir)
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

            training_history.append(epoch_result)

        # 최종 모델 저장 (eval이 없는 경우)
        if not eval_loader:
            self.save(output_dir)

        return {"history": training_history}

    def _evaluate(self, eval_loader: DataLoader) -> float:
        """평가 데이터셋에 대한 loss 계산"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(eval_loader)

    def save(self, output_dir: str):
        """모델 저장"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 모델 및 토크나이저 저장
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # 설정 저장
        self.config.save(str(path / "ner_config.json"))

        # 태그 정보 저장
        with open(path / "tags.json", "w", encoding="utf-8") as f:
            json.dump({
                "tags": TAGS,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            }, f, ensure_ascii=False, indent=2)

        print(f"모델이 '{output_dir}'에 저장되었습니다.")

    def predict(self, text: str) -> list[dict]:
        """
        단일 텍스트에서 개체명 추출

        Args:
            text: 입력 텍스트

        Returns:
            추출된 엔티티 리스트
            [{"text": "...", "label": "...", "start": int, "end": int}, ...]
        """
        self.model.eval()

        # 토큰화
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

        # 엔티티 추출
        entities = self._extract_entities(text, predictions, offset_mapping)

        return entities

    def predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """
        배치 예측

        Args:
            texts: 텍스트 리스트

        Returns:
            각 텍스트에 대한 엔티티 리스트
        """
        return [self.predict(text) for text in texts]

    def _extract_entities(
        self,
        text: str,
        predictions: list[int],
        offset_mapping: list[tuple[int, int]],
    ) -> list[dict]:
        """
        예측 결과에서 엔티티 추출 및 병합

        Args:
            text: 원본 텍스트
            predictions: 토큰별 예측 라벨 ID
            offset_mapping: 토큰별 원본 텍스트 위치

        Returns:
            엔티티 리스트
        """
        entities = []
        current_entity = None

        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            # 특수 토큰 스킵
            if start == 0 and end == 0:
                continue

            label = id2label[pred_id]

            if label.startswith("B-"):
                # 이전 엔티티 저장
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)

                # 새 엔티티 시작
                entity_type = label[2:]
                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                }

            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                # 같은 타입의 엔티티 계속
                if current_entity["label"] == entity_type:
                    current_entity["end"] = end
                else:
                    # 타입이 다르면 이전 엔티티 저장하고 새로 시작
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = {
                        "label": entity_type,
                        "start": start,
                        "end": end,
                    }

            else:  # "O" 태그
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = None

        # 마지막 엔티티 처리
        if current_entity:
            current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            entities.append(current_entity)

        return entities

    def extract_features(self, text: str) -> dict:
        """
        텍스트에서 NER Feature 추출 (KCD 예측 모델 입력용)

        Args:
            text: 입력 텍스트

        Returns:
            라벨별로 그룹화된 엔티티
        """
        entities = self.predict(text)

        # 라벨별 그룹화
        features = {label: [] for label in ENTITY_LABELS}
        for entity in entities:
            label = entity["label"]
            if label in features:
                features[label].append(entity["text"])

        return features


def load_model(model_path: str) -> NERModel:
    """저장된 모델 로드 (편의 함수)"""
    return NERModel(model_path=model_path)


if __name__ == "__main__":
    print("=" * 60)
    print("NER 모델 초기화 테스트")
    print("=" * 60)

    # 모델 초기화
    config = NERModelConfig(
        num_epochs=1,
        batch_size=8,
    )
    model = NERModel(config=config)

    print(f"\n모델: {config.model_name}")
    print(f"라벨 수: {config.num_labels}")
    print(f"디바이스: {model.device}")

    # 추론 테스트 (학습 전이라 결과는 무의미)
    test_text = "환자가 3일전 넘어지면서 좌측 무릎에 통증이 발생하였습니다."
    print(f"\n테스트 텍스트: {test_text}")

    entities = model.predict(test_text)
    print(f"추출된 엔티티 (미학습 상태): {entities}")

    features = model.extract_features(test_text)
    print(f"Feature 추출 결과: {features}")
