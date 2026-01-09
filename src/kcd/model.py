"""
KCD 예측 모델

KoELECTRA 기반 Sequence Classification 모델
입력: 텍스트 + NER Feature + Meta Feature
출력: KCD 코드 분류
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from src.kcd.data_format import KCDPredictionSample, NERFeatures, MetaFeatures
from src.kcd.kcd_dictionary import get_kcd_dictionary


# 기본 모델 설정
DEFAULT_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
DEFAULT_MAX_LENGTH = 256


@dataclass
class KCDModelConfig:
    """KCD 예측 모델 설정"""
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    num_labels: int = 0              # 동적으로 설정됨
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./kcd_output"

    def save(self, path: str):
        """설정을 JSON으로 저장"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "KCDModelConfig":
        """JSON에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class KCDPredictionModel:
    """
    KCD 예측 모델 클래스

    텍스트 + Feature → KCD 코드 분류
    """

    def __init__(
        self,
        config: Optional[KCDModelConfig] = None,
        model_path: Optional[str] = None,
        label2id: Optional[dict] = None,
        id2label: Optional[dict] = None,
    ):
        """
        Args:
            config: 모델 설정 (새 모델 생성 시)
            model_path: 저장된 모델 경로 (로드 시)
            label2id: KCD 코드 -> ID 매핑
            id2label: ID -> KCD 코드 매핑
        """
        self.kcd_dict = get_kcd_dictionary()

        if model_path:
            self._load_from_path(model_path)
        else:
            if label2id is None or id2label is None:
                raise ValueError("새 모델 생성 시 label2id, id2label이 필요합니다.")

            self.config = config or KCDModelConfig()
            self.label2id = label2id
            self.id2label = id2label
            self.config.num_labels = len(label2id)
            self._initialize_model()

    def _initialize_model(self):
        """모델 및 토크나이저 초기화"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label={str(k): v for k, v in self.id2label.items()},
            label2id=self.label2id,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_from_path(self, model_path: str):
        """저장된 모델 로드"""
        path = Path(model_path)

        # 설정 로드
        config_path = path / "kcd_config.json"
        if config_path.exists():
            self.config = KCDModelConfig.load(str(config_path))
        else:
            self.config = KCDModelConfig()

        # 라벨 매핑 로드
        label_path = path / "labels.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            self.label2id = label_data["label2id"]
            self.id2label = {int(k): v for k, v in label_data["id2label"].items()}
        else:
            raise ValueError(f"labels.json not found in {model_path}")

        self.config.num_labels = len(self.label2id)

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: Optional[str] = None,
    ):
        """
        모델 학습

        Args:
            train_dataset: 학습 데이터셋 (KCDClassificationDataset)
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
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 학습
        self.model.train()
        best_eval_acc = 0
        training_history = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                # 정확도 계산
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct/total:.4f}"
                })

            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
            }

            # 평가
            if eval_loader:
                eval_loss, eval_acc = self._evaluate(eval_loader)
                epoch_result["eval_loss"] = eval_loss
                epoch_result["eval_acc"] = eval_acc
                print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc:.4f}")

                # Best 모델 저장 (eval_acc >= best_eval_acc로 변경하여 첫 번째 에포크에서도 저장)
                if eval_acc >= best_eval_acc:
                    best_eval_acc = eval_acc
                    self.save(output_dir)
            else:
                print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")

            training_history.append(epoch_result)

        # 최종 모델 저장 (항상 마지막 모델 저장)
        self.save(output_dir)

        return {"history": training_history}

    def _evaluate(self, eval_loader: DataLoader) -> tuple[float, float]:
        """평가 데이터셋에 대한 loss 및 accuracy 계산"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

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
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()
        return total_loss / len(eval_loader), correct / total

    def save(self, output_dir: str):
        """모델 저장"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 모델 및 토크나이저 저장
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # 설정 저장
        self.config.save(str(path / "kcd_config.json"))

        # 라벨 매핑 저장
        with open(path / "labels.json", "w", encoding="utf-8") as f:
            json.dump({
                "label2id": self.label2id,
                "id2label": {str(k): v for k, v in self.id2label.items()},
            }, f, ensure_ascii=False, indent=2)

        print(f"모델이 '{output_dir}'에 저장되었습니다.")

    def predict(
        self,
        text: str,
        ner_features: Optional[NERFeatures] = None,
        meta_features: Optional[MetaFeatures] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        KCD 코드 예측

        Args:
            text: 원본 텍스트
            ner_features: NER 추출 Feature (선택)
            meta_features: 메타 정보 (선택)
            top_k: 상위 K개 예측 반환

        Returns:
            예측 결과 리스트 [{"code": "S82.0", "name": "...", "score": 0.95}, ...]
        """
        self.model.eval()

        # 입력 텍스트 구성
        input_parts = [text]
        if ner_features:
            ner_text = ner_features.to_text()
            if ner_text:
                input_parts.append(ner_text)
        if meta_features:
            meta_text = meta_features.to_text()
            if meta_text:
                input_parts.append(meta_text)

        input_text = " ".join(input_parts)

        # 토큰화
        encoding = self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 확률 계산
        probs = torch.softmax(outputs.logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.id2label)))

        results = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            kcd_code = self.id2label[idx]
            kcd_info = self.kcd_dict.get_code(kcd_code)
            kcd_name = kcd_info.name if kcd_info else ""

            results.append({
                "code": kcd_code,
                "name": kcd_name,
                "score": prob,
            })

        return results

    def predict_from_sample(
        self,
        sample: KCDPredictionSample,
        top_k: int = 3,
    ) -> list[dict]:
        """샘플 객체에서 예측"""
        return self.predict(
            text=sample.text,
            ner_features=sample.ner_features,
            meta_features=sample.meta_features,
            top_k=top_k,
        )


def load_model(model_path: str) -> KCDPredictionModel:
    """저장된 모델 로드 (편의 함수)"""
    return KCDPredictionModel(model_path=model_path)


if __name__ == "__main__":
    from src.kcd.data_format import create_sample_dataset
    from src.kcd.dataset import create_label_mappings

    print("=" * 60)
    print("KCD 예측 모델 초기화 테스트")
    print("=" * 60)

    # 샘플 데이터에서 라벨 매핑 생성
    sample_data = create_sample_dataset()
    label2id, id2label = create_label_mappings(sample_data.samples)

    # 모델 초기화
    config = KCDModelConfig(
        num_epochs=1,
        batch_size=4,
    )

    model = KCDPredictionModel(
        config=config,
        label2id=label2id,
        id2label=id2label,
    )

    print(f"\n모델: {config.model_name}")
    print(f"라벨 수: {config.num_labels}")
    print(f"디바이스: {model.device}")

    # 추론 테스트 (학습 전이라 결과는 무의미)
    test_sample = sample_data.samples[0]
    print(f"\n테스트 텍스트: {test_sample.text}")

    results = model.predict_from_sample(test_sample)
    print(f"\n예측 결과 (미학습 상태):")
    for r in results:
        print(f"  {r['code']}: {r['name']} (score: {r['score']:.4f})")
