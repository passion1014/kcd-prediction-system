import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import joblib

# 1. 설정 및 하이퍼파라미터
MODEL_NAME = "monologg/kobert"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"사용 장치: {DEVICE}")

# 2. 더미 데이터 생성
data = {
    'text': [
        '급성 비인두염', '감기 증상', '콧물과 기침', # J00
        '제2형 당뇨병', '성인 당뇨', '인슐린 비의존 당뇨병', # E11
        '상세불명의 위염', '급성 위염', '속쓰림 및 소화불량' # K29
    ],
    'label': ['J00', 'J00', 'J00', 'E11', 'E11', 'E11', 'K29', 'K29', 'K29']
}
df = pd.DataFrame(data)

# 라벨 인코딩
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['label'])
NUM_LABELS = len(le.classes_)
print(f"분류할 클래스(KCD 코드) 개수: {NUM_LABELS}개 {le.classes_}")

# 3. 데이터셋 클래스 정의
class KCDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 4. 토크나이저 및 데이터로더 로드
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dataset = KCDDataset(df['text'].values, df['encoded_label'].values, tokenizer, MAX_LEN)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. 모델 정의
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
model.to(DEVICE)

# 6. 학습 루프
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("--- 학습 시작 ---")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("--- 학습 완료 ---")

# 7. 모델 저장
OUTPUT_DIR = "./kcd_model_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
print(f"모델과 라벨 인코더가 '{OUTPUT_DIR}'에 저장되었습니다.")

# 8. 저장된 모델 로드 및 테스트
print("\n--- 저장된 모델 로드 테스트 ---")
loaded_tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)
loaded_le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
loaded_model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
loaded_model.to(DEVICE)
loaded_model.eval()

def predict_kcd_code(text):
    encoding = loaded_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
    
    predicted_label = loaded_le.inverse_transform(preds.cpu().data.numpy())[0]
    return predicted_label

test_texts = [
    "환자가 콧물이 심하고 기침을 계속함",
    "혈당 수치가 높고 당뇨 관리가 필요함",
    "위가 쓰리고 소화가 잘 안됨"
]

for text in test_texts:
    code = predict_kcd_code(text)
    print(f"입력: {text} -> 예측 코드: {code}")
