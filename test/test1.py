# 설치 필요한 라이브러리
# pip install torch transformers scikit-learn seqeval

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset

# ---------------------------------------------------------
# 1. 설정 및 태그 정의 (Configuration)
# ---------------------------------------------------------
MODEL_NAME = "monologg/koelectra-base-v3-discriminator" # 한국어 성능이 우수한 모델
MAX_LEN = 128

# 우리가 정의한 태그 리스트 (BIO Scheme)
# B: 시작, I: 중간, O: 관련없음
TAGS = [
    "O",
    "B-DIS-MAIN", "I-DIS-MAIN",   # 주진단 (예: 용종, 골절)
    "B-DIS-HIST", "I-DIS-HIST",   # 과거력 (예: 위식도 역류성 질환)
    "B-SYMPTOM", "I-SYMPTOM",     # 증상 (예: 호흡곤란, 통증)
    "B-BODY", "I-BODY",           # 신체부위 (예: 대장, 손)
    "B-TREATMENT", "I-TREATMENT", # 치료/수술 (예: 용종점막절제술)
    "B-CAUSE", "I-CAUSE"          # 사고원인 (예: 낙하물)
]

# 태그 <-> ID 매핑 생성
label2id = {tag: i for i, tag in enumerate(TAGS)}
id2label = {i: tag for i, tag in enumerate(TAGS)}

# ---------------------------------------------------------
# 2. 데이터셋 클래스 정의 (Dataset)
# ---------------------------------------------------------
class MedicalNERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        # 실제 학습시는 레이블링 툴(Doccano 등)에서 뽑은 정답 리스트가 들어와야 함
        # 여기서는 데모를 위해 텍스트만 처리하거나, 더미 레이블을 매핑하는 로직이 필요
        # *주의*: 실제 학습 데이터는 문장과 함께 [O, O, B-DIS, ...] 형태의 라벨 리스트가 있어야 함
        
        # 토크나이징 (Subword 단위 분리)
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # 데모용: 실제 정답 라벨이 있다고 가정하고 텐서 변환
        # (실제 프로젝트에선 여기서 subword align 로직이 복잡하게 들어감)
        labels = item.get('labels', [0] * self.max_len) 
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# ---------------------------------------------------------
# 3. 더미 데이터 생성 (Simulation Data)
# ---------------------------------------------------------
# 실제로는 엑셀이나 DB에서 로드한 데이터가 들어와야 합니다.
# 여기서는 학습이 돌아가는 것을 보여주기 위해 임의의 형식만 맞춥니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dummy_data = [
    {
        'text': "위식도 역류성 질환으로 통원 치료를 받던 중 용종점막절제술을 받음",
        # 실제로는 Tokenizer가 자른 토큰 개수에 맞춰 라벨링이 되어 있어야 함 (매우 중요)
        # 예시상 패딩 처리함
        'labels': [0] * MAX_LEN 
    },
    {
        'text': "손이 붓고 고름 생겨 응급실 진료",
        'labels': [0] * MAX_LEN
    }
]

# 데이터셋 인스턴스 생성
train_dataset = MedicalNERDataset(dummy_data, tokenizer, label2id, MAX_LEN)

# ---------------------------------------------------------
# 4. 모델 초기화 및 학습 설정 (Training Setup)
# ---------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(TAGS),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # 에포크 수 (NER은 금방 과적합되므로 적게)
    per_device_train_batch_size=8,
    learning_rate=5e-5,              # 미세조정용 낮은 학습률
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no",              # 데모용이라 저장 안 함
    use_cpu=False                    # GPU 있으면 False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 학습 시작 (더미 데이터라 금방 끝남)
print(">>> 학습 시작 (Dummy Data)...")
trainer.train()
print(">>> 학습 완료!")

# ---------------------------------------------------------
# 5. 추론 및 후처리 로직 (Inference & Post-processing)
# ---------------------------------------------------------
def predict_ner(text, model, tokenizer):
    """
    입력 텍스트에서 Entity를 추출하여 보기 좋게 반환하는 함수
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 입력 처리
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    # 결과 디코딩
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    # 결과 정리 (Subword 병합 및 Entity 그룹화)
    entities = []
    current_entity = {"word": "", "label": None}
    
    for token, label in zip(tokens, predicted_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # Subword 처리 ('##'으로 시작하는 토큰 병합)
        clean_token = token.replace("##", "")
        
        if label.startswith("B-"):
            # 이전 Entity 저장
            if current_entity["word"]:
                entities.append(current_entity)
            # 새 Entity 시작
            current_entity = {"word": clean_token, "label": label[2:]} # "B-" 제거
            
        elif label.startswith("I-") and current_entity["label"] == label[2:]:
            # 현재 Entity에 이어붙이기
            current_entity["word"] += clean_token
            
        else: # "O" 태그이거나 라벨이 끊긴 경우
            if current_entity["word"]:
                entities.append(current_entity)
                current_entity = {"word": "", "label": None}
                
    if current_entity["word"]:
        entities.append(current_entity)

    return entities

# ---------------------------------------------------------
# 6. 실제 테스트 (Demo)
# ---------------------------------------------------------
# *참고*: 학습 데이터가 더미(0)라서 결과는 엉망이겠지만, 로직 흐름은 확인 가능
test_text = "위식도 역류성 질환으로 통원 치료를 받던 중 호흡곤란으로 대장내시경 검사를 받았는데 용종이 커져서 영향을 받은 것으로 용종점막절제술을 받게 된 것입니다."

print("\n>>> [테스트 문장]:", test_text)
result = predict_ner(test_text, model, tokenizer)

print("\n>>> [NER 추출 결과]")
for entity in result:
    print(f"Entity: {entity['word']}  |  Label: {entity['label']}")

# ---------------------------------------------------------
# 7. KCD 코드 매핑용 데이터 구조화 (Tip)
# ---------------------------------------------------------
final_structure = {
    "주진단(MAIN)": [],
    "부위(BODY)": [],
    "증상(SYMPTOM)": [],
    "과거력(HIST)": [],
    "수술(TREATMENT)": []
}

for item in result:
    tag = item['label']
    word = item['word']
    
    if tag == "DIS-MAIN": final_structure["주진단(MAIN)"].append(word)
    elif tag == "BODY": final_structure["부위(BODY)"].append(word)
    elif tag == "SYMPTOM": final_structure["증상(SYMPTOM)"].append(word)
    elif tag == "DIS-HIST": final_structure["과거력(HIST)"].append(word)
    elif tag == "TREATMENT": final_structure["수술(TREATMENT)"].append(word)

print("\n>>> [KCD 매핑을 위한 최종 구조]")
print(final_structure)