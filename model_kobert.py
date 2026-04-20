import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔥 KoBERT 사전학습 모델 (SKT)
MODEL_NAME = "skt/kobert-base-v1"

LABELS = {
    0: "일상",
    1: "전조",
    2: "위험",
    3: "보이스피싱"
}

print("🧠 KoBERT 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False  # 🔥 핵심 (deadlock 방지)
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)

model.eval()

# 👉 만약 fine-tuning 모델 있다면 경로로 교체
# model = AutoModelForSequenceClassification.from_pretrained("/your/kobert_model")


def detect_batch(sentences):

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    results = []

    for i in range(len(sentences)):
        label = LABELS[preds[i].item()]
        conf = probs[i][preds[i]].item()
        results.append((label, conf))

    return results