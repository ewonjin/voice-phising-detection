import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH, LABELS

print("🧠 KLUE-BERT 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def detect_batch(sentences):

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    results = []

    for i in range(len(sentences)):
        label = LABELS[preds[i].item()]
        conf = probs[i][preds[i]].item()
        results.append((label, conf))

    return results