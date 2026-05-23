# model_kluebert.py  ─ KLUE-BERT 이진 분류 (모듈 로드 시 1회만 초기화)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH, LABELS

print("[KLUE-BERT] 모델 로딩 중...")
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model.to(_device).eval()
print(f"[KLUE-BERT] 로드 완료 ({_device})")

def detect_batch(sentences: list[str]) -> list[tuple[str, float]]:
    """
    문장 배치 → [(label, confidence), ...]
    label: "정상" | "보이스피싱"
    """
    if not sentences: return []
    inputs = _tokenizer(sentences, return_tensors="pt",
                        padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return [(LABELS[preds[i].item()], probs[i][preds[i]].item())
            for i in range(len(sentences))]
