# aicheck.py  ─ AI 합성 음성 단독 검사 유틸리티
#
# 이 파일은 단일 오디오 파일에 대해 독립적으로 AI 합성 여부를 검사합니다.
# 통합 시스템(파일main_v6, 인식main_v6)에서는 VoiceAnalyzer.accumulate_ai_prob()
# / finalize_ai_verdict() 방식으로 사후 종합 판정을 수행합니다.
# aicheck.py 는 개발·검증 목적의 독립 실행 도구입니다.

import torch
import librosa
import numpy as np
from PIL import Image
from torchvision import models, transforms
from config import AI_DETECTOR_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR     = 16000

# ── 모델 로드 (이 스크립트 단독 실행 시) ──────────────────────────────
print(f"[aicheck] AI 탐지 모델 로딩 중 ({DEVICE})...")
_model = models.densenet121(pretrained=False)
_model.classifier = torch.nn.Linear(_model.classifier.in_features, 2)
_model.load_state_dict(torch.load(AI_DETECTOR_PATH, map_location=DEVICE))
_model = _model.to(DEVICE).eval()
print("[aicheck] 로드 완료.")

_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def _audio_to_tensor(audio_path: str) -> torch.Tensor:
    """오디오 파일 → LFCC 특징 이미지 텐서 (학습 로직과 동일)."""
    y, _ = librosa.load(audio_path, sr=SR, duration=4.0)
    if len(y) < SR * 4:
        y = np.pad(y, (0, SR * 4 - len(y)))
    S    = np.abs(librosa.stft(y, n_fft=512))
    lfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(S), n_mfcc=40, dct_type=2)
    norm = (lfcc - lfcc.min()) / (lfcc.max() - lfcc.min() + 1e-8) * 255
    img  = Image.fromarray(norm.astype(np.uint8)).convert("RGB").resize((224, 224))
    return _tfm(img).unsqueeze(0).to(DEVICE)


def predict_voice(audio_path: str) -> tuple[str, float]:
    """
    단일 파일에 대한 AI 합성 음성 판정.

    Returns:
        (label: str, probability: float)
        label: "AI 생성 음성" | "실제 인간 음성"
    """
    tensor = _audio_to_tensor(audio_path)
    with torch.no_grad():
        probs     = torch.softmax(_model(tensor), dim=1)
        prob, idx = torch.max(probs, 1)
    label = "AI 생성 음성" if idx.item() == 1 else "실제 인간 음성"
    return label, prob.item()


def check_file(audio_path: str) -> None:
    """결과를 수사 보고서 형식으로 출력."""
    border = "═" * 50
    label, prob = predict_voice(audio_path)

    print(f"\n╔{border}╗")
    print(f"║{'  🔍  AI 음성 탐지 단독 분석 결과':^50}║")
    print(f"╚{border}╝")
    print(f"  · 분석 파일     : {audio_path}")
    print(f"  · 판정 결과     : {label}")
    print(f"  · AI 생성 확률  : {prob*100:.2f}%")
    print(f"{border}")
    if label == "AI 생성 음성" and prob >= 0.75:
        print("  ⚠️  주의: 높은 확률로 AI 합성 음성입니다.")
    elif label == "실제 인간 음성":
        print("  ✅  실제 인간이 발화한 음성으로 판단됩니다.")
    print(f"{border}\n")


# ── 단독 실행 예시 ────────────────────────────────────────────────────
if __name__ == "__main__":
    AUDIO_SAMPLE = "sample_call.mp3"   # 검사할 파일 경로로 변경
    check_file(AUDIO_SAMPLE)
