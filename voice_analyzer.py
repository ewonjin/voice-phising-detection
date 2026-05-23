# voice_analyzer.py  ─ 음성 감정 분석 + AI 합성 탐지 (사후 종합 판정)
#
# ════════════════════════════════════════════════════════════════════
#  핵심 설계: Deferred AI Verdict (사후 종합 판정)
# ════════════════════════════════════════════════════════════════════
#
#  [실시간 분석 단계]
#    · get_arousal_state()    → Arousal 추론 → RiskAccumulator 에 전달
#    · accumulate_ai_prob()   → AI 탐지 확률을 _buf 에만 저장, 화면 미출력
#
#  [리포트 생성 시점]
#    · finalize_ai_verdict()  → _buf 전체를 위험도 가중 평균으로 종합
#                               → "AI 생성 음성 확률: NN%" 단일 수치 반환
#                               → 리포트에만 포함 (정상 대화 시 호출 안 함)
#
#  이유:
#    매 문장 "AI 88% / 12%" 반복 출력 → 사용자 혼란, 가독성 저하
#    통화 전체 맥락 종합 → 더 신뢰할 수 있는 단일 판정값 제공

import gc
from collections import deque
import numpy as np
import librosa
import torch
from PIL import Image
from torchvision import models, transforms
from config import (EMOTION_MODEL_PATH, AI_DETECTOR_PATH,
                    CALIBRATION_SEC, HISTORY_LEN, SAMPLE_RATE)


class VoiceAnalyzer:
    _MEAN  = [0.485, 0.456, 0.406]
    _STD   = [0.229, 0.224, 0.225]
    _DBMIN = -80.0; _DBMAX = 0.0

    def __init__(
        self,
        emotion_model_path: str   = EMOTION_MODEL_PATH,
        ai_detector_path:   str   = AI_DETECTOR_PATH,
        calibration_sec:    float = CALIBRATION_SEC,
        history_len:        int   = HISTORY_LEN,
        sr:                 int   = SAMPLE_RATE,
    ):
        self.emotion_model_path = emotion_model_path
        self.ai_detector_path   = ai_detector_path
        self.calibration_sec    = calibration_sec
        self.sr                 = sr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VoiceAnalyzer] 장치: {self.device}")

        self._em_model: torch.nn.Module | None = None   # Lazy Loading
        self._ai_model: torch.nn.Module | None = None

        self._calib:    list[float] = []
        self._baseline: float | None = None
        self.is_calibrated = False
        self._hist: deque[float] = deque(maxlen=history_len)

        # ── AI 탐지 누적 버퍼 ─────────────────────────────────────────
        # (fake_probability, risk_weight) 쌍을 저장.
        # risk_weight = 해당 시점 누적 위험 점수 → 위험 구간의 판정이 더 중요
        self._buf: list[tuple[float, float]] = []

        self._tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ── Lazy Loading ─────────────────────────────────────────────────
    def _em(self) -> torch.nn.Module:
        if self._em_model is None:
            print("[VoiceAnalyzer] 감정 모델 로딩 중...")
            m = models.densenet121(num_classes=4)
            s = torch.load(self.emotion_model_path, map_location=self.device, weights_only=False)
            m.load_state_dict(s if isinstance(s, dict) else s.state_dict(), strict=False)
            self._em_model = m.to(self.device).eval()
        return self._em_model

    def _ai(self) -> torch.nn.Module:
        if self._ai_model is None:
            print("[VoiceAnalyzer] AI 탐지 모델 로딩 중...")
            m = models.densenet121(pretrained=False)
            m.classifier = torch.nn.Linear(m.classifier.in_features, 2)
            m.load_state_dict(torch.load(self.ai_detector_path, map_location=self.device))
            self._ai_model = m.to(self.device).eval()
        return self._ai_model

    # ── 전처리 ───────────────────────────────────────────────────────
    def _stft_tensor(self, audio: np.ndarray) -> torch.Tensor:
        mx = np.max(np.abs(audio))
        if mx > 1e-9: audio = audio / mx
        db  = librosa.amplitude_to_db(np.abs(librosa.stft(audio)))
        clp = np.clip(db, self._DBMIN, self._DBMAX)
        u8  = ((clp - self._DBMIN) / (self._DBMAX - self._DBMIN) * 255).astype(np.uint8)
        img = Image.fromarray(u8).convert("RGB").resize((224, 224))
        return self._tfm(img).unsqueeze(0).to(self.device)

    def _lfcc_tensor(self, audio: np.ndarray) -> torch.Tensor:
        t = self.sr * 4
        audio = np.pad(audio, (0, max(0, t - len(audio))))[:t]
        S    = np.abs(librosa.stft(audio, n_fft=512))
        lf   = librosa.feature.mfcc(S=librosa.amplitude_to_db(S), n_mfcc=40, dct_type=2)
        n    = (lf - lf.min()) / (lf.max() - lf.min() + 1e-8) * 255
        img  = Image.fromarray(n.astype(np.uint8)).convert("RGB").resize((224, 224))
        return self._tfm(img).unsqueeze(0).to(self.device)

    # ── 실시간 API ───────────────────────────────────────────────────
    def _raw_arousal(self, audio: np.ndarray) -> float:
        with torch.no_grad():
            p = torch.softmax(self._em()(self._stft_tensor(audio)), dim=1)[0]
        return (p[0] + p[1]).item()

    def feed_calibration(self, audio: np.ndarray, timestamp: float) -> bool:
        """초기 N초 Arousal 수집 → 베이스라인 확정."""
        if self.is_calibrated: return True
        self._calib.append(self._raw_arousal(audio))
        if timestamp >= self.calibration_sec and self._calib:
            self._baseline = float(np.mean(self._calib))
            self.is_calibrated = True
            print(f"\n[VoiceAnalyzer] ✅ 베이스라인 확정: {self._baseline:.4f}\n")
            return True
        return False

    def get_arousal_state(self, audio: np.ndarray) -> tuple[float, float]:
        """
        현재 이동평균 Arousal 과 베이스라인 반환.
        → risk_calculator._arousal_comp() 에 직접 전달.
        """
        if not self.is_calibrated or self._baseline is None:
            return 0.5, 0.5
        self._hist.append(self._raw_arousal(audio))
        return float(np.mean(self._hist)), self._baseline

    def accumulate_ai_prob(self, audio: np.ndarray, risk_weight: float = 1.0) -> None:
        """
        ★ 핵심: AI 탐지 확률을 내부 버퍼에만 저장. 결과를 출력하거나 반환하지 않음.

        실시간으로 "AI 확률 88%" 를 반복 출력하지 않고,
        (fake_prob, risk_weight) 쌍만 _buf 에 누적.
        → finalize_ai_verdict() 에서 가중 평균으로 종합.
        """
        with torch.no_grad():
            p = torch.softmax(self._ai()(self._lfcc_tensor(audio)), dim=1)
            fake_prob = p[0][1].item()
        self._buf.append((fake_prob, max(risk_weight, 0.1)))

    # ── 리포트 생성 시점 API ─────────────────────────────────────────
    def finalize_ai_verdict(self) -> tuple[float, str]:
        """
        ★ 핵심: 누적된 AI 탐지 확률을 종합하여 최종 판정.
        리포트 생성 시점에만 호출. 정상 대화일 때는 호출하지 않음.

        알고리즘: 위험도 가중 평균
            final = Σ(fake_prob × weight) / Σ(weight)
        위험 점수가 높았던 구간(피싱 발화 시점)의 판정이 더 중요하게 반영.

        Returns:
            (probability: float, label: str)
        """
        if not self._buf:
            return 0.0, "판정 불가 (데이터 없음)"
        probs   = np.array([p for p, _ in self._buf])
        weights = np.array([w for _, w in self._buf])
        prob    = float(np.average(probs, weights=weights))
        label   = "AI 생성 음성" if prob >= 0.5 else "실제 인간 음성"
        return prob, label

    def release(self) -> None:
        self._em_model = self._ai_model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("[VoiceAnalyzer] 메모리 해제 완료.")
