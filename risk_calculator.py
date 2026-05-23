# risk_calculator_v2.py  ─ 멀티모달 위험도 계산
#
# ════════════════════════════════════════════════════════════════════
#  수식 개요
# ════════════════════════════════════════════════════════════════════
#
#  Final_Risk = SafetyNet(
#      clip(
#          TextRisk × w_nlp × Boost(TextRisk)
#        + ArousalComp(current, baseline)      ← 비대칭 가중치
#        + FakeFlag × w_fake
#      )
#  )
#
#  ── 키워드 의존도 완화 (요구사항 반영) ─────────────────────────────
#  ▸ BERT 단독(키워드 0개) 위험: risk = conf × 0.75  → 0.41~0.71
#    (기존 conf × 0.4 → 0.22~0.38 이었으나, BERT 판단을 더 신뢰)
#  ▸ 키워드 기여: log1p(k) × 0.12, 상한 0.35  (기존 × 0.20 무제한)
#    → 키워드 급등 방지, 완만한 누적
#  ▸ 조합 보너스: +0.25  (기존 +0.40 → 완화)
#
#  ── 비대칭 Arousal 가중치 ──────────────────────────────────────────
#  Δ = current − baseline
#  Δ > 0: voice_weight = Δ × 1.5  (압박·흥분 → 최대 포착)
#  Δ < 0: voice_weight = |Δ| × 0.5 (회유·차분 → 소폭 포착)
#
#  ── Safety-Net ────────────────────────────────────────────────────
#  text_risk ≥ 0.85 → final_risk = max(final_risk, 0.85)
#  음성이 침착해도 텍스트 극명 위험 시 탐지 우선권 보장.

import numpy as np

# ── 상수 ────────────────────────────────────────────────────────────
_W_NLP          = 0.65    # 텍스트 채널 가중치
_W_VOICE        = 0.25    # 음성 채널 가중치
_W_FAKE         = 0.10    # AI 탐지 채널 가중치
_RISE_K         = 1.5     # Arousal 상승 계수
_FALL_K         = 0.5     # Arousal 하락 계수
_TEXT_SAFETY    = 0.85    # Safety-Net 임계값
_KW_CAP         = 0.35    # 키워드 기여 상한
_SHIFT_STD      = 0.15    # 태도변화 표준편차 임계값
_SHIFT_RNG      = 0.35    # 태도변화 범위 임계값


# ── 내부 헬퍼 ───────────────────────────────────────────────────────

def _text_boost(tr: float) -> float:
    """Sigmoid 부스터: text_risk ≥ 0.7 이면 최대 1.2배."""
    return 1.0 + 0.2 / (1.0 + np.exp(-10.0 * (tr - 0.7)))

def _arousal_comp(cur: float, base: float) -> tuple[float, str]:
    """
    비대칭 Arousal → voice_component + 감정 상태 레이블.

    상승(Δ>0): raw = Δ × 1.5  → 압박·흥분 최대 반영
    하락(Δ<0): raw = |Δ| × 0.5 → 회유·차분도 포착하되 오탐 완화
    """
    delta = cur - base
    if delta > 0:
        raw   = delta * _RISE_K
        label = f"고각성(Angry) ⬆️  (Baseline 대비 격앙됨)"
    elif delta < 0:
        raw   = abs(delta) * _FALL_K
        label = f"저각성(Neutral/Calm) ⬇️  (Baseline 대비 차분해짐·회유 의심)"
    else:
        raw   = 0.0
        label = f"중립(Neutral) ➡️  (Baseline 동일)"
    return float(np.clip(raw, 0.0, 1.0) * _W_VOICE), label

def _safety_net(final: float, tr: float) -> tuple[float, bool]:
    triggered = tr >= _TEXT_SAFETY
    if triggered:
        final = max(final, _TEXT_SAFETY)
    return final, triggered


# ── 공개 API ────────────────────────────────────────────────────────

def calculate_multimodal_risk(
    label:             str,
    confidence:        float,
    k_score:           int,
    has_combined_risk: bool  = False,
    current_arousal:   float = 0.5,
    baseline_arousal:  float = 0.5,
    is_ai_voice:       bool  = False,
    voice_available:   bool  = False,
    verbose:           bool  = True,
) -> tuple[float, str]:
    """
    텍스트 + 음성 통합 위험도 계산.

    Returns:
        (final_risk: float, tone_label: str)
        tone_label → 실시간 화면의 '🎭 감정 상태' 항목에 직접 사용
    """
    # 1. 텍스트 위험도 ─────────────────────────────────────────────────
    k_contrib = min(float(np.log1p(k_score) * 0.12), _KW_CAP)  # 상한 적용
    text_risk = float(np.clip(confidence + k_contrib, 0.0, 1.0))

    if has_combined_risk:
        text_risk = float(np.clip(text_risk + 0.25, 0.0, 1.0))  # 조합 보너스 완화

    # 키워드 0개 + 보이스피싱 → 기존 ×0.4 제거, conf×0.75 로 적절한 기본값 부여
    if label == "보이스피싱" and k_score == 0:
        text_risk = float(np.clip(confidence * 0.75, 0.0, 1.0))

    # 2. 텍스트 부스터 ─────────────────────────────────────────────────
    text_comp = text_risk * _W_NLP * _text_boost(text_risk)

    # 3. 비대칭 Arousal 음성 성분 ──────────────────────────────────────
    tone_label = "━ 음성 분석 대기 중"
    if voice_available:
        voice_comp, tone_label = _arousal_comp(current_arousal, baseline_arousal)
        fake_comp = _W_FAKE if is_ai_voice else 0.0
        raw = text_comp + voice_comp + fake_comp
    else:
        raw = text_comp

    final = float(np.clip(raw, 0.0, 1.0))

    # 4. Safety-Net ─────────────────────────────────────────────────────
    final, triggered = _safety_net(final, text_risk)
    if triggered and verbose:
        print("   [🛡️  Safety-Net] 텍스트 단독 고위험 → 최솟값 0.85 보장")

    return final, tone_label


def calculate_risk(
    label: str, confidence: float,
    k_score: int, has_combined_risk: bool = False,
) -> float:
    """텍스트 전용 하위 호환 래퍼 (텍스트main_v6.py 에서 사용)."""
    risk, _ = calculate_multimodal_risk(
        label=label, confidence=confidence,
        k_score=k_score, has_combined_risk=has_combined_risk,
        voice_available=False, verbose=False,
    )
    return risk


# ── 누적 위험도 관리 ─────────────────────────────────────────────────

class RiskAccumulator:
    """
    동적 감쇄율 + 연속 피싱 보너스 + 태도변화 탐지.

    update() 에 current_arousal 을 선택적으로 전달하면
    detect_attitude_shift() 가 Arousal 급변을 감지함.
    """
    def __init__(self, decay_rate: float = 0.9):
        self.base_decay       = decay_rate
        self.cumulative_score = 0.0
        self.phishing_streak  = 0
        self.normal_streak    = 0
        self._risk_hist:    list[float] = []
        self._arousal_hist: list[float] = []

    def _decay(self) -> float:
        r = self.cumulative_score / 5.0
        return min(self.base_decay + (1 - self.base_decay) * r * 0.9, 0.99)

    def _soft_w(self, r: float) -> float:
        return 2.0 / (1.0 + np.exp(-8.0 * (r - 0.5)))

    def _streak_bonus(self) -> float:
        return min((self.phishing_streak - 1) * 0.05, 0.3) if self.phishing_streak >= 2 else 0.0

    def update(
        self, risk: float, label: str,
        current_arousal: float | None = None,
    ) -> float:
        self.cumulative_score *= self._decay()
        if label in ("보이스피싱", "위험"):
            self.phishing_streak += 1; self.normal_streak = 0
            self.cumulative_score += risk * self._soft_w(risk) + self._streak_bonus()
        else:
            self.normal_streak += 1; self.phishing_streak = 0
            self.cumulative_score -= (1.0 - risk) * 0.3
        self._risk_hist.append(risk)
        if len(self._risk_hist) > 5: self._risk_hist.pop(0)
        if current_arousal is not None:
            self._arousal_hist.append(current_arousal)
            if len(self._arousal_hist) > 5: self._arousal_hist.pop(0)
        self.cumulative_score = max(0.0, min(self.cumulative_score, 5.0))
        return self.cumulative_score

    def detect_topic_shift(self) -> bool:
        """최근 3문장 위험도가 낮은데 누적 점수가 높으면 위장 정상화 전술."""
        if len(self._risk_hist) < 3: return False
        return float(np.mean(self._risk_hist[-3:])) < 0.35 and self.cumulative_score > 2.5

    def detect_attitude_shift(self) -> tuple[bool, str]:
        """Arousal 급변 (압박→회유) 탐지."""
        if len(self._arousal_hist) < 3: return False, ""
        arr = np.array(self._arousal_hist)
        std, rng = float(np.std(arr)), float(np.ptp(arr))
        shift = std > _SHIFT_STD and rng > _SHIFT_RNG
        msg = (f"⚡ [태도변화] Arousal 급변 감지 — std={std:.3f}, range={rng:.3f} "
               f"(압박→회유 전술 의심)") if shift else ""
        return shift, msg
