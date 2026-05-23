# 파일main_v6.py  ─ 오디오 파일 기반 멀티모달 보이스피싱 분석
#
# 데이터 흐름:
#   ① librosa.load()          → 원본 오디오 배열 y[]
#   ② run_stt_with_timestamps() → [(start, end, text), ...]
#   ③ 세그먼트별 루프:
#      · KLUE-BERT + keyword_score   → text_risk
#      · VoiceAnalyzer.get_arousal_state()    → (cur, base)  [베이스라인 후]
#      · VoiceAnalyzer.accumulate_ai_prob()   → _buf에 저장만 (출력 없음)
#      · calculate_multimodal_risk()          → final_risk, tone_label
#      · RiskAccumulator.update()             → total_score
#      · print_realtime_box()                 → 터미널 실시간 출력
#   ④ 루프 종료 후:
#      · VoiceAnalyzer.finalize_ai_verdict()  → AI 음성 확률 (한 번만)
#      · generate_report()                    → 수사 보고서 (리포트 내에만 AI 확률 포함)

import gc, os
import librosa
import numpy as np

from config import (AUDIO_PATH, LOG_FILE, SAMPLE_RATE, MIN_CHUNK_SAMPLES,
                    SCORE_WARN, SCORE_DANGER, DECAY_RATE_FILE)
from whisper_stt import run_stt_with_timestamps
from risk_calculator import RiskAccumulator, calculate_multimodal_risk
from keyword_score import keyword_score
from llm_report import generate_report
from voice_analyzer import VoiceAnalyzer
from model_kluebert import detect_batch


# ── 실시간 출력 UI ──────────────────────────────────────────────────────────

def print_realtime_box(
    sentence: str, score: float, tone_label: str,
    detected: list[str], timestamp: float,
) -> None:
    """
    각 문장 인식 시 출력하는 박스 UI.

    형식:
      [ 문장 실시간 분석 ] -----------------------------------
        🗣️  인식: "..."
        🔥 피싱 누적 지수: X.XX / 5.00  [████░░░░░░]
        🚨🚨 [강력 경고] ... (점수에 따라 조건부)
        🎭 감정 상태: ...
      ---------------------------------------------------
    """
    sep  = "-" * 54
    bar  = "█" * int(score / 5.0 * 20) + "░" * (20 - int(score / 5.0 * 20))
    disp = sentence if len(sentence) <= 45 else sentence[:42] + "..."

    print(f"\n[ 문장 실시간 분석 ] {sep}")
    print(f"  🕐 [{timestamp:6.1f}s]  🗣️  인식: \"{disp}\"")
    print(f"  🔥 피싱 누적 지수: {score:.2f} / 5.00  [{bar}]")

    if score >= SCORE_DANGER:
        print(f"  🚨🚨 [강력 경고] 보이스피싱 확실! 즉시 대화를 중단하세요! 🚨🚨")
    elif score >= SCORE_WARN:
        print(f"  ⚠️  [주의] 보이스피싱 의심 정황이 포착되었습니다. 주의하세요!")

    print(f"  🎭 감정 상태: {tone_label}")
    if detected:
        clean = [d for d in detected if not d.startswith("[")]
        if clean:
            print(f"  🔑 탐지 키워드: {', '.join(clean)}")

    print(f"  {sep}{'------'}")


def print_final_summary(score: float, all_detected: list[str]) -> None:
    """분석 완료 후 최종 수치 요약 출력."""
    border = "═" * 60
    level  = ("🚨 위험 단계" if score >= SCORE_DANGER
              else "⚠️ 주의 단계" if score >= SCORE_WARN
              else "✅ 일상 단계")
    kws    = list(dict.fromkeys(d for d in all_detected if not d.startswith("[")))
    print(f"\n{border}")
    print(f"  📊 분석 완료  |  최종 누적 위험도: {score:.2f}/5.00  |  {level}")
    print(f"  🔑 탐지 키워드: {', '.join(kws) if kws else '없음'}")
    print(f"{border}")


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'━'*60}")
    print(f"  📄 파일 분석 시작: {AUDIO_PATH}")
    print(f"{'━'*60}\n")

    if not os.path.exists(AUDIO_PATH):
        print(f"[오류] 파일을 찾을 수 없습니다: {AUDIO_PATH}"); return

    # ① 원본 오디오 로드
    print("[1/4] 오디오 로딩 중...")
    y, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    print(f"      완료 — 길이: {len(y)/SAMPLE_RATE:.1f}초\n")

    # ② STT (타임스탬프 포함) ─ 완료 후 모델 자동 해제
    print("[2/4] 음성 → 텍스트 변환 중...")
    segments = run_stt_with_timestamps(AUDIO_PATH)
    if not segments:
        print("[오류] STT 결과가 없습니다."); return
    print(f"      {len(segments)}개 세그먼트 추출 완료.\n")

    # ③ 분석 모듈 초기화
    print("[3/4] 분석 모듈 초기화 중...")
    voice       = VoiceAnalyzer()
    accumulator = RiskAccumulator(decay_rate=DECAY_RATE_FILE)
    full_texts: list[str]  = []
    all_detected: list[str] = []
    print("      완료.\n")

    print("[4/4] 세그먼트별 분석 시작...\n")

    # ③ 세그먼트별 분석 루프
    for start, end, text in segments:
        full_texts.append(text)
        chunk = y[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]

        # 텍스트 분석
        label, conf              = detect_batch([text])[0]
        k_score, detected, combo = keyword_score(text)
        if detected: all_detected.extend(detected)

        # 음성 분석 (이벤트 기반 + 사후 누적)
        cur_a, base_a    = 0.5, 0.5
        voice_available  = False

        if len(chunk) >= MIN_CHUNK_SAMPLES:
            if not voice.is_calibrated:
                voice.feed_calibration(chunk, end)
            elif k_score > 0 or label == "보이스피싱":
                # Arousal → 실시간 위험도 반영
                cur_a, base_a = voice.get_arousal_state(chunk)
                voice_available = True
                # AI 탐지 확률 → 내부 버퍼에만 저장 (출력 ✗)
                voice.accumulate_ai_prob(chunk, risk_weight=accumulator.cumulative_score)

        # 통합 위험도 계산
        risk, tone = calculate_multimodal_risk(
            label=label, confidence=conf, k_score=k_score,
            has_combined_risk=combo, current_arousal=cur_a,
            baseline_arousal=base_a, voice_available=voice_available,
            verbose=False,
        )
        total_score = accumulator.update(risk, label,
                                         current_arousal=cur_a if voice_available else None)

        # 화제전환 / 태도변화 감지
        if accumulator.detect_topic_shift():
            print("  ⚠️  [화제전환] 갑작스러운 일상 전환 감지 — 위험 점수 유지")
        shifted, smsg = accumulator.detect_attitude_shift()
        if shifted: print(f"  {smsg}")

        # 실시간 박스 출력
        print_realtime_box(text, total_score, tone, detected, start)

    # ④ 분석 종료 후 처리
    voice.release()
    gc.collect()

    final_score  = accumulator.cumulative_score
    detail_str   = ", ".join(list(dict.fromkeys(
        d for d in all_detected if not d.startswith("[조합:") and not d.startswith("[완곡:")
    ))) or "없음"

    print_final_summary(final_score, all_detected)

    # ★ AI 음성 최종 판정 — 리포트 생성 조건(≥ 2.0)일 때만 실행
    ai_prob, ai_label = None, None
    if final_score >= 2.0:
        print("\n  🔍 AI 생성 음성 종합 판정 중... (전체 통화 누적 분석)")
        ai_prob, ai_label = voice.finalize_ai_verdict()
        print(f"  결과: {ai_label}  ({ai_prob*100:.1f}%)")

    # 리포트 생성
    report = generate_report(" ".join(full_texts), final_score, detail_str,
                             ai_prob=ai_prob, ai_label=ai_label)

    if report:
        print(report)
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  💾 리포트 저장 완료: {LOG_FILE}")
    else:
        print("\n✅ 분석 결과: 일상 대화입니다. 상세 리포트를 생성하지 않습니다.")


if __name__ == "__main__":
    main()
