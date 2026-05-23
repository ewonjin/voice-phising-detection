# 인식main_v6.py  ─ 실시간 마이크 입력 멀티모달 보이스피싱 탐지
#
# 데이터 흐름:
#   ① AudioBuffer   → 원본 PCM 연속 저장 (타임스탬프 슬라이싱용)
#   ② STT 스트림     → (text, abs_start, abs_end)
#   ③ 세그먼트별:
#      · KLUE-BERT + keyword_score        → text_risk
#      · VoiceAnalyzer.get_arousal_state() → (cur, base)
#      · VoiceAnalyzer.accumulate_ai_prob()→ _buf 저장만 (출력 없음)
#      · calculate_multimodal_risk()       → final_risk, tone_label
#      · RiskAccumulator.update()          → total_score
#      · print_realtime_box()             → 박스 형식 실시간 출력
#   ④ Ctrl+C 종료 후:
#      · finalize_ai_verdict()            → AI 확률 종합 (한 번만)
#      · generate_report()               → 수사 보고서 (AI 확률 포함)

import gc, os, sys
import numpy as np
import speech_recognition as sr

from config import (SAMPLE_RATE, MIN_CHUNK_SAMPLES, LOG_FILE,
                    SCORE_WARN, SCORE_DANGER, DECAY_RATE_RT)
from utils_text import normalize_stt
from keyword_score import keyword_score
from risk_calculator import RiskAccumulator, calculate_multimodal_risk
from voice_analyzer import VoiceAnalyzer
from llm_report import generate_report
from model_kluebert import detect_batch

_STT_CHUNK_SEC = 5
_ENERGY_THRESH = 0.01


# ── 오디오 버퍼 ─────────────────────────────────────────────────────────────

class AudioBuffer:
    """
    실시간 캡처 오디오를 절대 타임스탬프로 인덱싱하는 원형 버퍼.

    _offset: _buf[0] 에 해당하는 전체 타임라인의 절대 샘플 인덱스.
    STT 세그먼트의 절대 타임스탬프로 get_chunk() 하면
    해당 발화 구간의 원본 오디오를 정확히 슬라이싱 가능.
    """
    def __init__(self, sr: int = SAMPLE_RATE, max_sec: int = 120):
        self.sr  = sr; self._max = sr * max_sec
        self._buf: np.ndarray = np.array([], dtype=np.float32)
        self._off = 0

    def append(self, chunk: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        if len(self._buf) > self._max:
            trim = len(self._buf) - self._max
            self._buf = self._buf[trim:]; self._off += trim

    def get_chunk(self, start_sec: float, end_sec: float) -> np.ndarray | None:
        s = max(0, int(start_sec * self.sr) - self._off)
        e = min(len(self._buf), int(end_sec * self.sr) - self._off)
        return self._buf[s:e].copy() if e > s else None

    @property
    def elapsed_sec(self) -> float:
        return (self._off + len(self._buf)) / self.sr


# ── 실시간 UI ────────────────────────────────────────────────────────────────

def print_realtime_box(
    sentence: str, score: float, tone_label: str,
    detected: list[str],
) -> None:
    """
    각 문장 인식 시 출력하는 박스 형식 UI.

    예시:
      [ 문장 실시간 분석 ] -----------------------------------
        🗣️  인식: "본인 명의로 대포통장이 개설되어 수사 중입니다."
        🔥 피싱 누적 지수: 3.85 / 5.00  [███████████████░░░░░]
        🚨🚨 [강력 경고] 보이스피싱 확실! 즉시 대화를 중단하세요! 🚨🚨
        🎭 감정 상태: 고각성(Angry) ⬆️  (Baseline 대비 격앙됨)
      ---------------------------------------------------
    """
    sep  = "-" * 54
    bar  = "█" * int(score / 5.0 * 20) + "░" * (20 - int(score / 5.0 * 20))
    disp = sentence if len(sentence) <= 45 else sentence[:42] + "..."

    print(f"\n[ 문장 실시간 분석 ] {sep}")
    print(f"  🗣️  인식: \"{disp}\"")
    print(f"  🔥 피싱 누적 지수: {score:.2f} / 5.00  [{bar}]")

    if score >= SCORE_DANGER:
        print(f"  🚨🚨 [강력 경고] 보이스피싱 확실! 즉시 대화를 중단하세요! 🚨🚨")
    elif score >= SCORE_WARN:
        print(f"  ⚠️  [주의] 보이스피싱 의심 정황이 포착되었습니다. 주의하세요!")

    print(f"  🎭 감정 상태: {tone_label}")
    if detected:
        clean = [d for d in detected if not d.startswith("[")]
        if clean: print(f"  🔑 탐지 키워드: {', '.join(clean)}")

    print(f"  {sep}{'------'}")


# ── STT 스트림 제너레이터 ────────────────────────────────────────────────────

def _stt_stream(audio_buf: AudioBuffer):
    """
    마이크 → 5초 청크 캡처 → STT → (text, abs_start, abs_end) yield.

    절대 타임스탬프 계산:
        Faster-Whisper 는 각 청크 내 로컬 타임스탬프(0초 시작)를 반환.
        청크 시작 시각(chunk_start_sec)을 누적하여 절대값으로 변환.
        → AudioBuffer.get_chunk(abs_start, abs_end) 로 정확한 슬라이싱 가능.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    r     = sr.Recognizer()
    chunk_start = 0.0

    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            print("🎤 주변 소음 분석 중...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("=" * 60)
            print("  🎙️  실시간 보이스피싱 탐지 시작  (Ctrl+C 로 종료)")
            print("=" * 60 + "\n")

            while True:
                data    = r.listen(source, phrase_time_limit=_STT_CHUNK_SEC)
                raw     = data.get_raw_data()
                arr     = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                dur     = len(arr) / SAMPLE_RATE

                audio_buf.append(arr)
                chunk_start += dur

                if np.sqrt(np.mean(arr ** 2)) < _ENERGY_THRESH:
                    continue   # 무음 스킵

                this_start = chunk_start - dur
                segs, _    = model.transcribe(arr, language="ko", beam_size=5)
                for seg in segs:
                    text = normalize_stt(seg.text.strip())
                    if text:
                        yield text, this_start + seg.start, this_start + seg.end

    except KeyboardInterrupt:
        pass
    finally:
        del model; gc.collect()
        print("\n[STT] 메모리 해제 완료.")


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  🎤 멀티모달 실시간 보이스피싱 감시 시스템")
    print("=" * 60 + "\n")

    audio_buf   = AudioBuffer(sr=SAMPLE_RATE)
    voice       = VoiceAnalyzer()
    accumulator = RiskAccumulator(decay_rate=DECAY_RATE_RT)
    full_texts: list[str]   = []
    all_detected: list[str] = []

    try:
        for sentence, seg_start, seg_end in _stt_stream(audio_buf):
            full_texts.append(sentence)

            # 텍스트 분석
            label, conf              = detect_batch([sentence])[0]
            k_score, detected, combo = keyword_score(sentence)
            if detected: all_detected.extend(detected)

            # 음성 분석 (이벤트 기반 + 사후 누적)
            cur_a, base_a   = 0.5, 0.5
            voice_available = False
            chunk = audio_buf.get_chunk(seg_start, seg_end)

            if chunk is not None and len(chunk) >= MIN_CHUNK_SAMPLES:
                if not voice.is_calibrated:
                    voice.feed_calibration(chunk, seg_end)
                elif k_score > 0 or label == "보이스피싱":
                    cur_a, base_a   = voice.get_arousal_state(chunk)
                    voice_available = True
                    # AI 확률 → 버퍼에만 저장, 실시간 화면 출력 없음
                    voice.accumulate_ai_prob(chunk,
                                             risk_weight=accumulator.cumulative_score)

            # 통합 위험도
            risk, tone = calculate_multimodal_risk(
                label=label, confidence=conf, k_score=k_score,
                has_combined_risk=combo, current_arousal=cur_a,
                baseline_arousal=base_a, voice_available=voice_available,
                verbose=False,
            )
            total_score = accumulator.update(risk, label,
                                             current_arousal=cur_a if voice_available else None)

            # 화제전환 / 태도변화
            if accumulator.detect_topic_shift():
                print("  ⚠️  [화제전환] 갑작스러운 일상 전환 — 위험 점수 유지")
            shifted, smsg = accumulator.detect_attitude_shift()
            if shifted: print(f"  {smsg}")

            # 실시간 박스 출력 (AI 확률 미포함)
            print_realtime_box(sentence, total_score, tone, detected)

    except KeyboardInterrupt:
        print("\n⏹ 감시 종료")
    finally:
        voice.release()

    if not full_texts:
        return

    final_score = accumulator.cumulative_score
    detail_str  = ", ".join(list(dict.fromkeys(
        d for d in all_detected if not d.startswith("[조합:") and not d.startswith("[완곡:")
    ))) or "없음"

    # ★ AI 음성 최종 판정 — 리포트 조건 충족 시에만 실행 및 표시
    ai_prob, ai_label = None, None
    if final_score >= 2.0:
        print("\n  🔍 AI 생성 음성 종합 판정 중...")
        ai_prob, ai_label = voice.finalize_ai_verdict()
        print(f"  결과: {ai_label}  ({ai_prob*100:.1f}%)")

    report = generate_report(" ".join(full_texts), final_score, detail_str,
                             ai_prob=ai_prob, ai_label=ai_label)

    if report:
        print(report)
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  💾 리포트 저장: {LOG_FILE}")
    else:
        print("\n✅ 분석 결과: 일상 대화입니다. 리포트를 생성하지 않습니다.")


if __name__ == "__main__":
    main()
