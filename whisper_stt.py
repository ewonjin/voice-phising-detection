# whisper_stt.py  ─ STT 모듈 (run_stt + run_stt_with_timestamps)
#
# ImportError 완전 해결:
#   파일main_v6.py 가 호출하는 run_stt_with_timestamps() 를 이 파일에 정의.
#   두 함수 모두 STT 완료 후 모델을 자동 해제하여 메모리 누수 방지.

import gc
from faster_whisper import WhisperModel
from utils_text import normalize_stt


def _load_model() -> WhisperModel:
    """STT 모델 로드. GPU 가용 시 자동 전환, 실패 시 CPU 폴백."""
    try:
        import torch
        if torch.cuda.is_available():
            return WhisperModel("small", device="cuda",
                                device_index=0, compute_type="int8_float16")
    except Exception:
        pass
    return WhisperModel("small", device="cpu", compute_type="int8")


def run_stt(audio_path: str, language: str = "ko") -> str:
    """
    오디오 파일 전체를 전사하여 단일 문자열로 반환.
    (텍스트main_v6 등 타임스탬프 불필요 모드에서 사용)
    """
    model = _load_model()
    try:
        segs, _ = model.transcribe(audio_path, language=language,
                                   beam_size=5, vad_filter=True)
        return " ".join(normalize_stt(s.text.strip()) for s in segs if s.text.strip())
    finally:
        del model; gc.collect()


def run_stt_with_timestamps(
    audio_path: str,
    language:   str = "ko",
) -> list[tuple[float, float, str]]:
    """
    오디오 파일을 전사하고 세그먼트별 타임스탬프를 포함하여 반환.

    Returns:
        [(start_sec, end_sec, normalized_text), ...]
        → 파일main_v6.py 에서 librosa 슬라이싱에 직접 사용.

    타임스탬프 활용:
        chunk = y[int(start * sr) : int(end * sr)]
        → VoiceAnalyzer 에 전달하여 정확한 구간 음성 분석 가능.
    """
    print(f"[STT] 타임스탬프 전사 시작: {audio_path}")
    model   = _load_model()
    results: list[tuple[float, float, str]] = []
    try:
        segs, _ = model.transcribe(audio_path, language=language,
                                   beam_size=5, vad_filter=True)
        for seg in segs:
            text = normalize_stt(seg.text.strip())
            if text and (seg.end - seg.start) >= 0.1:
                results.append((float(seg.start), float(seg.end), text))
    finally:
        del model; gc.collect()
        print(f"[STT] 완료 — {len(results)}개 세그먼트 추출.")
    return results
