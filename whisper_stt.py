import gc
from faster_whisper import WhisperModel

def run_stt(audio_path):

    print("🎤 STT 시작...")

    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        audio_path,
        language="ko",
        vad_filter=True
    )

    texts = [seg.text for seg in segments]
    transcript = " ".join(texts).strip()

    del model
    gc.collect()

    return transcript

# import gc
# import os
# from faster_whisper import WhisperModel

# def run_stt(audio_path):
#     print("🎤 STT 시작 (최적화 모드)...")

#     # 1. 모델 로드: 용량은 그대로 유지하되 빔 서치 등의 옵션을 강화
#     # 용량 부담이 없다면 "base" 대신 "base.en"은 아니니 "base"를 유지하되 옵션 튜닝
#     model = WhisperModel("base", device="cpu", compute_type="int8")

#     # 2. 전사 옵션 최적화
#     segments, info = model.transcribe(
#         audio_path,
#         language="ko",
#         vad_filter=True,
#         beam_size=10,
#         # 아래 프롬프트를 추가하면 유사한 발음을 피싱 용어로 우선 인식합니다.
#         initial_prompt="보이스피싱, 검찰청, 금융감독원, 수사관, 계좌 도용, 대포통장, 안전 계좌, 형사 사건"
#         )

#     texts = [seg.text for seg in segments]
#     transcript = " ".join(texts).strip()

#     # 메모리 해제
#     del model
#     gc.collect()

#     return transcript