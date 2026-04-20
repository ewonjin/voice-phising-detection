# 인식main.py
import sys
import numpy as np
from risk_calculator_v2 import RiskAccumulator, calculate_risk
from keyword_score import keyword_score
from llm_report import generate_report\

# 모델 선택 및 로드
MODEL_TYPE = "kluebert" # 학습하신 이진 분류 모델 경로를 config.py의 MODEL_PATH에 설정하세요
if MODEL_TYPE == "kluebert":
    from model_kluebert import detect_batch

def realtime_stt_stream():
    """문장이 생성될 때마다 yield로 반환하는 제너레이터"""
    model = whisper.load_model("base")
    r = sr.Recognizer()

    with sr.Microphone(sample_rate=16000) as source:
        print("🎤 주변 소음 분석 중...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("=== 실시간 탐지 중 (q 입력 시 종료) ===\n")

        while True:
            try:
                audio = r.listen(source, phrase_time_limit=5)
                raw = audio.get_raw_data()
                audio_np = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                result = model.transcribe(audio_np, language="ko", fp16=False)
                text = result['text'].strip()

                if text:
                    yield text # 문장이 인식되면 즉시 반환

                if check_exit():
                    if sys.stdin.read(1) == 'q': break
            except Exception as e:
                print(f"Error: {e}")
                break

def main():
    print("🎤 실시간 보이스피싱 감시 모드 작동 중...")
    accumulator = RiskAccumulator(decay_rate=0.85)
    full_texts = []

    # realtime_stt_stream()은 문장이 완성될 때마다 yield하는 제너레이터로 가정
    try:
        for sentence in realtime_stt_stream():
            full_texts.append(sentence)
            
            # 1. 모델 판별 및 키워드 확인
            label, conf = detect_batch([sentence])[0]
            k_score, _ = keyword_score(sentence)
            
            # 2. 위험도 및 누적 점수 계산 (label 전달 필수)
            curr_risk = calculate_risk(label, conf, k_score)
            total_score = accumulator.update(curr_risk, label)
            
            # 3. 실시간 결과 출력
            print(f"\n[인식]: {sentence}")
            print(f"🔥 지수: {total_score:.2f} | 판별: {label}({conf:.2f})")

            if total_score > 3.8: # 임계값 소폭 상향 조정
                print("🚨🚨 [강력 경고] 보이스피싱 의심! 대화를 중단하세요! 🚨🚨")

    except KeyboardInterrupt:
        print("\n⏹ 감시 종료")

    # 최종 리포트 생성
    if full_texts:
        report = generate_report(" ".join(full_texts))
        print("\n=== 최종 분석 리포트 ===\n", report)
        