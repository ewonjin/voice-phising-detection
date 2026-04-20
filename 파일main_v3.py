# 파일main.py
from config import AUDIO_PATH, LOG_FILE
from whisper_stt import run_stt
from risk_calculator_v2 import RiskAccumulator, calculate_risk
from keyword_score import keyword_score
from utils_text import split_sentences
from llm_report import generate_report

# 모델 선택 및 로드
MODEL_TYPE = "kluebert" # 학습하신 이진 분류 모델 경로를 config.py의 MODEL_PATH에 설정하세요
if MODEL_TYPE == "kluebert":
    from model_kluebert import detect_batch

def main():
    print(f"📄 파일 분석 시작: {AUDIO_PATH}")
    transcript = run_stt(AUDIO_PATH)
    if not transcript: return

    sentences = split_sentences(transcript)
    accumulator = RiskAccumulator(decay_rate=0.9) # 파일은 흐름 파악을 위해 감쇄율을 높게 설정
    
    print("\n[문장별 누적 위험도 분석]")
    for sentence in sentences:
        label, conf = detect_batch([sentence])[0]
        k_score, _ = keyword_score(sentence)
        
        curr_risk = calculate_risk(label, conf, k_score)
        total_score = accumulator.update(curr_risk, label)
        
        print(f"- {sentence} | 점수: {total_score:.2f} ({label})")

    # 최종 리포트
    report = generate_report(transcript)
    print("\n=== 분석 리포트 ===\n", report)
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write(report)

if __name__ == "__main__":
    main()