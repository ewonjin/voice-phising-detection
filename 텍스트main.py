import os
from config import LOG_FILE
from keyword_score import keyword_score
from risk_calculator_v2 import RiskAccumulator, calculate_risk
from llm_report import generate_report

# 모델 선택 및 로드
MODEL_TYPE = "kluebert" # 학습하신 이진 분류 모델 경로를 config.py의 MODEL_PATH에 설정하세요
if MODEL_TYPE == "kluebert":
    from model_kluebert import detect_batch

def main():
    print("🧪 보이스피싱 탐지 시뮬레이터 (Text Input Mode)")
    print("--------------------------------------------------")
    print("- 한 줄씩 입력 후 [Enter]를 누르세요.")
    print("- 종료하고 최종 리포트를 보려면 [q]를 입력하세요.")
    print("--------------------------------------------------\n")

    # 누적기 및 데이터 저장용 리스트 초기화
    accumulator = RiskAccumulator(decay_rate=0.9)
    conversation_history = []
    
    while True:
        user_input = input("🗣️ 입력: ").strip()

        if user_input.lower() == 'q':
            print("\n⏹ 테스트를 종료합니다. 최종 분석을 시작합니다...")
            break
        
        if not user_input:
            continue

        # 1. 모델 판별 (현재 문장 하나만 리스트로 전달)
        result = detect_batch([user_input])[0]
        label, conf = result
        
        # 2. 키워드 점수 계산
        k_score, detected_ks = keyword_score(user_input)
        
        # 3. 위험도 및 누적 점수 계산
        curr_risk = calculate_risk(label, conf, k_score)
        total_score = accumulator.update(curr_risk, label)
        
        # 대화 기록 저장
        conversation_history.append(user_input)

        # 4. 실시간 결과 출력
        print(f"   └ [판단]: {label} ({conf:.2f})")
        print(f"   └ [키워드]: {detected_ks if detected_ks else '없음'}")
        print(f"   └ [🔥 누적 피싱 지수]: {total_score:.2f} / 5.00")

        if total_score > 3.5:
            print("   🚨 [WARNING] 보이스피싱 의심 수준이 높습니다!")
        print("-" * 50)

    # 5. 최종 리포트 생성 및 저장
    if conversation_history:
        full_transcript = " ".join(conversation_history)
        
        print("\n" + "="*50)
        print("📝 최종 피싱 리포트 (LLM 분석)")
        print("="*50)
        
        # LLM 리포트 생성
        report = generate_report(full_transcript)
        print(report)
        
        # 파일 저장
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print("\n" + "="*50)
        print(f"✅ 리포트가 저장되었습니다: {LOG_FILE}")
    else:
        print("입력된 대화 내용이 없어 리포트를 생성하지 않습니다.")

if __name__ == "__main__":
    main()