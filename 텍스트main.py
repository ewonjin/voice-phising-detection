# 텍스트main_v6.py  ─ 텍스트 입력 전용 시뮬레이터 (음성 모델 없음)
import os
from config import LOG_FILE, SCORE_WARN, SCORE_DANGER
from keyword_score import keyword_score
from risk_calculator import RiskAccumulator, calculate_risk
from llm_report import generate_report
from model_kluebert import detect_batch

def print_realtime_box(sentence: str, score: float, detected: list[str], label: str, conf: float) -> None:
    sep = "-" * 54
    bar = "█" * int(score / 5.0 * 20) + "░" * (20 - int(score / 5.0 * 20))
    disp = sentence if len(sentence) <= 45 else sentence[:42] + "..."
    print(f"\n[ 문장 실시간 분석 ] {sep}")
    print(f"  🗣️  입력: \"{disp}\"")
    print(f"  🤖 BERT 판별: {label} (신뢰도 {conf:.2f})")
    print(f"  🔥 피싱 누적 지수: {score:.2f} / 5.00  [{bar}]")
    if score >= SCORE_DANGER:
        print(f"  🚨🚨 [강력 경고] 보이스피싱 확실! 즉시 대화를 중단하세요! 🚨🚨")
    elif score >= SCORE_WARN:
        print(f"  ⚠️  [주의] 보이스피싱 의심 정황이 포착되었습니다. 주의하세요!")
    if detected:
        clean = [d for d in detected if not d.startswith("[")]
        if clean: print(f"  🔑 탐지 키워드: {', '.join(clean)}")
    print(f"  {sep}{'------'}")

def main():
    print("\n" + "=" * 60)
    print("  🧪 보이스피싱 탐지 시뮬레이터 (텍스트 입력 모드)")
    print("  한 줄 입력 후 [Enter] → 분석 │ [q] → 종료 및 최종 리포트")
    print("=" * 60 + "\n")

    accumulator   = RiskAccumulator(decay_rate=0.9)
    history:       list[str]  = []
    all_detected:  list[str]  = []

    while True:
        try:
            user_input = input("🗣️  입력 ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "q":
            break
        if not user_input:
            continue

        label, conf              = detect_batch([user_input])[0]
        k_score, detected, combo = keyword_score(user_input)
        risk                     = calculate_risk(label, conf, k_score, combo)
        total_score              = accumulator.update(risk, label)

        history.append(user_input)
        if detected: all_detected.extend(detected)

        if accumulator.detect_topic_shift():
            print("  ⚠️  [화제전환] 주의 수준 유지")

        print_realtime_box(user_input, total_score, detected, label, conf)

    if not history:
        return

    final_score = accumulator.cumulative_score
    detail_str  = ", ".join(list(dict.fromkeys(
        d for d in all_detected if not d.startswith("[")
    ))) or "없음"

    # 텍스트 모드: AI 음성 데이터 없음 → ai_prob=None
    report = generate_report(" ".join(history), final_score, detail_str,
                             ai_prob=None, ai_label=None)

    if report:
        print(report)
        os.makedirs("logs", exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  💾 리포트 저장: {LOG_FILE}")
    else:
        print("\n✅ 보이스피싱 위험이 낮은 일반 대화입니다.")

if __name__ == "__main__":
    main()
