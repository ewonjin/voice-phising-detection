import csv
import os
from datetime import datetime

def main():
    # ── 1. [사용자 설정 항목] ──────────────────────────────────────────────
    # 읽어들일 CSV 파일의 경로를 지정해주세요.
    CSV_PATH = "./experiment_results/experiment_total_results.csv"
    # ───────────────────────────────────────────────────────────────────

    if not os.path.exists(CSV_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {CSV_PATH}")
        print("경로를 다시 확인해주세요.")
        return

    # 통계를 담을 사전
    intervals = ["~20s", "20s~40s", "40s~60s", "1m~1m20s", "1m20s~"]
    stats = {inv: {"total": 0, "correct": 0} for inv in intervals}
    
    # 실제 구분값(라벨)을 동적으로 수집할 세트
    actual_classes_found = set()

    print(f"🔍 CSV 파일 데이터를 분석 중입니다: {CSV_PATH}")

    try:
        with open(CSV_PATH, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                interval = row.get("길이구간", "알 수 없음").strip()
                actual = row.get("실제구분", "").strip()
                predicted = row.get("최종판정", "").strip()

                if not actual or not predicted:
                    continue  # 비어있는 행은 건너뜀

                # 💡 CSV에 기록된 실제 구분 라벨 저장 ("보이스피싱", "정상대화" 등)
                actual_classes_found.add(actual)

                if interval not in stats:
                    stats[interval] = {"total": 0, "correct": 0}

                stats[interval]["total"] += 1

                # 정탐(Correct) 판별 로직
                is_phishing_correct = (actual == "보이스피싱" and predicted == "보이스피싱")
                is_normal_correct = (actual == "정상대화" and predicted == "정상")

                if is_phishing_correct or is_normal_correct:
                    stats[interval]["correct"] += 1

    except Exception as e:
        print(f"❌ CSV 파일 읽기 중 오류가 발생했습니다: {e}")
        return

    # ── 2. 요약 리포트 텍스트 생성 ─────────────────────────────────────────
    classes_str = ", ".join(actual_classes_found) if actual_classes_found else "알 수 없음"
    
    summary_lines = []
    summary_lines.append("═" * 65)
    summary_lines.append(f"📊 CSV 기반 통계 요약 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    summary_lines.append(f"📂 분석 대상 파일: {CSV_PATH}")
    summary_lines.append(f"🏷️ 포함된 실제 데이터 그룹: [{classes_str}]")
    summary_lines.append("═" * 65)

    total_all = 0
    correct_all = 0

    all_intervals = intervals + [k for k in stats.keys() if k not in intervals]

    for inv in all_intervals:
        count_data = stats[inv]
        total = count_data["total"]
        
        if total == 0:
            continue
            
        correct = count_data["correct"]
        rate = (correct / total * 100) if total > 0 else 0.0
        
        summary_lines.append(f"  {inv:<12}: 정확한 판정 {correct:<3} / 전체 {total:<3} 개 (정탐률: {rate:>5.1f}%)")
        
        total_all += total
        correct_all += correct

    overall_rate = (correct_all / total_all * 100) if total_all > 0 else 0.0
    summary_lines.append("-" * 65)
    summary_lines.append(f"  {'TOTAL':<12}: 정확한 판정 {correct_all:<3} / 전체 {total_all:<3} 개 (종합 정탐률: {overall_rate:>5.1f}%)")
    summary_lines.append("═" * 65)

    report_string = "\n".join(summary_lines)

    # ── 3. 결과 출력 및 저장 ──────────────────────────────────────────────
    print("\n" + report_string)

    # 💡 [핵심 보완] CSV에서 추출한 클래스 세트를 조합하여 파일 이름 생성
    if actual_classes_found:
        # 가나다순 정렬 후 언더바(_)로 결합 (예: "정상대화", "보이스피싱")
        class_filename_part = "_".join(sorted(list(actual_classes_found)))
    else:
        class_filename_part = "결과"

    output_dir = os.path.dirname(CSV_PATH) or "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 📝 수동 타이핑 없이 고유 라벨로 매핑된 새 파일 경로 정의
    save_path = os.path.join(output_dir, f"summary_{class_filename_part}_{timestamp}.txt")

    try:
        # 중복 정의 버그 방지를 위해 파일 포인터 변수명 분리 (summary_f)
        with open(save_path, "w", encoding="utf-8") as summary_f:
            summary_f.write(report_string)
        print(f"💾 [텍스트 요약 리포트 저장 완료]: {save_path}\n")
    except Exception as e:
        print(f"❌ 텍스트 파일 저장 실패: {e}\n")

if __name__ == "__main__":
    main()