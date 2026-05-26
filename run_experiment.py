import gc
import os
import csv
import librosa
import soundfile as sf  # 임시 WAV 파일 생성용 라이브러리
import numpy as np
from datetime import datetime

# [기존 모듈 그대로 활용]
from config import (SAMPLE_RATE, MIN_CHUNK_SAMPLES, DECAY_RATE_FILE)
from whisper_stt import run_stt_with_timestamps
from risk_calculator import RiskAccumulator, calculate_multimodal_risk
from keyword_score import keyword_score
from voice_analyzer import VoiceAnalyzer
from model_kluebert import detect_batch

# 📊 피싱 판정 기준 임계값
PHISHING_THRESHOLD = 3.5

def load_completed_files(csv_path):
    """이미 분석이 완료되어 CSV에 저장된 파일 목록을 불러옵니다."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "파일명" in row:
                    completed.add(row["파일명"])
    return completed

def get_arousal_label(value):
    """Arousal 수치를 직관적인 텍스트 상태로 변환합니다."""
    if value >= 0.65:
        return "흥분/압박"
    elif value <= 0.40:
        return "차분/침착"
    else:
        return "평온/중립"

def main():
    # ── 1. [사용자 설정 항목] ──────────────────────────────────────────────
    # TARGET_DIR   = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/음성 분리"
    # TARGET_DIR   = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/fishingmp3"
    # ACTUAL_CLASS = "보이스피싱"                                         # 👈 폴더 안 파일들의 실제 정답 ("보이스피싱" 또는 "정상대화")
    
    # TARGET_DIR = "/Users/ewonjin/Downloads/call"
    TARGET_DIR = "/Users/ewonjin/Downloads/통녹"
    ACTUAL_CLASS = "정상대화" 
    # TARGET_DIR = "/Users/ewonjin/Downloads/임시"
    # ───────────────────────────────────────────────────────────────────

    OUTPUT_DIR   = "./experiment_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_file_path = os.path.join(OUTPUT_DIR, "experiment_total_results.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file_path = os.path.join(OUTPUT_DIR, f"summary_{ACTUAL_CLASS}_{timestamp}.txt")

    intervals = ["~20s", "20s~40s", "40s~60s", "1m~1m20s", "1m20s~"]
    stats = {inv: [0, 0] for inv in intervals}  # [전체 개수, 정탐 개수] 구조

    if not os.path.exists(TARGET_DIR):
        print(f"[오류] 해당 경로가 존재하지 않습니다: {TARGET_DIR}")
        return

    completed_files = load_completed_files(csv_file_path)
    audio_files = [f for f in os.listdir(TARGET_DIR)
               if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.pcm'))]
    file_exists = os.path.exists(csv_file_path)

    voice = VoiceAnalyzer()

    print(f"\n🚀 [{ACTUAL_CLASS}] 배치 실험 파이프라인 가동 (통합 정탐률 리포트 버전)")
    print(f"📂 대상 경로: {TARGET_DIR}")
    print(f"📊 발견된 음성 파일: 총 {len(audio_files)}개")
    print("=" * 75)

    with open(csv_file_path, mode="a", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "파일명", "실제구분", "음성길이(초)", "길이구간", 
            "최종위험점수", "AI음성확률", "태도변화여부", "감정변화상세(Arousal)", "최종판정"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for idx, filename in enumerate(audio_files):
            if filename in completed_files:
                print(f"⏭️  [{idx+1}/{len(audio_files)}] {filename} -> 스킵합니다.")
                continue

            audio_path = os.path.join(TARGET_DIR, filename)
            print(f"\n🔍 [{idx+1}/{len(audio_files)}] 분석 중: {filename}")
            
            temp_wav_path = None

            try:
                # ① 오디오 데이터 로드
                if audio_path.lower().endswith(".pcm"):
                    raw_data = np.fromfile(audio_path, dtype=np.int16)
                    if len(raw_data) == 0:
                        print(f"⚠️  {filename}: PCM 데이터가 비어있습니다.")
                        continue
                    y = raw_data.astype(np.float32) / 32768.0
                    
                    # Whisper STT 호환용 임시 WAV 파일 생성
                    temp_wav_path = os.path.join(TARGET_DIR, f"temp_{filename}.wav")
                    sf.write(temp_wav_path, y, SAMPLE_RATE)
                    stt_target_path = temp_wav_path
                else:
                    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    stt_target_path = audio_path
                
                duration = len(y) / SAMPLE_RATE
                if duration == 0:
                    print(f"⚠️  {filename}: 음성 길이가 0초입니다. 건너뜁니다.")
                    continue

                if duration <= 20: interval = "~20s"
                elif duration <= 40: interval = "20s~40s"
                elif duration <= 60: interval = "40s~60s"
                elif duration <= 80: interval = "1m~1m20s"
                else: interval = "1m20s~"

                # ② STT 실행
                segments = run_stt_with_timestamps(stt_target_path)
                if not segments:
                    print(f"⚠️  {filename}: STT 결과가 비어있어 분석을 건너뜁니다.")
                    continue

                accumulator = RiskAccumulator(decay_rate=DECAY_RATE_FILE)
                if not hasattr(accumulator, '_arousal_history'):
                    accumulator._arousal_history = []

                voice.is_calibrated = False
                voice._calib.clear()
                voice._hist.clear()
                voice._buf.clear()  

                has_attitude_shift = False
                latest_shift_msg = "변화 없음 (안정적 상태 유지)"

                # ③ 세그먼트 루프
                for start, end, text in segments:
                    start_idx = int(start * SAMPLE_RATE)
                    end_idx = min(int(end * SAMPLE_RATE), len(y))
                    if start_idx >= len(y):
                        continue
                        
                    chunk = y[start_idx:end_idx]

                    label, conf = detect_batch([text])[0]
                    k_score, _, combo = keyword_score(text)

                    cur_a, base_a = 0.5, 0.5
                    voice_available = False

                    if len(chunk) >= MIN_CHUNK_SAMPLES:
                        if not voice.is_calibrated:
                            voice.feed_calibration(chunk, end)
                        else:
                            cur_a, base_a = voice.get_arousal_state(chunk)
                            voice_available = True
                            
                            if k_score > 0 or label == "보이스피싱":
                                voice.accumulate_ai_prob(chunk, risk_weight=accumulator.cumulative_score)
                            else:
                                voice_available = False

                    risk, _ = calculate_multimodal_risk(
                        label=label, confidence=conf, k_score=k_score,
                        has_combined_risk=combo, current_arousal=cur_a,
                        baseline_arousal=base_a, voice_available=voice_available, verbose=False
                    )
                    
                    passed_arousal = cur_a if len(chunk) >= MIN_CHUNK_SAMPLES and voice.is_calibrated else None
                    accumulator.update(risk, label, current_arousal=passed_arousal)

                    if not hasattr(accumulator, '_arousal_history'):
                        accumulator._arousal_history = []
                    
                    if passed_arousal is not None and passed_arousal not in accumulator._arousal_history:
                        accumulator._arousal_history.append(passed_arousal)
                        if len(accumulator._arousal_history) > 5:
                            accumulator._arousal_history.pop(0)

                    # 태도 급변 패턴 탐지 및 감정 변화 흐름 매핑
                    shifted, _ = accumulator.detect_attitude_shift()
                    if shifted and len(accumulator._arousal_history) >= 2:
                        has_attitude_shift = True
                        
                        hist = accumulator._arousal_history
                        min_idx = int(np.argmin(hist))
                        max_idx = int(np.argmax(hist))
                        
                        if min_idx < max_idx:
                            start_val = hist[min_idx]
                            end_val = hist[max_idx]
                            start_state = get_arousal_label(start_val)
                            end_state = get_arousal_label(end_val)
                            latest_shift_msg = f"[{start_state}] ➔ [{end_state}] 상태로 감정 상승 (Δ+{end_val - start_val:.2f})"
                        else:
                            start_val = hist[max_idx]
                            end_val = hist[min_idx]
                            start_state = get_arousal_label(start_val)
                            end_state = get_arousal_label(end_val)
                            latest_shift_msg = f"[{start_state}] ➔ [{end_state}] 상태로 급격히 다운/회유 (Δ{end_val - start_val:.2f})"

                # ④ 사후 종합 판정 호출
                final_score = accumulator.cumulative_score
                
                ai_prob, ai_label = 0.0, "데이터 없음"
                if final_score >= 2.0 and voice._buf:
                    ai_prob, ai_label = voice.finalize_ai_verdict()

                is_phishing = final_score >= PHISHING_THRESHOLD
                final_verdict = "보이스피싱" if is_phishing else "정상"

                # 💡 [핵심 수정] 정탐(True Positive / True Negative) 조건 설정
                # 실제 피싱을 피싱으로 맞췄거나, 실제 정상대화를 정상으로 정확히 맞춘 경우
                is_correct = (is_phishing and ACTUAL_CLASS == "보이스피싱") or (not is_phishing and ACTUAL_CLASS == "정상대화")

                # 통계 사전 업데이트
                stats[interval][0] += 1       # 전체 분석 대상 수 개수 증가
                if is_correct:
                    stats[interval][1] += 1   # 정확하게 맞춘 정탐 수 개수 증가

                # 💾 CSV 파일에 행단위 저장
                writer.writerow({
                    "파일명": filename,
                    "실제구분": ACTUAL_CLASS,
                    "음성길이(초)": round(duration, 1),
                    "길이구간": interval,
                    "최종위험점수": round(final_score, 2),
                    "AI음성확률": f"{ai_prob*100:.1f}%" if ai_prob > 0 else "N/A",
                    "태도변화여부": "Y" if has_attitude_shift else "N",
                    "감정변화상세(Arousal)": latest_shift_msg,
                    "최종판정": final_verdict
                })
                f.flush()

                shift_tag = " ⚡[태도변화감지]" if has_attitude_shift else ""
                print(f"✨ 판정 완료 -> Score: {final_score:.2f} | AI: {ai_prob*100:.1f}% | 결과: [{final_verdict}]{shift_tag}")
                print(f"   └ 감정 흐름: {latest_shift_msg}")

            except Exception as e:
                print(f"❌ {filename} 처리 중 예상치 못한 치명적 오류 발생: {e}")
            
            finally:
                # 사용이 끝난 임시 wav 파일 즉시 클리닝
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                if 'y' in locals(): del y
                gc.collect()

    voice.release()

    # ── 5. 요약 통계 리포트 생성 및 저장 ────────────────
    summary_lines = []
    summary_lines.append("═" * 60)
    summary_lines.append(f"📊 20초 단위 실험 결과 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    summary_lines.append(f"📂 대상 폴더: {TARGET_DIR} (설정 라벨: [{ACTUAL_CLASS}])")
    summary_lines.append("═" * 60)
    
    total_all = 0
    correct_all = 0

    for interval, count in stats.items():
        total = count[0]
        correct = count[1]
        rate = (correct / total * 100) if total > 0 else 0.0
        
        # 💡 보이스피싱과 정상대화 모두 문구를 '정탐률'로 일치화하여 출력
        summary_lines.append(f"  {interval:<12}: 정확한 판정 {correct} / 이번 분석 {total} 개 (정탐률: {rate:.1f}%)")
        
        total_all += total
        correct_all += correct

    overall_rate = (correct_all / total_all * 100) if total_all > 0 else 0.0
    summary_lines.append("-" * 60)
    summary_lines.append(f"  {'TOTAL':<12}: 정확한 판정 {correct_all} / 전체 분석 {total_all} 개 (종합 정탐률: {overall_rate:.1f}%)")
    summary_lines.append("═" * 60)

    # 변수 이름 충돌 방지 (`summary_f` 사용)
    with open(summary_file_path, "w", encoding="utf-8") as summary_f:
        summary_f.write("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))
    print(f"💾 [통합 상세 목록 CSV 누적완료]: {csv_file_path}")
    print(f"💾 [이번 회차 요약 결과 TXT 저장완료]: {summary_file_path}\n")

if __name__ == "__main__":
    main()