# baseline_gui.py ─ 순수 STT + BERT 성능 비교용 (멀티모달 기능 제거 버전)
import sys
import os
import csv
import gc
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QTextCursor

# 기존 모듈 임포트 (키워드 분석, 음성 분석 모듈은 사용하지 않음)
from config import SAMPLE_RATE, DECAY_RATE_FILE
from whisper_stt import run_stt_with_timestamps
from model_kluebert import detect_batch
from risk_calculator import RiskAccumulator, calculate_risk

# ── 공통 UI 유틸리티 ──
class StreamRedirector(object):
    def __init__(self, signal):
        self._signal = signal
    def write(self, text):
        if text: self._signal.emit(text)
    def flush(self):
        pass

def append_colored(box: QTextEdit, text: str):
    cursor = box.textCursor()
    cursor.movePosition(QTextCursor.End)
    box.setTextCursor(cursor)
    fmt = box.currentCharFormat()
    
    color = "#c9d1d9"
    if "🚨" in text or "보이스피싱" in text: color = "#ff6b6b"
    elif "✅" in text or "정상" in text: color = "#69db7c"
    elif "📊" in text or "TOTAL" in text: color = "#74c0fc"
    
    fmt.setForeground(QColor(color))
    cursor.setCharFormat(fmt)
    cursor.insertText(text)
    box.ensureCursorVisible()

# ── 배치 처리 워커 (순수 STT + BERT만 구동) ──
class BaselineWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal = pyqtSignal(str)

    def __init__(self, target_dir: str, actual_class: str):
        super().__init__()
        self.target_dir = target_dir
        self.actual_class = actual_class
        self._stop_flag = False
        self._old_stdout = None

    def stop(self):
        self._stop_flag = True

    def run(self):
        self._old_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.log_signal)
        try:
            OUTPUT_DIR = "./experiment_results"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            csv_path = os.path.join(OUTPUT_DIR, "baseline_total_results.csv")
            
            audio_files = [f for f in os.listdir(self.target_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.pcm'))]
            total_files = len(audio_files)
            
            intervals = ["~20s", "20s~40s", "40s~60s", "1m~1m20s", "1m20s~"]
            stats = {inv: {"total": 0, "correct": 0} for inv in intervals}

            self.log_signal.emit(f"🧪 [Baseline] 순수 STT+BERT 배치 실험 시작 (총 {total_files}개)\n")
            self.log_signal.emit(f"⚠️ 키워드 분석, Arousal 감정 분석, AI 합성 탐지 로직이 모두 비활성화되었습니다.\n\n")

            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                fieldnames = ["파일명", "실제구분", "음성길이(초)", "길이구간", "베이스라인_최종점수", "최종판정"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                    writer.writeheader()

                for idx, filename in enumerate(audio_files):
                    if self._stop_flag: break
                    current_idx = idx + 1
                    self.progress_signal.emit(current_idx, total_files)

                    audio_path = os.path.join(self.target_dir, filename)
                    temp_wav = None

                    try:
                        if audio_path.lower().endswith(".pcm"):
                            raw = np.fromfile(audio_path, dtype=np.int16)
                            y = raw.astype(np.float32) / 32768.0
                            temp_wav = os.path.join(self.target_dir, f"temp_{filename}.wav")
                            sf.write(temp_wav, y, SAMPLE_RATE)
                            stt_path = temp_wav
                        else:
                            y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                            stt_path = audio_path

                        duration = len(y) / SAMPLE_RATE
                        if duration <= 20: interval = "~20s"
                        elif duration <= 40: interval = "20s~40s"
                        elif duration <= 60: interval = "40s~60s"
                        elif duration <= 80: interval = "1m~1m20s"
                        else: interval = "1m20s~"

                        # 1. STT 변환
                        segments = run_stt_with_timestamps(stt_path)
                        if not segments: continue

                        # 2. 누적기 초기화 (감쇠율만 적용)
                        accumulator = RiskAccumulator(decay_rate=DECAY_RATE_FILE)

                        # 3. 텍스트 세그먼트별 순수 BERT 분석
                        for start, end, text in segments:
                            # 멀티모달 기능 배제: BERT의 라벨과 신뢰도만 추출
                            label, conf = detect_batch([text])[0]
                            
                            # calculate_risk의 k_score=0, has_combined_risk=False 로 고정하여
                            # 키워드 및 조합 보너스를 원천 차단
                            risk = calculate_risk(label=label, confidence=conf, k_score=0, has_combined_risk=False)
                            
                            # 감정(arousal) 수치 전달 안 함
                            accumulator.update(risk, label, current_arousal=None)

                        # 4. 최종 판정 (임계값 3.5 기준)
                        final_score = accumulator.cumulative_score
                        is_phishing = final_score >= 3.5
                        final_verdict = "보이스피싱" if is_phishing else "정상"
                        
                        is_correct = ((is_phishing and self.actual_class == "보이스피싱") or 
                                      (not is_phishing and self.actual_class == "정상대화"))

                        stats[interval]["total"] += 1
                        if is_correct: stats[interval]["correct"] += 1

                        writer.writerow({
                            "파일명": filename, "실제구분": self.actual_class, 
                            "음성길이(초)": round(duration, 1), "길이구간": interval,
                            "베이스라인_최종점수": round(final_score, 2), "최종판정": final_verdict
                        })
                        f.flush()

                        res_tag = "🚨[보이스피싱]" if is_phishing else "✅[정상]"
                        self.log_signal.emit(f"📊 [{current_idx}/{total_files}] {filename} -> 점수: {final_score:.2f}/5.00 | 결과: {res_tag}\n")

                    except Exception as e:
                        self.log_signal.emit(f"❌ [{current_idx}] {filename} 실패: {e}\n")
                    finally:
                        if temp_wav and os.path.exists(temp_wav): os.remove(temp_wav)
                        gc.collect()

            # 최종 통계 요약
            summary_lines = ["\n" + "═" * 60, f"📈 베이스라인 (STT+BERT Only) 실험 통계 리포트", f"🏷️ 실제 데이터 그룹: [{self.actual_class}]", "═" * 60]
            total_all, correct_all = 0, 0

            for inv in intervals:
                t = stats[inv]["total"]
                if t == 0: continue
                c = stats[inv]["correct"]
                r = (c / t * 100) if t > 0 else 0.0
                summary_lines.append(f"  {inv:<12}: 정탐 {c:<3} / 전체 {t:<3} 개 (정탐률: {r:>5.1f}%)")
                total_all += t; correct_all += c

            if total_all > 0:
                overall_rate = (correct_all / total_all * 100)
                summary_lines.append("-" * 60)
                summary_lines.append(f"  {'TOTAL':<12}: 정탐 {correct_all:<3} / 전체 {total_all:<3} 개 (종합 정탐률: {overall_rate:>5.1f}%)")
            
            summary_lines.append("═" * 60)
            self.done_signal.emit("\n".join(summary_lines))

        finally:
            sys.stdout = self._old_stdout

# ── 메인 GUI 윈도우 ──
class BaselineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧪 베이스라인 단일 모델 테스트 (STT + BERT Only)")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")
        self._folder_path = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. 폴더 선택
        row1 = QHBoxLayout()
        self.btn_folder = QPushButton("📂 오디오 폴더 선택")
        self.btn_folder.setStyleSheet("background:#21262d; padding:10px; border-radius:5px; font-weight:bold;")
        self.btn_folder.clicked.connect(self._pick_folder)
        self.lbl_folder = QLabel("선택된 폴더 없음")
        row1.addWidget(self.btn_folder)
        row1.addWidget(self.lbl_folder)
        layout.addLayout(row1)

        # 2. 클래스 선택 (보이스피싱 vs 정상대화)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("데이터 실제 구분:"))
        self.btn_phish = QPushButton("보이스피싱")
        self.btn_phish.setCheckable(True)
        self.btn_phish.setChecked(True)
        self.btn_phish.setStyleSheet("QPushButton:checked { background-color: #da3633; color: white; border-radius:5px; padding:8px;} QPushButton { background-color:#21262d; border-radius:5px; padding:8px;}")
        
        self.btn_normal = QPushButton("정상대화")
        self.btn_normal.setCheckable(True)
        self.btn_normal.setStyleSheet("QPushButton:checked { background-color: #2ea043; color: white; border-radius:5px; padding:8px;} QPushButton { background-color:#21262d; border-radius:5px; padding:8px;}")
        
        self.btn_phish.clicked.connect(lambda: (self.btn_normal.setChecked(False), self.btn_phish.setChecked(True)))
        self.btn_normal.clicked.connect(lambda: (self.btn_phish.setChecked(False), self.btn_normal.setChecked(True)))
        
        row2.addWidget(self.btn_phish)
        row2.addWidget(self.btn_normal)
        row2.addStretch()
        layout.addLayout(row2)

        # 3. 컨트롤 버튼
        row3 = QHBoxLayout()
        self.btn_run = QPushButton("▶ 베이스라인 실험 실행")
        self.btn_run.setStyleSheet("background:#1f6feb; color:white; padding:10px; border-radius:5px; font-weight:bold;")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar { background:#21262d; border:none; height:10px;} QProgressBar::chunk { background:#1f6feb; }")
        
        row3.addWidget(self.btn_run)
        row3.addWidget(self.progress)
        layout.addLayout(row3)

        # 4. 로그 박스
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background:#161b22; border:1px solid #30363d; font-family:Consolas, monospace; padding:10px;")
        layout.addWidget(self.log_box)

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "오디오 폴더 선택")
        if path:
            self._folder_path = path
            self.lbl_folder.setText(path)
            self.btn_run.setEnabled(True)

    def _run(self):
        self.log_box.clear()
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        
        actual_class = "보이스피싱" if self.btn_phish.isChecked() else "정상대화"
        self.worker = BaselineWorker(self._folder_path, actual_class)
        self.worker.log_signal.connect(lambda t: append_colored(self.log_box, t))
        self.worker.progress_signal.connect(lambda c, t: (self.progress.setMaximum(t), self.progress.setValue(c)))
        self.worker.done_signal.connect(self._on_done)
        self.worker.start()

    def _on_done(self, summary_txt):
        append_colored(self.log_box, f"\n{summary_txt}\n")
        self.btn_run.setEnabled(True)
        append_colored(self.log_box, "\n💾 결과가 'experiment_results/baseline_total_results.csv'에 저장되었습니다.\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BaselineApp()
    win.show()
    sys.exit(app.exec_())