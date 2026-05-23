# app_gui.py  ─ 보이스피싱 탐지 시스템 통합 GUI (v2.5)
import sys
import os
import io
import gc
import threading
import traceback
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QFileDialog, QFrame,
    QSizePolicy, QProgressBar, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer,
)
from PyQt5.QtGui import (
    QFont, QColor, QTextCursor,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  공통 유틸리티 및 스타일 시스템
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StreamRedirector(io.TextIOBase):
    """sys.stdout을 가로채서 Qt 시그널로 전달하는 스트림 객체."""
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, text: str) -> int:
        if text and text != "\n":
            self._signal.emit(text)
        elif text == "\n":
            self._signal.emit("\n")
        return len(text)

    def flush(self):
        pass


def make_text_box() -> QTextEdit:
    """공통 결과 출력 텍스트 박스 생성."""
    box = QTextEdit()
    box.setReadOnly(True)
    box.setFont(QFont("D2Coding", 10) if sys.platform == "darwin" else QFont("Consolas", 10))
    box.setStyleSheet("""
        QTextEdit {
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px;
            selection-background-color: #1f6feb;
        }
    """)
    return box


def append_colored(box: QTextEdit, text: str):
    """텍스트 내용에 따른 실시간 컬러 하이라이팅 마킹 시스템."""
    color_map = [
        (["🚨", "보이스피싱 확실", "강력 경고", "위험"], "#ff6b6b"),
        (["⚠️", "주의", "의심", "경고", "화제전환", "태도변화"], "#ffa94d"),
        (["✅", "완료", "정상", "일상", "해제"], "#69db7c"),
        (["🔍", "분석 중", "로딩", "STT", "BERT", "초기화"], "#74c0fc"),
        (["💾", "저장", "리포트"], "#da77f2"),
        (["🔥", "피싱 누적 지수", "Score"], "#ff922b"),
    ]
    color = "#c9d1d9"
    for keywords, c in color_map:
        if any(kw in text for kw in keywords):
            color = c
            break

    cursor = box.textCursor()
    cursor.movePosition(QTextCursor.End)
    box.setTextCursor(cursor)

    fmt = box.currentCharFormat()
    fmt.setForeground(QColor(color))
    cursor.setCharFormat(fmt)
    cursor.insertText(text)
    box.setTextCursor(cursor)
    box.ensureCursorVisible()


def make_btn(label: str, color: str = "#1f6feb", hover: str = "#388bfd") -> QPushButton:
    """공통 디자인 버튼 생성 팩토리."""
    btn = QPushButton(label)
    btn.setFont(QFont("Malgun Gothic", 10, QFont.Medium) if sys.platform == "win32" else QFont("Apple SD Gothic Neo", 10, QFont.Medium))
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:disabled {{
            background-color: #30363d;
            color: #6e7681;
        }}
        QPushButton:pressed {{
            background-color: #0d419d;
        }}
    """)
    return btn


def ask_save_report(parent, report_text: str, mode_title: str):
    """공통 리포트 파일 저장 대화상자 함수."""
    if not report_text.strip():
        return
    reply = QMessageBox.question(
        parent, '수사 리포트 저장',
        f'[{mode_title}] 최종 수사 리포트 생성이 완료되었습니다.\n텍스트 파일(.txt)로 저장하시겠습니까?',
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
    )
    if reply == QMessageBox.Yes:
        default_name = f"phishing_{mode_title.lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(parent, "리포트 저장 경로 선택", default_name, "텍스트 파일 (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                append_colored(parent._log, f"💾 [저장 완료] 리포트가 성공적으로 저장되었습니다:\n   {path}\n")
            except Exception as e:
                append_colored(parent._log, f"❌ [저장 실패] 오류: {e}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tab 1: 파일 기반 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FileAnalysisWorker(QThread):
    log_signal   = pyqtSignal(str)
    done_signal  = pyqtSignal(float, str)   
    error_signal = pyqtSignal(str)

    def __init__(self, audio_path: str):
        super().__init__()
        self.audio_path  = audio_path
        self._old_stdout = None
        self._stop_flag  = False  # 중단 플래그 추가

    def stop(self):
        """외부에서 호출하여 스레드 작업을 안전하게 중단시킵니다."""
        self._stop_flag = True

    def run(self):
        self._old_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.log_signal)
        try:
            import librosa
            from config import SAMPLE_RATE, MIN_CHUNK_SAMPLES, DECAY_RATE_FILE
            from whisper_stt import run_stt_with_timestamps
            from risk_calculator import RiskAccumulator, calculate_multimodal_risk
            from keyword_score import keyword_score
            from llm_report import generate_report
            from voice_analyzer import VoiceAnalyzer
            from model_kluebert import detect_batch

            self.log_signal.emit(f"\n{'━'*55}\n")
            self.log_signal.emit(f"  📄 파일 분석 시작: {self.audio_path}\n")
            self.log_signal.emit(f"{'━'*55}\n\n")

            if not os.path.exists(self.audio_path):
                self.error_signal.emit(f"파일을 찾을 수 없습니다: {self.audio_path}")
                return

            self.log_signal.emit("[1/4] 오디오 로딩 중...\n")
            y, _ = librosa.load(self.audio_path, sr=SAMPLE_RATE, mono=True)
            self.log_signal.emit(f"      완료 — 길이: {len(y)/SAMPLE_RATE:.1f}초\n\n")

            if self._stop_flag: return self._abort_early()

            self.log_signal.emit("[2/4] 음성 → 텍스트 변환 중... (파일 크기에 따라 시간이 소요됩니다)\n")
            segments = run_stt_with_timestamps(self.audio_path)
            if not segments:
                self.error_signal.emit("STT 결과가 없습니다.")
                return
            self.log_signal.emit(f"      {len(segments)}개 세그먼트 추출 완료.\n\n")

            if self._stop_flag: return self._abort_early()

            self.log_signal.emit("[3/4] 분석 모듈 초기화 중...\n")
            voice       = VoiceAnalyzer()
            accumulator = RiskAccumulator(decay_rate=DECAY_RATE_FILE)
            full_texts, all_detected = [], []
            self.log_signal.emit("      완료.\n\n")

            self.log_signal.emit("[4/4] 세그먼트별 분석 시작...\n\n")

            for start, end, text in segments:
                # 💡 사용자가 중단 버튼을 누르면 즉시 루프 탈출
                if self._stop_flag:
                    self.log_signal.emit("\n⚠️ 사용자의 요청으로 파일 분석이 중간에 정지되었습니다. (현재까지의 데이터로 리포트를 산출합니다)\n")
                    break

                full_texts.append(text)
                chunk = y[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]

                label, conf              = detect_batch([text])[0]
                k_score, detected, combo = keyword_score(text)
                if detected: all_detected.extend(detected)

                cur_a, base_a, voice_available = 0.5, 0.5, False
                if len(chunk) >= MIN_CHUNK_SAMPLES:
                    if not voice.is_calibrated:
                        voice.feed_calibration(chunk, end)
                    elif k_score > 0 or label == "보이스피싱":
                        cur_a, base_a = voice.get_arousal_state(chunk)
                        voice_available = True
                        voice.accumulate_ai_prob(chunk, risk_weight=accumulator.cumulative_score)

                risk, tone = calculate_multimodal_risk(
                    label=label, confidence=conf, k_score=k_score,
                    has_combined_risk=combo, current_arousal=cur_a,
                    baseline_arousal=base_a, voice_available=voice_available, verbose=False
                )
                total_score = accumulator.update(risk, label, current_arousal=cur_a if voice_available else None)

                if accumulator.detect_topic_shift():
                    self.log_signal.emit("  ⚠️  [화제전환] 갑작스러운 일상 전환 감지\n")
                shifted, smsg = accumulator.detect_attitude_shift()
                if shifted: self.log_signal.emit(f"  {smsg}\n")

                sep  = "-" * 50
                bar  = "█" * int(total_score / 5.0 * 20) + "░" * (20 - int(total_score / 5.0 * 20))
                self.log_signal.emit(f"\n[ 문장 분석 ] {sep}\n")
                self.log_signal.emit(f"  🕐 [{start:6.1f}s]  🗣️  인식: \"{text}\"\n")
                self.log_signal.emit(f"  🔥 피싱 누적 지수: {total_score:.2f} / 5.00  [{bar}]\n")
                if total_score >= 3.5: self.log_signal.emit("  🚨🚨 [강력 경고] 보이스피싱 확실!\n")
                elif total_score >= 2.0: self.log_signal.emit("  ⚠️  [주의] 보이스피싱 의심 정황 포착\n")
                self.log_signal.emit(f"  🎭 감정 상태: {tone}\n")
                if detected:
                    clean = [d for d in detected if not d.startswith("[")]
                    if clean: self.log_signal.emit(f"  🔑 탐지 키워드: {', '.join(clean)}\n")
                self.log_signal.emit(f"  {sep}------\n")

            voice.release()
            gc.collect()

            if not full_texts:
                return self._abort_early()

            final_score = accumulator.cumulative_score
            detail_str  = ", ".join(list(dict.fromkeys(d for d in all_detected if not d.startswith("[조합:") and not d.startswith("[완곡:")))) or "없음"

            ai_prob, ai_label = None, None
            if final_score >= 2.0:
                self.log_signal.emit("\n  🔍 AI 생성 음성 종합 판정 중...\n")
                ai_prob, ai_label = voice.finalize_ai_verdict()
                self.log_signal.emit(f"  결과: {ai_label}  ({ai_prob*100:.1f}%)\n")

            report = generate_report(" ".join(full_texts), final_score, detail_str, ai_prob=ai_prob, ai_label=ai_label)
            self.done_signal.emit(final_score, report)
        except Exception as e:
            self.error_signal.emit(f"오류 발생: {e}\n{traceback.format_exc()}")
        finally:
            sys.stdout = self._old_stdout

    def _abort_early(self):
        """본격적인 분석 루프 전에 중단되었을 경우의 처리"""
        self.log_signal.emit("\n⚠️ 분석 시작 전에 중단되었습니다.\n")
        self.done_signal.emit(0.0, "")


class FileAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._audio_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        file_row = QHBoxLayout()
        self._path_label = QLabel("선택된 파일 없음")
        self._path_label.setStyleSheet("color:#8b949e; font-style:italic;")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_pick = make_btn("📁  파일 선택", "#21262d", "#30363d")
        btn_pick.clicked.connect(self._pick_file)

        file_row.addWidget(btn_pick)
        file_row.addWidget(self._path_label)
        layout.addLayout(file_row)

        ctrl_row = QHBoxLayout()
        self._btn_run = make_btn("▶  분석 실행", "#1f6feb")
        self._btn_run.clicked.connect(self._run_analysis)
        self._btn_run.setEnabled(False)

        # 💡 중단 버튼 추가
        self._btn_stop = make_btn("⏹  분석 중단", "#da3633", "#f85149")
        self._btn_stop.clicked.connect(self._stop_analysis)
        self._btn_stop.setEnabled(False)

        self._btn_clear = make_btn("🗑  로그 지우기", "#21262d", "#30363d")
        self._btn_clear.clicked.connect(lambda: self._log.clear())

        ctrl_row.addWidget(self._btn_run)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addWidget(self._btn_clear)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setStyleSheet("QProgressBar { background:#21262d; border:none; border-radius:4px; height:6px; } QProgressBar::chunk { background:#1f6feb; border-radius:4px; }")
        layout.addWidget(self._progress)

        self._log = make_text_box()
        layout.addWidget(self._log)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "오디오 파일 선택", "", "오디오 파일 (*.mp3 *.wav *.m4a *.mp4 *.pcm);;모든 파일 (*)")
        if path:
            self._audio_path = path
            self._path_label.setText(os.path.basename(path))
            self._path_label.setStyleSheet("color:#c9d1d9;")
            self._btn_run.setEnabled(True)

    def _run_analysis(self):
        if not self._audio_path: return
        self._log.clear()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)  # 실행 시 중단 버튼 활성화
        self._progress.setVisible(True)

        self._worker = FileAnalysisWorker(self._audio_path)
        self._worker.log_signal.connect(lambda t: append_colored(self._log, t))
        self._worker.done_signal.connect(self._on_done)
        self._worker.error_signal.connect(lambda e: append_colored(self._log, f"\n❌ 오류: {e}\n"))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop_analysis(self):
        """작업 중단 신호를 워커 스레드에 보냄"""
        if self._worker:
            self._worker.stop()
        self._btn_stop.setEnabled(False)
        append_colored(self._log, "\n⏳ 분석 중단 요청됨... (현재 진행 중인 작업 완료 직후 안전하게 정지합니다)\n")

    def _on_done(self, score: float, report: str):
        if not report:  # 완전 극초기에 중단되어 출력할 내용이 없는 경우
            return

        append_colored(self._log, f"\n\n{'═'*55}\n")
        append_colored(self._log, f"  📊 최종 누적 위험도: {score:.2f}/5.00\n")
        append_colored(self._log, f"{'═'*55}\n\n")
        
        if report and "일상 단계" not in report:
            append_colored(self._log, report + "\n")
            ask_save_report(self, report, "File")
        else:
            append_colored(self._log, "✅ 위험 점수가 2.0 미만이므로 보이스피싱 위험이 낮은 일반 대화로 분류되었습니다. (상세 리포트 생략)\n")

    def _on_finished(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tab 2: 텍스트 입력 시뮬레이터 (QThread & Signal 완벽 버그 수정본)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextInitWorker(QThread):
    """무거운 모델 파일 패키지를 GUI 동결 없이 완전 비동기로 최초 1회 로드하는 워커."""
    loaded_signal = pyqtSignal(dict)
    error_signal  = pyqtSignal(str)

    def run(self):
        try:
            from model_kluebert import detect_batch
            from keyword_score import keyword_score
            from risk_calculator import RiskAccumulator, calculate_risk
            from llm_report import generate_report
            funcs = {
                'detect_batch': detect_batch, 'keyword_score': keyword_score,
                'RiskAccumulator': RiskAccumulator, 'calculate_risk': calculate_risk,
                'generate_report': generate_report
            }
            self.loaded_signal.emit(funcs)
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())


class TextAnalyzeWorker(QThread):
    """입력된 단일 문장에 대해 파이토치 BERT 및 키워드 누적 연산을 스레드 격리하여 실행하는 워커."""
    done_signal  = pyqtSignal(float, str, float, list, int)
    error_signal = pyqtSignal(str)

    def __init__(self, text, funcs, accumulator):
        super().__init__()
        self.text = text
        self.funcs = funcs
        self.accumulator = accumulator

    def run(self):
        try:
            label, conf = self.funcs['detect_batch']([self.text])[0]
            k_score, detected, combo = self.funcs['keyword_score'](self.text)
            risk = self.funcs['calculate_risk'](label, conf, k_score, combo)
            total_score = self.accumulator.update(risk, label)
            self.done_signal.emit(total_score, label, conf, detected, k_score)
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())


class TextReportWorker(QThread):
    """GGUF LLM 수사 지침서 작성을 메인 UI 스레드 방해 없이 장기 연산하는 리포트 워커."""
    done_signal  = pyqtSignal(str, float)
    error_signal = pyqtSignal(str)

    def __init__(self, history, all_detected, accumulator, generate_report_fn):
        super().__init__()
        self.history = history
        self.all_detected = all_detected
        self.accumulator = accumulator
        self.generate_report_fn = generate_report_fn

    def run(self):
        try:
            final_score = self.accumulator.cumulative_score
            detail_str = ", ".join(list(dict.fromkeys(d for d in self.all_detected if not d.startswith("[")))) or "없음"
            report = self.generate_report_fn(" ".join(self.history), final_score, detail_str, None, None)
            self.done_signal.emit(report, final_score)
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())


class TextSimTab(QWidget):
    def __init__(self):
        super().__init__()
        self._history = []
        self._all_detected = []
        self._funcs = None
        self._accumulator = None
        self._build_ui()
        self._start_init_modules()

    def _start_init_modules(self):
        self._set_loading(True, "모듈 초기화 중...")
        self._init_worker = TextInitWorker()
        self._init_worker.loaded_signal.connect(self._on_modules_loaded)
        self._init_worker.error_signal.connect(self._on_modules_error)
        self._init_worker.start()

    def _on_modules_loaded(self, funcs):
        self._funcs = funcs
        self._accumulator = funcs['RiskAccumulator'](decay_rate=0.9)
        append_colored(self._log, "[KLUE-BERT] 모듈 로드 완료. 문장 입력 준비됨.\n")
        self._set_loading(False)

    def _on_modules_error(self, err_trace):
        append_colored(self._log, f"❌ 모듈 로드 치명적 에러 발생:\n{err_trace}\n")
        self._set_loading(False)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("의심 문장을 입력하세요... (Enter 또는 버튼 클릭)")
        self._input.setStyleSheet("QLineEdit { background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; padding:8px 12px; font-size:11pt; } QLineEdit:focus { border-color:#1f6feb; } QLineEdit:disabled { background:#161b22; color:#6e7681; }")
        self._input.returnPressed.connect(self._send_sentence)

        self._btn_send = make_btn("입력 분석  ↵", "#1f6feb")
        self._btn_send.clicked.connect(self._send_sentence)

        input_row.addWidget(self._input)
        input_row.addWidget(self._btn_send)
        layout.addLayout(input_row)

        ctrl_row = QHBoxLayout()
        self._btn_report = make_btn("📋  분석 종료 & 최종 리포트", "#388bfd")
        self._btn_report.clicked.connect(self._generate_final_report)

        self._btn_reset = make_btn("🔄  초기화", "#21262d", "#30363d")
        self._btn_reset.clicked.connect(self._reset)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setStyleSheet("QProgressBar { background:#21262d; border:none; border-radius:4px; height:6px; max-width: 130px; margin-left: 10px;} QProgressBar::chunk { background:#1f6feb; border-radius:4px; }")
        
        self._progress_label = QLabel("준비 중...")
        self._progress_label.setStyleSheet("color:#8b949e; font-size:10pt; margin-left: 5px;")

        ctrl_row.addWidget(self._btn_report)
        ctrl_row.addWidget(self._btn_reset)
        ctrl_row.addWidget(self._progress)
        ctrl_row.addWidget(self._progress_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self._log = make_text_box()
        layout.addWidget(self._log)
        append_colored(self._log, "⏳ KLUE-BERT 핵심 연산 구조 분석기 기동 중...\n")

    def _set_loading(self, is_loading: bool, msg: str = ""):
        self._progress.setVisible(is_loading)
        self._progress_label.setText(msg if is_loading else "")
        self._input.setEnabled(not is_loading)
        self._btn_send.setEnabled(not is_loading)
        self._btn_report.setEnabled(not is_loading and len(self._history) > 0)
        self._btn_reset.setEnabled(not is_loading)
        if not is_loading: self._input.setFocus()

    def _send_sentence(self):
        text = self._input.text().strip()
        if not text or not self._funcs: return
        self._input.clear()
        append_colored(self._log, f"\n🗣️  입력: \"{text}\"\n")
        self._set_loading(True, "문장 분석 중...")

        self._analyze_worker = TextAnalyzeWorker(text, self._funcs, self._accumulator)
        self._analyze_worker.done_signal.connect(self._on_sentence_analyzed)
        self._analyze_worker.error_signal.connect(self._on_worker_error)
        self._history.append(text)
        self._analyze_worker.start()

    def _on_sentence_analyzed(self, score, label, conf, detected, k_score):
        if detected: self._all_detected.extend(detected)
        bar = "█" * int(score / 5.0 * 20) + "░" * (20 - int(score / 5.0 * 20))
        append_colored(self._log, f"  🤖 BERT: {label} (신뢰도 {conf:.2f})\n")
        append_colored(self._log, f"  🔥 피싱 누적 지수: {score:.2f} / 5.00  [{bar}]\n")
        if score >= 3.5: append_colored(self._log, "  🚨🚨 [강력 경고] 보이스피싱 확실!\n")
        elif score >= 2.0: append_colored(self._log, "  ⚠️  [주의] 보이스피싱 의심 정황 포착\n")
        if detected:
            clean = [d for d in detected if not d.startswith("[")]
            if clean: append_colored(self._log, f"  🔑 탐지 키워드: {', '.join(clean)}\n")
        append_colored(self._log, "  " + "─"*50 + "\n")
        self._set_loading(False)

    def _on_worker_error(self, err_trace):
        append_colored(self._log, f"❌ 내부 분석 동작 처리 중 에러:\n{err_trace}\n")
        self._set_loading(False)

    def _generate_final_report(self):
        if not self._history: return
        self._set_loading(True, "LLM 리포트 생성 중...")
        append_colored(self._log, "\n🔍 최종 수사 리포트를 생성하는 중입니다...\n")
        append_colored(self._log, "   (LLM 인공지능 요약서 작성에 최대 1분 내외 소요될 수 있으니 잠시 대기바랍니다.)\n")

        self._report_worker = TextReportWorker(self._history, self._all_detected, self._accumulator, self._funcs['generate_report'])
        self._report_worker.done_signal.connect(self._on_report_done)
        self._report_worker.error_signal.connect(self._on_worker_error)
        self._report_worker.start()

    def _on_report_done(self, report, score):
        append_colored(self._log, f"\n{'═'*55}\n")
        append_colored(self._log, f"  📊 최종 누적 위험도: {score:.2f}/5.00\n")
        append_colored(self._log, f"{'═'*55}\n\n")
        if report:
            append_colored(self._log, report + "\n")
            self._set_loading(False)
            ask_save_report(self, report, "Text")
        else:
            append_colored(self._log, "✅ 위험 점수가 2.0 미만이므로 보이스피싱 위험이 낮은 일반 대화로 분류되었습니다. (상세 리포트 생략)\n")
            self._set_loading(False)

    def _reset(self):
        self._history.clear()
        self._all_detected.clear()
        if self._funcs and self._accumulator:
            self._accumulator = self._funcs['RiskAccumulator'](decay_rate=0.9)
        self._log.clear()
        append_colored(self._log, "🔄 초기화 완료. 새 대화를 입력하세요.\n")
        self._set_loading(False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tab 3: 실시간 마이크 감시
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MicWorker(QThread):
    log_signal  = pyqtSignal(str)
    done_signal = pyqtSignal(float, str)

    def __init__(self):
        super().__init__()
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.log_signal)
        try:
            import speech_recognition as sr_lib
            from faster_whisper import WhisperModel
            from config import SAMPLE_RATE, MIN_CHUNK_SAMPLES, DECAY_RATE_RT
            from utils_text import normalize_stt
            from keyword_score import keyword_score
            from risk_calculator import RiskAccumulator, calculate_multimodal_risk
            from voice_analyzer import VoiceAnalyzer
            from llm_report import generate_report
            from model_kluebert import detect_batch

            _STT_CHUNK_SEC, _ENERGY_THRESH = 5, 0.01

            class _AudioBuffer:
                def __init__(self):
                    self._buf = np.array([], dtype=np.float32)
                    self._off = 0
                    self._max = SAMPLE_RATE * 120
                def append(self, chunk):
                    self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
                    if len(self._buf) > self._max:
                        trim = len(self._buf) - self._max
                        self._buf = self._buf[trim:]; self._off += trim
                def get_chunk(self, s, e):
                    si = max(0, int(s * SAMPLE_RATE) - self._off)
                    ei = min(len(self._buf), int(e * SAMPLE_RATE) - self._off)
                    return self._buf[si:ei].copy() if ei > si else None

            audio_buf = _AudioBuffer()
            voice     = VoiceAnalyzer()
            accumulator = RiskAccumulator(decay_rate=DECAY_RATE_RT)
            full_texts, all_detected = [], []

            model = WhisperModel("small", device="cpu", compute_type="int8")
            r     = sr_lib.Recognizer()
            chunk_start = 0.0

            self.log_signal.emit("🎤 주변 소음 캘리브레이션 분석 중...\n")
            with sr_lib.Microphone(sample_rate=SAMPLE_RATE) as source:
                r.adjust_for_ambient_noise(source, duration=2)
                self.log_signal.emit("=" * 55 + "\n")
                self.log_signal.emit("  🎙️  실시간 보이스피싱 탐지 시작  (종료 버튼 선택 전까지 가동)\n")
                self.log_signal.emit("=" * 55 + "\n\n")

                while not self._stop_flag:
                    try: data = r.listen(source, phrase_time_limit=_STT_CHUNK_SEC, timeout=1)
                    except sr_lib.WaitTimeoutError: continue

                    raw = data.get_raw_data()
                    arr = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                    dur = len(arr) / SAMPLE_RATE
                    audio_buf.append(arr)
                    chunk_start += dur

                    if np.sqrt(np.mean(arr ** 2)) < _ENERGY_THRESH: continue

                    this_start = chunk_start - dur
                    segs, _    = model.transcribe(arr, language="ko", beam_size=5)

                    for seg in segs:
                        if self._stop_flag: break
                        text = normalize_stt(seg.text.strip())
                        if not text: continue
                        full_texts.append(text)

                        label, conf              = detect_batch([text])[0]
                        k_score, detected, combo = keyword_score(text)
                        if detected: all_detected.extend(detected)

                        cur_a, base_a, voice_available = 0.5, 0.5, False
                        chunk = audio_buf.get_chunk(this_start + seg.start, this_start + seg.end)
                        if chunk is not None and len(chunk) >= MIN_CHUNK_SAMPLES:
                            if not voice.is_calibrated: voice.feed_calibration(chunk, this_start + seg.end)
                            elif k_score > 0 or label == "보이스피싱":
                                cur_a, base_a = voice.get_arousal_state(chunk)
                                voice_available = True
                                voice.accumulate_ai_prob(chunk, risk_weight=accumulator.cumulative_score)

                        risk, tone = calculate_multimodal_risk(
                            label=label, confidence=conf, k_score=k_score,
                            has_combined_risk=combo, current_arousal=cur_a,
                            baseline_arousal=base_a, voice_available=voice_available, verbose=False
                        )
                        total_score = accumulator.update(risk, label, current_arousal=cur_a if voice_available else None)

                        bar  = "█" * int(total_score / 5.0 * 20) + "░" * (20 - int(total_score / 5.0 * 20))
                        self.log_signal.emit(f"\n[ 실시간 탐지 결과 ] {'─'*35}\n")
                        self.log_signal.emit(f"  🗣️  인식: \"{text}\"\n")
                        self.log_signal.emit(f"  🔥 피싱 누적 지수: {total_score:.2f} / 5.00  [{bar}]\n")
                        if total_score >= 3.5: self.log_signal.emit("  🚨🚨 [강력 경고] 보이스피싱 확실!\n")
                        elif total_score >= 2.0: self.log_signal.emit("  ⚠️  [주의] 보이스피싱 의심 단계\n")
                        self.log_signal.emit(f"  🎭 감정 상태: {tone}\n")

            del model; gc.collect()
            voice.release()

            final_score = accumulator.cumulative_score
            detail_str  = ", ".join(list(dict.fromkeys(d for d in all_detected if not d.startswith("[조합:") and not d.startswith("[완곡:")))) or "없음"

            ai_prob, ai_label = None, None
            if final_score >= 2.0 and voice._buf:
                self.log_signal.emit("\n  🔍 AI 생성 음성 종합 판정 중...\n")
                ai_prob, ai_label = voice.finalize_ai_verdict()
                self.log_signal.emit(f"  결과: {ai_label}  ({ai_prob*100:.1f}%)\n")

            report = generate_report(" ".join(full_texts), final_score, detail_str, ai_prob=ai_prob, ai_label=ai_label) if full_texts else ""
            self.done_signal.emit(final_score, report)
        except Exception as e:
            self.log_signal.emit(f"\n❌ 마이크 모듈 오류: {e}\n{traceback.format_exc()}\n")
        finally:
            sys.stdout = old_stdout


class MicTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._status = QLabel("⏸  대기 중")
        self._status.setStyleSheet("color:#8b949e; font-size:11pt; padding:4px 0;")
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._btn_start = make_btn("🎙  감시 시작", "#2ea043", "#3fb950")
        self._btn_start.clicked.connect(self._start)

        self._btn_stop = make_btn("⏹  감시 종료", "#da3633", "#f85149")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)

        self._btn_clear = make_btn("🗑  로그 지우기", "#21262d", "#30363d")
        self._btn_clear.clicked.connect(lambda: self._log.clear())

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._log = make_text_box()
        layout.addWidget(self._log)

    def _start(self):
        self._log.clear()
        self._status.setText("🔴  감시 중 (실시간 음성 분석 기동)")
        self._status.setStyleSheet("color:#f85149; font-size:11pt; padding:4px 0;")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = MicWorker()
        self._worker.log_signal.connect(lambda t: append_colored(self._log, t))
        self._worker.done_signal.connect(self._on_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop(self):
        if self._worker: self._worker.stop()
        self._btn_stop.setEnabled(False)
        self._status.setText("⏳  최종 결과서 연산 중...")

    def _on_done(self, score: float, report: str):
        append_colored(self._log, f"\n\n{'═'*55}\n")
        append_colored(self._log, f"  📊 최종 누적 위험도: {score:.2f}/5.00\n")
        append_colored(self._log, f"{'═'*55}\n\n")
        if report:
            append_colored(self._log, report + "\n")
            ask_save_report(self, report, "Mic")
        else:
            append_colored(self._log, "✅ 분석 결과: 보이스피싱 위험이 낮은 일상 대화입니다.\n")

    def _on_finished(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText("⏸  대기 중")
        self._status.setStyleSheet("color:#8b949e; font-size:11pt; padding:4px 0;")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tab 4: 일괄 배치 실험
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BatchWorker(QThread):
    log_signal     = pyqtSignal(str)
    progress_signal= pyqtSignal(int, int)   
    done_signal    = pyqtSignal(str)        

    def __init__(self, target_dir: str, actual_class: str):
        super().__init__()
        self.target_dir   = target_dir
        self.actual_class = actual_class
        self._old_stdout  = None
        self._stop_flag   = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        self._old_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.log_signal)
        try:
            import csv, soundfile as sf, librosa
            import numpy as np
            from config import SAMPLE_RATE, MIN_CHUNK_SAMPLES, DECAY_RATE_FILE
            from whisper_stt import run_stt_with_timestamps
            from risk_calculator import RiskAccumulator, calculate_multimodal_risk
            from keyword_score import keyword_score
            from voice_analyzer import VoiceAnalyzer
            from model_kluebert import detect_batch
            from datetime import datetime

            PHISHING_THRESHOLD, OUTPUT_DIR = 3.5, "./experiment_results"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            csv_path = os.path.join(OUTPUT_DIR, "experiment_total_results.csv")

            completed = set()
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        if "파일명" in row: completed.add(row["파일명"])

            audio_files = [f for f in os.listdir(self.target_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.pcm'))]
            total_files = len(audio_files)
            intervals = ["~20s", "20s~40s", "40s~60s", "1m~1m20s", "1m20s~"]
            stats = {inv: {"total": 0, "correct": 0} for inv in intervals}

            # Arousal 수치를 직관적인 텍스트 상태로 변환하는 함수
            def get_arousal_label(value):
                if value >= 0.65: return "흥분/압박"
                elif value <= 0.40: return "차분/침착"
                else: return "평온/중립"

            self.log_signal.emit(f"🚀 [{self.actual_class}] 배치 파일 다중 실험 시작 (총 {total_files}개 파일)\n")
            voice = VoiceAnalyzer()

            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                fieldnames = ["파일명","실제구분","음성길이(초)","길이구간","최종위험점수","AI음성확률","태도변화여부","감정변화상세(Arousal)","최종판정"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: writer.writeheader()

                for idx, filename in enumerate(audio_files):
                    current_idx = idx + 1
                    
                    if self._stop_flag:
                        self.log_signal.emit("\n⚠️ 사용자의 요청으로 배치 실험이 중간에 정지되었습니다. (현재까지의 데이터로 리포트를 산출합니다)\n")
                        break

                    self.progress_signal.emit(current_idx, total_files)
                    
                    if filename in completed:
                        self.log_signal.emit(f"⏩ [{current_idx}/{total_files}] {filename} (이미 분석 완료됨)\n")
                        continue

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

                        segments = run_stt_with_timestamps(stt_path)
                        if not segments: continue

                        accumulator = RiskAccumulator(decay_rate=DECAY_RATE_FILE)
                        voice.is_calibrated = False
                        voice._calib.clear(); voice._hist.clear(); voice._buf.clear()
                        
                        # 파일 전체 가청 구간의 감정 지표 추적용 배열
                        file_arousals = []

                        for start, end, text in segments:
                            si = int(start * SAMPLE_RATE)
                            ei = min(int(end * SAMPLE_RATE), len(y))
                            chunk = y[si:ei]

                            label, conf       = detect_batch([text])[0]
                            k_score, _, combo= keyword_score(text)
                            cur_a, base_a, voice_available = 0.5, 0.5, False

                            if len(chunk) >= MIN_CHUNK_SAMPLES:
                                if not voice.is_calibrated: voice.feed_calibration(chunk, end)
                                else:
                                    cur_a, base_a = voice.get_arousal_state(chunk)
                                    voice_available = True
                                    file_arousals.append(cur_a) # 실시간 Arousal 지표 축적
                                    if k_score > 0 or label == "보이스피싱":
                                        voice.accumulate_ai_prob(chunk, risk_weight=accumulator.cumulative_score)

                            risk, _ = calculate_multimodal_risk(label=label, confidence=conf, k_score=k_score, has_combined_risk=combo, current_arousal=cur_a, baseline_arousal=base_a, voice_available=voice_available, verbose=False)
                            accumulator.update(risk, label, current_arousal=cur_a if voice_available else None)

                        # ── 💡 [핵심 구현] 임계값 0.25 기반 양방향 감정 격변 매핑 알고리즘 ──
                        has_attitude_shift = False
                        shift_message = "변화 없음"
                        clean_smsg = shift_message

                        if len(file_arousals) >= 2:
                            # 통화 초반 3문장 평균을 기준으로 설정
                            start_a = float(np.mean(file_arousals[:3])) if len(file_arousals) >= 3 else file_arousals[0]
                            max_a = max(file_arousals)
                            min_a = min(file_arousals)
                            
                            delta_up = max_a - start_a     # 격앙/압박 방향 변화량
                            delta_down = start_a - min_a   # 차분/회유 방향 변화량
                            
                            # 요구사항에 맞춰 임계값을 0.25로 변경하고 양방향 탐지 활성화
                            if delta_up >= 0.25 or delta_down >= 0.25:
                                has_attitude_shift = True
                                # 상승 변화가 더 지배적일 때 (격앙)
                                if delta_up >= delta_down:
                                    clean_smsg = f"[{get_arousal_label(start_a)}] ➔ [{get_arousal_label(max_a)}] 상태로 감정 상승 (Δ+{delta_up:.2f})"
                                    shift_message = f"⚡[태도변화감지]\n     └ 감정 흐름: {clean_smsg}"
                                # 하락 변화가 더 지배적일 때 (회유/차분)
                                else:
                                    clean_smsg = f"[{get_arousal_label(start_a)}] ➔ [{get_arousal_label(min_a)}] 상태로 급격히 다운 (Δ-{delta_down:.2f})"
                                    shift_message = f"⚡[태도변화감지]\n     └ 감정 흐름: {clean_smsg}"

                        final_score = accumulator.cumulative_score
                        ai_prob = 0.0
                        if final_score >= 2.0 and voice._buf: ai_prob, _ = voice.finalize_ai_verdict()

                        is_phishing = final_score >= PHISHING_THRESHOLD
                        final_verdict = "보이스피싱" if is_phishing else "정상"
                        is_correct = ((is_phishing and self.actual_class == "보이스피싱") or (not is_phishing and self.actual_class == "정상대화"))

                        if interval in stats:
                            stats[interval]["total"] += 1
                            if is_correct: stats[interval]["correct"] += 1

                        # CSV 쓰기 작업 (상세 감정 텍스트 반영)
                        writer.writerow({
                            "파일명": filename, "실제구분": self.actual_class, "음성길이(초)": round(duration, 1), "길이구간": interval,
                            "최종위험점수": round(final_score, 2), "AI음성확률": f"{ai_prob*100:.1f}%" if ai_prob > 0 else "N/A",
                            "태도변화여부": "Y" if has_attitude_shift else "N", "감정변화상세(Arousal)": clean_smsg,
                            "최종판정": final_verdict
                        })
                        f.flush() 

                        # 실시간 결과 터미널 스타일 로그 렌더링
                        ai_str = f"{ai_prob*100:.1f}%" if (final_score >= 2.0 and voice._buf) else "N/A"
                        res_tag = "🚨[보이스피싱]" if is_phishing else "✅[정상]"
                        
                        log_txt = f"📊 [{current_idx}/{total_files}] {filename}\n"
                        log_txt += f"   └ 점수: {final_score:.2f} / 5.00 | AI: {ai_str:>5} | 결과: {res_tag}"
                        
                        if has_attitude_shift:
                            log_txt += f"  {shift_message}\n"
                        else:
                            log_txt += "\n"
                            
                        self.log_signal.emit(log_txt)

                    except Exception as e: 
                        self.log_signal.emit(f"❌ [{current_idx}/{total_files}] {filename} 실패: {e}\n")
                    finally:
                        if temp_wav and os.path.exists(temp_wav): os.remove(temp_wav)
                        gc.collect()

            voice.release()

            # 최종 통계 요약 리포트 산출 및 동기화
            summary_lines = []
            summary_lines.append("═" * 65)
            summary_lines.append(f"📊 배치 실험 통계 요약 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            summary_lines.append(f"🏷️ 포함된 실제 데이터 그룹: [{self.actual_class}]")
            summary_lines.append("═" * 65)

            total_all = 0
            correct_all = 0

            for inv in intervals:
                count_data = stats.get(inv, {"total": 0, "correct": 0})
                total = count_data["total"]
                if total == 0: continue
                correct = count_data["correct"]
                rate = (correct / total * 100) if total > 0 else 0.0
                summary_lines.append(f"  {inv:<12}: 정확한 판정 {correct:<3} / 전체 {total:<3} 개 (정탐률: {rate:>5.1f}%)")
                total_all += total
                correct_all += correct

            if total_all > 0:
                overall_rate = (correct_all / total_all * 100) if total_all > 0 else 0.0
                summary_lines.append("-" * 65)
                summary_lines.append(f"  {'TOTAL':<12}: 정확한 판정 {correct_all:<3} / 전체 {total_all:<3} 개 (종합 정탐률: {overall_rate:>5.1f}%)")
                summary_lines.append("═" * 65)
                self.done_signal.emit("\n".join(summary_lines))
            else:
                self.done_signal.emit("")

        except Exception as e: self.log_signal.emit(f"오류: {e}")
        finally: sys.stdout = self._old_stdout


class BatchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._folder_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        folder_row = QHBoxLayout()
        self._folder_label = QLabel("선택된 실험 대상 폴더 없음")
        self._folder_label.setStyleSheet("color:#8b949e; font-style:italic;")
        btn_folder = make_btn("📂  폴더 선택", "#21262d", "#30363d")
        btn_folder.clicked.connect(self._pick_folder)
        folder_row.addWidget(btn_folder)
        folder_row.addWidget(self._folder_label)
        layout.addLayout(folder_row)

        class_row = QHBoxLayout()
        lbl = QLabel("폴더 데이터 실제 구분 라벨 선택: ")
        
        # 버튼 토글 디자인
        toggle_style = """
            QPushButton {
                background-color: #21262d; color: #8b949e;
                border: 1px solid #30363d; border-radius: 6px;
                padding: 8px 18px; font-weight: 600;
            }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:checked { background-color: %s; color: #ffffff; border: none; }
            QPushButton:checked:hover { background-color: %s; }
        """
        
        self._class_fishing = make_btn("보이스피싱")
        self._class_fishing.setCheckable(True)
        self._class_fishing.setStyleSheet(toggle_style % ("#da3633", "#f85149"))
        self._class_fishing.setChecked(True)
        
        self._class_normal  = make_btn("정상대화")
        self._class_normal.setCheckable(True)
        self._class_normal.setStyleSheet(toggle_style % ("#2ea043", "#3fb950"))
        
        self._class_fishing.clicked.connect(lambda: (
            self._class_normal.setChecked(False), 
            self._class_fishing.setChecked(True)
        ))
        self._class_normal.clicked.connect(lambda: (
            self._class_fishing.setChecked(False), 
            self._class_normal.setChecked(True)
        ))
        
        class_row.addWidget(lbl)
        class_row.addWidget(self._class_fishing)
        class_row.addWidget(self._class_normal)
        class_row.addStretch()
        layout.addLayout(class_row)

        ctrl_row = QHBoxLayout()
        self._btn_run = make_btn("▶  배치 실험 실행", "#1f6feb")
        self._btn_run.clicked.connect(self._run)
        self._btn_run.setEnabled(False)

        self._btn_stop = make_btn("⏹  실험 중단", "#da3633", "#f85149")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)

        self._btn_clear = make_btn("🗑  로그 지우기", "#21262d", "#30363d")
        self._btn_clear.clicked.connect(lambda: self._log.clear())

        ctrl_row.addWidget(self._btn_run)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addWidget(self._btn_clear)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self._progress = QProgressBar()
        self._progress.setStyleSheet("QProgressBar { background:#21262d; border:none; border-radius:4px; height:6px; } QProgressBar::chunk { background:#1f6feb; border-radius:4px; }")
        layout.addWidget(self._progress)

        self._log = make_text_box()
        layout.addWidget(self._log)

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "오디오 폴더 선택")
        if path:
            self._folder_path = path
            self._folder_label.setText(path)
            self._btn_run.setEnabled(True)

    def _run(self):
        if not self._folder_path: return
        self._log.clear()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setValue(0)
        
        lbl = "보이스피싱" if self._class_fishing.isChecked() else "정상대화"
        self._worker = BatchWorker(self._folder_path, lbl)
        self._worker.log_signal.connect(lambda t: append_colored(self._log, t))
        self._worker.progress_signal.connect(lambda c, t: (self._progress.setMaximum(t), self._progress.setValue(c)))
        self._worker.done_signal.connect(self._on_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._btn_stop.setEnabled(False)
        append_colored(self._log, "\n⏳ 배치 실험 중단 요청됨... (현재 분석 중인 오디오 파일까지만 완료 후 정지합니다)\n")

    def _on_done(self, report_str: str):
        if report_str:
            append_colored(self._log, f"\n\n{report_str}\n")
            ask_save_report(self, report_str, "Batch")
        else:
            append_colored(self._log, "\n✅ 새로 분석된 파일이 없어 리포트를 생성하지 않습니다.\n")

    def _on_finished(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tab 5: CSV 분석 리포트 및 그래프 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CsvReportWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            import csv
            import matplotlib.pyplot as plt
            plt.rc('font', family='AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic')
            plt.rcParams['axes.unicode_minus'] = False

            intervals = ["~20s", "20s~40s", "40s~60s", "1m~1m20s", "1m20s~"]
            stats = {inv: {"total": 0, "correct": 0} for inv in intervals}

            with open(self.csv_path, mode="r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    inv = row.get("길이구간", "알 수 없음").strip()
                    act = row.get("실제구분", "").strip()
                    pred = row.get("최종판정", "").strip()
                    if inv not in stats: stats[inv] = {"total": 0, "correct": 0}
                    stats[inv]["total"] += 1
                    if (act == "보이스피싱" and pred == "보이스피싱") or (act == "정상대화" and pred == "정상"):
                        stats[inv]["correct"] += 1

            lines = ["═"*55, "📈  시간대 구간별 실효 정탐률 리포트 수립", "═"*55]
            x_labels, y_rates = [], []
            for inv in intervals:
                t = stats[inv]["total"]
                if t == 0: continue
                c = stats[inv]["correct"]
                r = c / t * 100
                lines.append(f"  {inv:<12}: 정탐 {c}/{t} 개 (정탐률: {r:.1f}%)")
                x_labels.append(inv)
                y_rates.append(r)

            self.log_signal.emit("\n".join(lines) + "\n")

            if x_labels:
                plt.figure(figsize=(8, 4.5))
                bars = plt.bar(x_labels, y_rates, color='#58a6ff', width=0.5)
                plt.title("보이스피싱 시스템 시간대별 구간 정탐률 흐름")
                plt.ylim(0, 110)
                for b in bars:
                    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center')
                g_path = os.path.join(os.path.dirname(self.csv_path), "experiment_graph_result.png")
                plt.tight_layout()
                plt.savefig(g_path, dpi=150)
                plt.close()
                self.log_signal.emit(f"📈 [시각화 그래프 세이브 완료]: {g_path}\n")
                os.system(f"open '{g_path}'" if sys.platform == 'darwin' else f"start {g_path}")

            self.done_signal.emit("✅ 종합 분석 완수.")
        except Exception as e: self.log_signal.emit(f"오류: {e}")


class CsvReportTab(QWidget):
    def __init__(self):
        super().__init__()
        self._csv_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        row = QHBoxLayout()
        self._lbl = QLabel("선택된 결과 로그 CSV 파일 없음")
        btn = make_btn("📊  실험 통계 CSV 파일 로드", "#21262d", "#30363d")
        btn.clicked.connect(self._pick)
        row.addWidget(btn)
        row.addWidget(self._lbl)
        layout.addLayout(row)

        self._btn_run = make_btn("📈  시각화 분석 그래프 렌더링", "#1f6feb")
        self._btn_run.clicked.connect(self._run)
        self._btn_run.setEnabled(False)
        layout.addWidget(self._btn_run)

        self._log = make_text_box()
        layout.addWidget(self._log)

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV 로드", "", "CSV 파일 (*.csv)")
        if p:
            self._csv_path = p
            self._lbl.setText(os.path.basename(p))
            self._btn_run.setEnabled(True)

    def _run(self):
        self._log.clear()
        self._worker = CsvReportWorker(self._csv_path)
        self._worker.log_signal.connect(lambda t: append_colored(self._log, t))
        self._worker.done_signal.connect(lambda s: append_colored(self._log, f"\n{s}\n"))
        self._worker.start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 윈도우 인터페이스 디자인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAIN_STYLE = """
QMainWindow, QWidget { background-color: #0d1117; color: #c9d1d9; }
QTabWidget::pane { border: 1px solid #30363d; border-radius: 0 6px 6px 6px; background: #0d1117; }
QTabBar::tab { background: #161b22; color: #8b949e; border: 1px solid #30363d; border-bottom: none; border-radius: 6px 6px 0 0; padding: 10px 22px; font-size: 10pt; font-weight: 600; margin-right: 2px; }
QTabBar::tab:selected { background: #0d1117; color: #58a6ff; border-color: #30363d; }
QTabBar::tab:hover:!selected { background: #1c2128; color: #c9d1d9; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️  보이스피싱 탐지 시스템  v2.5")
        self.resize(1120, 780)
        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)

        header_layout = QHBoxLayout()
        header = QLabel("🛡️  AI 기반 보이스피싱 탐지 멀티모달 프레임워크")
        header.setFont(QFont("Malgun Gothic", 14, QFont.Bold) if sys.platform == "win32" else QFont("Apple SD Gothic Neo", 14, QFont.Bold))
        header.setStyleSheet("color: #58a6ff; padding: 2px 0;")
        
        btn_exit = make_btn("❌  프로그램 강제 종료", "#da3633", "#f85149")
        btn_exit.clicked.connect(self.close)
        
        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(btn_exit)
        layout.addLayout(header_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #30363d; max-height: 1px; margin-bottom: 10px;")
        layout.addWidget(line)

        tabs = QTabWidget()
        tabs.addTab(FileAnalysisTab(), "📄  파일 분석")
        tabs.addTab(TextSimTab(),      "💬  텍스트 시뮬레이터")
        tabs.addTab(MicTab(),          "🎙️  실시간 마이크")
        tabs.addTab(BatchTab(),        "📊  배치 실험")
        tabs.addTab(CsvReportTab(),    "📈  CSV 분석 리포트")
        layout.addWidget(tabs)

        self.statusBar().setStyleSheet("background:#161b22; color:#6e7681; border-top:1px solid #30363d;")
        self.statusBar().showMessage("  시스템 정상 가동 중  |  캡스톤 디자인 종합 설계")

    def closeEvent(self, event):
        QApplication.quit()
        event.accept()


def main():
    app = QApplication(sys.argv)
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception: pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()