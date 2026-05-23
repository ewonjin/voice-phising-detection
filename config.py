# config.py  ─ 전역 설정
# 실행 전 각 경로를 본인 환경에 맞게 수정하세요.

# ── 파일 경로 ─────────────────────────────────────────────────────────
# AUDIO_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/fishingmp3/[SHANA]3차-(9).4차례 신고된 남성 전화금융사기범 음성 1 - 3of3.mp3"
# AUDIO_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/음성 분리/4차례 신고된 남성 전화금융사기범 음성 2-2.m4a"
# AUDIO_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/음성 분리/4차례 신고된 여성 전화금융사기범-3.m4a"
# AUDIO_PATH = '/Users/ewonjin/Downloads/무제.m4a'
# AUDIO_PATH = '/Users/ewonjin/Downloads/통화 녹음 0317557504_260406_131945.m4a'
# AUDIO_PATH = '/Users/ewonjin/Downloads/call/통화 녹음 ㅈㅍ_260514_222159.m4a'
# AUDIO_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/음성 분리/6번_3차례 신고된 남성 전화금융사기범 (음성_6)-2.m4a"
AUDIO_PATH = "/Users/ewonjin/Downloads/New Projectver2.wav"

# MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v2_1"
# MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v3(합성2500)"
# MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v4(합성400)"
# MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v5(합성100)"
# MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v6(각300)"
MODEL_PATH         = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v7(600,800)"

EMOTION_MODEL_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/가중치/DenseNet121_Data Augmentation.pth"
AI_DETECTOR_PATH   = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/가중치/AI_Voice_Detector_Korean_v3.pth"
LLM_MODEL_PATH     = "/Users/ewonjin/ggml-model-Q4_K_M.gguf"
LOG_FILE           = "logs/phishing_report.txt"

# ── BERT 레이블 ───────────────────────────────────────────────────────
LABELS = {0: "정상", 1: "보이스피싱"}

# ── 피싱 탐지 키워드 ──────────────────────────────────────────────────
PHISHING_KEYWORDS = [
    "검찰", "검찰청", "경찰", "금감원", "금융감독원", "수사관", "검사",
    "수사", "사건", "계좌", "압류", "이체", "송금", "명의도용", "연루",
    "OTP", "보안카드", "인증번호", "대출", "저금리", "대환",
    "안전계좌", "자금세탁", "불법자금", "국가환수", "비밀유지",
    "원격제어", "팀뷰어", "퀵서포트", "애플리케이션", "설치", "링크",
    "액정", "수리", "임시폰", "문화상품권", "문상", "핀번호", "신분증",
    "소상공인", "지원금", "현금 인출", "직접 전달", "조용한 곳", "누설",
    ".apk", "가로채기", "상환용", "보증보험",
    "주민번호", "상품권", "기프트카드", "응모", "당첨",
]

# ── 음성 분석기 설정 ──────────────────────────────────────────────────
CALIBRATION_SEC   = 5.0
HISTORY_LEN       = 5
SAMPLE_RATE       = 16000
MIN_CHUNK_SAMPLES = 1600

# ── 위험도 임계값 ─────────────────────────────────────────────────────
SCORE_WARN        = 2.0    # 주의 단계
SCORE_DANGER      = 3.5    # 강력 경고 단계
DECAY_RATE_RT     = 0.85   # 실시간 모드 감쇄율
DECAY_RATE_FILE   = 0.90   # 파일 모드 감쇄율
