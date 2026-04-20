# AUDIO_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/금감원 보이스피싱 음성 1~6페이지/4차례 신고된 남성 전화금융사기범 음성 1.m4a"
AUDIO_PATH = '/Users/ewonjin/Downloads/무제.m4a'

MODEL_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_bert_model"
# MODEL_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kobert_model"
# MODEL_PATH = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_kluebert_binary_v2"

LLM_MODEL_PATH = "/Users/ewonjin/ggml-model-Q4_K_M.gguf"

LOG_FILE = "logs/phishing_report.txt"

# 4진
LABELS = {
    0: "일상",
    1: "전조",
    2: "위험",
    3: "보이스피싱"
}

# Binary
# LABELS = {
#     0: "정상", 
#     1: "보이스피싱"
# }

PHISHING_KEYWORDS = [
"검찰","검찰청","경찰","금감원","금융감독원",
"수사","사건","계좌","압류","이체","송금",
"OTP","보안카드","인증번호","대출"
]