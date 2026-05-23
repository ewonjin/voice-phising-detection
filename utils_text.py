# utils_text.py  ─ STT 오인식 정규화 + 문장 분리
import re

# Whisper 오인식 패턴 → 올바른 표현
_STT_REGEX = {
    r"검찰[처쳥칭]":                "검찰청",
    r"금\s*감\s*원":               "금감원",
    r"금융\s*감독\s*원":           "금융감독원",
    r"경\s*찰\s*청":               "경찰청",
    r"오\s*티\s*피|o\.t\.p":       "OTP",
    r"보안\s*카\s*드":             "보안카드",
    r"인\s*증\s*번\s*호":         "인증번호",
    r"계\s*좌\s*번\s*호":         "계좌번호",
    r"팀\s*뷰\s*어|team\s*viewer": "팀뷰어",
    r"퀵\s*서\s*포\s*트|quick\s*support": "퀵서포트",
    r"에이\s*피\s*케이|apk\s*파일": ".apk",
}
_PHONETIC = {
    "검찰처": "검찰청", "금감완": "금감원",
    "수삿관": "수사관", "안전게좌": "안전계좌",
    "이채": "이체",     "계좌번": "계좌번호",
}

def normalize_stt(text: str) -> str:
    """Whisper STT 오인식 정규화."""
    if not text:
        return text
    for pat, rep in _STT_REGEX.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    for wrong, right in _PHONETIC.items():
        text = text.replace(wrong, right)
    return re.sub(r"\s{2,}", " ", text).strip()

def split_sentences(text: str) -> list[str]:
    """텍스트를 의미 단위 문장으로 분리."""
    text = normalize_stt(text)
    text = text.replace("?", ".").replace("!", ".").replace("\n", ". ")
    text = re.sub(r"([가-힣]{2,}(다|요|죠|나|까|어|아))(\s+)", r"\1. ", text)
    result, buf = [], ""
    for s in [s.strip() for s in text.split(".") if s.strip()]:
        buf = (buf + " " + s).strip() if buf else s
        if len(buf) >= 10:
            result.append(buf); buf = ""
    if buf:
        if result: result[-1] += " " + buf
        else: result.append(buf)
    return result
