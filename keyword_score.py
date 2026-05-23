# keyword_score.py  ─ 키워드 분석 (정규식 유연 매칭 + 의도 기반 조합 규칙)
import re
from config import PHISHING_KEYWORDS

_JOSA = r"[을를에서의도로은는이가까지로부터에게]?"
_FLEX = {kw: re.compile(re.escape(kw) + _JOSA) for kw in PHISHING_KEYWORDS}

_INTENT_RULES = [
    {"name": "기관_사칭_금전요구",
     "cat_a": ["검찰","경찰","금감원","수사관","검사","금융위","법원"],
     "cat_b": ["계좌","이체","송금","안전계좌","보안카드","비밀","인출"],
     "weight": 0.4},
    {"name": "가족_사칭_긴급요청",
     "cat_a": ["엄마","아빠","아들","딸","아이","자녀","형","누나","오빠"],
     "cat_b": ["급해","빌려줘","보내줘","문화상품권","기프트카드","핀번호","임시폰"],
     "weight": 0.35},
    {"name": "원격제어_앱설치_유도",
     "cat_a": ["설치","다운로드","링크","앱",".apk","클릭"],
     "cat_b": ["원격","팀뷰어","퀵서포트","화면공유","접근권한","제어"],
     "weight": 0.4},
    {"name": "정부지원금_사칭",
     "cat_a": ["지원금","환급","보조금","복지","혜택","당첨","선정"],
     "cat_b": ["계좌","신청","링크","확인","클릭","입력","등록"],
     "weight": 0.3},
    {"name": "비밀유지_압박",
     "cat_a": ["비밀","누설","가족한테","아무한테도","절대","조용히"],
     "cat_b": ["수사","계좌","이체","도와줘","처리"],
     "weight": 0.35},
]

_EUPH = [
    (re.compile(r"잠깐.{0,5}(알려|보내|이체|송금)"),              "완곡_금전요청"),
    (re.compile(r"(지금|바로|빨리).{0,10}(계좌|이체|송금|입금)"), "긴급송금유도"),
    (re.compile(r"(저희|당사|기관).{0,5}(안내|도움|처리)"),       "기관사칭어조"),
]

def keyword_score(text: str) -> tuple[int, list[str], bool]:
    """
    Returns:
        score            : 탐지 신호 총 개수
        detected         : 탐지된 키워드/패턴 목록
        has_combined_risk: 조합 규칙 발동 여부
    """
    score, detected = 0, []
    has_combined = False

    for kw, pat in _FLEX.items():
        if pat.search(text):
            score += 1; detected.append(kw)

    combo_bonus = 0.0
    for rule in _INTENT_RULES:
        if any(w in text for w in rule["cat_a"]) and any(w in text for w in rule["cat_b"]):
            has_combined = True
            combo_bonus += rule["weight"]
            detected.append(f"[조합:{rule['name']}]")
    score += int(combo_bonus / 0.15)

    for pat, label in _EUPH:
        if pat.search(text):
            score += 1; detected.append(f"[완곡:{label}]")

    return score, detected, has_combined
