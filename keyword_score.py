from config import PHISHING_KEYWORDS

def keyword_score(text):

    score = 0
    detected = []

    for k in PHISHING_KEYWORDS:
        if k in text:
            score += 1
            detected.append(k)

    return score, detected