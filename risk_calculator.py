import numpy as np

def calculate_risk(label, confidence, k_score):

    risk = confidence + (k_score * 0.15)

    if label == "보이스피싱" and k_score == 0:
        risk *= 0.4

    return min(risk, 1.0)


def calculate_average_risk(sentences, results, keyword_fn):

    risks = []
    all_keywords = []

    print("\n--- 문장별 분석 ---")

    for i, s in enumerate(sentences):

        label, conf = results[i]

        k_score, keywords = keyword_fn(s)
        risk = calculate_risk(label, conf, k_score)

        risks.append(risk)
        all_keywords.extend(keywords)

    avg = np.mean(risks)
    mx = np.max(risks)

    # 🔥 핵심: 평균 + 최대값 혼합
    final_risk = (avg * 0.7) + (mx * 0.3)

    return final_risk, list(set(all_keywords))