import numpy as np

class RiskAccumulator:
    def __init__(self, decay_rate=0.9):
        self.cumulative_score = 0.0
        self.decay_rate = decay_rate
    
    def update(self, current_risk, label):
        # 1. 기본 감쇄
        self.cumulative_score *= self.decay_rate
    
        # 2. 라벨에 따른 차등 적용
        if label == "보이스피싱" or label == "위험":
            # 위험할 때는 가중치 부여하여 합산
            weight = 1.5 if current_risk > 0.7 else 1.0
            self.cumulative_score += (current_risk * weight)
        else:
            # 핵심: 정상/일상 판정 시 점수를 오히려 감점 (-0.5 등)
            # 이렇게 하면 일상 대화가 길어질수록 점수가 0에 수렴합니다.
            self.cumulative_score -= (1.0 - current_risk) * 0.5 
        
        # 3. 하한선(0)과 상한선(5) 제한
        self.cumulative_score = max(0.0, min(self.cumulative_score, 5.0))
        return self.cumulative_score

# 기존 calculate_risk 함수는 그대로 유지하거나 가중치만 조정하세요.
def calculate_risk(label, confidence, k_score):
    risk = confidence + (k_score * 0.15)
    if label == "보이스피싱" and k_score == 0:
        risk *= 0.4
    return min(risk, 1.0)

# 기존 함수는 파일 분석용으로 유지
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