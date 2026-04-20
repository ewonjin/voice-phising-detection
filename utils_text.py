# utils_text.py
import re

def split_sentences(text):
    # 1. 영어 문장과 한국어 문장이 섞인 경우를 위해 줄바꿈이나 기호 처리
    text = text.replace("?", ".").replace("!", ".").replace("\n", ". ")
    
    # 2. 마침표가 없어도 '다, 요, 죠' 등으로 끝나고 공백이 있으면 문장으로 간주
    text = re.sub(r'([가-힣]{2,}(다|요|죠|나|까|어|아))(\s+)', r'\1. ', text)
    
    # 3. 영어의 경우 단어 10개 단위 혹은 특정 길이마다 강제로 마침표 삽입 (선택 사항)
    # 여기서는 단순 분리만 수행
    sentences = text.split(".")
    
    # 4. 빈 문장 제거 및 최소 길이 필터링
    result = [s.strip() for s in sentences if len(s.strip()) > 5]
    return result