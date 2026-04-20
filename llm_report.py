import gc
from llama_cpp import Llama
from config import LLM_MODEL_PATH

print("🧠 LLM 로딩 중...")
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=1024,
    n_gpu_layers=-1,
    verbose=False
)

def generate_report(transcript):

    prompt = f"""
당신은 보이스피싱 수사 전문가입니다.

반드시 한국어로 작성하세요.

다음 녹취를 분석하세요:

1. 대화 요약
2. 보이스피싱 여부
3. 판단 근거
4. 신고 리포트

녹취:
{transcript}
"""

    output = llm(prompt, max_tokens=512, temperature=0.1)
    return output['choices'][0]['text']