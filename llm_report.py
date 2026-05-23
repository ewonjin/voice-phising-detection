# llm_report_v2.py  ─ LLM 리포트 생성 (Lazy Loading + 수사 보고서 형식)
#
# 출력 형식 예시:
#
# ╔══════════════════════════════════════════════════════════╗
# ║         🚨  보이스피싱 종합 수사 분석 보고서  🚨          ║
# ╚══════════════════════════════════════════════════════════╝
#  [분석 지표]
#  · 최종 위험 점수  : 4.12 / 5.00
#  · AI 생성 음성    : 확률 87.3%  →  AI 생성 음성
#  ...
# ══════════════════════════════════════════════════════════

from config import LLM_MODEL_PATH

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        print("[LLM] 모델 로딩 중 (최초 리포트 요청)...")
        _llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048,
                     n_gpu_layers=-1, verbose=False)
        print("[LLM] 로드 완료.")
    return _llm


def _report_header(score: float) -> str:
    """수사 보고서 헤더 박스."""
    if score >= 3.5:
        title = "🚨  보이스피싱 종합 수사 분석 보고서  🚨"
        level = "【위험】 보이스피싱 고위험군으로 판단됩니다."
    else:
        title = "⚠️   보이스피싱 주의 단계 분석 요약   ⚠️"
        level = "【주의】 보이스피싱 의심 정황이 포착되었습니다."

    w = 60
    border = "═" * w
    pad    = (w - len(title)) // 2
    return (
        f"\n╔{border}╗\n"
        f"║{' '*pad}{title}{' '*(w-pad-len(title))}║\n"
        f"╚{border}╝\n"
        f"{level}\n"
        f"{border}\n"
    )


def generate_report(
    transcript:       str,
    bert_score:       float,
    detected_details: str,
    ai_prob:          float | None = None,   # VoiceAnalyzer.finalize_ai_verdict()
    ai_label:         str   | None = None,
) -> str:
    """
    누적 위험도 기반 조건부 리포트 생성.

    Args:
        transcript      : 전체 대화 원문
        bert_score      : RiskAccumulator.cumulative_score (0~5)
        detected_details: 탐지 키워드 문자열
        ai_prob         : AI 음성 확률 (리포트 시점 finalize_ai_verdict() 결과)
        ai_label        : "AI 생성 음성" | "실제 인간 음성"

    Returns:
        리포트 문자열 (일상 단계 < 2.0 은 "" 반환)

    ★ AI 음성 확률은 반드시 이 함수 안에서만 최종 출력됨.
       실시간 화면에서는 절대 표시하지 않음.
    """
    if bert_score < 2.0:
        return ""   # 일상 단계: 리포트 미생성, AI 판독도 미표시

    if bert_score >= 3.5:
        instruction = """\
[작성 지침: 상세 리포트 모드]
1. 사건 요약: 대화 내용을 구체적으로 요약하세요.
2. 보이스피싱 여부: '위험'으로 명시하세요.
3. 판단 근거: 탐지 키워드와 수치를 제시하세요.
4. 대응 지침: 즉각 행동 요령과 신고 절차를 포함하세요.
   (금융감독원 1332, 경찰청 112)"""
        verdict = "위험 (보이스피싱 고위험군)"
    else:
        instruction = """\
[작성 지침: 간략 요약 모드]
1. 대화 핵심 요약: 2~3줄로 작성하세요.
2. 주의 이유: 탐지 키워드를 근거로 설명하세요."""
        verdict = "주의 (의심 정황 포착)"

    # AI 음성 판정 문구 (리포트 내부에만 포함)
    if ai_prob is not None:
        ai_section = (
            f"· AI 생성 음성 판정 : 확률 {ai_prob*100:.1f}%  →  {ai_label}\n"
            f"  ※ 전체 통화 구간의 음성을 종합 분석한 결과입니다.\n"
        )
    else:
        ai_section = "· AI 생성 음성 판정 : 데이터 없음 (음성 분석 미실시)\n"

    border  = "═" * 60
    prompt  = f"""\
당신은 보이스피싱 수사 전문가입니다.
아래 분석 지표와 대화 원문을 바탕으로 리포트를 작성하세요.

[분석 지표]
· 최종 위험 점수  : {bert_score:.2f} / 5.00
· 시스템 판단     : {verdict}
· 탐지 내용       : {detected_details}
{ai_section}
[대화 원문]
{transcript}

{instruction}
"""
    llm    = _get_llm()
    output = llm(prompt, max_tokens=1024, temperature=0.1,
                 stop=["[대화 원문]", "###"])
    body   = output["choices"][0]["text"].strip()

    # ── 수사 보고서 형식으로 래핑 ──────────────────────────────────
    header  = _report_header(bert_score)
    metrics = (
        f"[분석 지표 요약]\n"
        f"  · 최종 위험 점수  : {bert_score:.2f} / 5.00\n"
        f"  · 탐지 키워드     : {detected_details or '없음'}\n"
        f"  · AI 음성 확률    : {f'{ai_prob*100:.1f}%  →  {ai_label}' if ai_prob is not None else '판정 불가'}\n"
        f"{border}\n"
    )
    if ai_prob is not None:
        conclusion = (
            f"\n{border}\n"
            f"[결론]\n"
            f"  본 음성은 분석 결과 AI 생성 음성일 확률이 "
            f"{ai_prob*100:.1f}% 로 판단됩니다.\n"
            f"  종합 피싱 위험 점수: {bert_score:.2f}/5.00\n"
            f"{'  🚨 즉시 경찰(112) 또는 금융감독원(1332)에 신고하세요!' if bert_score >= 3.5 else ''}\n"
            f"{border}\n"
        )
    else:
        conclusion = ""

    return header + metrics + body + conclusion
