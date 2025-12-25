import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.schemas.report import ReportAnalysisRequest, ReportAnalysisResponse, ReportDayStat
from app.core.langchain_rag import rag_chat_answer


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _macro_ratio(carb: float, protein: float, fat: float) -> tuple[float, float, float]:
    total = carb + protein + fat
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (carb / total * 100, protein / total * 100, fat / total * 100)


def _prep_stats(days: List[ReportDayStat]) -> dict:
    if not days:
        return {"kcal": 0.0, "carb": 0.0, "protein": 0.0, "fat": 0.0, "ratio": (0.0, 0.0, 0.0)}
    kcal = _avg([d.calories or 0 for d in days])
    carb = _avg([d.carbohydrate or 0 for d in days])
    protein = _avg([d.protein or 0 for d in days])
    fat = _avg([d.fat or 0 for d in days])
    ratio = _macro_ratio(carb, protein, fat)
    return {"kcal": kcal, "carb": carb, "protein": protein, "fat": fat, "ratio": ratio}


def _llm():
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("LLM_ENDPOINT")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    return ChatOpenAI(api_key=api_key, base_url=endpoint, model=model, temperature=0.2, max_tokens=300)


def _goal_text(payload: ReportAnalysisRequest) -> str:
    goal_parts = []
    if payload.kcal is not None:
        goal_parts.append(f"목표 칼로리 {payload.kcal:.0f} kcal")

    def _goal_line(name: str, target: float | None, gmin: float | None, gmax: float | None) -> str:
        vals = []
        if target is not None:
            vals.append(f"목표 {target:.1f}g")
        if gmin is not None or gmax is not None:
            vals.append(f"범위 {gmin if gmin is not None else '?'}~{gmax if gmax is not None else '?'}g")
        return f"{name} " + ", ".join(vals) if vals else ""

    goal_parts.extend(
        [
            _goal_line("탄수화물", payload.carbohydrate, payload.carbohydrateMin, payload.carbohydrateMax),
            _goal_line("단백질", payload.protein, payload.proteinMin, payload.proteinMax),
            _goal_line("지방", payload.fat, payload.fatMin, payload.fatMax),
        ]
    )
    return "; ".join([g for g in goal_parts if g])


def _format_context(stats: dict, label: str, payload: ReportAnalysisRequest) -> str:
    goal_text = _goal_text(payload)
    return (
        f"{label}: 평균 {stats['kcal']:.0f} kcal, "
        f"탄수화물 {stats['carb']:.1f}g, 단백질 {stats['protein']:.1f}g, 지방 {stats['fat']:.1f}g, "
        f"비율 {stats['ratio'][0]:.1f}/{stats['ratio'][1]:.1f}/{stats['ratio'][2]:.1f}%"
        + (f"\n목표/범위: {goal_text}" if goal_text else "")
    )


def analyze_report(payload: ReportAnalysisRequest) -> ReportAnalysisResponse:
    weekly_stats = _prep_stats(payload.weekly)
    monthly_stats = _prep_stats(payload.monthly)
    note_text = "\n".join(payload.notes) if payload.notes else ""
    note_clean = "".join(note_text.split())
    has_meaningful_notes = bool(note_clean) and len(note_clean) >= 6
    user_profile_parts = []
    if payload.diseases:
        user_profile_parts.append("질환: " + ", ".join(payload.diseases))
    goal_text = _goal_text(payload)
    if goal_text:
        user_profile_parts.append("목표/범위: " + goal_text)
    user_profile = "; ".join(user_profile_parts) if user_profile_parts else "추가 정보 없음"

    def disease_prompt(stats: dict, label: str) -> str:
        goals = goal_text if goal_text else "목표/범위 정보 없음"
        return (
            f"{label} 동안의 섭취 패턴입니다. "
            f"평균 에너지 {stats['kcal']:.0f} kcal, "
            f"탄수화물 {stats['carb']:.1f}g, 단백질 {stats['protein']:.1f}g, 지방 {stats['fat']:.1f}g "
            f"(비율 {stats['ratio'][0]:.1f}/{stats['ratio'][1]:.1f}/{stats['ratio'][2]:.1f}%). "
            f"사용자 정보: {user_profile}. 목표/범위: {goals}. 추가 메모: {note_text if note_text else '없음'}. "
            "이런 식사 패턴에서 발생하거나 악화될 수 있는 질병과 주의사항을 알려줘. "
            "근거에 없는 숫자는 만들지 말고, 질병명이 없으면 근거 부족이라고 말해."
        )

    weekly_rag: str | None = None
    monthly_rag: str | None = None
    weekly_ev: list[dict] = []
    monthly_ev: list[dict] = []
    if has_meaningful_notes:
        try:
            weekly_res = rag_chat_answer(disease_prompt(weekly_stats, "최근 7일"), [], None)
            monthly_res = rag_chat_answer(disease_prompt(monthly_stats, "최근 30일"), [], None)
            if weekly_res:
                weekly_rag, weekly_ev = weekly_res
            if monthly_res:
                monthly_rag, monthly_ev = monthly_res
        except Exception:
            pass

    llm = _llm()

    def llm_summary(stats: dict, label: str) -> str:
        if not llm:
            return _format_context(stats, label, payload)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너는 식단/영양 분석을 한국어로 짧게 요약하는 어시스턴트다. "
                    "제공된 숫자만 사용하고, 근거 없는 숫자는 만들지 않는다.",
                ),
                (
                    "user",
                    "{label} 섭취 통계를 요약하고, 위험할 수 있는 질병/주의사항을 5문장으로 알려줘.\n"
                    "{ctx}\n"
                    "사용자 정보: {profile}\n"
                    "추가 메모: {notes}\n"
                    "- 숫자/비율은 그대로 사용\n"
                    "- 질병 위험이 불명확하면 '질병 위험을 특정하기 어렵습니다.'라고 말해",
                ),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        try:
            return chain.invoke(
                {
                    "label": label,
                    "ctx": _format_context(stats, label, payload),
                    "notes": note_text or "없음",
                    "profile": user_profile,
                }
            )
        except Exception:
            return _format_context(stats, label, payload)

    if not has_meaningful_notes:
        overall = "추가 메모가 없어 질병 위험을 특정하기 어렵습니다. 건강검진 결과나 질환 이력을 알려주시면 더 정확히 분석할게요."
        weekly_txt = _format_context(weekly_stats, "최근 7일", payload)
        monthly_txt = _format_context(monthly_stats, "최근 30일", payload)
        return ReportAnalysisResponse(
            overall=overall,
            weekly=weekly_txt,
            monthly=monthly_txt,
            weeklyEvidence=[],
            monthlyEvidence=[],
        )

    weekly_txt = weekly_rag or llm_summary(weekly_stats, "최근 7일")
    monthly_txt = monthly_rag or llm_summary(monthly_stats, "최근 30일")
    overall = "최근 섭취 데이터와 제공된 메모를 바탕으로 분석했습니다."
    return ReportAnalysisResponse(
        overall=overall,
        weekly=weekly_txt,
        monthly=monthly_txt,
        weeklyEvidence=weekly_ev,
        monthlyEvidence=monthly_ev,
    )
