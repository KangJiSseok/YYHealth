from fastapi import FastAPI
import httpx
import os
import re
from collections import defaultdict

from app.core.calculator import calculate_nutrition
from app.schemas.nutrition import NutritionResult
from app.schemas.evidence import NutritionWithEvidence
from app.core.disease_mapping import normalize_diseases
from app.core.rag_search import retrieve_evidence
from app.schemas.survey import SurveyRequest

app = FastAPI()


@app.post("/nutrition", response_model=NutritionResult)
def compute_nutrition(survey: SurveyRequest) -> NutritionResult:
    return calculate_nutrition(survey)


@app.post("/nutrition_with_evidence", response_model=NutritionWithEvidence)
def compute_nutrition_with_evidence(survey: SurveyRequest) -> NutritionWithEvidence:
    calc = calculate_nutrition(survey)
    normalized = normalize_diseases(survey.diseases)
    evidence = retrieve_evidence(normalized)
    rationale = _build_rationale(calc, normalized)
    summaries = _summarize_evidence(evidence)
    return NutritionWithEvidence(calculation=calc.model_dump(), evidence=evidence, rationale=rationale, summaries=summaries)


def _build_rationale(calc: NutritionResult, diseases: list[str]) -> list[str]:
    reasons: list[str] = []
    disease_set = set(diseases)

    if "dyslipidemia" in disease_set or "ascvd" in disease_set:
        reasons.append("이상지질혈증/ASCVD: 근거 기반으로 총지방 상한을 25%로 조정(포화지방은 7% 미만 권고).")

    if not reasons:
        reasons.append("명시된 질환별 비율 조정 근거가 없어 기본 AMDR 비율을 유지했습니다.")
    return reasons


def _summarize_evidence(evidence: list) -> dict[str, str]:
    """
    Summarize per disease using LLM when available; fallback otherwise.
    """
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    grouped = defaultdict(list)
    for ev in evidence:
        disease = None
        text = None
        score = None
        if isinstance(ev, dict):
            disease = ev.get("disease")
            text = ev.get("text")
            score = ev.get("score")
        else:
            disease = getattr(ev, "disease", None)
            text = getattr(ev, "text", None)
            score = getattr(ev, "score", None)
        grouped[disease or "general"].append({"text": text, "score": score})

    summaries: dict[str, str] = {}
    for disease, items in grouped.items():
        top_items = sorted(items, key=lambda x: x.get("score") or 0.0, reverse=True)[:1]
        snippets = []
        for ev in top_items:
            if ev.get("text"):
                snippets.append(_clean_snippet(ev["text"][:300]))
        if not snippets:
            continue
        if not api_key:
            summaries[disease] = _fallback_summary(disease)
            continue

        prompt = (
            f"질환: {disease}\n"
            "아래 근거를 참고해, 해당 질환이 있는 사람이 탄수화물/단백질/지방 섭취 시 주의할 점과 권장 행동을 한국어 2문장으로 설명해 주세요.\n"
            "요약 규칙:\n"
            "- 쉬운 한국어, 전문용어 최소화\n"
            "- 숫자·비율은 원문 그대로 유지, 과장 금지\n"
            "- 원문을 복사하지 말고 핵심만 재구성\n"
            "- 근거가 부족하면 '요약할 근거가 부족합니다.'만 출력\n"
            "근거:\n"
            + "\n".join([f"- {s}" for s in snippets])
        )

        try:
            resp = httpx.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "너는 의학 근거를 간결하고 안전하게 요약하는 한국어 어시스턴트다."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 160,
                },
                timeout=1.5,
            )
            data = resp.json()
            summary = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not summary or "요약할 근거가 부족합니다" in summary:
                summaries[disease] = _fallback_summary(disease)
            else:
                summaries[disease] = summary
        except Exception:
            summaries[disease] = _fallback_summary(disease)
    return summaries


def _clean_snippet(text: str) -> str:
    no_newline = text.replace("\n", " ")
    no_hyphen = re.sub(r"-\s+", "", no_newline)
    collapsed = re.sub(r"\s+", " ", no_hyphen)
    return collapsed.strip()


def _fallback_summary(disease: str) -> str:
    """
    Fallback 요약: 질환별 기본 주의사항을 한글로 제공.
    """
    disease_lower = disease.lower()
    base = {
        "type2_diabetes": "혈당 관리를 위해 정제 탄수화물을 줄이고 식이섬유가 풍부한 복합탄수화물로 교체하세요. 단백질은 적정량(권고 범위 내)으로 유지하고, 포화·트랜스지방은 제한하세요. 식사량은 접시/그릇 크기로 눈금화해 과식을 피하고, 체중 5~10% 감량이 도움이 됩니다.",
        "prediabetes": "혈당 상승을 막기 위해 정제 탄수화물을 줄이고 섬유질이 많은 식사를 하세요. 단백질과 건강한 지방을 적정량 포함해 식후 혈당 급등을 막고, 체중 관리와 규칙적 활동을 병행하세요.",
        "obesity": "체중 감량을 위해 총 에너지를 줄이고 단백질은 부족하지 않게, 포화지방을 줄이며 정제 탄수화물 대신 통곡·채소를 늘리세요. 과음/과식을 피하고 균형 식단을 유지하면 체중 5~10% 감량에 도움이 됩니다.",
        "ascvd": "포화·트랜스지방을 줄이고 건강한 지방으로 대체하세요. 탄수화물은 50~60% 범위에서 섬유질 위주로, 단백질은 15~20%로 적정 섭취하며 콜레스테롤·당분을 줄이세요.",
        "dyslipidemia": "LDL/TG 개선을 위해 포화·트랜스지방을 줄이고 식물성 불포화지방을 늘리세요. 정제 탄수화물을 제한하고 식이섬유가 많은 식품을 선택해 중성지방 상승을 막으세요.",
        "metabolic_syndrome": "복부비만·TG·혈당을 낮추기 위해 정제 탄수화물과 포화지방을 줄이고, 채소·통곡·살코기 위주의 균형 식단을 유지하세요. 과도한 칼로리 섭취와 음주를 줄이세요.",
    }
    note = base.get(disease_lower, "탄수화물은 섬유질 위주로, 포화지방을 줄이고 단백질은 권장 범위에서 적정 섭취하세요.")
    return note
