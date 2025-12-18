from fastapi import FastAPI
import httpx
import os
import re
from collections import defaultdict
from dotenv import load_dotenv
import logging
import json

from app.core.calculator import calculate_nutrition
from app.schemas.nutrition import NutritionResult
from app.schemas.evidence import NutritionWithEvidence
from app.core.disease_mapping import normalize_diseases
from app.core.rag_search import retrieve_evidence, retrieve_general
from app.schemas.survey import SurveyRequest
from app.schemas.chat import ChatRequest, ChatResponse

app = FastAPI()
# 로컬 .env 자동 로드 (uvicorn 환경변수 미설정 시 대비)
load_dotenv()
logger = logging.getLogger(__name__)


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
    calc_summaries = _summarize_calculation(calc, evidence, normalized)
    return NutritionWithEvidence(
        calculation=calc.model_dump(),
        evidence=evidence,
        rationale=rationale,
        summaries=summaries,
        calc_summaries=calc_summaries,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(query: ChatRequest) -> ChatResponse:
    diseases = normalize_diseases(query.diseases or [])
    answer = _chat_answer_direct(query.message, diseases)
    return ChatResponse(answer=answer, evidence=[], conversation_id=query.conversation_id)


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


def _chat_answer_direct(question: str, diseases: list[str]) -> str:
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    # 질환 리스트 질의 처리
    if "질환" in question and ("무엇" in question or "뭐" in question or "가지고" in question):
        if diseases:
            return "등록된 질환: " + ", ".join(diseases) + ". 각 질환에 맞춰 포화지방을 줄이고, 정제 탄수화물 대신 섬유질이 많은 식품을 선택하며, 단백질은 권장 범위에서 적정 섭취하세요."
        return "회원 정보에 질환이 등록되어 있지 않습니다. 가입 시 질환을 추가하면 더 맞춤형 안내가 가능합니다."

    if api_key:
        try:
            prompt = (
                "너는 영양/식단 상담을 안전하게 돕는 한국어 어시스턴트다.\n"
                "아래 사용자의 질문에 대해, 과장 없이 2~3문장으로 답하세요.\n"
                "- 쉬운 한국어 사용\n"
                "- 숫자/비율은 근거가 있을 때만 명시, 없으면 일반적 권장 범위 언급\n"
                "- 포화지방/트랜스지방 제한, 섬유질 많은 탄수화물, 적정 단백질 섭취를 기본 가이드로 삼으세요.\n"
                f"질문: {question}\n"
            )
            resp = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "너는 영양/식단 상담을 안전하게 돕는 한국어 어시스턴트다."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 200,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if answer:
                return answer
        except Exception:
            pass

    # fallback: LLM 미사용 시 질문을 반영한 기본 안내
    base = "포화·트랜스지방을 줄이고, 섬유질이 많은 탄수화물과 충분한 단백질을 균형 있게 섭취하세요."
    return f"질문을 확인했어요: \"{question}\". {base} 더 구체적인 목표나 식사 정보를 주시면 맞춤 조언을 드릴 수 있어요."


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
    return base.get(
        disease_lower,
        "지금 가진 정보만으로는 딱 맞는 답을 찾기 어렵습니다. 식사 기록과 목표(체중, 질환, 활동량)를 더 알려주시면 구체적인 가이드를 드릴게요.",
    )


def _summarize_calculation(calc: NutritionResult, evidence: list, diseases: list[str]) -> dict[str, str]:
    """
    Explain why kcal/탄/단/지 값이 그렇게 나왔는지 요약.
    LLM이 없으면 숫자 기반의 간단한 한글 설명을 반환.
    """
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    def _as_dict(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj.__dict__ if hasattr(obj, "__dict__") else {}

    mac = {k: _as_dict(v) for k, v in calc.macronutrients.items()}
    targets = {
        "energy": {"kcal": float(calc.adjusted_eer or calc.eer)},
        "carbohydrate": mac.get("carbohydrate", {}),
        "protein": mac.get("protein", {}),
        "fat": mac.get("fat", {}),
    }

    def format_macro(name, obj):
        if not obj:
            return "-"
        tgt = obj.get("recommended_g")
        mn = obj.get("min_g")
        mx = obj.get("max_g")
        if tgt is None or mn is None or mx is None:
            return "-"
        return f"{tgt:.1f}g (범위 {mn:.1f}~{mx:.1f}g)"

    formula_desc = _formula_text(calc.formula_name)

    if not api_key:
        return {
            "energy": f"EER 공식({calc.formula_name}): {formula_desc} / 활동지수 PA {calc.pa:.2f} 적용 → {targets['energy']['kcal']:.0f} kcal",
            "carbohydrate": f"AMDR 55~65% 적용: {format_macro('carbohydrate', targets['carbohydrate'])}",
            "protein": f"AMDR 7~20% 적용: {format_macro('protein', targets['protein'])}",
            "fat": f"AMDR 15~30% 적용: {format_macro('fat', targets['fat'])}",
        }

    top_evi = []
    for ev in sorted(evidence, key=lambda x: getattr(x, "score", None) or 0.0, reverse=True)[:3]:
        text = getattr(ev, "text", None) or (ev.get("text") if isinstance(ev, dict) else None)
        if text:
            top_evi.append(_clean_snippet(text)[:300])

    prompt = (
        "다음 개인화 영양 계산 결과가 왜 이렇게 나왔는지 한국어로 2~3문장씩 설명하세요.\n"
        "질병/조건: " + (", ".join(diseases) if diseases else "특이사항 없음") + "\n"
        f"권장 칼로리: {targets['energy']['kcal']:.0f} kcal (EER 공식 {calc.formula_name}: {formula_desc}, PA {calc.pa:.2f})\n"
        f"탄수화물: {format_macro('carbohydrate', targets['carbohydrate'])}\n"
        f"단백질: {format_macro('protein', targets['protein'])}\n"
        f"지방: {format_macro('fat', targets['fat'])}\n"
        "출력은 JSON 형태로, 키는 energy, carbohydrate, protein, fat 입니다.\n"
        "규칙:\n"
        "- 쉬운 한국어로 요약, 숫자는 그대로 유지\n"
        "- 과장 금지, 부족한 근거는 언급하지 않음\n"
        "- 근거는 아래 텍스트를 참고\n"
        "근거:\n" + "\n".join(f"- {t}" for t in top_evi)
    )

    try:
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "너는 영양/식단 계산을 설명하는 한국어 어시스턴트다. JSON만 답한다."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {
                "energy": parsed.get("energy") or parsed.get("kcal") or "",
                "carbohydrate": parsed.get("carbohydrate") or "",
                "protein": parsed.get("protein") or "",
                "fat": parsed.get("fat") or "",
            }
    except Exception as e:
        logger.warning("LLM calc summary failed: %s", str(e))

    return {
        "energy": f"EER 공식({calc.formula_name}): {formula_desc} / 활동지수 PA {calc.pa:.2f} 적용 → {targets['energy']['kcal']:.0f} kcal",
        "carbohydrate": f"AMDR 55~65% 적용: {format_macro('carbohydrate', targets['carbohydrate'])}",
        "protein": f"AMDR 7~20% 적용: {format_macro('protein', targets['protein'])}",
        "fat": f"AMDR 15~30% 적용: {format_macro('fat', targets['fat'])}",
    }


def _formula_text(formula_name: str) -> str:
    """
    Return human-readable EER formula.
    """
    mapping = {
        # 성인 남성/여성 WHO/FAO/UNU 2004 style
        "male_20_plus": "EER = 662 - 9.53*age + PA*(15.91*weight_kg + 539.6*height_m)",
        "female_20_plus": "EER = 354 - 6.91*age + PA*(9.36*weight_kg + 726*height_m)",
        "male_3_18": "EER = 88.5 - 61.9*age + PA*(26.7*weight_kg + 903*height_m) + 20",
        "female_3_18": "EER = 135.3 - 30.8*age + PA*(10*weight_kg + 934*height_m) + 20",
        # fallback
    }
    return mapping.get(formula_name, "EER = basemetabolic + PA * (체중·키·연령 계수)")
