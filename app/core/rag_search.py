import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


DEFAULT_DISEASE_QUERIES = {
    "type2_diabetes": {
        "guideline": "당뇨병 환자의 탄수화물/단백질/지방 섭취 시 주의사항과 권장 비율(한국 지침, 조절 목표, 저탄수/고탄수 안전성)",
        "evidence": "type 2 diabetes macronutrient safety and precautions (hypoglycemia risk, carb quality, fat quality)",
    },
    "prediabetes": {
        "guideline": "당뇨 전단계에서 혈당 상승을 막기 위한 탄수화물/단백질/지방 섭취 주의사항과 한국 권고",
        "evidence": "prediabetes macronutrient precautions and outcomes (glycemic control, weight, fat quality)",
    },
    "obesity": {
        "guideline": "비만/체중감량 시 에너지 적자와 탄수화물/단백질/지방 섭취 시 주의사항, 금기, 권장 비율",
        "evidence": "obesity weight loss macronutrient precautions (protein adequacy, fat quality, carb quality)",
    },
    "metabolic_syndrome": {
        "guideline": "대사증후군 환자의 탄수화물/지방 섭취 주의사항과 권장 비율(트리글리세라이드, HDL 개선)",
        "evidence": "metabolic syndrome macronutrient precautions and outcomes (TG, HDL, insulin resistance)",
    },
    "ascvd": {
        "guideline": "심혈관질환/ASCVD 환자의 포화지방 제한, 탄수화물/단백질/지방 섭취 시 주의사항과 권고",
        "evidence": "cardiovascular disease macronutrient precautions (fat quality, carb quality, LDL impact)",
    },
    "dyslipidemia": {
        "guideline": "이상지질혈증 환자의 포화지방 제한과 탄수화물/단백질/지방 섭취 주의사항, 권장 비율",
        "evidence": "dyslipidemia macronutrient precautions (LDL/TG response, fat quality, refined carb caution)",
    },
}


def _get_env():
    load_dotenv()
    return {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_api_base": os.getenv("OPENAI_API_BASE"),
    }


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    env = _get_env()
    return QdrantClient(url_env=env["qdrant_url"], api_key=env["qdrant_api_key"])


def _embed(query: str) -> List[float]:
    env = _get_env()
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    if not env["openai_api_key"]:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=env["openai_api_key"], base_url=env["openai_api_base"])
    resp = client.embeddings.create(model="text-embedding-3-large", input=query)
    return resp.data[0].embedding


def _rerank(points, query: str):
    query_lower = query.lower()
    prefer_numeric = any(k in query_lower for k in ["distribution", "%", "proportion", "range", "intake", "mean"])
    prefer_guideline = any(k in query_lower for k in ["recommend", "guideline", "range", "should", "intake"])

    def score(p):
        s = p.score
        payload = p.payload or {}
        ctype = payload.get("chunk_type")
        text = (payload.get("text") or "").lower()

        priority = {
            "guideline_recommendation": 1.2,
            "recommendation_summary": 1.15,
            "numeric_evidence": 1.0,
            "explanation": 0.95,
            "reference": 0.7,
            None: 1.0,
        }
        s *= priority.get(ctype, 1.0)

        if prefer_numeric and ctype == "numeric_evidence":
            s += 0.05
        if prefer_guideline and ctype == "guideline_recommendation":
            s += 0.08
        if prefer_guideline and ctype == "numeric_evidence":
            s -= 0.05
        if "table" in text:
            s += 0.02
        return s

    return sorted(points, key=score, reverse=True)


def retrieve_general(query: str, limit: int = 3) -> List[dict]:
    env = _get_env()
    client = QdrantClient(url=env["qdrant_url"], api_key=env["qdrant_api_key"])
    vector = _embed(query)
    points = client.search(
        collection_name="nutrition_rag_chunks",
        query_vector=vector,
        limit=limit,
        with_payload=True,
    )
    reranked = _rerank(points, query)[:limit]
    evidence: List[dict] = []
    for p in reranked:
        payload = p.payload or {}
        evidence.append(
            {
                "disease": payload.get("disease"),
                "source_pdf": payload.get("source_pdf"),
                "page_number": payload.get("page_number"),
                "chunk_type": payload.get("chunk_type"),
                "text": payload.get("text"),
                "score": p.score,
            }
        )
    return evidence


def retrieve_evidence(diseases: List[str], limit_per_disease: int = 2) -> List[dict]:
    env = _get_env()
    client = QdrantClient(url=env["qdrant_url"], api_key=env["qdrant_api_key"])
    evidence: List[dict] = []
    for disease in diseases:
        queries = DEFAULT_DISEASE_QUERIES.get(
            disease,
            {
                "guideline": f"recommended carbohydrate percentage in {disease}",
                "evidence": f"evidence for macronutrient distribution in {disease}",
            },
        )
        for qtype, qtext in queries.items():
            vector = _embed(qtext)
            collected = []

            if qtype == "guideline":
                guideline_filter = rest.Filter(
                    must=[rest.FieldCondition(key="chunk_type", match=rest.MatchValue(value="guideline_recommendation"))]
                )
                collected.extend(
                    client.search(
                        collection_name="nutrition_rag_chunks",
                        query_vector=vector,
                        query_filter=guideline_filter,
                        limit=limit_per_disease,
                        with_payload=True,
                    )
                )
            else:
                numeric_filter = rest.Filter(
                    must=[rest.FieldCondition(key="chunk_type", match=rest.MatchValue(value="numeric_evidence"))]
                )
                collected.extend(
                    client.search(
                        collection_name="nutrition_rag_chunks",
                        query_vector=vector,
                        query_filter=numeric_filter,
                        limit=limit_per_disease,
                        with_payload=True,
                    )
                )

            if len(collected) < limit_per_disease:
                remaining = limit_per_disease - len(collected)
                collected.extend(
                    client.search(
                        collection_name="nutrition_rag_chunks",
                        query_vector=vector,
                        limit=remaining,
                        with_payload=True,
                    )
                )

            reranked = _rerank(collected, qtext)[:limit_per_disease]
            for p in reranked:
                payload = p.payload or {}
                evidence.append(
                    {
                        "disease": disease,
                        "source_pdf": payload.get("source_pdf"),
                        "page_number": payload.get("page_number"),
                        "chunk_type": payload.get("chunk_type"),
                        "text": payload.get("text"),
                        "score": p.score,
                    }
                )
    return evidence
