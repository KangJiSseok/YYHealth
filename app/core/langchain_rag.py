import logging
import os
from functools import lru_cache
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from app.schemas.chat import UserInfo

logger = logging.getLogger(__name__)


def _env() -> dict:
    load_dotenv()
    return {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "llm_endpoint": os.getenv("LLM_ENDPOINT"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    }


@lru_cache(maxsize=1)
def _embeddings():
    env = _env()
    if not env["llm_api_key"]:
        raise RuntimeError("LLM_API_KEY(또는 OPENAI_API_KEY)가 설정되지 않았습니다.")
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=env["llm_api_key"],
        base_url=env["llm_endpoint"],
    )


@lru_cache(maxsize=1)
def _vectorstore() -> Qdrant:
    env = _env()
    client = QdrantClient(url=env["qdrant_url"], api_key=env["qdrant_api_key"])
    return Qdrant(
        client=client,
        collection_name="nutrition_rag_chunks",
        embeddings=_embeddings(),
        content_payload_key="text",
    )


@lru_cache(maxsize=1)
def _llm():
    env = _env()
    if not env["llm_api_key"]:
        raise RuntimeError("LLM_API_KEY(또는 OPENAI_API_KEY)가 설정되지 않았습니다.")
    return ChatOpenAI(
        api_key=env["llm_api_key"],
        base_url=env["llm_endpoint"],
        model=env["llm_model"],
        temperature=0.2,
        max_tokens=320,
    )


def _build_query(question: str, diseases: List[str]) -> str:
    disease_text = f"질환: {', '.join(diseases)}" if diseases else "질환 정보 없음"
    return f"{question}\n{disease_text}"


def _format_user_profile(user_info: Optional[UserInfo], diseases: List[str]) -> str:
    parts: list[str] = []
    if diseases:
        parts.append("질환: " + ", ".join(diseases))
    if user_info:
        body: list[str] = []
        if user_info.height:
            body.append(f"키 {user_info.height}cm")
        if user_info.weight:
            body.append(f"몸무게 {user_info.weight}kg")
        if user_info.birthdate:
            body.append(f"생년월일 {user_info.birthdate}")
        if user_info.kcal:
            body.append(f"목표열량 {user_info.kcal}kcal")
        macro_parts: list[str] = []
        if user_info.protein:
            macro_parts.append(f"단백질 {user_info.protein}g")
        if user_info.protein_min or user_info.protein_max:
            macro_parts.append(
                f"단백질 범위 {user_info.protein_min or '?'}~{user_info.protein_max or '?'}g"
            )
        if user_info.fat:
            macro_parts.append(f"지방 {user_info.fat}g")
        if user_info.fat_min or user_info.fat_max:
            macro_parts.append(f"지방 범위 {user_info.fat_min or '?'}~{user_info.fat_max or '?'}g")
        if macro_parts:
            body.append(" / ".join(macro_parts))
        if body:
            parts.append(" ".join(body))
    return "; ".join(parts) if parts else "제공된 추가 정보 없음"


def _to_context(docs_with_scores: List[Tuple]) -> str:
    snippets = []
    for doc, score in docs_with_scores:
        meta = doc.metadata or {}
        prefix = meta.get("disease") or meta.get("chunk_type") or ""
        snippets.append(f"[{score:.3f} {prefix}] {doc.page_content}")
    return "\n\n".join(snippets)


def _to_evidence(docs_with_scores: List[Tuple]) -> list[dict]:
    evidences: list[dict] = []
    for doc, score in docs_with_scores:
        meta = doc.metadata or {}
        source = (
            meta.get("source")
            or meta.get("source_pdf")
            or meta.get("source_document")
            or meta.get("source_file")
        )
        page = meta.get("page") or meta.get("page_number") or meta.get("page_no")
        evidences.append(
            {
                "source": source,
                "page": page,
                "chunk_type": meta.get("chunk_type"),
                "disease_tags": meta.get("disease_tags"),
                "score": score,
                "text": doc.page_content,
                "snippet": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
            }
        )
    return evidences


def rag_chat_answer(
    question: str, diseases: List[str], user_info: Optional[UserInfo]
) -> Optional[Tuple[str, List[dict]]]:
    """
    LangChain 기반 RAG:
    - Qdrant에서 유사도 검색 (k=4, score_threshold 사용)
    - 히트가 없으면 None 반환하여 상위 로직이 직접 LLM 호출
    """
    try:
        vectorstore = _vectorstore()
        query = _build_query(question, diseases)

        # 1) 질환 태그 우선 필터 검색
        base_filter = None
        if diseases:
            base_filter = {"must": [{"key": "disease_tags", "match": {"any": diseases}}]}
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=6, filter=base_filter)
        filtered = [(doc, score) for doc, score in docs_with_scores if score is not None and score >= 0.45]

        # 2~4) chunk_type별 보강 검색 (guideline/numeric) + 일반 검색을 추가로 수행해 최대 4회 보강
        for ctype, k, thresh in [
            ("guideline_recommendation", 4, 0.40),
            ("numeric_evidence", 4, 0.38),
            (None, 4, 0.42),  # 필터 없는 일반 보강
        ]:
            qfilter = None
            if ctype:
                qfilter = {"must": [{"key": "chunk_type", "match": {"value": ctype}}]}
                if base_filter:
                    qfilter["must"].append({"key": "disease_tags", "match": {"any": diseases}})
            elif base_filter:
                qfilter = base_filter
            extra = vectorstore.similarity_search_with_score(query, k=k, filter=qfilter)
            filtered.extend([(d, s) for d, s in extra if s is not None and s >= thresh])

        # 동일 포인트(id) 중복 제거 (여러 검색 라운드에 걸쳐 반환될 수 있음)
        dedup: dict = {}
        for doc, score in filtered:
            doc_id = getattr(doc, "id", None) or getattr(doc.metadata, "get", lambda k, d=None: None)("id")
            key = doc_id or (doc.page_content[:50], tuple(sorted((doc.metadata or {}).items())))
            prev = dedup.get(key)
            if prev is None or (score or 0) > (prev[1] or 0):
                dedup[key] = (doc, score)
        filtered = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:6]
        if not filtered:
            return None

        profile = _format_user_profile(user_info, diseases)
        context = _to_context(filtered)
        evidence = _to_evidence(filtered)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "너는 영양/식단 안전 가이드를 한국어로 간단히 답하는 어시스턴트다.\n"
                    "- 주어진 근거(context)를 최우선으로 사용하고, 과장하거나 새로운 수치를 만들지 말 것\n"
                    "- 숫자/비율은 근거에 있을 때만 언급\n"
                    "- 답변은 2~3문장, 쉽고 명료하게\n"
                    "- 근거가 부족하면 추가 질문을 유도",
                ),
                (
                    "user",
                    "사용자 정보: {profile}\n"
                    "질문: {question}\n"
                    "검색된 근거:\n{context}\n"
                    "규칙:\n"
                    "- 근거에 나온 숫자/비율만 사용, 새로운 숫자는 만들지 말 것\n"
                    "- 근거가 부족하면 추가 질문을 유도\n"
                    "위 근거를 토대로 짧게 답변하세요.",
                ),
            ]
        )

        chain = prompt | _llm() | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context, "profile": profile})
        return answer, evidence
    except Exception as e:  # pragma: no cover - 방어적 로깅
        logger.warning("LangChain RAG chat failed: %s", e)
        return None
