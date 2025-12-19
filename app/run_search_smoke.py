import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent


def embed_query(query: str) -> list[float]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url=api_base)
    resp = client.embeddings.create(model="text-embedding-3-large", input=query)
    return resp.data[0].embedding


def search(query: str, limit: int = 5, filters: rest.Filter | None = None):
    load_dotenv(BASE_DIR / ".env")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    queries = [
        ("guideline", "recommended carbohydrate percentage in type 2 diabetes"),
        ("evidence", "evidence for low vs high carbohydrate diet in type 2 diabetes"),
    ]
    all_points = []
    for qtype, qtext in queries:
        vector = embed_query(qtext)
        qfilter = None
        if qtype == "guideline":
            qfilter = rest.Filter(
                must=[rest.FieldCondition(key="chunk_type", match=rest.MatchValue(value="guideline_recommendation"))]
            )
        elif qtype == "evidence":
            qfilter = rest.Filter(
                must=[rest.FieldCondition(key="chunk_type", match=rest.MatchValue(value="numeric_evidence"))]
            )
        pts = client.search(
            collection_name="nutrition_rag_chunks",
            query_vector=vector,
            query_filter=qfilter or filters,
            limit=limit,
            with_payload=True,
        )
        all_points.extend(pts)

    reranked = _rerank(all_points, query)[:limit]
    print(f"[search] Query: {query} (combined {len(all_points)}; showing {limit})")
    for i, point in enumerate(reranked):
        payload = point.payload or {}
        print("-" * 60)
        print(f"Rank {i+1} | score={point.score:.4f}")
        print(f"source: {payload.get('source_pdf')} page: {payload.get('page_number')} type: {payload.get('chunk_type')}")
        print(f"diseases: {payload.get('disease_tags')} nutrients: {payload.get('nutrient_focus')} population: {payload.get('population')}")
        print(payload.get("text", "")[:500])


def _rerank(points, query: str):
    query_lower = query.lower()
    prefer_numeric = any(k in query_lower for k in ["distribution", "%", "proportion", "range", "intake", "mean"])
    prefer_guideline = any(k in query_lower for k in ["recommend", "guideline", "range", "should"])

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
        if "table" in text:
            s += 0.02
        return s

    return sorted(points, key=score, reverse=True)


if __name__ == "__main__":
    # Example: simple keyword search
    search("type 2 diabetes carbohydrate distribution", limit=5)
