import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.qdrant_setup import COLLECTION_NAME, EMBEDDING_DIM

try:
    from openai import OpenAI
    from openai import BadRequestError
except ImportError:
    OpenAI = None  # type: ignore
    BadRequestError = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

# Batching knobs
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def dedup_chunks(chunks: List[Dict]) -> List[Dict]:
    """텍스트 해시로 중복 청크를 제거한다."""
    seen = set()
    result: List[Dict] = []
    for c in chunks:
        text = c.get("text", "")
        norm = " ".join(text.split()).strip().lower()
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        result.append(c)
    if len(result) != len(chunks):
        print(f"[embed_upsert] Deduped {len(chunks)} -> {len(result)} chunks (by text hash)")
    return result


def _batch(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key, base_url=api_base)
    embeddings: List[List[float]] = []

    for start, batch_texts in _batch(texts, EMBED_BATCH_SIZE):
        # Filter out empty texts
        filtered = [(i, t) for i, t in enumerate(batch_texts) if t and t.strip()]
        if not filtered:
            print(f"[embed_upsert][WARN] Skipping empty batch {start}")
            continue
        indices, inputs = zip(*filtered)
        try:
            resp = client.embeddings.create(model="text-embedding-3-large", input=list(inputs))
            embeds = resp.data
        except BadRequestError as e:
            print(f"[embed_upsert][ERROR] Batch {start} BadRequest: {e}. Retrying per-item.")
            embeds = []
            for idx, txt in filtered:
                try:
                    r = client.embeddings.create(model="text-embedding-3-large", input=txt)
                    embeds.append(r.data[0])
                except Exception as ie:  # noqa: BLE001
                    print(f"[embed_upsert][ERROR] Skipping item {start + idx} due to error: {ie}")
                    embeds.append(None)

        embed_map = {}
        for idx, item in zip(indices, embeds):
            if item:
                embed_map[idx] = item.embedding
        # Reconstruct full batch order with placeholders
        for local_idx, _txt in enumerate(batch_texts):
            if local_idx in embed_map:
                embeddings.append(embed_map[local_idx])
        print(f"[embed_upsert] Embedded batch {start}..{start + len(batch_texts) - 1}")

    return embeddings


def upsert_chunks(chunks: Iterable[Dict]) -> None:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60.0,
    )

    if os.getenv("RESET_COLLECTION", "false").lower() == "true":
        print(f"[embed_upsert] RESET_COLLECTION enabled -> recreating {COLLECTION_NAME}")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=EMBEDDING_DIM, distance=rest.Distance.COSINE),
        )

    chunk_list = list(chunks)
    total = len(chunk_list)

    for start, batch in _batch(chunk_list, UPSERT_BATCH_SIZE):
        texts = [c["text"] for c in batch]
        vectors = embed_texts(texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            if vector is None:
                continue
            payload = {k: v for k, v in chunk.items() if k != "chunk_id"}
            points.append(
                rest.PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )
        print(f"[embed_upsert] Upserted batch {start}..{start + len(batch) - 1} / {total} ({len(points)} points)")


def run() -> None:
    load_dotenv(BASE_DIR / ".env")

    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks = dedup_chunks(load_chunks(CHUNKS_PATH))
    if not chunks:
        print("[embed_upsert] No chunks to upsert.")
        return

    print(f"[embed_upsert] Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    upsert_chunks(chunks)
    print("[embed_upsert] Upsert complete.")


if __name__ == "__main__":
    run()
