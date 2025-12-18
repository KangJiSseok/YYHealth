import os

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


COLLECTION_NAME = "nutrition_rag_chunks"
EMBEDDING_DIM = 3072  # text-embedding-3-large
DISTANCE = rest.Distance.COSINE


def create_collection(
    url: str | None = None,
    api_key: str | None = None,
    collection_name: str = COLLECTION_NAME,
    embedding_dim: int = EMBEDDING_DIM,
) -> None:
    """
    Idempotently create the Qdrant collection for chunk embeddings.
    """
    client = QdrantClient(
        url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=api_key or os.getenv("QDRANT_API_KEY"),
    )

    if client.collection_exists(collection_name):
        return

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=embedding_dim,
            distance=DISTANCE,
        ),
    )


if __name__ == "__main__":
    create_collection()
