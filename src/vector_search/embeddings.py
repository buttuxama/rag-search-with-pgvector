from collections.abc import Sequence

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def create_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Creates an embedding model using the specified model name."""
    return SentenceTransformer(model_name)


def document_text(document: dict) -> str:
    """Extracts the text from a document dictionary for embedding."""
    return document["question"] + " " + document["answer"]


def embed_documents(
    documents: list[dict],
    model: SentenceTransformer,
    batch_size: int = 50,
):
    """Embeds a list of documents using the specified model."""
    texts = [document_text(doc) for doc in documents]
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_vectors = model.encode(batch)
        vectors.extend(batch_vectors)

    return vectors


def embed_query(query: str, model: SentenceTransformer):
    """Embeds a query string using the specified model."""
    return model.encode(query)


def vector_to_string(vector: Sequence[float]) -> str:
    """Converts a vector to a string representation."""
    return "[" + ",".join(str(x) for x in vector) + "]"
