import psycopg

from vector_search.config import DATABASE_URL
from vector_search.embeddings import create_embedder, embed_documents
from vector_search.ingest import load_faq_data
from vector_search.pgvector_store import PgVectorStore


def main():
    documents = load_faq_data()
    embedder = create_embedder()
    vectors = embed_documents(documents, embedder)

    with psycopg.connect(DATABASE_URL) as conn:
        store = PgVectorStore(conn)
        store.initialize_schema(reset=True)
        store.insert_documents(documents, vectors)
        store.create_index()


if __name__ == "__main__":
    main()
