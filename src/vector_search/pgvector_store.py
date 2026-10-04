from collections.abc import Sequence

from tqdm.auto import tqdm

from vector_search.embeddings import vector_to_string


class PgVectorStore:
    def __init__(self, connection):
        self.conn = connection

    def initialize_schema(self, reset: bool = False):
        """Initializes the database schema."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        if reset:
            self.conn.execute("DROP TABLE IF EXISTS documents")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                course TEXT NOT NULL,
                section TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding vector(384) NOT NULL
            )
            """
        )
        self.conn.commit()

    def insert_documents(self, documents, vectors):
        """Inserts documents and their embeddings into the database."""
        for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
            self.conn.execute(
                """
                INSERT INTO documents (course, section, question, answer, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (
                    doc["course"],
                    doc["section"],
                    doc["question"],
                    doc["answer"],
                    vector_to_string(vec),
                ),
            )
        self.conn.commit()

    def create_index(self):
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            """
        )
        self.conn.commit()

    def search(
        self,
        query_vector: Sequence[float],
        course: str,
        num_results: int = 5,
    ):
        """Searches for the most similar documents to the query vector."""
        query_str = vector_to_string(query_vector)
        rows = self.conn.execute(
            """
            SELECT course, section, question, answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """,
            (query_str, course, query_str, num_results),
        ).fetchall()

        return [
            {
                "course": r[0],
                "section": r[1],
                "question": r[2],
                "answer": r[3],
                "similarity": r[4],
            }
            for r in rows
        ]
