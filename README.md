# Vector Search with PostgreSQL and pgvector

This project implements the vector-search lesson using PostgreSQL and
pgvector. The Python modules contain the application logic; the notebook is
only a demonstration client.

## Requirements

- Python 3.14
- PostgreSQL with the pgvector extension
- An Ollama Cloud API key for the RAG demonstration

The pgvector lesson uses Docker to run PostgreSQL. Start the container using
the command from the lesson, or use an existing PostgreSQL installation with
pgvector enabled.

## Setup

Install dependencies:

```bash
uv sync --no-editable
```

The `--no-editable` option installs the project package into the virtual
environment so the `ingest-pgvector` command can import `vector_search`
reliably. When running project commands after setup, use `--no-sync` so uv
does not replace the regular installation with an editable installation:

```bash
uv run --no-sync ingest-pgvector
uv run --no-sync jupyter lab
```

Create the environment file:

```bash
touch .env
```

Add the following variables to `.env`, replacing the placeholders with your
actual values:

```dotenv
OLLAMA_API_KEY=your-ollama-api-key
CONNECTION_STRING=postgresql://your-username:your-password@localhost:5432/faq
```

`your-username` and `your-password` must match the PostgreSQL credentials.
Change `localhost`, `5432`, or `faq` if your PostgreSQL server uses a
different host, port, or database name. Never commit `.env`.

## Application flow

The application has two separate flows: ingestion prepares the searchable
database, and the notebook uses that database for vector search and RAG
answer generation.

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion flow: uv run ingest-pgvector"]
        A["FAQ API"] --> B["ingest.py<br/>load_faq_data()"]
        B --> C["embeddings.py<br/>create embeddings in batches"]
        C --> D["PgVectorStore<br/>initialize schema"]
        D --> E["PostgreSQL + pgvector<br/>documents table"]
        C --> F["PgVectorStore<br/>insert documents and vectors"]
        F --> E
        E --> G["HNSW cosine index"]
    end

    subgraph Query["Query flow: notebook"]
        H["User question"] --> I["SentenceTransformer<br/>embed query"]
        I --> J["RAGPgVector.search()"]
        J --> K["PgVectorStore.search()"]
        K --> E
        E --> L["Top matching FAQ documents"]
        L --> M["RAGBase.build_context()"]
        M --> N["RAGBase.build_prompt()"]
        N --> O["Ollama Cloud<br/>OpenAI-compatible API"]
        O --> P["Answer"]
    end
```

### Flow responsibilities

1. `ingest_pgvector.py` downloads the FAQ documents and coordinates ingestion.
2. `embeddings.py` converts each FAQ question and answer into a 384-dimensional
   embedding.
3. `PgVectorStore` stores documents and embeddings in PostgreSQL and creates
   the HNSW index for efficient cosine similarity search.
4. The notebook embeds a user's question and passes the vector to
   `RAGPgVector`.
5. `PgVectorStore` retrieves the closest FAQ documents for the selected
   course.
6. `RAGBase` builds a context-rich prompt from those documents.
7. The Ollama-compatible client generates the final answer.

## Load the FAQ data

The ingestion command downloads the FAQ documents, creates embeddings with
`all-MiniLM-L6-v2`, initializes the `documents` table, inserts the embeddings,
and creates the HNSW cosine index:

```bash
uv run ingest-pgvector
```

The ingestion command resets the `documents` table. Run it again whenever you
want to rebuild the dataset from scratch.

## Run the demonstration

Open `src/vector_search/vector_search.ipynb` after ingestion. The notebook
connects to PostgreSQL, performs a vector search, and runs the RAG assistant.
It does not download data, create the schema, or insert documents.

## Project layout

- `src/vector_search/ingest.py`: downloads the FAQ documents.
- `src/vector_search/embeddings.py`: creates and reuses the embedding model.
- `src/vector_search/pgvector_store.py`: owns pgvector schema, ingestion, and search SQL.
- `src/vector_search/ingest_pgvector.py`: command-line ingestion workflow.
- `src/vector_search/rag_base.py`: RAG prompt and LLM behavior, including the PostgreSQL search adapter.
- `src/vector_search/vector_search.ipynb`: demonstration-only notebook.
