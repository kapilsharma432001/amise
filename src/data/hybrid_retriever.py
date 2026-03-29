"""
AMISE - Hybrid Retriever
The production grade hybrid RAG retieval engine that combines:-
    1. Dense Retrival - pgvector cosine similarity over embeddings
    2. Sparse Retrieval - BM25 okapi lexical scoring (same algorithm as used by ElasticSearch)
    3. RRF Fusion - Reciprocal rank fusion to merge both rankings
"""
import hashlib
import json
from dataclass import dataclass, field
from typing import Optional

import asyncpg # better than psycopg2 for async operations
import litellm
import struclog
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector
from rank_bm25 import BM25Okapi

load_dotenv()
logger = struclog.get_logger(__name__)

# Embedding model for dense retrieval (semantic search)
# text-embedding-3-small: 1536 dimensions
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

@dataclass
class Document:
    """
    A chunk of text ready for ingestion into the retrieval system.

    Why a dataclass and not a dict?
    - Enforces structure
    - IDE automcompletion and type checking
    - Immutable by default mindset

    """
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.sha256(
                self.content.encode()
            ).hexdigest()[:16]
    

@dataclass
class RetrievalResult:
    """
    A single search result with its score and origin.

    Tracking 'source' (dense/sparse/hybrid) is critical for:
    - Debugging: why did this result rank high?
    - Evaluation: which retrieval method is contributing more?
    - Tuning: should we adjust the RRF weight?
    """

    doc_id: str
    content: str
    score: float
    metadata: dict
    source: str  # "dense", "sparse", or "hybrid"

    
class DocumentStore:
    """
    Manages the PostgreSQL + pgvector storage layer.

    Responsibility: ONLY handles database operations.
    Does NOT handle search logic — that's the retriever's job.

    This separation follows the Single Responsibility Principle:
    - DocumentStore = storage (CRUD)
    - HybridRetriever = search intelligence (ranking, fusion)
    """

    def __init__(self, dsn: str):
        """
        Args:
            dsn: PostgreSQL connection string.
                Format: postgresql://user:pass@host:port/dbname
        """
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """
        Create connection pool and set up schema.

        Why a connection pool?
        - Creating a new DB connection takes ~50-100ms (TCP + TLS + auth).
        - A pool keeps connections warm and reuses them.
        - For an agent making 10 retrieval calls in parallel,
        this means 10x50ms=500ms saved per agent cycle.
        """
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=2,      # Always keep 2 connections warm
            max_size=10,     # Never exceed 10 (protects the DB)
            command_timeout=30,
        )

        # Register pgvector type so asyncpg can handle vector columns
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await self._create_schema(conn)

        logger.info("document_store.initialized", dsn=self.dsn[:30] + "...")

    async def _create_schema(self, conn: asyncpg.Connection) -> None:
        """
        Create the documents table with vector column and indexes.

        Design decisions in the schema:
        - UNIQUE(doc_id): prevents duplicate ingestion
        - vector(1536): matches text-embedding-3-small output dimension
        - ivfflat index: approximate nearest neighbor for fast search
        (exact search is O(n), IVFFlat is O(√n) with minor recall loss)
        - GIN index on content: accelerates full-text search if we add
        PostgreSQL tsvector search later
        """
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          SERIAL PRIMARY KEY,
                doc_id      TEXT UNIQUE NOT NULL,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{}',
                embedding   vector(1536),
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # IVFFlat index: lists=100 is good for up to ~100k documents.
        # For 1M+ docs, switch to HNSW index.
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        logger.info("document_store.schema_ready")

    async def insert_document(
        self, doc_id: str, content: str, metadata: dict, embedding: list[float]
    ) -> bool:
        """
        Insert a single document. Returns True if inserted, False if duplicate.

        ON CONFLICT DO NOTHING: idempotent ingestion.
        Running the same ingestion pipeline twice won't create duplicates.
        This is critical in production where pipelines get retried.
        """
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            result = await conn.execute(
                """
                INSERT INTO documents (doc_id, content, metadata, embedding)
                VALUES ($1, $2, $3::jsonb, $4::vector)
                ON CONFLICT (doc_id) DO NOTHING;
                """,
                doc_id,
                content,
                json.dumps(metadata),
                embedding,
            )
            # result is "INSERT 0 1" if inserted, "INSERT 0 0" if skipped
            return result == "INSERT 0 1"

    async def fetch_all_contents(self) -> list[dict]:
        """
        Fetch all document IDs and contents for BM25 index building.

        Why load all content into memory for BM25?
        - BM25 is an in-memory algorithm — it needs the full corpus to
        compute IDF (inverse document frequency) scores.
        - For <100k chunks, this fits comfortably in memory (~500MB).
        - Beyond that, switch to Elasticsearch or PostgreSQL tsvector.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id, content, metadata FROM documents ORDER BY id;"
            )
            return [
                {
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]

    async def dense_search(
        self, query_embedding: list[float], top_k: int = 20
    ) -> list[dict]:
        """
        Perform vector cosine similarity search using pgvector.

        The SQL uses <=> operator which is pgvector's cosine distance.
        We compute similarity as (1 - distance) so higher = more similar.

        Why top_k=20 when we might only need 5 final results?
        - We over-retrieve from each method, then let RRF pick the best.
        - This gives RRF more candidates to work with, improving recall.
        """
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(
                """
                SELECT
                    doc_id,
                    content,
                    metadata,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2;
                """,
                query_embedding,
                top_k,
            )
            return [
                {
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "score": float(row["similarity"]),
                }
                for row in rows
            ]

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("document_store.closed")

# Embedding Engine - convert text to vectors using LiteLLM's aembedding() which abstracts away the provider (OpenAI, Anthropic, etc.)
class EmbeddingEngine:
    """
    Handles all text-to-vector conversions via LiteLLM.

    Why a separate class instead of a function?
    - Future: add embedding cache (avoid re-embedding the same text)
    - Future: add batch optimization (embed 100 chunks in one API call)
    - Testability: mock this class in unit tests
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.

        Uses LiteLLM's aembedding() which routes to the correct
        provider based on model name (OpenAI, Cohere, etc.).
        """
        response = await litellm.aembedding(
            model=self.model,
            input=[text],
        )
        return response.data[0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single API call.

        Why batch? Because:
        - 1 API call with 100 texts ≈ 200ms
        - 100 API calls with 1 text each ≈ 100 × 200ms = 20,000ms
        Batching gives ~100x speedup for ingestion.
        """
        response = await litellm.aembedding(
            model=self.model,
            input=texts,
        )
        return [item["embedding"] for item in response.data]

# BM25 Engine - in-memory lexical search using rank_bm25 library (keyword-based, not semantic)
class BM25Engine:
    """
    In-memory BM25 index for lexical search.

    BM25 (Best Matching 25) scores documents based on:
    - TF: How often does the query term appear in the doc? (saturating)
    - IDF: How rare is the query term across all docs? (rarer = better)
    - DL: How long is the doc relative to average? (normalizes for length)

    The formula: score(q,d) = Σ IDF(t) x [TF(t,d) x (k1+1)] / [TF(t,d) + k1 x (1 - b + b x |d|/avgdl)]
    Default k1=1.5, b=0.75 — these are battle-tested values from IR research.
    """

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.corpus_docs: list[dict] = []  # Parallel array to BM25 internal corpus

    def build_index(self, documents: list[dict]) -> None:
        """
        Build the BM25 index from a list of documents.

        Tokenization here is simple whitespace split + lowercasing.
        Production improvement: use spaCy or NLTK for proper tokenization,
        stopword removal, and stemming. But for AMISE, this is sufficient.
        """
        self.corpus_docs = documents
        tokenized_corpus = [
            doc["content"].lower().split() for doc in documents
        ]
        self.index = BM25Okapi(tokenized_corpus)

        logger.info("bm25_engine.index_built", num_documents=len(documents))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Score all documents against the query and return top_k.

        Returns documents with BM25 scores (not normalized to 0-1).
        RRF doesn't need normalized scores — it only uses ranks.
        """
        if self.index is None:
            logger.warning("bm25_engine.search_called_before_build")
            return []

        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        # Pair scores with documents and sort descending
        scored_docs = [
            {
                "doc_id": doc["doc_id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
            }
            for doc, score in zip(self.corpus_docs, scores)
            if score > 0  # Skip zero-score documents
        ]

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

# the orchestrator
class HybridRetriever:
    """
    Orchestrates Dense + Sparse retrieval and fuses results using RRF.

    This is the class that the rest of AMISE (agents, API) interacts with.
    It delegates to DocumentStore (storage), EmbeddingEngine (vectors),
    and BM25Engine (lexical), then merges everything via RRF.

    Design pattern: **Mediator**
    - Coordinates interactions between DocumentStore, EmbeddingEngine,
      and BM25Engine without them knowing about each other.
    """
    class HybridRetriever:
        """
        Orchestrates Dense + Sparse retrieval and fuses results using RRF.

        This is the class that the rest of AMISE (agents, API) interacts with.
        It delegates to DocumentStore (storage), EmbeddingEngine (vectors),
        and BM25Engine (lexical), then merges everything via RRF.

        Design pattern: **Mediator**
        - Coordinates interactions between DocumentStore, EmbeddingEngine,
        and BM25Engine without them knowing about each other.
        """

        def __init__(self, database_url: str, rrf_k: int = 60):
            """
            Args:
                database_url: PostgreSQL DSN string.
                rrf_k: RRF smoothing constant (default 60 from original paper).
                    Higher k = less emphasis on top-ranked results.
                    Lower k  = more winner-take-all behavior.
            """
            self.store = DocumentStore(dsn=database_url)
            self.embedder = EmbeddingEngine()
            self.bm25 = BM25Engine()
            self.rrf_k = rrf_k