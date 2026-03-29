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
    Manages the PostgresSQL + pgvector storage layer
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