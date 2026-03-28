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

