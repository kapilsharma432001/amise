"""
AMISE - Hybrid Retriever
The production grade hybrid RAG retieval engine that combines:-
    1. Dense Retrival - pgvector cosine similarity over embeddings
    2. Sparse Retrieval - BM25 okapi lexical scoring (same algorithm as used by ElasticSearch)
    3. RRF Fusion - Reciprocal rank fusion to merge both rankings
"""
