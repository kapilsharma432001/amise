import psycopg2
import json

class HybridRetriever:
    def __init__(self, db_params: dict):
        self.conn = psycopg2.connect(**db_params)
        self.setup_database()

    def setup_database(self):
        """Initializes the pgvector extension and our documents table."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # We store the text, a 1536-dimensional vector (standard for OpenAI text-embedding-3-small),
            # and a tsvector column for fast lexical search.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_reports (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536),
                    fts_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                );
            """)
            # Create indexes for faster retrieval
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fts ON market_reports USING GIN (fts_tokens);")
            self.conn.commit()

    def hybrid_search(self, query_text: str, query_embedding: list, limit: int = 5) -> list:
        """
        Executes an RRF hybrid search combining semantic and lexical results.
        """
        sql = """
        WITH semantic_search AS (
            SELECT id, content, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM market_reports
            ORDER BY embedding <=> %s::vector
            LIMIT 20
        ),
        keyword_search AS (
            SELECT id, content, RANK() OVER (ORDER BY ts_rank_cd(fts_tokens, plainto_tsquery('english', %s)) DESC) AS rank
            FROM market_reports
            WHERE fts_tokens @@ plainto_tsquery('english', %s)
            LIMIT 20
        )
        SELECT 
            COALESCE(s.id, k.id) AS id,
            COALESCE(s.content, k.content) AS content,
            COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
        FROM semantic_search s
        FULL OUTER JOIN keyword_search k ON s.id = k.id
        ORDER BY rrf_score DESC
        LIMIT %s;
        """
        
        with self.conn.cursor() as cur:
            # Pass the embedding twice (for the window function and the ORDER BY), 
            # then the text twice, then the final limit.
            cur.execute(sql, (query_embedding, query_embedding, query_text, query_text, limit))
            results = cur.fetchall()
            
        return [{"id": row[0], "content": row[1], "score": row[2]} for row in results]

# Quick test configuration block
if __name__ == "__main__":
    db_config = {
        "dbname": "amise",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432"
    }
    
    print("Connecting to PostgreSQL and setting up schema...")
    retriever = HybridRetriever(db_config)
    print("Schema initialized successfully. Database is ready for Hybrid RAG.")