from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os


class Settings(BaseModel):
    app_name: str = "Ask My Docs Backend"
    app_version: str = "0.1.0"
    database_path: Path = Path(os.getenv("ASK_MY_DOCS_DB", "data/ask_my_docs.sqlite3"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "1.0"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "1.0"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    preliminary_top_n: int = int(os.getenv("PRELIMINARY_TOP_N", "50"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.05"))
    fallback_answer: str = "The provided documents do not contain information to answer this query."


@lru_cache
def get_settings() -> Settings:
    return Settings()

