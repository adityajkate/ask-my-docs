import json
from dataclasses import dataclass
from app.config import Settings
from app.dependencies import Principal
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import LexicalReranker
from app.retrieval.vector import LocalVectorIndex
from app.storage import Storage


@dataclass(frozen=True)
class RetrievalResult:
    selected: list[RetrievedChunk]
    dense_count: int
    bm25_count: int
    fused_count: int
    confidence: float


def is_authorized(row, principal: Principal) -> bool:
    allowed_users = set(json.loads(row["allowed_user_ids_json"] or "[]"))
    allowed_groups = set(json.loads(row["allowed_group_ids_json"] or "[]"))
    if not allowed_users and not allowed_groups:
        return True
    return principal.user_id in allowed_users or bool(allowed_groups & set(principal.group_ids))


class RetrievalService:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.reranker = LexicalReranker()

    def _load_authorized_chunks(self, principal: Principal) -> list[RetrievedChunk]:
        rows = self.storage.list_chunks()
        chunks: list[RetrievedChunk] = []
        for row in rows:
            if not is_authorized(row, principal):
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    title=row["title"],
                    source_url=row["source_url"],
                    section_title=row["section_title"],
                    text=row["text"],
                    score=0.0,
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
        return chunks

    def retrieve(self, query: str, principal: Principal, top_k: int | None = None) -> RetrievalResult:
        chunks = self._load_authorized_chunks(principal)
        if not chunks:
            return RetrievalResult([], 0, 0, 0, 0.0)

        preliminary_top_n = self.settings.preliminary_top_n
        dense_results = LocalVectorIndex(chunks).search(query, preliminary_top_n)
        bm25_results = BM25Index(chunks).search(query, preliminary_top_n)
        fused = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
            k=self.settings.rrf_k,
        )[:preliminary_top_n]
        selected = self.reranker.rerank(query, fused, top_k or self.settings.final_top_k)
        confidence = selected[0].score if selected else 0.0
        if confidence < self.settings.min_confidence:
            selected = []
        return RetrievalResult(
            selected=selected,
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
            confidence=confidence,
        )

