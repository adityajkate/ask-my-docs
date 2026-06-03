from app.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    *,
    dense_weight: float,
    bm25_weight: float,
    k: int,
) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        by_id[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + dense_weight / (k + rank)

    for rank, chunk in enumerate(bm25_results, start=1):
        by_id.setdefault(chunk.chunk_id, chunk)
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + bm25_weight / (k + rank)

    fused = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            source_url=chunk.source_url,
            section_title=chunk.section_title,
            text=chunk.text,
            score=scores[chunk_id],
            metadata=chunk.metadata,
        )
        for chunk_id, chunk in by_id.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)

