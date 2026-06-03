from app.retrieval.models import RetrievedChunk
from app.retrieval.text import terms


class LexicalReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        query_terms = set(terms(query))
        reranked: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_terms = set(terms(chunk.text))
            overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
            combined = chunk.score + overlap
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    score=combined,
                    metadata={**chunk.metadata, "rerank_overlap": overlap},
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

