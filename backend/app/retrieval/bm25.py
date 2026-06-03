import math
from collections import Counter
from app.retrieval.models import RetrievedChunk
from app.retrieval.text import terms


class BM25Index:
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(terms(chunk.text)) for chunk in chunks]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_terms]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.document_frequency: Counter[str] = Counter()
        for counter in self.doc_terms:
            self.document_frequency.update(counter.keys())

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_terms = terms(query)
        scored: list[RetrievedChunk] = []
        total_docs = max(1, len(self.chunks))
        for chunk, counter, doc_len in zip(self.chunks, self.doc_terms, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                frequency = counter.get(term, 0)
                if frequency == 0:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = frequency + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_len))
                score += idf * (frequency * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        source_url=chunk.source_url,
                        section_title=chunk.section_title,
                        text=chunk.text,
                        score=score,
                        metadata=chunk.metadata,
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

