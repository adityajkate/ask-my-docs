import hashlib
import math
from app.retrieval.models import RetrievedChunk
from app.retrieval.text import terms


class HashingVectorizer:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term in terms(text):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LocalVectorIndex:
    def __init__(self, chunks: list[RetrievedChunk], vectorizer: HashingVectorizer | None = None):
        self.chunks = chunks
        self.vectorizer = vectorizer or HashingVectorizer()
        self.vectors = [self.vectorizer.embed(chunk.text) for chunk in chunks]

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_vector = self.vectorizer.embed(query)
        scored: list[RetrievedChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = cosine(query_vector, vector)
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

