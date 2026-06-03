from app.config import Settings
from app.generation.citations import validate_citations
from app.retrieval.models import RetrievedChunk
from app.retrieval.text import terms


class GroundedGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool]:
        if not chunks:
            return self.settings.fallback_answer, True

        query_terms = set(terms(question))
        sentences: list[str] = []
        for chunk in chunks:
            candidates = [part.strip() for part in chunk.text.replace("\n", " ").split(".") if part.strip()]
            best = sorted(
                candidates,
                key=lambda sentence: len(set(terms(sentence)) & query_terms),
                reverse=True,
            )[:2]
            for sentence in best:
                if sentence:
                    sentences.append(f"{sentence}. [Chunk: {chunk.chunk_id}]")
            if len(sentences) >= 4:
                break

        answer = " ".join(sentences).strip()
        if not answer:
            return self.settings.fallback_answer, True
        if not validate_citations(answer, chunks):
            return self.settings.fallback_answer, True
        return answer, False

    async def stream_answer(self, question: str, chunks: list[RetrievedChunk]):
        answer, fallback = self.answer(question, chunks)
        for token in answer.split(" "):
            yield token + " "
        yield {"fallback": fallback}

