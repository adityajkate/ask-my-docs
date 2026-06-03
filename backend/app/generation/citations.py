import re
from app.retrieval.models import RetrievedChunk


CITATION_PATTERN = re.compile(r"\[Chunk:\s*([a-f0-9]{8,64})\]")


def extract_citation_ids(answer: str) -> set[str]:
    return set(CITATION_PATTERN.findall(answer))


def validate_citations(answer: str, selected_chunks: list[RetrievedChunk]) -> bool:
    selected_ids = {chunk.chunk_id for chunk in selected_chunks}
    cited_ids = extract_citation_ids(answer)
    return bool(cited_ids) and cited_ids.issubset(selected_ids)


def citation_payload(chunk: RetrievedChunk) -> dict:
    quote = chunk.text[:320].strip()
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "section_title": chunk.section_title,
        "source_url": chunk.source_url,
        "quote": quote,
    }

