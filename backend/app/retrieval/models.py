from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    source_url: str | None
    section_title: str | None
    text: str
    score: float
    metadata: dict[str, Any]

