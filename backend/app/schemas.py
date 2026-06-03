from pydantic import BaseModel, Field
from typing import Any


class DocumentIngestRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_url: str | None = None
    author: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_group_ids: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    source_url: str | None
    author: str | None
    checksum: str
    chunk_count: int
    created_at: str
    updated_at: str


class ChunkSummary(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    section_title: str | None
    token_count: int
    text_preview: str


class DocumentDetail(DocumentSummary):
    chunks: list[ChunkSummary]


class IngestResponse(BaseModel):
    document: DocumentSummary
    chunks_created: int
    skipped_unchanged: bool


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    section_title: str | None
    source_url: str | None
    quote: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=12)
    include_debug: bool = False


class RetrievalDebug(BaseModel):
    dense_results: int
    bm25_results: int
    fused_results: int
    selected_results: int
    confidence: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    fallback: bool
    session_id: str
    message_id: str
    debug: RetrievalDebug | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    accepted: bool


class EvaluationCase(BaseModel):
    query: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase]


class EvaluationResult(BaseModel):
    total_cases: int
    context_recall: float
    citation_validity: float
    results: list[dict[str, Any]]

