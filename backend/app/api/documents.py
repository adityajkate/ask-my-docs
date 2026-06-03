from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.dependencies import Principal, get_principal
from app.ingestion.parsers import parse_upload
from app.main_services import get_ingestion_service, get_storage
from app.schemas import DocumentDetail, DocumentIngestRequest, DocumentSummary, IngestResponse
from app.storage import Storage


router = APIRouter(prefix="/api/documents", tags=["documents"])


def to_document_summary(row) -> DocumentSummary:
    return DocumentSummary(
        document_id=row["document_id"],
        title=row["title"],
        source_url=row["source_url"],
        author=row["author"],
        checksum=row["checksum"],
        chunk_count=int(row["chunk_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: DocumentIngestRequest, principal: Principal = Depends(get_principal)):
    service = get_ingestion_service()
    document, chunks_created, skipped = service.ingest_text(
        title=request.title,
        text=request.text,
        source_url=request.source_url,
        author=request.author,
        metadata={**request.metadata, "ingested_by": principal.user_id},
        allowed_user_ids=request.allowed_user_ids,
        allowed_group_ids=request.allowed_group_ids,
    )
    row = get_storage().get_document(document["document_id"])
    return IngestResponse(
        document=to_document_summary(row),
        chunks_created=chunks_created,
        skipped_unchanged=skipped,
    )


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    principal: Principal = Depends(get_principal),
):
    try:
        text = parse_upload(file.filename or "upload.txt", file.file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = get_ingestion_service()
    document, chunks_created, skipped = service.ingest_text(
        title=file.filename or "Untitled upload",
        text=text,
        source_url=None,
        author=None,
        metadata={"filename": file.filename, "content_type": file.content_type, "ingested_by": principal.user_id},
        allowed_user_ids=[principal.user_id],
        allowed_group_ids=[],
    )
    row = get_storage().get_document(document["document_id"])
    return IngestResponse(
        document=to_document_summary(row),
        chunks_created=chunks_created,
        skipped_unchanged=skipped,
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents(storage: Storage = Depends(get_storage)):
    return [to_document_summary(row) for row in storage.list_documents()]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, storage: Storage = Depends(get_storage)):
    row = storage.get_document(document_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = storage.list_chunks(document_id)
    return DocumentDetail(
        **to_document_summary(row).model_dump(),
        chunks=[
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "section_title": chunk["section_title"],
                "token_count": chunk["token_count"],
                "text_preview": chunk["text"][:240],
            }
            for chunk in chunks
        ],
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str, storage: Storage = Depends(get_storage)):
    deleted = storage.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "document_id": document_id}

