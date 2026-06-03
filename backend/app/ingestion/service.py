import hashlib
from dataclasses import asdict
from app.config import Settings
from app.ingestion.chunker import chunk_text, stable_hash
from app.storage import Storage


class IngestionService:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings

    def ingest_text(
        self,
        *,
        title: str,
        text: str,
        source_url: str | None,
        author: str | None,
        metadata: dict,
        allowed_user_ids: list[str],
        allowed_group_ids: list[str],
    ) -> tuple[dict, int, bool]:
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.storage.get_document_by_checksum(checksum)
        if existing:
            document = dict(existing)
            return document, int(document.get("chunk_count", 0) or 0), True

        document_id = stable_hash(f"{source_url or title}:{checksum}")[:24]
        self.storage.upsert_document(
            document_id=document_id,
            title=title,
            source_url=source_url,
            author=author,
            checksum=checksum,
            metadata=metadata,
            allowed_user_ids=allowed_user_ids,
            allowed_group_ids=allowed_group_ids,
        )
        chunks = chunk_text(
            document_id=document_id,
            document_version=checksum,
            text=text,
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        self.storage.replace_chunks(document_id, [asdict(chunk) for chunk in chunks])
        document = dict(self.storage.get_document(document_id))
        return document, len(chunks), False

