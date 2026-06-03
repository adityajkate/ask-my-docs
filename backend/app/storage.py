import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    author TEXT,
                    checksum TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    allowed_user_ids_json TEXT NOT NULL,
                    allowed_group_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section_title TEXT,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    fallback INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def get_document_by_checksum(self, checksum: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM documents WHERE checksum = ?", (checksum,)).fetchone()

    def upsert_document(
        self,
        *,
        document_id: str,
        title: str,
        source_url: str | None,
        author: str | None,
        checksum: str,
        metadata: dict[str, Any],
        allowed_user_ids: list[str],
        allowed_group_ids: list[str],
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, title, source_url, author, checksum, metadata_json,
                    allowed_user_ids_json, allowed_group_ids_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    title,
                    source_url,
                    author,
                    checksum,
                    json.dumps(metadata),
                    json.dumps(allowed_user_ids),
                    json.dumps(allowed_group_ids),
                    created_at,
                    now,
                ),
            )

    def replace_chunks(self, document_id: str, chunks: Iterable[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, section_title, text,
                    token_count, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk["chunk_index"],
                        chunk.get("section_title"),
                        chunk["text"],
                        chunk["token_count"],
                        chunk["content_hash"],
                        json.dumps(chunk.get("metadata", {})),
                    )
                    for chunk in chunks
                ],
            )

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return cursor.rowcount > 0

    def list_documents(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.updated_at DESC
                """
            ).fetchall()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                WHERE d.document_id = ?
                GROUP BY d.document_id
                """,
                (document_id,),
            ).fetchone()

    def list_chunks(self, document_id: str | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT c.*, d.title, d.source_url, d.allowed_user_ids_json, d.allowed_group_ids_json
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
        """
        params: tuple[Any, ...] = ()
        if document_id:
            query += " WHERE c.document_id = ?"
            params = (document_id,)
        query += " ORDER BY c.document_id, c.chunk_index"
        with self.connect() as conn:
            return conn.execute(query, params).fetchall()

    def save_chat_message(
        self,
        *,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        fallback: bool,
    ) -> str:
        message_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, user_id, question, answer,
                    citations_json, fallback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    user_id,
                    question,
                    answer,
                    json.dumps(citations),
                    1 if fallback else 0,
                    utc_now(),
                ),
            )
        return message_id

    def save_feedback(self, *, message_id: str, user_id: str, rating: int, comment: str | None) -> str:
        feedback_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (feedback_id, message_id, user_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, message_id, user_id, rating, comment, utc_now()),
            )
        return feedback_id

