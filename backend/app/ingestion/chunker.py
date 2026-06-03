import hashlib
import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chunk_index: int
    section_title: str | None
    text: str
    token_count: int
    content_hash: str
    metadata: dict


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def chunk_text(
    *,
    document_id: str,
    document_version: str,
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    stride = max(1, chunk_size_tokens - chunk_overlap_tokens)
    for section_title, section_text in split_sections(text):
        tokens = tokenize(section_text)
        if not tokens:
            continue
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + chunk_size_tokens]
            if not window:
                continue
            chunk_body = " ".join(window)
            content_hash = stable_hash(chunk_body)
            section_path = section_title or "root"
            chunk_id = stable_hash(
                f"{document_id}:{document_version}:{section_path}:{chunk_index}:{content_hash}"
            )[:24]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    section_title=section_title,
                    text=chunk_body,
                    token_count=len(window),
                    content_hash=content_hash,
                    metadata={"section_path": section_path, "token_start": start},
                )
            )
            chunk_index += 1
            if start + chunk_size_tokens >= len(tokens):
                break
    return chunks

