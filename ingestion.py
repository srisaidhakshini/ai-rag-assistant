"""Document loading, size/dedup guarding, and boundary-aware chunking.

Handles PDF / TXT / MD / DOCX (PRD requirement #1) and produces
page-tagged, semantically-bounded chunks (requirements #2-#3).
"""
import hashlib
import os

import config

SEPARATORS = ["\n\n", "\n", ". ", " "]


class DocumentError(Exception):
    """Raised for anything that stops a document from being indexed
    (unsupported type, oversized file, unreadable/scanned PDF, ...)."""


def hash_file(path):
    """SHA-256 of file contents, used to detect duplicate uploads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _check_file_size(path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise DocumentError(
            f"'{os.path.basename(path)}' is {size_mb:.1f} MB, which exceeds the "
            f"{config.MAX_FILE_SIZE_MB} MB per-file limit."
        )


def _load_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    total_chars = 0
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        total_chars += len(text.strip())
        pages.append({"text": text, "page": i})

    if total_chars < 20:
        raise DocumentError(
            f"'{os.path.basename(path)}' appears to be a scanned/image-only PDF with "
            "no extractable text. OCR is out of scope for v1 (see PRD Section 4) - "
            "try a text-based PDF, or export/copy the content to a .txt/.md file."
        )
    return pages


def _load_docx(path):
    from docx import Document as DocxDocument

    doc = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"text": text, "page": None}]


def _load_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [{"text": text, "page": None}]


def load_document(path):
    """Returns a list of {"text": str, "page": int|None} dicts, one per PDF
    page (or a single entry for TXT/MD/DOCX which have no page concept)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise DocumentError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        )
    _check_file_size(path)

    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    return _load_text(path)  # .txt, .md


def _split_text(text, chunk_size, overlap, separators):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = next((s for s in separators if s in text), None)
    if sep is None:
        # No usable separator left - hard-split on raw characters.
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end
        return chunks

    parts = text.split(sep)
    next_seps = separators[separators.index(sep) + 1:]
    chunks = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > chunk_size:
            chunks.extend(_split_text(part, chunk_size, overlap, next_seps))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i - 1][-overlap:] + chunks[i])
        return overlapped
    return chunks


def chunk_document(pages, source, chunk_size=None, overlap=None):
    """Splits each page's text recursively (paragraph -> line -> sentence ->
    word) into overlapping chunks, tagged with source filename + page."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    chunks = []
    for page in pages:
        pieces = _split_text(page["text"], chunk_size, overlap, SEPARATORS)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                chunks.append({"text": piece, "source": source, "page": page.get("page")})
    return chunks
