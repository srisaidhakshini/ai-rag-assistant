"""Central, env-overridable configuration for the RAG assistant.

Reading these as module attributes (rather than binding them into other
modules at import time) is what lets tests monkeypatch e.g. MAX_FILE_SIZE_MB
without needing to reload modules.
"""
import os

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# How many extra candidates to pull per TOP_K slot before dedup/truncation -
# CHUNK_OVERLAP means adjacent chunks are often near-duplicates, so without
# over-fetching, dedup would silently shrink the context below TOP_K.
RETRIEVAL_FANOUT = int(os.getenv("RAG_RETRIEVAL_FANOUT", "3"))
# Overlap-coefficient (intersection / smaller set) above which two retrieved
# chunks are considered near-duplicates and the lower-scoring one is dropped.
DEDUP_OVERLAP_THRESHOLD = float(os.getenv("RAG_DEDUP_OVERLAP_THRESHOLD", "0.7"))
MAX_FILE_SIZE_MB = float(os.getenv("RAG_MAX_FILE_SIZE_MB", "20"))
HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "3"))

# Cosine-similarity score distributions differ by embedding backend, so the
# "reject low-confidence retrieval" threshold (PRD requirement #8) is
# per-backend rather than one global number.
RELEVANCE_THRESHOLDS = {
    "openai": 0.25,
    "sentence-transformers": 0.30,
    "tfidf": 0.10,
}
DEFAULT_RELEVANCE_THRESHOLD = 0.2

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
