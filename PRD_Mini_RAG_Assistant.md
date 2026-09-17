# PRD: Mini AI Knowledge Assistant (RAG)

**Author:** [Your name]
**Status:** Draft → Implemented (v1)
**Last updated:** 2026-09-17

---

## 1. Problem Statement

People have knowledge locked inside documents (handbooks, policies, reports,
contracts) that is slow to search manually and easy to misremember. Generic
LLM chat can *sound* authoritative while answering from general training
data instead of the actual document — which is dangerous for anything
policy- or fact-sensitive.

**Goal:** let a user drop in their own document(s) and get answers that are
grounded in that content, with visible proof (citations) of where each
answer came from.

## 2. Objective / Success Criteria

Build a working Retrieval-Augmented Generation (RAG) application that:

| # | Requirement | Success bar |
|---|---|---|
| 1 | Accepts a document or set of documents as knowledge source | PDF, TXT, MD, DOCX supported |
| 2 | Extracts and processes content | Clean text, page-level metadata preserved for PDFs |
| 3 | Splits content into chunks | Semantically coherent chunks, no context lost at boundaries |
| 4 | Generates embeddings | Pluggable backend, works with or without an API key |
| 5 | Stores/retrieves via vector search | Sub-second retrieval, cosine similarity |
| 6 | Accepts user questions | Simple chat interface |
| 7 | Retrieves + generates answers via LLM | Cited, grounded answers |
| 8 | Answers are primarily based on the source | Out-of-scope questions are declined, not hallucinated |
| 9 | Usable interface | Non-technical user can upload → ask → get an answer in under a minute |

**Definition of done for v1:** a user can upload a PDF, ask 5 questions
about it, get grounded answers with source citations for in-scope
questions, and get an honest "not covered" response for out-of-scope ones
— with zero required configuration.

## 3. Target User

Primarily this is an internship deliverable / portfolio piece, but the
product framing is: **a non-technical knowledge worker** (HR person,
analyst, support agent) who wants to query a handbook, report, or policy
doc without reading the whole thing or pinging someone else.

## 4. Scope

### In scope (v1 — built)
- Multi-format ingestion: PDF, TXT, MD, DOCX
- Recursive, boundary-aware chunking (paragraph → line → sentence → word),
  with configurable size/overlap
- Pluggable embedding backends, auto-selected by what's available:
  1. OpenAI (`text-embedding-3-small`)
  2. Sentence-Transformers (local, free, `all-MiniLM-L6-v2`)
  3. TF-IDF (zero-dependency fallback — always works)
- FAISS vector store (cosine similarity via normalized inner product)
- Pluggable LLM backends, auto-selected:
  1. Anthropic Claude
  2. OpenAI GPT
  3. Extractive fallback (returns top passages directly — no API needed)
- Relevance threshold to reject low-confidence retrievals instead of
  forcing an answer out of noise
- Source citations (filename + page/section) on every answer
- Multi-document indexing (cumulative index across uploads)
- Conversation history (last 3 turns fed back into the prompt)
- Gradio web UI: upload, index, chat, reset
- Smoke tests covering indexing, in-scope Q&A, out-of-scope rejection,
  multi-doc retrieval

### Out of scope (v1)
- User accounts / multi-tenant document isolation
- Persistent database (index currently lives in memory / local FAISS file)
- Fine-grained access control on documents
- Advanced retrieval (hybrid search, re-ranking, HyDE) — noted as future work
- Automated retrieval quality evaluation (e.g. RAGAS) — noted as future work
- Streaming token-by-token responses
- Non-English documents (untested, likely works for TF-IDF/embeddings but not verified)
- OCR for scanned/image-only PDFs — detected explicitly and rejected with a clear error rather than silently indexing nothing

## 5. Key Design Decisions (and why)

| Decision | Rationale |
|---|---|
| Recursive character splitter over fixed-size chunking | Fixed-size chunking cuts mid-sentence and mid-idea, hurting retrieval precision. Splitting on `\n\n` → `\n` → `. ` → words keeps chunks semantically whole where possible. |
| ~900 char chunks, 150 char overlap | Small enough to keep retrieval precise and stay within LLM context budgets across multiple chunks; overlap prevents losing a fact that straddles a chunk boundary. |
| Pluggable embedding + LLM backends with automatic fallback | Makes the app **always runnable** — a reviewer with no API key still sees a working demo (TF-IDF + extractive answers), while a configured key unlocks full semantic quality. This was a deliberate reliability choice, not a shortcut. |
| Cosine similarity via normalized FAISS `IndexFlatIP` | Standard, fast, exact (not approximate) for the small-to-medium corpus sizes this tool targets. |
| Relevance threshold before generation | Directly enforces requirement #8 — without it, retrieval always returns *something*, even if irrelevant, and an LLM will often try to answer anyway. Thresholding forces an honest "not covered" response. |
| Gradio over Streamlit | Faster to stand up a chat-style interface with file upload in one file; trivially deployable to Hugging Face Spaces for a public demo link. |
| Top-k = 4, per-backend relevance threshold (OpenAI 0.25, Sentence-Transformers 0.30, TF-IDF 0.10) | Cosine-similarity score distributions differ by embedding backend, so a single global threshold either over-rejects semantic backends or under-rejects lexical ones. Values chosen empirically, overridable via env vars. |

## 5a. Gap Resolutions (added after PRD review, before implementation)

A review pass on this PRD surfaced six gaps between what was written and what
a reviewer/user would actually hit. All six are addressed in the v1 implementation:

| Gap | Resolution |
|---|---|
| Data privacy — sensitive doc types (contracts/policies) sent to third-party APIs | UI displays a persistent privacy note when OpenAI/Anthropic backends are active; unsetting both API keys keeps everything fully local (TF-IDF + extractive). |
| Retrieval params unspecified | Pinned: top-k = 4, per-backend relevance threshold (see table above), both env-overridable. |
| Duplicate re-uploads | SHA-256 file hash checked before indexing; duplicates are skipped with a clear message, not re-added. |
| Scanned/image-only PDFs silently producing empty answers | Detected at ingestion (near-zero extractable text) and rejected with an explicit "OCR not supported in v1" error rather than indexing nothing. |
| API cost/rate-limit/outage mid-session | Embedding and LLM calls that fail at runtime (not just missing key) trigger an automatic one-time downgrade to the next local-capable backend, with the switch surfaced in the indexing log. |
| No file size bound | `RAG_MAX_FILE_SIZE_MB` (default 20MB) enforced at ingestion; oversized files are rejected before processing. |

## 6. Architecture

```
 ┌────────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────────────┐
 │  Document   │ →  │  Chunking      │ →  │  Embedding    │ →  │  FAISS Vector │
 │  (PDF/TXT/  │    │  (recursive,   │    │  (OpenAI/ST/  │    │  Store        │
 │  DOCX/MD)   │    │   overlap)     │    │  TF-IDF)      │    │               │
 └────────────┘    └───────────────┘    └──────────────┘    └───────┬───────┘
                                                                      │
 ┌────────────┐    ┌───────────────┐    ┌──────────────┐            │
 │   Answer    │ ←  │  LLM Generate  │ ←  │  Top-k        │ ← ── ── ─┘
 │ + Citations │    │  (Claude/GPT/  │    │  Retrieval    │
 │             │    │  extractive)   │    │  (threshold)  │
 └────────────┘    └───────────────┘    └──────────────┘
        ↑
   Gradio UI (upload, chat, reset, status)
```

## 7. Risks / Open Questions

- **TF-IDF quality ceiling:** the zero-key fallback is sparse/lexical, not
  semantic — it'll miss synonym-based matches (e.g. "time off" vs "PTO"
  worded differently). Flagged clearly in the UI; real embeddings fix this.
- **In-memory index by default:** fine for a demo, not for production scale
  or restart-persistence without wiring up `VectorStore.save/load`.
- **No automated retrieval evaluation yet** — currently validated with
  targeted smoke tests, not a scored eval set (see Future Work).

## 8. Future Work (bonus / next iteration)

1. Retrieval evaluation harness (precision@k against a labeled Q&A set)
2. Hybrid retrieval (TF-IDF + dense) with re-ranking
3. Persist FAISS index to disk between sessions (already scaffolded in `VectorStore.save/load`)
4. Deploy to Hugging Face Spaces / Streamlit Cloud for a public demo link
5. Per-document access control if this became multi-user

## 9. Deliverables Checklist

- [x] GitHub-ready repo (`ingestion.py`, `embeddings.py`, `vectorstore.py`,
      `llm_providers.py`, `rag_pipeline.py`, `app.py`, tests, sample docs)
- [x] README with setup + architecture explanation
- [ ] Public working demo link (pending deployment step)
- [x] Source citations
- [x] Conversation history
- [x] Multi-document support
- [x] File size limit, dedup, scanned-PDF rejection, backend-failure fallback (see Section 5a)
- [ ] Formal evaluation / benchmark (stretch)
