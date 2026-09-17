import pytest

from embeddings import TfidfEmbeddingBackend
from llm_providers import ExtractiveProvider
from rag_pipeline import RagPipeline

HANDBOOK = """# Leave Policy

Employees receive 20 days of paid time off (PTO) per year, accrued monthly.
Unused PTO up to 5 days may roll over to the next calendar year.

# Expense Policy

Travel expenses must be submitted within 30 days with receipts attached.
"""

EXPENSES = "# Refunds\n\nApproved refunds are processed within 14 business days.\n"


@pytest.fixture
def pipeline():
    # Deterministic, offline, no API keys needed - the point of the
    # TF-IDF + extractive fallback tier (PRD requirement #4/#7).
    return RagPipeline(embedding_backend=TfidfEmbeddingBackend(), llm_provider=ExtractiveProvider())


@pytest.fixture
def handbook_file(tmp_path):
    p = tmp_path / "handbook.md"
    p.write_text(HANDBOOK)
    return str(p)


@pytest.fixture
def expenses_file(tmp_path):
    p = tmp_path / "expenses.md"
    p.write_text(EXPENSES)
    return str(p)


def test_index_and_in_scope_question_is_grounded_with_citations(pipeline, handbook_file):
    pipeline.index_document(handbook_file)
    result = pipeline.ask("How many PTO days do employees get?")
    assert "20" in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["source"] == "handbook.md"


def test_out_of_scope_question_is_declined_not_hallucinated(pipeline, handbook_file):
    pipeline.index_document(handbook_file)
    result = pipeline.ask("What is the capital of France?")
    assert result["citations"] == []
    assert "couldn't find" in result["answer"].lower()


def test_duplicate_upload_is_skipped(pipeline, handbook_file):
    pipeline.index_document(handbook_file)
    msg2 = pipeline.index_document(handbook_file)
    assert "already indexed" in msg2.lower()
    assert pipeline.status()["num_documents"] == 1


def test_multi_document_retrieval_finds_the_right_source(pipeline, handbook_file, expenses_file):
    pipeline.index_document(handbook_file)
    pipeline.index_document(expenses_file)
    result = pipeline.ask("How long do refunds take?")
    assert any(c["source"] == "expenses.md" for c in result["citations"])


def test_reset_clears_index_and_history(pipeline, handbook_file):
    pipeline.index_document(handbook_file)
    pipeline.reset()
    assert pipeline.status()["num_documents"] == 0
    result = pipeline.ask("Anything?")
    assert "no documents" in result["answer"].lower()


def test_empty_question_is_handled_without_erroring(pipeline, handbook_file):
    pipeline.index_document(handbook_file)
    result = pipeline.ask("   ")
    assert result["citations"] == []


def test_save_and_load_index_restores_state_across_pipelines(pipeline, handbook_file, tmp_path):
    persist_path = str(tmp_path / "store" / "index")
    pipeline.index_document(handbook_file)
    pipeline.save_index(persist_path)

    restored = RagPipeline(embedding_backend=TfidfEmbeddingBackend(), llm_provider=ExtractiveProvider())
    loaded = restored.load_index(persist_path)
    assert loaded is True
    assert restored.status()["num_documents"] == 1

    result = restored.ask("How many PTO days do employees get?")
    assert "20" in result["answer"]
    assert result["citations"][0]["source"] == "handbook.md"


def test_load_index_returns_false_when_nothing_persisted(pipeline, tmp_path):
    assert pipeline.load_index(str(tmp_path / "missing" / "index")) is False


def test_load_index_returns_false_on_embedding_backend_mismatch(pipeline, handbook_file, tmp_path):
    persist_path = str(tmp_path / "store" / "index")
    pipeline.index_document(handbook_file)
    pipeline.save_index(persist_path)

    class OtherBackend(TfidfEmbeddingBackend):
        name = "other-backend"

    mismatched = RagPipeline(embedding_backend=OtherBackend(), llm_provider=ExtractiveProvider())
    assert mismatched.load_index(persist_path) is False
    assert mismatched.status()["num_documents"] == 0


def test_reset_clears_the_persisted_index(pipeline, handbook_file, tmp_path):
    persist_path = str(tmp_path / "store" / "index")
    pipeline.index_document(handbook_file)
    pipeline.save_index(persist_path)

    pipeline.clear_persisted_index(persist_path)

    fresh = RagPipeline(embedding_backend=TfidfEmbeddingBackend(), llm_provider=ExtractiveProvider())
    assert fresh.load_index(persist_path) is False
