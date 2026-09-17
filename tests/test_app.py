import app
from embeddings import TfidfEmbeddingBackend
from llm_providers import ExtractiveProvider
from rag_pipeline import RagPipeline


def test_app_ui_handlers(tmp_path):
    # Initialize pipeline with deterministic offline components
    app.pipeline = RagPipeline(
        embedding_backend=TfidfEmbeddingBackend(),
        llm_provider=ExtractiveProvider(),
    )

    # Test upload with temporary file
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("The secret project code is Nebula-42.")
    log, status = app.handle_upload([str(sample_file)])
    assert "Indexed 'sample.txt'" in log
    assert "1" in status

    # Test handle_chat produces dictionary messages format for Gradio
    chat_history, empty_input = app.handle_chat("What is the project code?", [])
    assert empty_input == ""
    assert len(chat_history) == 2
    assert chat_history[0] == {"role": "user", "content": "What is the project code?"}
    assert chat_history[1]["role"] == "assistant"
    assert "Nebula-42" in chat_history[1]["content"]

    # Test reset handler
    chat_history, status, reset_msg = app.handle_reset()
    assert chat_history == []
    assert reset_msg == "Index reset."
