"""Gradio UI: upload documents, index them, ask grounded questions, reset."""
import gradio as gr

from ingestion import DocumentError
from rag_pipeline import RagPipeline

pipeline = RagPipeline()

PRIVACY_NOTE = (
    "_Privacy: when the OpenAI/Anthropic backends are active, document text and "
    "questions are sent to that provider's API for embedding/generation. Leave "
    "`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` unset to keep everything fully "
    "local (TF-IDF + extractive answers)._"
)


def _status_text():
    s = pipeline.status()
    return (
        f"**Embedding backend:** {s['embedding_backend']} &nbsp;|&nbsp; "
        f"**LLM backend:** {s['llm_backend']} &nbsp;|&nbsp; "
        f"**Documents indexed:** {s['num_documents']} &nbsp;|&nbsp; "
        f"**Chunks:** {s['num_chunks']}\n\n{PRIVACY_NOTE}"
    )


def handle_upload(files):
    if not files:
        return "No files selected.", _status_text()
    messages = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            messages.append(pipeline.index_document(path))
        except DocumentError as e:
            messages.append(f"Skipped: {e}")
    return "\n".join(messages), _status_text()


def handle_chat(message, chat_history):
    if not message or not message.strip():
        return chat_history, ""
    if chat_history is None:
        chat_history = []
    result = pipeline.ask(message)
    answer = result["answer"]
    if result["citations"]:
        cite_lines = "\n".join(
            f"- {c['source']}" + (f", p.{c['page']}" if c.get("page") else "") + f" (relevance {c['score']})"
            for c in result["citations"]
        )
        answer = f"{answer}\n\n**Sources:**\n{cite_lines}"
    
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""


def handle_reset():
    pipeline.reset()
    return [], _status_text(), "Index reset."


with gr.Blocks(title="Mini RAG Assistant") as demo:
    gr.Markdown(
        "# Mini AI Knowledge Assistant\n"
        "Upload a document (PDF / TXT / MD / DOCX), then ask questions grounded in its content. "
        "Out-of-scope questions are declined rather than guessed at."
    )
    status_box = gr.Markdown(_status_text())

    with gr.Row():
        file_input = gr.File(
            label="Upload document(s)",
            file_count="multiple",
            file_types=[".pdf", ".txt", ".md", ".docx"],
        )
        index_btn = gr.Button("Index document(s)", variant="primary")
    index_log = gr.Textbox(label="Indexing log", interactive=False)

    chatbot = gr.Chatbot(label="Ask questions about your documents", height=400)
    question_box = gr.Textbox(label="Your question", placeholder="e.g. What is the PTO policy?")
    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        reset_btn = gr.Button("Reset index")

    index_btn.click(handle_upload, inputs=file_input, outputs=[index_log, status_box])
    ask_btn.click(handle_chat, inputs=[question_box, chatbot], outputs=[chatbot, question_box])
    question_box.submit(handle_chat, inputs=[question_box, chatbot], outputs=[chatbot, question_box])
    reset_btn.click(handle_reset, outputs=[chatbot, status_box, index_log])

if __name__ == "__main__":
    demo.launch()
