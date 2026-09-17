"""Pluggable LLM backends, auto-selected by what's available
(PRD requirement #7): Anthropic Claude -> OpenAI GPT -> extractive fallback.
"""
import os

import config


def build_prompt(question, context_chunks, history):
    context_block = "\n\n".join(
        f"[{i + 1}] (Source: {c['source']}" + (f", page {c['page']}" if c.get("page") else "") + f")\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )
    history_block = ""
    if history:
        turns = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        history_block = f"Previous conversation:\n{turns}\n\n"

    return (
        "You are a document Q&A assistant. Answer the question using ONLY the "
        "context passages below, and cite them by their [number]. If the context "
        "does not contain enough information to answer, say so explicitly instead "
        "of guessing or using outside knowledge.\n\n"
        f"{history_block}"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


class LLMProvider:
    name = "base"

    def generate(self, question, context_chunks, history):
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = config.ANTHROPIC_MODEL

    def generate(self, question, context_chunks, history):
        prompt = build_prompt(question, context_chunks, history)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = config.OPENAI_LLM_MODEL

    def generate(self, question, context_chunks, history):
        prompt = build_prompt(question, context_chunks, history)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class ExtractiveProvider(LLMProvider):
    """No-API-needed fallback: returns the retrieved passages directly
    instead of generating a synthesized answer."""

    name = "extractive"

    def generate(self, question, context_chunks, history):
        lines = ["No LLM is configured, so here are the most relevant passages found:\n"]
        for i, c in enumerate(context_chunks, 1):
            loc = c["source"] + (f", p.{c['page']}" if c.get("page") else "")
            lines.append(f"{i}. ({loc}) {c['text'].strip()}")
        return "\n\n".join(lines)


def get_llm_provider():
    """Auto-select the best available provider. Only catches missing-package
    / missing-key failures at construction time; runtime API failures (bad
    key, rate limit, outage) are handled by the caller degrading per-call."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except Exception:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except Exception:
            pass
    return ExtractiveProvider()
