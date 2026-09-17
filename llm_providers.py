"""Pluggable LLM backends, auto-selected by what's available
(PRD requirement #7): Anthropic Claude -> OpenAI GPT -> extractive fallback.
"""
import os
import re

import config

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
}


def _words(text):
    return set(re.findall(r"\w+", text.lower())) - _STOPWORDS


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


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
    instead of generating a synthesized answer, trimmed to the sentences
    that actually match the question so it reads like an answer rather
    than a wall of chunk text."""

    name = "extractive"
    MAX_SENTENCES = 2

    def generate(self, question, context_chunks, history):
        query_words = _words(question)
        lines = ["No LLM is configured, so here are the most relevant excerpts found:\n"]
        for i, c in enumerate(context_chunks, 1):
            loc = c["source"] + (f", p.{c['page']}" if c.get("page") else "")
            excerpt = self._best_excerpt(c["text"], query_words)
            lines.append(f"{i}. ({loc}) {excerpt}")
        return "\n\n".join(lines)

    def _best_excerpt(self, text, query_words):
        sentences = _split_sentences(text)
        if len(sentences) <= self.MAX_SENTENCES or not query_words:
            return text.strip()

        scored = sorted(
            range(len(sentences)),
            key=lambda i: len(_words(sentences[i]) & query_words),
            reverse=True,
        )
        if len(_words(sentences[scored[0]]) & query_words) == 0:
            return text.strip()  # no keyword overlap anywhere - don't guess which part matters

        top_idx = sorted(scored[: self.MAX_SENTENCES])
        return " ".join(sentences[i] for i in top_idx)


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
