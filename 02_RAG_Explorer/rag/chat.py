"""RAG chat: retrieve → format → Ollama."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import ollama

from .embed import embed_query
from .store import Hit, VectorStore

OLLAMA_MODEL = os.getenv("RAG_MODEL", "llama3.2:1b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """You are ShopBot for ShopSphere, an e-commerce store. Answer ONLY using the retrieved context below. If the answer is not in the context, say "I don't have that information in my knowledge base — please contact support@shopsphere.com."

- Be concise (under 150 words).
- Quote exact figures from the context — do not invent numbers, SKUs, or timeframes.
- Cite sources inline like [refund_policy.md].
"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    retrieval_context: list[str]
    hits: list[Hit]
    mode: str
    model: str


def is_ollama_configured() -> bool:
    try:
        ollama.Client(host=OLLAMA_HOST).list()
        return True
    except Exception:
        return False


def answer_with_rag(
    question: str,
    store: VectorStore,
    top_k: int = 4,
    history: Sequence[dict] | None = None,
) -> RagAnswer:
    q_emb = embed_query(question)
    hits = store.search(q_emb, top_k=top_k)
    retrieval_context = [h.text for h in hits]
    sources = sorted({h.source for h in hits})

    context_block = "\n\n".join(
        f"[{h.source} #{h.metadata.get('index')}]\n{h.text}" for h in hits
    ) or "(no documents retrieved)"

    if not is_ollama_configured():
        mock = (
            f"[mock mode — Ollama not running at {OLLAMA_HOST}] Top retrieved chunks: "
            + "; ".join(f"{h.source}#{h.metadata.get('index')}" for h in hits)
        )
        return RagAnswer(
            answer=mock,
            sources=sources,
            retrieval_context=retrieval_context,
            hits=hits,
            mode="mock",
            model="mock",
        )

    client = ollama.Client(host=OLLAMA_HOST)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history or []:
        messages.append(h)
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nRetrieved context:\n{context_block}",
    })
    
    completion = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0.2,
        }
    )
    return RagAnswer(
        answer=completion["message"]["content"],
        sources=sources,
        retrieval_context=retrieval_context,
        hits=hits,
        mode="live",
        model=OLLAMA_MODEL,
    )
