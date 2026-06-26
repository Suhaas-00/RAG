"""Groq LLM wrapper with reliable .env loading and diagnostics."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

FALLBACK = "Not found in the document"


def load_groq_api_key(start: Path | None = None) -> str | None:
    root = (start or Path.cwd()).resolve()
    candidates = [root / ".env", *root.parents]
    env_files = [path if path.name == ".env" else path / ".env" for path in candidates]
    loaded_from = None
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)
            loaded_from = env_file
            break
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print(f"[diagnostic] GROQ_API_KEY is missing. Checked .env from {root} upward.", flush=True)
        if loaded_from:
            print(f"[diagnostic] Loaded .env file: {loaded_from}", flush=True)
    return key


def answer_with_groq(question: str, context: str, *, model: str = "llama-3.1-8b-instant") -> str:
    api_key = load_groq_api_key()
    if not api_key:
        return context or FALLBACK

    from groq import Groq

    system = (
        "You are a grounded biomedical RAG assistant. Answer only from the supplied context. "
        "Every answer must cite source, section, page number, and confidence. "
        f"If the context is insufficient, answer exactly: {FALLBACK}."
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer with concise medical language and include citations in this form: "
        "(Source: file.pdf; Section: section; Page: n; Confidence: x)."
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip() or FALLBACK

