"""LangChain wrapper around a local Ollama model.

Reads the model name and base URL from environment variables so it
matches ``ollama list`` on the host.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_URL = "http://localhost:11434"


def get_chat_ollama(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatOllama:
    """Return a LangChain ``ChatOllama`` client.

    Reads ``OLLAMA_MODEL`` and ``OLLAMA_BASE_URL`` from the environment
    (defaults are the standard local install). Temperature defaults to
    0 so classification calls are reproducible.
    """
    load_dotenv()
    return ChatOllama(
        model=model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        base_url=base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_URL),
        temperature=temperature,
        **kwargs,
    )
