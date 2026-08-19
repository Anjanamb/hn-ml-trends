"""Classifiers used by nb03 (TF-IDF baseline) and nb04 (Ollama LLM).

Both classifier heads share the same interface: fit / predict on a
list of strings, return a list of category names from
``src.labels.CATEGORIES``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.labels import CATEGORIES, prompt_snippet


# ---------------------------------------------------------------- baseline

def tfidf_logreg_pipeline(
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
) -> Pipeline:
    """TF-IDF + logistic regression, wrapped in a sklearn Pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced",
        )),
    ])


# ---------------------------------------------------------------- LLM head

CLASSIFY_PROMPT_TEMPLATE = """You classify Hacker News story submissions into one of a fixed set of categories.

{categories}

Rules:
- Reply with exactly one category name from the list above. Nothing else.
- No explanations, no punctuation, no markdown, no quotes.

Story title: {title}

Story text (may be empty; when empty the URL is external):
{text}

Category:"""


@dataclass
class LLMClassifier:
    """Thin wrapper over a LangChain chat model that emits a category name."""

    chat: object   # a LangChain BaseChatModel

    def _predict_one(self, title: str, text: str) -> str:
        from langchain_core.messages import HumanMessage
        prompt = CLASSIFY_PROMPT_TEMPLATE.format(
            categories=prompt_snippet(),
            title=title,
            text=text[:1500] if text else "(no body; external URL)",
        )
        resp = self.chat.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip().splitlines()[0].strip().lower()
        # Snap to the closest known label; default to "other" if the LLM
        # replies with something outside the set.
        for name in CATEGORIES:
            if raw == name or raw.startswith(name):
                return name
        return "other"

    def predict(self, titles: Iterable[str], texts: Iterable[str]) -> list[str]:
        return [self._predict_one(t, x) for t, x in zip(titles, texts)]
