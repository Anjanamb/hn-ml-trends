"""Subtopic categories for Hacker News (AI/ML slice) + gold-label I/O.

The label set is deliberately small (~7 categories) so gold labelling
stays tractable and classification metrics are stable on a few-hundred
example set.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# Category definitions. Kept short and unambiguous; used both in the
# LLM prompt (nb04) and as reference for the human labeller (nb02).
CATEGORIES: dict[str, str] = {
    "research":   "New research paper, arXiv link, or discussion of academic results.",
    "product":    "A product launch, Show HN, startup announcement, or feature release.",
    "tool":       "An open-source library, framework, dataset, benchmark, or model release.",
    "tutorial":   "A blog post, walkthrough, guide, or explanatory article.",
    "opinion":    "Opinion, essay, hot take, or industry commentary.",
    "news":       "Company / industry news, funding round, acquisition, personnel change.",
    "other":      "Anything that does not clearly fit above (jobs, memes, off-topic).",
}


def category_names() -> list[str]:
    return list(CATEGORIES.keys())


def prompt_snippet() -> str:
    """Human-readable multi-line description used in LLM prompts."""
    lines = ["Categories:"]
    for name, desc in CATEGORIES.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def load_gold(path: str | Path) -> pd.DataFrame:
    """Load the hand-labelled gold set from CSV.

    Expected columns: ``id, category`` (plus any extras kept for
    reference). Returns a DataFrame indexed by story id.
    """
    df = pd.read_csv(path)
    required = {"id", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gold CSV missing columns: {sorted(missing)}")
    unknown = set(df["category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown labels in gold set: {sorted(unknown)}")
    return df.set_index("id")


def save_gold(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a gold-label DataFrame to CSV."""
    out = df.reset_index() if df.index.name == "id" else df
    out.to_csv(path, index=False)
