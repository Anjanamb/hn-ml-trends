"""LangGraph state machine that orchestrates the trend-tracker pipeline.

Nodes (roughly):
    fetch         ->  pull the latest Hacker News stories
    classify      ->  LLM classifier assigns each story a category
    embed_cluster ->  sentence-transformer embeddings + KMeans / HDBSCAN
    detect_trend  ->  compare current window vs previous window shares
    drill_down    ->  (conditional) if any category delta > threshold,
                       fetch more stories on that topic and re-summarise
    summarise     ->  LLM writes the final human-readable trend report

The full graph is assembled in notebook 06; this module hosts the
state type + pure node functions so they are unit-testable.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    """The state carried between LangGraph nodes.

    Kept as a TypedDict rather than a dataclass because LangGraph
    serialises state between nodes and TypedDicts play nicer with
    JSON round-trips.
    """

    # Ingested stories (populated by ``node_fetch``).
    stories: pd.DataFrame
    # Per-story category predictions (populated by ``node_classify``).
    predictions: list[str]
    # Sentence embeddings + cluster assignments.
    embeddings: Any               # ndarray, shape (N, dim)
    clusters: list[int]
    # Category share deltas between current and previous window.
    trend: pd.DataFrame
    # Names of subtopics flagged for drill-down.
    drill_categories: list[str]
    # Final LLM-generated trend report.
    report: str
    # Working thresholds and knobs (defaults set at graph construction).
    drill_threshold: float


DEFAULT_DRILL_THRESHOLD = 0.05    # 5-percentage-point share swing
